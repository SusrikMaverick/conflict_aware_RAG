"""Load completed C/D card artifacts without changing their content or provenance."""

import json
from pathlib import Path
import re

from src.card_extraction import norm, segments, source_span
from src.compare import FIELDS, digest


def _question_map(rows, label):
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("question_id"), str) or not row["question_id"]:
            raise ValueError(f"{label} contains an invalid question_id")
        if row["question_id"] in result:
            raise ValueError(f"Duplicate question_id in {label}: {row['question_id']}")
        result[row["question_id"]] = row
    return result


def _raw_cards(path, cases):
    """Validate one complete artifact, including all full source card records."""
    path = Path(path).resolve()
    try:
        manifest_bytes = (path / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        cards_bytes = (path / "cards.jsonl").read_bytes()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read card artifact at {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise ValueError("Card artifact manifest must be complete")
    if manifest.get("stage") not in ("extract", "review"):
        raise ValueError("Card artifact stage must be extract or review")
    if not isinstance(manifest.get("model"), str) or not manifest["model"]:
        raise ValueError("Card artifact must identify its model")
    if not isinstance(manifest.get("bundle_hashes"), dict):
        raise ValueError("Card artifact must record its bundle_hashes")
    if not isinstance(manifest.get("cost", {}), dict):
        raise ValueError("Card artifact cost must be an object")
    observed_hash = digest(cards_bytes)
    if manifest.get("cards_sha256") != observed_hash:
        raise ValueError("Card artifact cards.jsonl hash differs from its manifest")
    expected = _question_map(cases, "requested cases")
    question_ids = manifest.get("question_ids")
    if (not isinstance(question_ids, list) or any(not isinstance(qid, str) for qid in question_ids)
            or len(question_ids) != len(set(question_ids)) or set(question_ids) != set(expected)):
        raise ValueError("Card artifact manifest question IDs differ from the requested cases")
    if "questions" in manifest and manifest["questions"] != len(expected):
        raise ValueError("Card artifact question count differs from the requested cases")
    try:
        rows = _question_map((json.loads(line) for line in cards_bytes.splitlines() if line.strip()), "card rows")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid card artifact rows: {exc}") from exc
    if set(rows) != set(expected):
        raise ValueError("Card artifact rows do not cover the requested question IDs exactly")
    seen_cards = set()
    for qid, case in expected.items():
        row = rows[qid]
        if row.get("source_status") != "complete":
            raise ValueError(f"Card artifact contains an incomplete or errored question: {qid}")
        if row.get("question") != case["question"]:
            raise ValueError(f"Card artifact question text differs: {qid}")
        if row.get("documents_sha256") != digest(case["documents"]):
            raise ValueError(f"Card artifact document fingerprint differs: {qid}")
        documents = {}
        for document in case["documents"]:
            doc_id = document.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id or doc_id in documents or not isinstance(document.get("text"), str):
                raise ValueError(f"Requested case has invalid or duplicate documents: {qid}")
            documents[doc_id] = document
        cards = row.get("cards")
        if not isinstance(cards, list):
            raise ValueError(f"Card artifact cards must be a list: {qid}")
        per_document = {}
        for card in cards:
            if not isinstance(card, dict):
                raise ValueError(f"Card artifact contains a non-object card: {qid}")
            doc_id, card_id = card.get("doc_id"), card.get("card_id")
            if not isinstance(doc_id, str) or doc_id not in documents:
                raise ValueError(f"Card artifact card belongs to an unknown document: {qid}")
            if (not isinstance(card_id, str) or not re.fullmatch(re.escape(doc_id) + r":c[0-2]", card_id)
                    or card_id in seen_cards):
                raise ValueError(f"Card artifact has an invalid or duplicate card_id: {qid}")
            seen_cards.add(card_id)
            per_document[doc_id] = per_document.get(doc_id, 0) + 1
            if per_document[doc_id] > 3:
                raise ValueError(f"Card artifact exceeds the three-card source limit: {qid}")
            if any(not isinstance(card.get(key), str) for key in ("answer", "entity", "scope", "claim", "quote")):
                raise ValueError(f"Card artifact fields must be strings: {card_id}")
            if not card["answer"].strip() or not card["claim"].strip() or norm(card["answer"]) not in norm(card["claim"]):
                raise ValueError(f"Card artifact answer/claim fails the extraction rule: {card_id}")
            document = documents[doc_id]
            span = source_span(card.get("source_ids"), segments(document["text"]), document["text"])
            if (card["quote"] != span["quote"]
                    or type(card.get("span_start")) is not int or type(card.get("span_end")) is not int
                    or card["span_start"] != span["span_start"] or card["span_end"] != span["span_end"]):
                raise ValueError(f"Card artifact quote or offsets differ from the source: {card_id}")
    return path, manifest, rows, digest(manifest_bytes), observed_hash


def load_card_set(path, cases, *, expected_stage=None, expected_model=None):
    """Load verified artifact records and return projected answerer cases + metadata.

    A D artifact is accepted only when its accessible C parent still matches the
    recorded hashes, model, input set and unchanged ordered card subsets.
    """
    path, manifest, rows, manifest_hash, cards_hash = _raw_cards(path, cases)
    stage, model = manifest["stage"], manifest["model"]
    if expected_stage is not None and stage != expected_stage:
        raise ValueError(f"Expected card stage {expected_stage}, found {stage}")
    if expected_model is not None and model != str(expected_model):
        raise ValueError(f"Card artifact model differs: expected {expected_model}, found {model}")
    parent_manifest = None
    if stage == "review":
        parent = manifest.get("parent")
        if not isinstance(parent, dict) or not isinstance(parent.get("path"), str) or not parent["path"]:
            raise ValueError("Reviewed cards require their C parent artifact")
        parent_path = Path(parent["path"])
        if not parent_path.is_absolute():
            parent_path = path / parent_path
        _, parent_manifest, parent_rows, parent_manifest_hash, parent_cards_hash = _raw_cards(parent_path, cases)
        if parent_manifest["stage"] != "extract":
            raise ValueError("Reviewed cards must have an extraction-stage C parent")
        if parent.get("manifest_sha256") != parent_manifest_hash or parent.get("cards_sha256") != parent_cards_hash:
            raise ValueError("Reviewed card parent hashes differ from the recorded C artifact")
        if parent_manifest["model"] != model:
            raise ValueError("Reviewed cards and their C parent must use the same model")
        if parent_manifest["bundle_hashes"] != manifest["bundle_hashes"]:
            raise ValueError("Reviewed cards and their C parent use different bundles")
        for qid, row in rows.items():
            selected_ids = {card["card_id"] for card in row["cards"]}
            expected_cards = [card for card in parent_rows[qid]["cards"] if card["card_id"] in selected_ids]
            if [digest(card) for card in row["cards"]] != [digest(card) for card in expected_cards]:
                raise ValueError(f"Reviewed cards must be unchanged ordered subsets of C: {qid}")
    updated = []
    for case in cases:
        numbers = {doc["doc_id"]: index for index, doc in enumerate(case["documents"], 1)}
        cards = [{**{key: card[key] for key in FIELDS},
                  "original_document_number": numbers[card["doc_id"]]}
                 for card in rows[case["question_id"]]["cards"]]
        updated.append({**case, "cards": cards, "source_status": "complete"})
    extraction_model = parent_manifest["model"] if parent_manifest else model
    metadata = {"card_variant": "C" if stage == "extract" else "D",
                "extraction_model": extraction_model,
                "card_policy": "All newly extracted source-located cards" if stage == "extract" else "Question-fit checked, unchanged subset of C cards",
                "card_source": {"path": str(path), "manifest_sha256": manifest_hash,
                                "cards_sha256": cards_hash, "stage": stage, "model": model},
                "card_stage_cost": manifest.get("cost", {}), "card_stage_manifest": manifest}
    if parent_manifest is not None:
        metadata["card_parent_cost"] = parent_manifest.get("cost", {})
    return updated, metadata
