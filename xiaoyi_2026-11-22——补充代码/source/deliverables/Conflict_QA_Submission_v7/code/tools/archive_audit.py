"""Read-only archive integrity checks, using explicit historical namespaces."""

import collections
import gzip
import json


def records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def resolve_reference(reference, archived_paths, aliases, owner=""):
    # Corrected records were saved under corrected/ but kept run-relative links.
    if owner.startswith("corrected/") and "corrected/" + reference in archived_paths:
        return "corrected/" + reference
    if reference in archived_paths:
        return reference
    return aliases.get(reference, reference)


def audit_package(base):
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    inputs = read(base / "data/inputs.json")
    by_question = {row["question_id"]: row for row in inputs}
    labels = read(base / "data/evaluation.json")
    errors, request_keys = [], set()
    requests = 0
    for file in sorted((base / "data").glob("requests*.jsonl.gz")):
        for record in records(file):
            requests += 1
            if record["key"] in request_keys:
                errors.append("Duplicate request key: " + record["key"])
            request_keys.add(record["key"])
            choice = record["response"]["choices"][0]
            if (
                record["text"] != choice["message"]["content"]
                or record["finish_reason"] != choice["finish_reason"]
            ):
                errors.append("Request/response mismatch: " + record["key"])
    stages = {}
    for record in records(base / "data/stages.jsonl.gz"):
        if record["path"] in stages:
            errors.append("Duplicate stage path: " + record["path"])
        stages[record["path"]] = record["output"]
    aliases = read(base / "provenance/archive_aliases.json")["prediction_sources"]
    paths = set(stages)
    mapped_links = set()
    links = set()
    for path, stage in stages.items():
        for request in stage.get("requests", []):
            if request["key"] not in request_keys:
                errors.append("Stage has missing request: " + path)
        for reference in stage.get("upstream_outputs", []):
            links.add((path, reference))
            target = resolve_reference(reference, paths, aliases, owner=path)
            if target != reference:
                mapped_links.add((path, reference))
            if target not in paths:
                errors.append(
                    "Stage has missing upstream: " + path + " -> " + reference
                )
    predictions = list(records(base / "data/predictions.jsonl.gz"))
    seen = set()
    mapped_sources = 0
    methods = {"ours_1024", "FR4_M03", "WR5_M02", "WR3_M01", "WR2_M08", "WR1_M05"}
    for row in predictions:
        pair = (row["question_id"], row["method"])
        if pair in seen:
            errors.append("Duplicate prediction: " + str(pair))
        seen.add(pair)
        if (
            row["question_id"] not in labels
            or row["dataset"] != labels[row["question_id"]]["dataset"]
        ):
            errors.append("Prediction label dataset mismatch: " + str(pair))
        source = resolve_reference(row["source"], paths, aliases)
        mapped_sources += source != row["source"]
        stage = stages.get(source)
        if stage is None or any(
            row[k] != stage[k] for k in ["question_id", "accepted_answers", "status"]
        ):
            errors.append("Prediction source mismatch: " + row["source"])
    expected = {(q, method) for q in by_question for method in methods}
    if seen != expected:
        errors.append("Prediction question/method coverage mismatch")
    if len(by_question) != len(inputs) or set(by_question) != set(labels):
        errors.append("Input question/label ID mismatch")
    # Check source-local quotes; this is not an entailment or truth judgment.
    card_count = 0
    for q, row in by_question.items():
        prefix = (
            "runs/phase3f-transfer-v3/ramdocs/F_C1/"
            if q.startswith("ramdocs-")
            else "runs/cross_dataset_v1/outputs/extract/"
        )
        reference = prefix + q + ".json"
        corrected = "corrected/runs/cross_dataset_v1/outputs/extract/" + q + ".json"
        extraction = stages.get(corrected, stages.get(reference))
        if extraction is None:
            errors.append("Missing extraction: " + q)
            continue
        documents = {
            d["doc_id"]: (i, d["text"]) for i, d in enumerate(row["documents"], 1)
        }
        for card in extraction.get("cards", []):
            card_count += 1
            document = documents.get(card["doc_id"])
            if (
                not document
                or not card.get("quote")
                or card["quote"] not in document[1]
            ):
                errors.append("Card document/quote mismatch: " + q)
                continue
            if (
                "span_start" in card
                and document[1][card["span_start"] : card["span_end"]] != card["quote"]
            ):
                errors.append("Card character offset mismatch: " + q)
            if (
                "original_document_number" in card
                and card["original_document_number"] != document[0]
            ):
                errors.append("Card document number mismatch: " + q)
    return dict(
        passed=not errors,
        errors=errors,
        questions=len(inputs),
        documents=sum(len(row["documents"]) for row in inputs),
        dataset_questions=dict(
            collections.Counter(labels[q]["dataset"] for q in by_question)
        ),
        predictions=len(predictions),
        stages=len(stages),
        requests=requests,
        cards_checked=card_count,
        upstream_links=len(links),
        resolved_namespaced_links=len(mapped_links),
        resolved_prediction_sources=mapped_sources,
        scope="Frozen six-method archives only; no fresh inference or baseline reconstruction",
    )
