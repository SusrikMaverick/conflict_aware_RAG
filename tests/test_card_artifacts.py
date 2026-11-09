import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.card_artifacts import load_card_set
from src.card_extraction import segments, source_span
from src.compare import FIELDS, digest


def examples():
    return [{"question_id": "q1", "question": "When was the courthouse built?",
             "documents": [{"doc_id": "q1:d0", "text": "The courthouse was built in 1902. It was restored in 1950."}],
             "cards": [], "gold_answers": ["SCORING_ONLY_GOLD"], "wrong_answers": ["SCORING_ONLY_WRONG"],
             "source_status": "old-status"},
            {"question_id": "q2", "question": "When was the library built?",
             "documents": [{"doc_id": "q2:d0", "text": "The library was built in 1910."}],
             "cards": [], "gold_answers": ["1910"], "wrong_answers": [], "source_status": "old-status"}]


def source_card(case, *, index=0, source_id="s0", answer="1902", claim=None):
    doc = case["documents"][0]
    span = source_span([source_id], segments(doc["text"]), doc["text"])
    return {"card_id": f"{doc['doc_id']}:c{index}", "doc_id": doc["doc_id"],
            "answer": answer, "entity": "courthouse", "scope": "built",
            "claim": claim or f"The courthouse was built in {answer}.", **span,
            "support_status": "source_located_only", "verification": {"status": "not_run", "probability": None}}


def write_artifact(path, cases, *, stage="extract", cards=None, parent=None, model="qwen72b"):
    path.mkdir(parents=True, exist_ok=True)
    cards = cards if cards is not None else {"q1": [source_card(cases[0])], "q2": []}
    rows = [{"question_id": case["question_id"], "question": case["question"],
             "documents_sha256": digest(case["documents"]), "cards": cards[case["question_id"]],
             "source_status": "complete"} for case in reversed(cases)]
    card_file = path / "cards.jsonl"
    card_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {"status": "complete", "stage": stage, "model": model,
                "question_ids": [case["question_id"] for case in cases], "questions": len(cases),
                "bundle_hashes": {"inputs": "example-sha"}, "cards_sha256": digest(card_file.read_bytes()),
                "cost": {"requests": 2, "prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200}}
    if parent is not None:
        manifest["parent"] = {"path": str(parent.resolve()),
                              "manifest_sha256": digest((parent / "manifest.json").read_bytes()),
                              "cards_sha256": digest((parent / "cards.jsonl").read_bytes())}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return rows, manifest


def rewrite_rows(path, rows):
    (path / "cards.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cards_sha256"] = digest((path / "cards.jsonl").read_bytes())
    manifest_path.write_text(json.dumps(manifest, indent=2))


class CardArtifactTests(unittest.TestCase):
    def test_c_loads_by_id_projects_only_final_fields_and_preserves_scoring_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            cases = examples()
            original = copy.deepcopy(cases)
            write_artifact(path, cases)
            updated, metadata = load_card_set(path, cases, expected_stage="extract", expected_model="qwen72b")
            self.assertEqual(cases, original)
            self.assertEqual(updated[0]["cards"][0]["answer"], "1902")
            self.assertEqual(set(updated[0]["cards"][0]), set(FIELDS) | {"original_document_number"})
            self.assertEqual(updated[0]["cards"][0]["original_document_number"], 1)
            self.assertEqual(updated[0]["gold_answers"], ["SCORING_ONLY_GOLD"])
            self.assertEqual(updated[1]["cards"], [])
            self.assertEqual(metadata["card_variant"], "C")
            self.assertEqual(metadata["card_stage_cost"]["total_tokens"], 200)

    def test_wrong_model_or_stage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            cases = examples()
            write_artifact(path, cases)
            for kwargs in ({"expected_model": "another-model"}, {"expected_stage": "review"}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    load_card_set(path, cases, **kwargs)

    def test_file_hash_and_incomplete_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            cases = examples()
            write_artifact(path, cases)
            (path / "cards.jsonl").write_text("[]\n")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                load_card_set(path, cases)
            _, manifest = write_artifact(path, cases)
            manifest["status"] = "failed"
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "must be complete"):
                load_card_set(path, cases)

    def test_missing_duplicate_and_errored_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            cases = examples()
            rows, _ = write_artifact(path, cases)
            bad_sets = [rows[:1], rows + [rows[0]], [{**row, "source_status": "error"} for row in rows]]
            for bad in bad_sets:
                rewrite_rows(path, bad)
                with self.subTest(rows=bad), self.assertRaises(ValueError):
                    load_card_set(path, cases)

    def test_changed_question_documents_or_source_quote_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            cases = examples()
            for field in ("question", "documents", "quote", "doc_id", "duplicate_card"):
                rows, _ = write_artifact(path, cases)
                q1 = next(row for row in rows if row["question_id"] == "q1")
                if field == "question":
                    q1["question"] = "A different question"
                elif field == "documents":
                    q1["documents_sha256"] = "different-sha"
                elif field == "quote":
                    q1["cards"][0]["quote"] = "Fabricated source quote"
                elif field == "doc_id":
                    q1["cards"][0]["doc_id"] = "another-document"
                else:
                    q1["cards"].append(copy.deepcopy(q1["cards"][0]))
                rewrite_rows(path, rows)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_card_set(path, cases)

    def test_d_requires_accessible_unchanged_parent_and_loads_unchanged_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = examples()
            c, d = root / "c", root / "d"
            first = source_card(cases[0])
            second = source_card(cases[0], index=1, source_id="s1", answer="1950", claim="The courthouse was restored in 1950.")
            write_artifact(c, cases, cards={"q1": [first, second], "q2": []})
            write_artifact(d, cases, stage="review", cards={"q1": [second], "q2": []}, parent=c)
            updated, metadata = load_card_set(d, cases, expected_stage="review")
            self.assertEqual(updated[0]["cards"][0]["answer"], "1950")
            self.assertEqual(metadata["card_variant"], "D")
            self.assertEqual(metadata["extraction_model"], "qwen72b")
            (c / "manifest.json").write_text((c / "manifest.json").read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "parent hashes differ"):
                load_card_set(d, cases)

    def test_d_cannot_rewrite_card_metadata_add_cards_or_reorder_kept_cards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = examples()
            c, d = root / "c", root / "d"
            first = source_card(cases[0])
            second = source_card(cases[0], index=1, source_id="s1", answer="1950", claim="The courthouse was restored in 1950.")
            write_artifact(c, cases, cards={"q1": [first, second], "q2": []})
            mutations = [[{**first, "entity": "A rewritten entity"}],
                         [{**first, "card_id": "q1:d0:c2"}], [second, first]]
            for selected in mutations:
                write_artifact(d, cases, stage="review", cards={"q1": selected, "q2": []}, parent=c)
                with self.subTest(cards=selected), self.assertRaisesRegex(ValueError, "unchanged ordered subsets"):
                    load_card_set(d, cases)

    def test_d_with_missing_parent_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = examples()
            c, d = root / "c", root / "d"
            write_artifact(c, cases)
            _, manifest = write_artifact(d, cases, stage="review", parent=c)
            manifest["parent"]["path"] = str(root / "not-present")
            (d / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Cannot read card artifact"):
                load_card_set(d, cases)


if __name__ == "__main__":
    unittest.main()
