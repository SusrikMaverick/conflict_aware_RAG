import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.experiment_report import ANNOTATIONS, INPUTS, build_report, load_runs, write_report


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def record(qid, method, correct=False, *, status="answered", strict=None, wrong=False):
    return {"question_id": qid, "question": f"Question {qid}?", "method": method,
            "gold_answers": ["right"], "wrong_answers": ["wrong"],
            "raw_output": "" if status == "error" else 'All Correct Answers: ["right"]',
            "accepted_answers": [] if status == "error" else ["right"],
            "official_correct": correct, "strict_correct": correct if strict is None else strict,
            "wrong_answer_accepted": wrong, "status": status,
            "finish_reason": None if status == "error" else "stop",
            "usage": {} if status == "error" else {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def make_runs(root):
    paths = [root / name for name in ("ab", "c", "d")]
    base = {"status": "complete", "final_model": "/models/qwen72b-awq",
            "generation": {"temperature": 0, "seed": 576, "max_tokens": 128},
            "question_ids": ["q1", "q2"], "questions": 2,
            "prompt_template_sha256": "shared-prompt", "scorer_source_sha256": "shared-scorer",
            "source_hashes": {INPUTS: "shared-inputs", ANNOTATIONS: "shared-labels"},
            "subset": True, "scope": "test-only controlled comparison"}
    manifests = [copy.deepcopy(base) for _ in paths]
    extraction_cost = {"requests": 3, "prompt_tokens": 60, "completion_tokens": 30, "total_tokens": 90}
    review_cost = {"requests": 2, "prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60}
    manifests[1].update(card_variant="C", card_stage_cost=extraction_cost,
                        card_source={"path": "/cards/c", "stage": "extract",
                                     "manifest_sha256": "c-manifest", "cards_sha256": "c-cards"},
                        card_stage_manifest={"stage": "extract", "cards_sha256": "c-cards", "cost": extraction_cost})
    manifests[2].update(card_variant="D", card_stage_cost=review_cost, card_parent_cost=extraction_cost,
                        card_source={"path": "/cards/d", "stage": "review",
                                     "manifest_sha256": "d-manifest", "cards_sha256": "d-cards"},
                        card_stage_manifest={"stage": "review", "cards_sha256": "d-cards", "cost": review_cost,
                                             "parent": {"path": "/cards/c", "manifest_sha256": "c-manifest",
                                                        "cards_sha256": "c-cards", "extraction_model": base["final_model"]}})
    for path, manifest in zip(paths, manifests):
        path.mkdir()
        write_json(path / "manifest.json", manifest)

    def rows(path, name, values):
        (path / name).write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")

    rows(paths[0], "plain.jsonl", [record("q1", "plain", True), record("q2", "plain")])
    rows(paths[0], "plus_cards.jsonl", [record("q2", "plus_cards", True), record("q1", "plus_cards", wrong=True)])
    rows(paths[1], "plus_cards.jsonl", [record("q2", "plus_cards", status="error"), record("q1", "plus_cards", True)])
    rows(paths[2], "plus_cards.jsonl", [record("q1", "plus_cards", True), record("q2", "plus_cards", True)])
    return paths


class ExperimentReportTests(unittest.TestCase):
    def test_first_run_must_use_b_cards_and_include_both_answer_methods(self):
        for field, value in (("card_variant", "C"), ("card_variant", "D"),
                             ("methods", ["plus_cards"])):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as temporary:
                paths = make_runs(Path(temporary))
                path = paths[0] / "manifest.json"
                manifest = json.loads(path.read_text())
                manifest[field] = value
                write_json(path, manifest)
                with self.assertRaisesRegex(ValueError, "first"):
                    load_runs(paths)

    def test_join_aligns_ids_keeps_failed_answers_and_carries_stage_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = make_runs(Path(temporary))
            records, provenance = load_runs(paths)
            summary, details = build_report(records, provenance)
            self.assertEqual(summary["questions"], 2)
            self.assertEqual(summary["variants"]["C"]["official_accuracy"], 0.5)
            self.assertEqual(summary["variants"]["C"]["request_errors"], 1)
            self.assertEqual(summary["comparisons"]["A_to_B"]["official"]["gains"], 1)
            self.assertEqual(summary["comparisons"]["A_to_B"]["official"]["losses"], 1)
            self.assertEqual(summary["comparisons"]["B_to_C"]["strict"]["gains"], 1)
            self.assertEqual(summary["comparisons"]["B_to_C"]["strict"]["losses"], 1)
            self.assertEqual(summary["comparisons"]["C_to_D"]["official"]["gains"], 1)
            self.assertEqual(provenance["card_stage_cost"]["C"]["total_tokens"], 90)
            self.assertEqual(provenance["card_stage_cost"]["D"]["total_tokens"], 60)
            self.assertEqual(provenance["card_parent_cost"]["D"]["total_tokens"], 90)
            self.assertEqual(provenance["D_card_pipeline_cost"]["observed_totals"]["total_tokens"], 150)
            self.assertFalse(provenance["card_stage_cost"]["B"]["included"])
            self.assertEqual([row["question_id"] for row in details], ["q1", "q2"])
            self.assertEqual(details[1]["C"]["status"], "error")
            self.assertEqual(details[1]["changes"]["C_to_D"]["official"], 1)

    def test_d_must_review_the_exact_c_artifact_used_by_the_c_answer_run(self):
        for key in ("cards_sha256", "manifest_sha256"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                paths = make_runs(Path(temporary))
                path = paths[2] / "manifest.json"
                manifest = json.loads(path.read_text())
                manifest["card_stage_manifest"]["parent"][key] = "different-c-artifact"
                write_json(path, manifest)
                with self.assertRaisesRegex(ValueError, "D parent.*does not match"):
                    load_runs(paths)

    def test_different_model_decoder_scorer_or_dataset_is_rejected(self):
        for field, value in (("final_model", "/models/qwen7b"),
                             ("generation", {"max_tokens": 1024}),
                             ("scorer_source_sha256", "new-score"),
                             ("source_hashes", {INPUTS: "different-inputs", ANNOTATIONS: "shared-labels"})):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                paths = make_runs(Path(temporary))
                manifest = json.loads((paths[1] / "manifest.json").read_text())
                manifest[field] = value
                write_json(paths[1] / "manifest.json", manifest)
                with self.assertRaisesRegex(ValueError, "Incompatible"):
                    load_runs(paths)

    def test_missing_or_duplicate_result_ids_are_rejected(self):
        for rows in ([record("q1", "plus_cards", True)],
                     [record("q1", "plus_cards", True)] * 2):
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as temporary:
                paths = make_runs(Path(temporary))
                (paths[2] / "plus_cards.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                with self.assertRaisesRegex(ValueError, "question ID"):
                    load_runs(paths)

    def test_changed_labels_or_wrong_variant_or_unfinished_run_is_rejected(self):
        for change in ("labels", "variant", "unfinished"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                paths = make_runs(Path(temporary))
                if change == "labels":
                    path = paths[2] / "plus_cards.jsonl"
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[0]["gold_answers"] = ["different-gold"]
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                else:
                    path = paths[2] / "manifest.json"
                    manifest = json.loads(path.read_text())
                    manifest["card_variant" if change == "variant" else "status"] = "C" if change == "variant" else "interrupted"
                    write_json(path, manifest)
                with self.assertRaises(ValueError):
                    load_runs(paths)

    def test_new_report_is_written_and_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = make_runs(root)
            out = root / "report"
            write_report(paths, out)
            original = (out / "summary.json").read_text()
            details = [json.loads(line) for line in (out / "per_question.jsonl").read_text().splitlines()]
            self.assertEqual(len(details), 2)
            with self.assertRaises(FileExistsError):
                write_report(paths, out)
            self.assertEqual((out / "summary.json").read_text(), original)


if __name__ == "__main__":
    unittest.main()
