from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import cards
from src.card_artifacts import load_card_set
from src.compare import load_bundle, run
from test_compare import FakeAnswerer, make_bundle


class FakeExtractor:
    def __init__(self, fail_q1=False):
        self.calls = []
        self.fail_q1 = fail_q1

    def __enter__(self): return self
    def __exit__(self, *args): pass

    def complete(self, messages, **kwargs):
        prompt = messages[0]["content"]
        self.calls.append(prompt)
        visible = json.loads(prompt.split("\nInput: ", 1)[1])
        if self.fail_q1 and "q1" in visible["question"]:
            raise RuntimeError("Simulated interrupted extraction")
        output = {"cards": [{"answer": "1902", "entity": "courthouse", "scope": "built",
                             "claim": "The courthouse was built in 1902.", "source_ids": ["s0"]}]}
        return {"text": json.dumps(output), "finish_reason": "stop",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}


class CardPipelineTests(unittest.TestCase):
    def test_failed_documents_resume_without_regenerating_successful_cards(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stderr(io.StringIO()):
            root = Path(temporary)
            bundle = root / "bundle"
            make_bundle(bundle)
            cases, provenance = load_bundle(bundle)
            out = root / "C"
            cards.prepare_stage(out, cases, provenance, "test72b", "extract", workers=1)
            first = FakeExtractor(fail_q1=True)
            result = cards.run_stage(first, cases, out=out, stage="extract", workers=1)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failed_documents"], 1)
            with self.assertRaisesRegex(ValueError, "complete"):
                load_card_set(out, cases)
            cards.prepare_stage(out, cases, provenance, "test72b", "extract", resume=True, workers=1)
            second = FakeExtractor()
            result = cards.run_stage(second, cases, out=out, stage="extract", workers=1)
            self.assertEqual(len(first.calls), 2)
            self.assertEqual(len(second.calls), 1)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["cost"]["requests"], 3)
            updated, metadata = load_card_set(out, cases, expected_model="test72b")
            self.assertEqual(metadata["card_variant"], "C")
            self.assertTrue(all(len(case["cards"]) == 1 for case in updated))
            self.assertTrue(all("verification" not in case["cards"][0] for case in updated))
            final = FakeAnswerer()
            answer_out = root / "answers"
            summary = run(final, updated, out=answer_out, workers=1, methods=("plus_cards",))
            self.assertEqual(len(final.calls), 2)
            self.assertFalse((answer_out / "plain.jsonl").exists())
            self.assertEqual(summary["plus_cards"]["questions"], 2)
            self.assertIsNone(summary["paired"]["accuracy_difference"])

    def test_resume_requires_identical_model_settings_and_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            make_bundle(bundle)
            cases, provenance = load_bundle(bundle)
            out = root / "C"
            cards.prepare_stage(out, cases, provenance, "test72b", "extract")
            with self.assertRaisesRegex(ValueError, "changed"):
                cards.prepare_stage(out, cases, provenance, "different-model", "extract", resume=True)
            with self.assertRaises(FileExistsError):
                cards.prepare_stage(out, cases, provenance, "test72b", "extract")

    def test_resuming_completed_stage_preserves_parent_hash_without_loading_model(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            root = Path(temporary)
            bundle = root / "bundle"
            make_bundle(bundle)
            out = root / "C"
            args = ["extract", "--bundle", str(bundle), "--out", str(out), "--model", "test72b", "--workers", "1"]
            with patch("src.interface.LLM", return_value=FakeExtractor()):
                self.assertEqual(cards.main(args), 0)
            original = (out / "manifest.json").read_bytes()
            with patch("src.interface.LLM", side_effect=AssertionError("Completed stages need no model")):
                self.assertEqual(cards.main([*args, "--resume"]), 0)
            self.assertEqual((out / "manifest.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
