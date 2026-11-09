from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import cards, compare
from src.card_review import DIMENSIONS
from src.experiment_report import write_report
from test_cards_pipeline import FakeExtractor
from test_compare import FakeAnswerer, make_bundle


class FakeReviewer:
    def __enter__(self): return self
    def __exit__(self, *args): pass

    def complete(self, messages, **kwargs):
        visible = json.loads(messages[0]["content"].split("\nInput: ", 1)[1])
        keep = "q1" in visible["question"]
        audits = []
        for card in visible["candidates"]:
            conditions = []
            for dimension in DIMENSIONS:
                required = dimension in ("referent", "requested_attribute")
                verdict = "satisfied" if required else "not_applicable"
                ids = ["s0"] if required else []
                if dimension == "requested_attribute" and not keep:
                    verdict, ids = "not_established", []
                conditions.append({"dimension": dimension, "verdict": verdict,
                                   "source_ids": ids, "reason": "Test source-grounded condition."})
            audits.append({"card_id": card["card_id"], "supported": keep, "conditions": conditions})
        return {"text": json.dumps({"audits": audits}), "finish_reason": "stop",
                "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}}


class FinalAnswerer(FakeAnswerer):
    def __enter__(self): return self
    def __exit__(self, *args): pass


class CheckedPipelineTests(unittest.TestCase):
    def test_c_d_and_final_report_work_together_without_model_calls(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(temporary)
            bundle = root / "bundle"
            make_bundle(bundle)
            common = ["--bundle", str(bundle), "--model", "test72b", "--workers", "1"]
            c_cards, d_cards = root / "c-cards", root / "d-cards"
            with patch("src.interface.LLM", return_value=FakeExtractor()):
                self.assertEqual(cards.main(["extract", *common, "--out", str(c_cards)]), 0)
            c_manifest_before = (c_cards / "manifest.json").read_bytes()
            with patch("src.interface.LLM", return_value=FakeReviewer()):
                self.assertEqual(cards.main(["review", *common, "--cards", str(c_cards), "--out", str(d_cards)]), 0)
            self.assertEqual((c_cards / "manifest.json").read_bytes(), c_manifest_before)
            d_rows = [json.loads(row) for row in (d_cards / "cards.jsonl").read_text().splitlines()]
            self.assertEqual([len(row["cards"]) for row in d_rows], [1, 0])
            out_paths = [root / "ab", root / "c-answers", root / "d-answers"]
            for source, out in zip((None, c_cards, d_cards), out_paths):
                args = [*common, "--out", str(out)]
                if source is not None:
                    args += ["--cards", str(source), "--only-cards"]
                with patch("src.interface.LLM", return_value=FinalAnswerer()):
                    self.assertEqual(compare.main(args), 0)
            summary = write_report(out_paths, root / "report")
            self.assertEqual(summary["questions"], 2)
            self.assertEqual(set(summary["variants"]), {"A", "B", "C", "D"})
            self.assertEqual(summary["comparisons"]["C_to_D"]["official"]["questions"], 2)


if __name__ == "__main__":
    unittest.main()
