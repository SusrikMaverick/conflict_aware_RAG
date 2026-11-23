"""Behavior checks for baseline packaging, replay and failure handling."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
import run
import baselines
from baseline_engine import BaselineEngine
from baseline_prompts import plain_parse
from baseline_nlp import Preprocessor
from methods import BASELINE_REGISTRY, resolve_method
from replay import payload

ROW = {
    "question_id": "example",
    "question": "When did the observatory open?",
    "documents": [{"doc_id": "d1", "text": "The observatory opened in 1982."}],
}


class BaselineTests(unittest.TestCase):
    def client(self):
        client = baselines.ArchiveClient.__new__(baselines.ArchiveClient)
        client.responses = {
            "a": {
                "key": "a",
                "request": payload("prompt", 128, False, 1.05, False),
                "text": "saved answer",
            }
        }
        client.expected = [{"key": "a"}]
        client.position = client.matched = 0
        return client

    def test_replay_matches_entire_request(self):
        client = self.client()
        for prompt, cap, schema, penalty in [
            ("changed", 128, False, 1.05),
            ("prompt", 129, False, 1.05),
            ("prompt", 128, True, 1.05),
            ("prompt", 128, False, 1.0),
        ]:
            with self.assertRaises(ValueError):
                client.chat(prompt, cap, schema, penalty)
        self.assertEqual(client.position, 0)
        self.assertEqual(
            client.chat("prompt", 128, False, 1.05)["text"], "saved answer"
        )
        with self.assertRaises(ValueError):
            client.chat("prompt", 128, False, 1.05)

    def test_readable_names_have_stable_ids(self):
        self.assertEqual(len(BASELINE_REGISTRY), 7)
        self.assertEqual(len({m.name for m in BASELINE_REGISTRY}), 7)
        for method in BASELINE_REGISTRY:
            self.assertEqual(resolve_method(method.name), resolve_method(method.legacy))

    def test_data_and_upstream_are_protected(self):
        for directory in ["data/baselines", "upstream", "assets"]:
            with self.assertRaises(ValueError):
                run.checked_output(BASE / directory / "output")

    def test_labels_rejected_before_generation(self):
        bad = dict(ROW, gold_answers=["1982"])
        with self.assertRaises(ValueError):
            run.validate_inputs([bad])

    def test_frozen_preprocessing_requires_exact_input(self):
        prep = Preprocessor.__new__(Preprocessor)
        prep.replay = True
        prep.inputs = {"example": ROW}
        prep.check_row(ROW)
        with self.assertRaises(ValueError):
            prep.check_row(dict(ROW, question="Changed question"))

    def test_frozen_keywords_reject_changed_isolated_answers(self):
        prep = Preprocessor.__new__(Preprocessor)
        prep.replay = True
        prep.inputs = {"example": ROW}
        prep.isolated = {"example": ["1982"]}
        prep.hints = {"example": {"hints": "1982"}}
        self.assertEqual(prep.keywords(ROW, ["1982"]), {"hints": "1982"})
        with self.assertRaises(ValueError):
            prep.keywords(ROW, ["1990"])

    def test_resume_identity_covers_prompt_scorer_and_encoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in [
                "baselines.py",
                "run.py",
                "methods.py",
                "evaluation.py",
                "download_model.py",
                "download_nlp_models.py",
                "requirements.txt",
                "data/evaluation.json",
                "configs/plain_prompt.txt",
                "tokenizer/config.json",
                "models/sup-simcse-bert-base-uncased/model.safetensors",
            ]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original")
            with patch.object(baselines, "BASE", root), patch.dict(
                "os.environ", {"QA_NLP_MODELS": str(root / "models")}
            ):
                previous = baselines.identity([ROW], ["trust"], False)
                for name in [
                    "configs/plain_prompt.txt",
                    "evaluation.py",
                    "requirements.txt",
                    "tokenizer/config.json",
                    "models/sup-simcse-bert-base-uncased/model.safetensors",
                ]:
                    path = root / name
                    path.write_text("changed")
                    current = baselines.identity([ROW], ["trust"], False)
                    self.assertNotEqual(previous, current, name)
                    path.write_text("original")
                    self.assertEqual(
                        previous, baselines.identity([ROW], ["trust"], False)
                    )

    def test_transport_failure_is_not_an_abstention(self):
        class Broken:
            def chat(self, *args):
                raise ConnectionError("offline")

        result = BaselineEngine(Broken(), None).one(ROW, "plain_1024")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["accepted_answers"], [])

    def test_invalid_json_is_retained_as_failure(self):
        class Invalid:
            def chat(self, *args):
                return dict(
                    key="a",
                    text="{broken",
                    usage={},
                    finish_reason="length",
                    cache_hit=False,
                    latency_seconds=0,
                )

        result = BaselineEngine(Invalid(), None).one(ROW, "plain_1024")
        self.assertEqual(result["status"], "error")
        self.assertEqual(len(result["requests"]), 1)

    def test_unknown_method_fails(self):
        result = BaselineEngine(None, None).one(ROW, "typo")
        self.assertEqual(result["status"], "error")

    def test_plain_parser_rejects_explanation_only(self):
        with self.assertRaises(ValueError):
            plain_parse("The answer may be 1982.")
        self.assertEqual(
            plain_parse("All Correct Answers: [1982]. Explanation: ..."), ["1982"]
        )
        with self.assertRaises(ValueError):
            plain_parse('All Correct Answers: [{"answer": "1982"}]')


if __name__ == "__main__":
    unittest.main()
