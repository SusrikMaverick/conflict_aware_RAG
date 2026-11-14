"""Behavior and regression checks for result-sensitive mechanisms.

Mock client tests exercise HTTP routing/cache behavior, not LLM inference.
"""

import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
from source_cards import segments, source_span
from answer_review import select
from answer_format import parse, SCHEMA
from evaluation import score
import run
from tools.archive_audit import resolve_reference


class Behavior(unittest.TestCase):
    def test_unicode(self):
        self.assertEqual(
            score(["São Paulo"], "answered", [["São Paulo"]], [["Rio"]])[
                "answer_match"
            ],
            1,
        )

    def test_alias_groups(self):
        self.assertEqual(
            score(["USA"], "answered", [["USA", "United States"]], [])["strict_set"], 1
        )
        self.assertEqual(
            score(["USA", "France"], "answered", [["USA", "United States"]], [])[
                "strict_set"
            ],
            0,
        )

    def test_gold_wrong_overlap(self):
        self.assertEqual(
            score(["Alice"], "answered", [["Alice"]], [["Alice"]])["labeled_wrong"], 0
        )

    def test_failure_denominator(self):
        result = score([], "error", [["Alice"]], [])
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["abstained"], 0)
        self.assertEqual(result["answer_match"], 0)

    def test_partial_or_extra_answers(self):
        self.assertEqual(
            score(["Alice"], "answered", [["Alice"], ["Bob"]], [])["answer_match"], 0
        )
        self.assertEqual(
            score(["Alice", "Carol"], "answered", [["Alice"]], [])["strict_set"], 0
        )

    def test_candidate_intersection(self):
        self.assertEqual(
            select(
                "M08", [], [" Alice  Smith ", "Bob"], ["alice smith"], [], [], {}, {}
            ),
            [" Alice  Smith "],
        )

    def test_two_rejections_required(self):
        inventory = [dict(id="C0", answer="Alice")]
        for verdict, expected in [
            ("accept", ["Alice"]),
            ("uncertain", ["Alice"]),
            ("reject", []),
        ]:
            self.assertEqual(
                select(
                    "M05",
                    ["Alice"],
                    [],
                    [],
                    [],
                    inventory,
                    {"C0": {"verdict": "reject"}},
                    {"C0": {"verdict": verdict}},
                ),
                expected,
            )

    def test_source_indices_are_literal(self):
        text = "Alice played football. Bob coached."
        parts = segments(text)
        span = source_span(["s0"], parts, text)
        self.assertEqual(span["quote"], text[span["span_start"] : span["span_end"]])
        with self.assertRaises(ValueError):
            source_span(["not-a-source"], parts, text)

    def test_no_semantic_alias_in_intersection(self):
        self.assertEqual(
            select("M08", [], ["United States"], ["USA"], [], [], {}, {}), []
        )

    def test_parser_drops_only_sentinels(self):
        self.assertEqual(
            parse(
                json.dumps(
                    {"accepted_answers": ["unknown", "Alice"], "explanation": ""}
                )
            ),
            ["Alice"],
        )

    def test_gold_cannot_cross_answers(self):
        args = (["New", "York"], "answered", [["New York"]], [])
        self.assertEqual(score(*args)["answer_match"], 0)
        self.assertEqual(score(*args, metric="legacy")["answer_match"], 1)

    def test_wrong_cannot_cross_answers(self):
        result = score(
            ["Alice", "Warner Bros.", "Discovery Nordic"],
            "answered",
            [["Alice"]],
            [["Bros Discovery"]],
        )
        self.assertEqual(result["answer_match"], 1)
        self.assertEqual(result["labeled_wrong"], 0)

    def test_real_wrong_still_detected(self):
        result = score(["Alice", "London"], "answered", [["Alice"]], [["London"]])
        self.assertEqual(result["labeled_wrong"], 1)
        self.assertEqual(result["answer_match"], 0)

    def test_empty_normalized_gold_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty after normalization"):
            score([], "abstain", [["A"]], [])

    def test_empty_prediction_not_credited(self):
        result = score(["", "the"], "answered", [["Alice"]], [])
        self.assertEqual(result["answer_match"], 0)
        self.assertEqual(score([], "abstain", [["Alice"]], [])["abstained"], 1)


class RuntimeGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = [
            dict(
                question_id="sample-1",
                question="Where?",
                documents=[dict(doc_id="d1", text="\u5317\u4eac São Paulo")],
            )
        ]

    def fixture(self):
        files = [
            "run.py",
            "evaluation.py",
            "methods.py",
            "src/client.py",
            "configs/plain_prompt.txt",
            "tokenizer/tokenizer.json",
            "requirements.txt",
            "data/stages.jsonl.gz",
            "data/predictions.jsonl.gz",
        ]
        for name in files:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("initial", encoding="utf-8")

    def test_mode_changes_fingerprint_and_rejects_reuse(self):
        self.fixture()
        with patch.object(run, "BASE", self.root):
            replay = run.generation_identity(self.inputs, True)
            fresh = run.generation_identity(self.inputs, False)
        output = self.root / "output"
        run.write(output / "generation_manifest.json", replay)
        before = (output / "generation_manifest.json").read_bytes()
        with self.assertRaises(ValueError):
            run.check_output_identity(output, fresh)
        self.assertEqual((output / "generation_manifest.json").read_bytes(), before)

    def test_prompt_and_tokenizer_changes_reject_reuse(self):
        self.fixture()
        with patch.object(run, "BASE", self.root):
            original = run.generation_identity(self.inputs, True)
            for name in [
                "configs/plain_prompt.txt",
                "tokenizer/tokenizer.json",
                "requirements.txt",
            ]:
                path = self.root / name
                previous = path.read_bytes()
                path.write_text("changed", encoding="utf-8")
                self.assertNotEqual(
                    original["fingerprint"],
                    run.generation_identity(self.inputs, True)["fingerprint"],
                )
                path.write_bytes(previous)

    def test_nonempty_unowned_output_rejected(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError, "no generation manifest"):
            run.check_output_identity(
                self.root, {"fingerprint": "test", "mode": "fresh"}
            )

    def test_malformed_input_errors_are_actionable(self):
        invalid = [
            {},
            [1],
            [dict(question_id="x", question="?", documents=[{}])],
            [dict(question_id="x", question="?", documents=["bad"])],
        ]
        for value in invalid:
            with self.assertRaises(ValueError):
                run.validate_inputs(value)
        valid = copy.deepcopy(self.inputs)
        valid[0]["gold"] = "leak"
        with self.assertRaisesRegex(ValueError, "labels must be separate"):
            run.validate_inputs(valid)

    def test_utf8_json_roundtrip(self):
        path = self.root / "\u4e2d\u6587.json"
        run.write(path, self.inputs)
        self.assertEqual(run.read(path), self.inputs)
        self.assertIn("\u5317\u4eac".encode(), path.read_bytes())

    def test_summary_mismatch_raises_under_optimization(self):
        code = "import run; run.check_summaries({},[dict(dataset='RGB',method='ours_1024')])"
        result = subprocess.run(
            [sys.executable, "-O", "-c", code], cwd=BASE, text=True, capture_output=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("coverage differs", result.stderr)

    def test_replay_detects_final_answer_drift(self):
        expected = run.records(BASE / "data/predictions.jsonl.gz")[0]
        self.assertTrue(run.compare_replay([expected])["passed"])
        altered = dict(expected, accepted_answers=["wrong drift answer"])
        self.assertFalse(run.compare_replay([altered])["passed"])

    def test_empty_duplicate_and_incomplete_evaluation_rejected(self):
        with self.assertRaises(ValueError):
            run.evaluate([], self.root / "score")
        row = run.records(BASE / "data/predictions.jsonl.gz")[0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            run.evaluate([row, row], self.root / "score")
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            run.evaluate([row], self.root / "score", expected_ids=[row["question_id"]])

    def test_corrected_reference_resolution_is_contextual(self):
        paths = {"runs/a", "corrected/runs/a", "corrected/runs/b"}
        self.assertEqual(
            resolve_reference("runs/a", paths, {}, owner="runs/old"), "runs/a"
        )
        self.assertEqual(
            resolve_reference("runs/a", paths, {}, owner="corrected/runs/new"),
            "corrected/runs/a",
        )
        self.assertEqual(
            resolve_reference(
                "legacy-marker", paths, {"legacy-marker": "corrected/runs/b"}
            ),
            "corrected/runs/b",
        )

    def test_metadata_cannot_carry_answer_labels(self):
        row = dict(
            question_id="x",
            question="Who?",
            documents=[
                dict(doc_id="d1", text="Alice.", metadata={"gold_answer": "Alice"})
            ],
        )
        with self.assertRaisesRegex(ValueError, "metadata must be empty"):
            run.validate_inputs([row])
        row["documents"][0]["metadata"] = {}
        self.assertEqual(run.validate_inputs([row]), [row])

    def test_outputs_cannot_overwrite_packaged_results_or_evidence(self):
        for directory in [
            "",
            "results/recomputed",
            "data",
            "provenance/review_v4",
            "src",
            "tools",
        ]:
            with self.assertRaisesRegex(ValueError, "separate output"):
                run.checked_output(BASE / directory)
        self.assertEqual(
            run.checked_output(self.root / "new_output"), self.root / "new_output"
        )

    def test_public_names_preserve_archive_identity(self):
        from methods import METHOD_REGISTRY, resolve_method

        self.assertEqual(len({m.name for m in METHOD_REGISTRY}), 6)
        archived = run.records(BASE / "data/predictions.jsonl.gz")[0]
        named = dict(archived, method=resolve_method(archived["method"]).name)
        self.assertTrue(run.compare_replay([named])["passed"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            run.evaluate([archived, named], self.root / "duplicate_score")
        result = run.evaluate([named], self.root / "named_score")
        row = next(iter(result.values()))
        self.assertEqual(row["method"], named["method"])
        self.assertEqual(row["legacy_method"], archived["method"])

    def test_mock_main_aux_endpoints_and_cache(self):
        import real_runtime
        import compact_client

        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return [1, 2, 3]

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: Tokenizer()
            )
        )
        calls = []

        def respond(request, timeout):
            calls.append((request.full_url, json.loads(request.data)))
            return io.BytesIO(
                json.dumps(
                    dict(
                        choices=[dict(message=dict(content="{}"), finish_reason="stop")]
                    )
                ).encode()
            )

        endpoints = {
            "SIX_MAIN_URL": "http://mock-main/v1/chat/completions",
            "SIX_AUX_URL": "http://mock-aux/v1/chat/completions",
        }
        with (
            patch.dict(sys.modules, {"transformers": fake_transformers}),
            patch.object(real_runtime, "ROOT", self.root),
            patch.dict(os.environ, endpoints),
            patch("urllib.request.urlopen", side_effect=respond),
        ):
            main = real_runtime.Client()
            aux = compact_client.CompactClient()
            self.assertFalse(main.chat("mock", json_mode=SCHEMA)["cache_hit"])
            self.assertTrue(main.chat("mock", json_mode=SCHEMA)["cache_hit"])
            result = aux.chat("mock-aux", json_mode={"type": "object"})
            self.assertEqual(result["endpoint"], endpoints["SIX_AUX_URL"])
        self.assertEqual(
            [url for url, _ in calls],
            [endpoints["SIX_MAIN_URL"], endpoints["SIX_AUX_URL"]],
        )
        self.assertEqual(
            calls[1][1]["guided_decoding_backend"], "xgrammar:disable-any-whitespace"
        )


if __name__ == "__main__":
    unittest.main()
