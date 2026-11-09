import json
from pathlib import Path
import tempfile
import unittest

from src import compare


def example(question_id="q1", cards=True):
    return {
        "question_id": question_id,
        "question": f"When was the courthouse in {question_id} built?",
        "documents": [{"doc_id": f"{question_id}:d0", "text": "The courthouse was built in 1902."}],
        "cards": [{"doc_id": f"{question_id}:d0", "card_id": f"{question_id}:d0:c0",
                   "answer": "1902", "entity": "courthouse", "scope": "built",
                   "claim": "The courthouse was built in 1902.",
                   "quote": "The courthouse was built in 1902.",
                   "original_document_number": 1}] if cards else [],
        "gold_answers": ["GOLD_LABEL_NOT_FOR_PROMPT"],
        "wrong_answers": ["WRONG_LABEL_NOT_FOR_PROMPT"],
        "source_status": "EXTRACTION_STATUS_NOT_FOR_PROMPT",
    }


def make_bundle(root):
    """Small frozen bundle with shuffled annotations and an uncertain card."""
    cases = [example(), example("q2", cards=False)]
    cases[0]["gold_answers"] = ["1902"]
    cases[1]["gold_answers"] = ["1903"]
    for case in cases:
        case["cards"] = [{**{k: card[k] for k in compare.FIELDS},
                          "original_document_number": card["original_document_number"]}
                         for card in case["cards"]]

    def write(name, value, lines=False):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in value) if lines else json.dumps(value))

    write(compare.INPUTS, [{k: c[k] for k in ("question_id", "question", "documents")} for c in cases], True)
    write(compare.ANNOTATIONS, [
        {k: c[k] for k in ("question_id", "gold_answers", "wrong_answers")} for c in reversed(cases)
    ], True)
    frozen = []
    for case in cases:
        source = f"runs/cards/{case['question_id']}.json"
        cards = [{k: v for k, v in c.items() if k != "original_document_number"} for c in case["cards"]]
        for card in cards:
            card.update(support_status="uncertain", verification={"probability": 0.1, "status": "uncertain", "threshold": 0.5})
        write(source, {"question_id": case["question_id"], "cards": cards})
        frozen.append({"question_id": case["question_id"], "cards": case["cards"],
                       "source": source, "source_status": case["source_status"],
                       "prompt": compare.build_prompts(case)["plus_cards"]})
    write(f"{compare.PLAN}/inputs.json", frozen)
    write(f"{compare.PLAN}/protocol.json", {"n": 2, "model": "Qwen2.5-7B-Instruct BF16",
                                          "intervention": "Append all saved cards"})
    snapshot = root / "conflict_aware_RAG/src/eval.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("PROMPT = " + repr(compare.PROMPT) + "\n")


class FakeAnswerer:
    def __init__(self, fail_plain=False):
        self.calls = []
        self.fail_plain = fail_plain

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        augmented = "Additional extracted evidence from the SAME documents" in messages[0]["content"]
        if self.fail_plain and not augmented:
            raise RuntimeError("Simulated final-answer failure")
        return {"text": 'All Correct Answers: ["1902"]. Explanation: Supported.',
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 100, "completion_tokens": 16, "total_tokens": 116}}


class PairedComparisonTests(unittest.TestCase):
    def test_bundle_aligns_annotations_by_id_and_preserves_uncertain_cards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_bundle(root)
            cases, provenance = compare.load_bundle(root)
            self.assertEqual([c["gold_answers"] for c in cases], [["1902"], ["1903"]])
            self.assertEqual(provenance["available_questions"], 2)
            self.assertEqual(cases[0]["cards"][0]["answer"], "1902")
            self.assertNotIn("verification", cases[0]["cards"][0])
            self.assertEqual(cases[1]["cards"], [])

    def test_bundle_rejects_documents_that_no_longer_match_the_frozen_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_bundle(root)
            path = root / compare.INPUTS
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["documents"][0]["text"] = "A different document."
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "frozen prompt differs"):
                compare.load_bundle(root)

    def test_prompts_differ_only_by_the_saved_card_appendix(self):
        case = example()
        prompts = compare.build_prompts(case)
        self.assertTrue(prompts["plus_cards"].startswith(prompts["plain"]))
        self.assertIn('"answer": "1902"', prompts["plus_cards"])
        for prompt in prompts.values():
            for private in ("GOLD_LABEL_NOT_FOR_PROMPT", "WRONG_LABEL_NOT_FOR_PROMPT",
                            "EXTRACTION_STATUS_NOT_FOR_PROMPT"):
                self.assertNotIn(private, prompt)
        empty = compare.build_prompts(example(cards=False))
        self.assertTrue(empty["plus_cards"].endswith("[]"))

    def test_both_methods_make_fresh_calls_with_identical_settings(self):
        client = FakeAnswerer()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "comparison"
            compare.run(client, [example(), example("q2", cards=False)], out=out,
                        workers=1, max_tokens=256, seed=576)
            self.assertEqual(len(client.calls), 4)
            settings = [kwargs for _, kwargs in client.calls]
            self.assertTrue(all(kwargs == settings[0] for kwargs in settings))
            self.assertEqual(settings[0]["max_tokens"], 256)
            self.assertEqual(settings[0]["temperature"], 0)
            self.assertEqual(settings[0]["seed"], 576)
            self.assertEqual(settings[0]["extra_body"]["repetition_penalty"], 1.05)
            for method in ("plain", "plus_cards"):
                records = [json.loads(line) for line in (out / f"{method}.jsonl").read_text().splitlines()]
                self.assertEqual({r["question_id"] for r in records}, {"q1", "q2"})
                self.assertEqual(len(records), 2)
            pairs = [json.loads(line) for line in (out / "paired.jsonl").read_text().splitlines()]
            self.assertEqual(len(pairs), 2)
            self.assertTrue(all(p["plain"]["accepted_answers"] == p["plus_cards"]["accepted_answers"] for p in pairs))

    def test_a_failed_method_does_not_drop_the_pair_or_skip_the_other_method(self):
        client = FakeAnswerer(fail_plain=True)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "comparison"
            summary = compare.run(client, [example()], out=out, workers=1)
            plain = json.loads((out / "plain.jsonl").read_text())
            cards = json.loads((out / "plus_cards.jsonl").read_text())
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(plain["status"], "error")
            self.assertNotEqual(cards["status"], "error")
            self.assertEqual(plain["question_id"], cards["question_id"])
            self.assertEqual(summary["questions"], 1)
            self.assertEqual(summary["paired"]["completed_pairs"], 1)
            self.assertEqual(summary["plain"]["official_accuracy"], 0)
            self.assertEqual(summary["plain"]["request_errors"], 1)
            pair = json.loads((out / "paired.jsonl").read_text())
            self.assertEqual(pair["plain"]["status"], "error")
            self.assertNotEqual(pair["plus_cards"]["status"], "error")

    def test_legacy_score_and_malformed_answer_diagnostic_stay_separate(self):
        result = compare.assess('All Correct Answers: [42.7 years, 46.2 years]. Explanation: {}',
                                ["42.7 years"], [])
        self.assertEqual(result["status"], "parse_error")
        self.assertTrue(result["official_correct"])
        self.assertFalse(result["strict_correct"])

    def test_existing_results_are_not_overwritten(self):
        client = FakeAnswerer()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            marker = out / "old-result.json"
            marker.write_text("preserve this result")
            with self.assertRaises(FileExistsError):
                compare.run(client, [example()], out=out)
            self.assertEqual(client.calls, [])
            self.assertEqual(marker.read_text(), "preserve this result")


if __name__ == "__main__":
    unittest.main()
