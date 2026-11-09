import json
import unittest

from src.card_extraction import ExtractionError, extract_document, extraction_schema, segments, source_span


class FakeModel:
    def __init__(self, cards=None, text=None, finish_reason="stop"):
        self.cards = [] if cards is None else cards
        self.text = text
        self.finish_reason = finish_reason
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"text": self.text if self.text is not None else json.dumps({"cards": self.cards}),
                "finish_reason": self.finish_reason,
                "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50}}


def candidate(**changes):
    return {"answer": "1902", "entity": "courthouse", "scope": "built",
            "claim": "The courthouse was built in 1902.", "source_ids": ["s0"], **changes}


class CardExtractionTests(unittest.TestCase):
    def test_quote_is_reconstructed_from_source_and_not_generated(self):
        text = "The courthouse was built in 1902. It was restored in 1990."
        model = FakeModel([candidate()])
        result = extract_document(model, "When was it built?", {"doc_id": "d1", "text": text})
        card = result["cards"][0]
        self.assertEqual(card["quote"], "The courthouse was built in 1902.")
        self.assertEqual(card["quote"], text[card["span_start"]:card["span_end"]])
        self.assertEqual(card["card_id"], "d1:c0")
        self.assertEqual(card["support_status"], "source_located_only")
        self.assertEqual(card["verification"], {"status": "not_run", "probability": None})
        self.assertEqual(result["request"]["usage"]["total_tokens"], 50)
        kwargs = model.calls[0][1]
        self.assertEqual((kwargs["temperature"], kwargs["max_tokens"], kwargs["seed"], kwargs["top_p"]), (0, 1024, 576, 1.0))
        self.assertEqual(kwargs["extra_body"]["repetition_penalty"], 1.0)

    def test_evaluation_annotations_and_metadata_do_not_enter_prompt(self):
        model = FakeModel([candidate()])
        document = {"doc_id": "d1", "text": "The courthouse was built in 1902.",
                    "type": "SECRET_SOURCE_LABEL", "answer": "SECRET_GOLD_LABEL",
                    "verification": "SECRET_SCORE"}
        extract_document(model, "When was it built?", document)
        prompt = model.calls[0][0][0]["content"]
        for secret in ("SECRET_SOURCE_LABEL", "SECRET_GOLD_LABEL", "SECRET_SCORE", '"doc_id"'):
            self.assertNotIn(secret, prompt)

    def test_invalid_card_stays_in_issues_and_does_not_discard_valid_sibling(self):
        bad = candidate(source_ids=["invented-segment"])
        result = extract_document(FakeModel([bad, candidate()]), "When?", {
            "doc_id": "d1", "text": "The courthouse was built in 1902."})
        self.assertEqual([c["card_id"] for c in result["cards"]], ["d1:c1"])
        self.assertEqual(result["issues"], [{"index": 0, "error": "invalid_source_ids", "raw_card": bad}])

    def test_missing_answer_in_claim_is_rejected_without_repair(self):
        result = extract_document(FakeModel([candidate(answer="1885")]), "When?", {
            "doc_id": "d1", "text": "The courthouse was built in 1902."})
        self.assertEqual(result["cards"], [])
        self.assertEqual(result["issues"][0]["error"], "claim_missing_answer")

    def test_empty_source_skips_model(self):
        model = FakeModel()
        result = extract_document(model, "When?", {"doc_id": "empty", "text": " \n "})
        self.assertTrue(result["skipped"])
        self.assertEqual(result["cards"], [])
        self.assertEqual(model.calls, [])

    def test_schema_bounds_and_valid_ids_match_c1(self):
        parts = segments("One sentence. Another sentence.")
        schema = extraction_schema(parts)
        cards = schema["properties"]["cards"]
        self.assertEqual(cards["maxItems"], 3)
        self.assertFalse(cards["items"]["additionalProperties"])
        ids = cards["items"]["properties"]["source_ids"]
        self.assertEqual(ids["items"]["enum"], ["s0", "s1"])
        self.assertEqual((ids["minItems"], ids["maxItems"]), (1, 4))
        for bad in ([], ["s0", "s0"], ["s9"], [1], [["s0"]]):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "invalid_source_ids"):
                source_span(bad, parts, "One sentence. Another sentence.")

    def test_discontiguous_ids_preserve_intervening_text_but_not_paragraph_crossing(self):
        text = "First. Middle. Last."
        span = source_span(["s2", "s0"], segments(text), text)
        self.assertEqual(span["quote"], text)
        paragraph = "First.\n\nLast."
        with self.assertRaisesRegex(ValueError, "source_span_cross_paragraph_or_cap"):
            source_span(["s0", "s1"], segments(paragraph), paragraph)
        long = "x" * 2501
        with self.assertRaisesRegex(ValueError, "source_span_cross_paragraph_or_cap"):
            source_span(["s0"], segments(long), long)

    def test_truncated_or_malformed_json_keeps_request_trace(self):
        for model in (FakeModel([candidate()], finish_reason="length"), FakeModel(text='{"cards":['),
                      FakeModel([candidate()] * 4)):
            with self.subTest(model=model), self.assertRaises(ExtractionError) as caught:
                extract_document(model, "When?", {"doc_id": "d1", "text": "The courthouse was built in 1902."})
            self.assertIn("raw_output", caught.exception.request)
            self.assertEqual(caught.exception.request["usage"]["total_tokens"], 50)


if __name__ == "__main__":
    unittest.main()
