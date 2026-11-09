import copy
import json
import unittest

from src.card_extraction import segments, source_span
from src.card_review import DIMENSIONS, ReviewError, review_document, validate_decisions


DOCUMENT = {"doc_id": "q1:d0", "text": "Louis Alexandre died in Rambouillet on 1 December 1737."}


def card(answer="1 December 1737", *, card_id="q1:d0:c0", entity="Louis Alexandre", document=None):
    document = document or DOCUMENT
    span = source_span(["s0"], segments(document["text"]), document["text"])
    return {"card_id": card_id, "doc_id": document["doc_id"], "answer": answer,
            "entity": entity, "scope": "death", "claim": f"{entity} died on {answer}.",
            **span, "support_status": "source_located_only",
            "verification": {"status": "not_run", "probability": None}}


def audit(item, *, supported=True, changes=None):
    conditions = []
    for dimension in DIMENSIONS:
        applies = dimension in ("referent", "requested_attribute")
        conditions.append({"dimension": dimension,
                           "verdict": "satisfied" if applies else "not_applicable",
                           "source_ids": ["s0"] if applies else [],
                           "reason": "The passage supports this requested fact." if applies else "The question has no such restriction."})
    for dimension, values in (changes or {}).items():
        next(condition for condition in conditions if condition["dimension"] == dimension).update(values)
    return {"card_id": item["card_id"], "supported": supported, "conditions": conditions}


class FakeReviewer:
    def __init__(self, audits, *, finish_reason="stop", text=None):
        self.audits = audits
        self.finish_reason = finish_reason
        self.text = text
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"text": self.text if self.text is not None else json.dumps({"audits": self.audits}),
                "finish_reason": self.finish_reason,
                "usage": {"prompt_tokens": 100, "completion_tokens": 80, "total_tokens": 180}}


class CardReviewTests(unittest.TestCase):
    def test_supported_date_card_is_kept_without_any_rewrite_or_mutation(self):
        item = card()
        original = copy.deepcopy(item)
        reviewer = FakeReviewer([audit(item)])
        result = review_document(reviewer, "When did Louis Alexandre die?", DOCUMENT, [item])
        self.assertIs(result["cards"][0], item)
        self.assertEqual(item, original)
        self.assertTrue(result["decisions"][0]["kept"])
        condition = result["decisions"][0]["conditions"][0]
        self.assertEqual(condition["quote"], DOCUMENT["text"])
        self.assertEqual(result["request"]["usage"]["total_tokens"], 180)
        self.assertEqual(reviewer.calls[0][1]["max_tokens"], 2048)
        self.assertEqual(reviewer.calls[0][1]["seed"], 576)

    def test_place_of_death_is_rejected_as_an_answer_to_a_date_question(self):
        item = card("Rambouillet")
        item["claim"] = "Louis Alexandre died in Rambouillet."
        record = audit(item, supported=False, changes={"requested_attribute": {
            "verdict": "contradicted", "reason": "The candidate is a place, while the question asks for a date.",
        }})
        result = review_document(FakeReviewer([record]), "When did Louis Alexandre die?", DOCUMENT, [item])
        self.assertEqual(result["cards"], [])
        self.assertEqual(result["decisions"][0]["decision"], "reject")
        self.assertTrue(any("requested_attribute:contradicted" in reason for reason in result["decisions"][0]["reasons"]))

    def test_missing_duplicate_and_unknown_card_ids_fail_complete_coverage(self):
        item = card()
        unknown = audit(item)
        unknown["card_id"] = "invented-card"
        for records in ([], [audit(item), audit(item)], [unknown]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                validate_decisions({"audits": records}, DOCUMENT, [item])

    def test_fabricated_or_empty_supporting_source_ids_fail(self):
        item = card()
        for ids in (["made-up"], [], ["s0", "s0"]):
            record = audit(item, changes={"referent": {"source_ids": ids}})
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_decisions({"audits": [record]}, DOCUMENT, [item])

    def test_unestablished_condition_rejects_without_claiming_it_is_false(self):
        item = card()
        record = audit(item, supported=False, changes={"time_direction": {
            "verdict": "not_established", "source_ids": [], "reason": "No evidence establishes the requested before/after relation.",
        }})
        result = validate_decisions({"audits": [record]}, DOCUMENT, [item])
        self.assertEqual(result["cards"], [])
        checked = result["decisions"][0]["conditions"][2]
        self.assertEqual(checked["verdict"], "not_established")
        self.assertEqual(checked["quote"], "")

    def test_required_question_dimension_cannot_be_marked_not_applicable(self):
        item = card()
        record = audit(item, changes={"requested_attribute": {
            "verdict": "not_applicable", "source_ids": [], "reason": "No attribute checked.",
        }})
        result = validate_decisions({"audits": [record]}, DOCUMENT, [item])
        self.assertFalse(result["decisions"][0]["kept"])
        self.assertIn("supported_flag_inconsistent_with_conditions", result["decisions"][0]["reasons"])

    def test_duplicate_dimensions_fail_instead_of_silently_skipping_a_check(self):
        item = card()
        record = audit(item)
        record["conditions"][-1] = copy.deepcopy(record["conditions"][0])
        with self.assertRaises(ValueError):
            validate_decisions({"audits": [record]}, DOCUMENT, [item])

    def test_wrong_document_mutated_quote_and_duplicate_input_ids_fail_before_model(self):
        item = card()
        mutations = [{**item, "doc_id": "another-document"}, {**item, "quote": "Fabricated quote."},
                     {**item, "span_start": 1}]
        for bad in mutations:
            reviewer = FakeReviewer([])
            with self.subTest(card=bad), self.assertRaises(ValueError):
                review_document(reviewer, "When did Louis Alexandre die?", DOCUMENT, [bad])
            self.assertEqual(reviewer.calls, [])
        reviewer = FakeReviewer([])
        with self.assertRaises(ValueError):
            review_document(reviewer, "When did Louis Alexandre die?", DOCUMENT, [item, item])
        self.assertEqual(reviewer.calls, [])

    def test_eval_labels_and_verifier_metadata_are_not_sent_to_reviewer(self):
        item = card()
        item["gold_answers"] = ["GOLD_SENTINEL"]
        item["verification"] = {"status": "VERIFIER_SENTINEL", "probability": 0.9}
        document = {**DOCUMENT, "wrong_answers": ["WRONG_SENTINEL"], "type": "TYPE_SENTINEL"}
        reviewer = FakeReviewer([audit(item)])
        result = review_document(reviewer, "When did Louis Alexandre die?", document, [item])
        prompt = reviewer.calls[0][0][0]["content"]
        for secret in ("GOLD_SENTINEL", "VERIFIER_SENTINEL", "WRONG_SENTINEL", "TYPE_SENTINEL"):
            self.assertNotIn(secret, prompt)
        self.assertIs(result["cards"][0], item)

    def test_two_legitimate_entities_can_both_be_retained(self):
        document = {"doc_id": "q2:d0", "text": "Jordan the player was born in 1963. Jordan the scientist was born in 1956."}
        first = card("1963", card_id="q2:d0:c0", entity="Jordan the player", document=document)
        second = card("1956", card_id="q2:d0:c1", entity="Jordan the scientist", document=document)
        first["claim"] = "Jordan the player was born in 1963."
        second["claim"] = "Jordan the scientist was born in 1956."
        result = review_document(FakeReviewer([audit(first), audit(second)]),
                                 "When was Jordan born?", document, [first, second])
        self.assertEqual(result["cards"], [first, second])

    def test_empty_cards_skip_the_model(self):
        reviewer = FakeReviewer([])
        result = review_document(reviewer, "When did Louis Alexandre die?", DOCUMENT, [])
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(reviewer.calls, [])
        self.assertEqual(result["decisions"], [])

    def test_truncation_and_malformed_json_keep_request_metadata_on_the_error(self):
        item = card()
        reviewers = [FakeReviewer([audit(item)], finish_reason="length"), FakeReviewer([], text="not JSON")]
        for reviewer in reviewers:
            with self.subTest(reviewer=reviewer), self.assertRaises(ReviewError) as raised:
                review_document(reviewer, "When did Louis Alexandre die?", DOCUMENT, [item])
            self.assertEqual(raised.exception.request["usage"]["total_tokens"], 180)
            self.assertTrue(raised.exception.request["raw_output"])

    def test_model_cannot_add_rewritten_answer_fields_to_a_decision(self):
        item = card()
        record = audit(item)
        record["answer"] = "A different answer"
        with self.assertRaises(ValueError):
            validate_decisions({"audits": [record]}, DOCUMENT, [item])


if __name__ == "__main__":
    unittest.main()
