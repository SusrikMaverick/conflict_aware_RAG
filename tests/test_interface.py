import unittest
from types import SimpleNamespace

from src.interface import LLM


class CompletionMetadataTests(unittest.TestCase):
    def test_completion_records_limit_and_usage_without_changing_chat_text(self):
        calls = []
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='All Correct Answers: ["1902"]. Explanation: {}'),
                finish_reason="length",
            )],
            usage=SimpleNamespace(model_dump=lambda: {
                "prompt_tokens": 100, "completion_tokens": 128, "total_tokens": 228,
            }),
        )

        def create(**kwargs):
            calls.append(kwargs)
            return response

        llm = LLM.__new__(LLM)
        llm.model = "same-local-model"
        llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        messages = [{"role": "user", "content": "When was it built?"}]
        result = llm.complete(messages, temperature=0, max_tokens=128)
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["usage"]["total_tokens"], 228)
        self.assertEqual(llm.chat(messages, temperature=0, max_tokens=128), result["text"])
        self.assertEqual(calls, [{"model": "same-local-model", "messages": messages,
                                  "temperature": 0, "max_tokens": 128}] * 2)


if __name__ == "__main__":
    unittest.main()
