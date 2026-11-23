"""Compact JSON request client."""

import os, threading
from real_runtime import Client, MODEL, MODEL_REV, digest
from answer_format import SCHEMA

COMPACT_RUNTIME = "vllm0.8.5.post1-bf16-compactaux-v3"
BACKEND = "xgrammar:disable-any-whitespace"


class CompactClient(Client):

    def chat(self, prompt, max_tokens=768, json_mode=False, penalty=1.0):
        if json_mode == SCHEMA:
            return super().chat(prompt, max_tokens, json_mode, penalty)
        if not isinstance(json_mode, dict):
            raise ValueError("Compact auxiliary requests require a JSON schema")
        messages = [{"role": "user", "content": prompt}]
        with self.lock:
            n = len(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            )
        if n + max_tokens > 32768:
            raise ValueError(f"context_overflow:{n}+{max_tokens}>32768; no truncation")
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "seed": 576,
            "repetition_penalty": penalty,
            "top_p": 1.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "decision", "schema": json_mode},
            },
            "guided_decoding_backend": BACKEND,
        }
        key = digest(
            {"revision": MODEL_REV, "runtime": COMPACT_RUNTIME, "payload": payload}
        )
        path = self.cache / (key + ".json")
        with self.lock:
            lock = self.keylocks.setdefault(key, threading.Lock())
        with lock:
            endpoint = os.environ.get(
                "SIX_AUX_URL", "http://127.0.0.1:8577/v1/chat/completions"
            )
            return self._request(
                path,
                key,
                payload,
                n,
                endpoint=endpoint,
                extra={"runtime_epoch": COMPACT_RUNTIME, "endpoint": endpoint},
            )
