"""Shared question-answering helpers."""

import os, hashlib, json, re, threading, time, urllib.request, urllib.error, uuid
from pathlib import Path

ROOT = Path(
    os.environ.get("SIX_METHOD_ROOT", Path(__file__).resolve().parents[1])
).resolve()
MODEL_REV = "a09a35458c702b33eeacc393d103063234e8bc28"
MODEL = "nlp576-qwen7b"


def digest(x):
    return hashlib.sha256(
        json.dumps(x, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def parse_json(text):
    s = text.strip()
    if s.startswith("```"):
        s = re.sub("^```(?:json)?\\s*|\\s*```$", "", s)
    out = json.loads(s)
    if not isinstance(out, dict):
        raise ValueError("JSON object required")
    return out


class Client:

    def __init__(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            Path(
                os.environ.get(
                    "SIX_TOKENIZER",
                    str(Path(__file__).resolve().parents[1] / "tokenizer"),
                )
            ),
            local_files_only=True,
        )
        self.cache = ROOT / "runs/request_cache/v1"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.keylocks = {}

    def chat(self, prompt, max_tokens=768, json_mode=False, penalty=1.0):
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
        }
        if isinstance(json_mode, dict):
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "decision", "schema": json_mode},
            }
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}
        key = digest(
            {
                "revision": MODEL_REV,
                "runtime": "vllm0.8.5.post1-bf16",
                "payload": payload,
            }
        )
        path = self.cache / (key + ".json")
        with self.lock:
            keylock = self.keylocks.setdefault(key, threading.Lock())
        with keylock:
            return self._request(path, key, payload, n)

    def _request(self, path, key, payload, n, endpoint=None, extra=None):
        endpoint = endpoint or os.environ.get(
            "SIX_MAIN_URL", "http://127.0.0.1:8576/v1/chat/completions"
        )
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")) | {"cache_hit": True}
        start = time.monotonic()
        errors = []
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    endpoint,
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=600) as f:
                    out = json.load(f)
                result = {
                    "key": key,
                    "model_revision": MODEL_REV,
                    "request": payload,
                    "response": out,
                    "text": out["choices"][0]["message"]["content"],
                    "finish_reason": out["choices"][0]["finish_reason"],
                    "usage": out.get("usage", {}),
                    "preflight_tokens": n,
                    "latency_seconds": time.monotonic() - start,
                    "attempts": attempt + 1,
                    "retry_errors": errors,
                    "cache_hit": False,
                    "is_model_inference": True,
                }
                if extra:
                    result.update(extra)
                write_json(path, result)
                return result
            except (urllib.error.URLError, TimeoutError) as exc:
                errors.append(type(exc).__name__ + ":" + str(exc)[:200])
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("backend failure: " + str(errors))


def load_plain_prompt():
    return (Path(__file__).resolve().parents[1] / "configs/plain_prompt.txt").read_text(
        encoding="utf-8"
    )
