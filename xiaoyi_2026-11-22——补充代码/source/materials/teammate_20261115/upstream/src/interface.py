"""The one place the project talks to the LLM.

`LLM` puts Qwen2.5-72B-Instruct on this node's GPU and frees the GPU when you are done.
It has two backends, which take the same calls:

- "vllm" (the default): a vLLM server with the AWQ 4-bit weights. Fast. The server stops
  on normal exit, Ctrl-C, an exception, or even if this Python process is killed.
- "hf": Hugging Face transformers inside this process, with the full weights quantized to
  NF4 4-bit on load, as in the MADAM-RAG paper's code. Much slower; it is for final
  numbers. Needs `pip install -e ".[hf]"`.

Run it with the vllm env's python.

    from src.interface import LLM

    with LLM() as llm:
        print(llm.chat([{"role": "user", "content": "Hello"}]))

Chat in the terminal:  python -m src.interface [--backend hf]
"""
import argparse
import atexit
import ctypes
import os
import secrets
import signal
import site
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from openai import OpenAI

MODELS = {  # the weights each backend loads
    "vllm": "/scratch/smukhe61/conflict_aware_RAG/models/Qwen2.5-72B-Instruct-AWQ",
    "hf": "/scratch/smukhe61/conflict_aware_RAG/models/Qwen2.5-72B-Instruct",
}
LOG_FILE = Path(__file__).resolve().parents[1] / "logs" / "vllm.log"

# ~/.local has leftovers of an old torch install. Keep them out of this process, as
# PYTHONNOUSERSITE does for the server.
sys.path[:] = [p for p in sys.path if p != site.getusersitepackages()]


class LLM:
    def __init__(self, backend="vllm", port=8000, max_model_len=32768,
                 gpu_memory_utilization=0.90, serve_args=()):
        """Everything after `backend` is for the vLLM server. `serve_args` are extra
        `vllm serve` flags, e.g. ["--max-num-seqs", "1"]."""
        self.backend, self.model = backend, MODELS[backend]
        if backend == "hf":
            self._load_hf()
            return
        key = secrets.token_hex(16)  # other users on the node can reach localhost
        LOG_FILE.parent.mkdir(exist_ok=True)
        self._log = open(LOG_FILE, "w")
        self._proc = subprocess.Popen(
            [Path(sys.executable).with_name("vllm"), "serve", self.model,
             "--host", "127.0.0.1", "--port", str(port),
             "--max-model-len", str(max_model_len),
             "--gpu-memory-utilization", str(gpu_memory_utilization), *serve_args],
            env={**os.environ, "VLLM_API_KEY": key, "VLLM_NO_USAGE_STATS": "1",
                 "VLLM_CACHE_ROOT": "/scratch/smukhe61/conflict_aware_RAG/cache/vllm",
                 "PYTHONNOUSERSITE": "1",  # ~/.local has an old torch that breaks vllm
                 "VLLM_USE_FLASHINFER_SAMPLER": "0"},  # it JIT-compiles with nvcc, which this node lacks
            stdout=self._log, stderr=subprocess.STDOUT,
            start_new_session=True,  # Ctrl-C goes to us only; we stop the server ourselves
            preexec_fn=lambda: ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM),  # die with us, even on kill -9
        )
        atexit.register(self.close)
        try:
            self._wait_until_ready(f"http://127.0.0.1:{port}/health")
        except BaseException:
            self.close()
            raise
        self.client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key=key)

    def _wait_until_ready(self, health_url, timeout=3600):  # a cold /scratch needs over 20 min
        print(f"Starting vLLM ({Path(self.model).name}), log: {LOG_FILE}", file=sys.stderr)
        start = time.time()
        while time.time() - start < timeout:
            if self._proc.poll() is not None:
                tail = "\n".join(LOG_FILE.read_text().splitlines()[-20:])
                raise RuntimeError(f"vLLM exited during startup:\n{tail}")
            try:
                urllib.request.urlopen(health_url, timeout=2).close()
                print(f"vLLM ready in {time.time() - start:.0f}s", file=sys.stderr)
                return
            except OSError:
                time.sleep(2)
        raise TimeoutError(f"vLLM not ready after {timeout}s, see {LOG_FILE}")

    def _load_hf(self):
        """Load the model as MADAM-RAG's run_madam_rag.py does: NF4 with double
        quantization, computing in float16."""
        import torch  # imported here: the vLLM backend doesn't need them in this process
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"Loading {Path(self.model).name} in NF4 with transformers", file=sys.stderr)
        start = time.time()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model, torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
        print(f"Model ready in {time.time() - start:.0f}s", file=sys.stderr)

    def _call_hf(self, messages, temperature=None, max_tokens=1024, extra_body=None):
        """`call` for the hf backend. It takes the OpenAI-style arguments this project
        uses; `extra_body` holds options named alike in both APIs, e.g. repetition_penalty."""
        options = {**(extra_body or {}), "max_new_tokens": max_tokens}
        if temperature == 0:  # greedy; the unused sampling defaults are cleared to avoid warnings
            options.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        elif temperature is not None:
            options["temperature"] = temperature
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self._model.device)
        prompt_tokens = inputs["input_ids"].shape[1]
        new = self._model.generate(**inputs, **options)[0, prompt_tokens:]
        eos = self._model.generation_config.eos_token_id
        ended = new[-1].item() in (eos if isinstance(eos, list) else [eos])
        return {"text": self._tokenizer.decode(new, skip_special_tokens=True),
                "prompt_tokens": prompt_tokens, "completion_tokens": len(new),
                "finish_reason": "stop" if ended else "length"}

    def call(self, messages, **kwargs) -> dict:
        """One request and what it cost, as counted by the backend: {"text", "prompt_tokens",
        "completion_tokens", "finish_reason"}. Use it wherever calls are measured."""
        if self.backend == "hf":
            return self._call_hf(messages, **kwargs)
        resp = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        choice = resp.choices[0]
        return {"text": choice.message.content or "",
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "finish_reason": choice.finish_reason}

    def chat(self, messages, **kwargs) -> str:
        """Return the reply. Extra kwargs (temperature, max_tokens, ...) go to the OpenAI API."""
        return self.call(messages, **kwargs)["text"]

    def stream(self, messages, **kwargs):
        """Like `chat`, but yields the reply as it is generated (hf: all at once)."""
        if self.backend == "hf":
            yield self.chat(messages, **kwargs)
            return
        chunks = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True, **kwargs
        )
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def close(self):
        """Stop the server and free the GPU. Safe to call more than once."""
        if self.backend == "hf":
            self._model = None  # its GPU memory is released when this process exits
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self._proc.pid, signal.SIGKILL)
                self._proc.wait()
        self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main():
    """Terminal chat. /reset clears the history; Ctrl-D or Ctrl-C quits."""
    parser = argparse.ArgumentParser(description="Chat with the model in the terminal.")
    parser.add_argument("--backend", choices=list(MODELS), default="vllm")
    backend = parser.parse_args().backend
    try:
        with LLM(backend) as llm:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            print("Type a message. /reset clears the history, Ctrl-D or Ctrl-C quits.")
            while True:
                user = input("\n> ").strip()
                if user == "/reset":
                    messages = messages[:1]
                elif user:
                    messages.append({"role": "user", "content": user})
                    reply = ""
                    for piece in llm.stream(messages):
                        print(piece, end="", flush=True)
                        reply += piece
                    print()
                    messages.append({"role": "assistant", "content": reply})
    except (EOFError, KeyboardInterrupt):
        pass
    print("\nModel stopped.")


if __name__ == "__main__":
    main()
