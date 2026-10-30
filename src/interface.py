"""The one place the project talks to the LLM.

`LLM` starts a vLLM server on this node's GPU, waits until it is ready, and stops
it when you are done: on normal exit, Ctrl-C, an exception, or even if this Python
process is killed. Run it with the vllm env's python.

    from src.interface import LLM

    with LLM() as llm:
        print(llm.chat([{"role": "user", "content": "Hello"}]))

Chat in the terminal:  python -m src.interface
"""
import atexit
import ctypes
import os
import secrets
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from openai import OpenAI

MODEL = "/scratch/smukhe61/conflict_aware_RAG/models/Qwen2.5-72B-Instruct-AWQ"
LOG_FILE = Path(__file__).resolve().parents[1] / "logs" / "vllm.log"


class LLM:
    def __init__(self, model=MODEL, port=8000, max_model_len=32768, gpu_memory_utilization=0.90):
        self.model = model
        key = secrets.token_hex(16)  # other users on the node can reach localhost
        LOG_FILE.parent.mkdir(exist_ok=True)
        self._log = open(LOG_FILE, "w")
        self._proc = subprocess.Popen(
            [Path(sys.executable).with_name("vllm"), "serve", model,
             "--host", "127.0.0.1", "--port", str(port),
             "--max-model-len", str(max_model_len),
             "--gpu-memory-utilization", str(gpu_memory_utilization)],
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

    def _wait_until_ready(self, health_url, timeout=1200):
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

    def chat(self, messages, **kwargs) -> str:
        """Return the reply. Extra kwargs (temperature, max_tokens, ...) go to the OpenAI API."""
        resp = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        return resp.choices[0].message.content

    def stream(self, messages, **kwargs):
        """Like `chat`, but yields the reply as it is generated."""
        chunks = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True, **kwargs
        )
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def close(self):
        """Stop the server and free the GPU. Safe to call more than once."""
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
    try:
        with LLM() as llm:
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
    print("\nServer stopped.")


if __name__ == "__main__":
    main()
