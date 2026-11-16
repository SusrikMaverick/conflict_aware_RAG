"""RAMDocs: MADAM-RAG, run from the authors' own code.

MADAM-RAG (arXiv:2504.13079) gives each document its own agent, lets the agents debate
for up to three rounds, and has an aggregator write the answer after each round. The
paper reports 26.4 for Qwen2.5-72B-Instruct.

The debate is not reimplemented here. `multi_agent_debate` is imported from the authors'
script (github.com/HanNight/RAMDocs, MIT), unchanged and checked against a pinned hash,
so the prompts, rounds and stopping rule are theirs. That script reaches the model through
one call, `generator(messages, ...)`. `Generator` answers it with our `LLM` in place of
their Hugging Face pipeline, and counts what the calls cost. Scoring and resuming are in
`src/ramdocs.py`.

    python -m src.benchmark_ramdocs_madam [--backend hf]
"""
import functools
import hashlib
import importlib.util
import sys
import urllib.request

from src import ramdocs

REFERENCE = ramdocs.ROOT / "benchmarks" / "MADAM-RAG" / "run_madam_rag.py"
REFERENCE_URL = ("https://raw.githubusercontent.com/HanNight/RAMDocs/"
                 "d44454c9ebb0bf513d8236a03decc9fb58704cad/run_madam_rag.py")
REFERENCE_SHA256 = "6869c70c954bd37525008db631f05b52946a198f8b314155b696ab1b4e00576b"
ROUNDS = 3  # the script's default


class Generator:
    """Stands in for the script's Hugging Face pipeline: the same call, answered by our
    `LLM`. One per question, so `cost` is that question's."""

    def __init__(self, llm):
        self.llm = llm
        self.cost = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "capped": 0}

    def __call__(self, messages, max_new_tokens, top_p, do_sample):
        # The script asks for greedy decoding and 128 new tokens, which is our DECODING.
        assert (max_new_tokens, top_p, do_sample) == (ramdocs.DECODING["max_tokens"], None, False)
        reply = self.llm.call(messages, **ramdocs.DECODING)
        self.cost["calls"] += 1
        self.cost["prompt_tokens"] += reply["prompt_tokens"]
        self.cost["completion_tokens"] += reply["completion_tokens"]
        self.cost["capped"] += reply["finish_reason"] == "length"
        return [{"generated_text": [*messages, {"role": "assistant", "content": reply["text"]}]}]


def fetch_reference():
    """Download the authors' script on first use, and stop unless it is the pinned file."""
    if not REFERENCE.exists():
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(REFERENCE_URL, REFERENCE)
    if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != REFERENCE_SHA256:
        sys.exit(f"{REFERENCE} is not the pinned MADAM-RAG script. Delete it to download it again.")


@functools.cache
def reference():
    """The authors' script as a module. Importing it loads torch and transformers, so it
    happens at the first question, once the model is up."""
    spec = importlib.util.spec_from_file_location("run_madam_rag", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def answer(llm, row):
    generator = Generator(llm)
    debate = reference().multi_agent_debate(
        row["question"], [d["text"] for d in row["documents"]], generator, num_rounds=ROUNDS)
    return {"output": debate["final_aggregation"], **generator.cost, "debate": debate}


if __name__ == "__main__":
    fetch_reference()
    ramdocs.main("madam", "MADAM-RAG", answer,
                 {"reference_sha256": REFERENCE_SHA256, "rounds": ROUNDS})
