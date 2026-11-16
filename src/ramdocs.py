"""RAMDocs: what every method's benchmark shares. The data, the paper's score, and a
resumable run.

A method is a function `answer(llm, row)` that answers one question and returns
{"output", "calls", "prompt_tokens", "completion_tokens", "capped", ...}. `main` runs it
over the 500 questions and prints exact-match accuracy, plus the LLM calls and tokens used.

Resumable, without changing the result:
- Each answer is appended to runs/ramdocs-<method>-<backend>.jsonl as soon as it arrives,
  and a rerun asks only the questions missing from that file.
- Requests go to the model one at a time, and the vLLM server keeps no cached prompt
  prefixes, so every answer is computed the same way whatever ran before it. (With
  batching, vLLM's greedy output can depend on which other requests share the batch.)
- A resumed run first re-asks a few answered questions and stops unless they come back
  identical.
- The file's first line records everything that decides the answers (data, method,
  backend, model revision, library versions, GPU, server flags, decoding). A rerun under
  different settings refuses to add to the file.
- No score is printed until all 500 are in. Once they are, a rerun just rescores,
  without the GPU.

A method with arguments of its own adds them to `parser(title)`, parses them, and passes
the backend on to `main`.
"""
import argparse
import hashlib
import json
import re
import string
import subprocess
import sys
import time
import urllib.request
from importlib.metadata import version
from pathlib import Path

from src.interface import LLM, MODELS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "RAMDocs" / "RAMDocs_test.jsonl"
DATA_URL = ("https://huggingface.co/datasets/HanNight/RAMDocs/resolve/"
            "9c041bfd158c603b615883d9a931b00cbc141494/RAMDocs_test.jsonl")  # unchanged since the paper

# The paper's decoding: greedy, 128 new tokens. Their Hugging Face greedy decoding still
# applied Qwen's default repetition penalty, so we set it too.
DECODING = {"temperature": 0, "max_tokens": 128, "extra_body": {"repetition_penalty": 1.05}}

# One sequence at a time and no reuse of cached prompt prefixes, so each answer depends
# only on its own prompt.
SERVE_ARGS = ["--max-num-seqs", "1", "--no-enable-prefix-caching"]
CHECKS = 3  # answered questions re-asked when a run resumes

PACKAGES = {"vllm": ["vllm"], "hf": ["transformers", "torch", "bitsandbytes", "accelerate"]}
PUNCT = set(string.punctuation + "‘’“”–—")


def normalize(s):
    """Lowercase, drop punctuation and articles, collapse spaces (as in MADAM-RAG's code)."""
    s = "".join(ch for ch in s.lower() if ch not in PUNCT)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def words(answer):
    """Regex matching a normalized answer as whole words of normalized text."""
    return re.compile(rf"(?<!\S){re.escape(answer)}(?!\S)")


def score(output, gold, wrong):
    """The paper's strict exact match: the answer list has every gold answer and no
    answer that only misinformation documents support."""
    listed = re.search(r"All Correct Answers:(.*?)(?:Explanation:|$)", output, re.S | re.I)
    text = normalize(listed.group(1) if listed else output)
    gold = {normalize(g) for g in gold}
    if not all(words(g).search(text) for g in gold):
        return False
    for g in gold:  # so "football" inside "american football" doesn't count as a wrong answer
        text = words(g).sub("|", text)
    wrong = {normalize(w) for w in wrong} - gold  # 11 questions also list a gold answer as wrong
    return not any(words(w).search(text) for w in wrong)


def load():
    """RAMDocs rows and the file's SHA-256, downloading the file on first use."""
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA)
    raw = DATA.read_bytes()
    return [json.loads(line) for line in raw.decode().splitlines()], hashlib.sha256(raw).hexdigest()


def settings(data_sha256, backend, method):
    """Everything that decides the answers. A run file only takes answers made with these.
    `method` is the method's own part: its name and what fixes its prompts."""
    model = MODELS[backend]
    downloads = Path(model, ".cache", "huggingface", "download").glob("*.metadata")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.splitlines()[0]
    return json.loads(json.dumps({
        "benchmark": "RAMDocs", "data_sha256": data_sha256, "backend": backend,
        "model": model, "model_revision": sorted({p.read_text().split("\n")[0] for p in downloads}),
        "versions": {p: version(p) for p in PACKAGES[backend]}, "gpu": gpu.strip(),
        "serve_args": SERVE_ARGS if backend == "vllm" else [], "decoding": DECODING,
        **method,  # last: a method that decodes differently says so here
    }))


def read_run(path):
    """The run file's settings and its answers by question index; (None, {}) if there are
    none yet. Drops a partial last line left by a kill mid-write."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
        if path.exists():
            path.write_text(text, encoding="utf-8")
    if not text:
        return None, {}
    first, *answers = text.splitlines()
    return json.loads(first)["config"], {a["id"]: a for a in map(json.loads, answers)}


def ask(llm, rows, i, answer):
    """Question `i`'s line for the run file, in the form `read_run` gives back."""
    return json.loads(json.dumps({"id": i, "question": rows[i]["question"], **answer(llm, rows[i])}))


def check_reproducible(llm, rows, answers, answer, path):
    """Re-ask a few answered questions, spread over the run. Resuming is only valid if
    they come back identical. These calls are neither saved nor counted."""
    ids = list(answers)[:: max(1, len(answers) // CHECKS)][:CHECKS]
    for i in ids:
        if ask(llm, rows, i, answer) != answers[i]:
            sys.exit(f"Question {i} came back different after the restart, so resuming would "
                     f"mix two runs. Move {path} aside and start fresh.")
    print(f"Reproducibility check: {len(ids)} answered questions came back identical.",
          file=sys.stderr)


def run(llm, rows, todo, path, answer):
    """Ask the questions in `todo` one at a time, appending each answer to `path` as it
    arrives."""
    start = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for done, i in enumerate(todo, 1):
            try:
                f.write(json.dumps(ask(llm, rows, i, answer)) + "\n")
                f.flush()
            except Exception as e:  # left out of the file, so the next run asks it again
                print(f"\nQuestion {i} failed: {e!r}", file=sys.stderr)
            minutes = (time.time() - start) / 60
            left = minutes / done * (len(todo) - done)
            print(f"\r{done}/{len(todo)} answered, {minutes:.1f} min, ~{left:.0f} min left ",
                  end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)


def report(rows, answers, title, path):
    n = len(rows)
    k = sum(score(a["output"], rows[i]["gold_answers"], rows[i]["wrong_answers"])
            for i, a in answers.items())
    p = k / n
    ci = 1.96 * (p * (1 - p) / n) ** 0.5
    calls = sum(a["calls"] for a in answers.values())
    tokens_in = sum(a["prompt_tokens"] for a in answers.values()) / n
    tokens_out = sum(a["completion_tokens"] for a in answers.values()) / n
    capped = sum(a["capped"] for a in answers.values())
    print(f"RAMDocs, {title}: {k}/{n} correct = {100 * p:.1f} ± {100 * ci:.1f} (95% CI)")
    print(f"LLM calls: {calls} ({calls / n:.1f} per question); per question {tokens_in:.0f} "
          f"input and {tokens_out:.0f} output tokens; {capped}/{calls} calls hit their token cap")
    print(f"Outputs: {path}")


def parser(title):
    """The arguments every RAMDocs benchmark takes. A method may add its own."""
    p = argparse.ArgumentParser(description=f"RAMDocs, {title}.")
    p.add_argument("--backend", choices=list(MODELS), default="vllm",
                   help="vllm: AWQ weights, fast (default). hf: NF4 with transformers, "
                        "as in the MADAM-RAG paper, much slower.")
    return p


def main(name, title, answer, fingerprint, backend=None):
    """Run the method `answer` over RAMDocs, or finish its interrupted run, and print the
    score. `name` goes in the run file's name, `title` in the report, and `fingerprint`
    (what fixes the method's prompts) in the run file's settings. `backend` is read from
    the command line unless the caller has parsed it. Returns the rows and the answers."""
    backend = backend or parser(title).parse_args().backend
    path = ROOT / "runs" / f"ramdocs-{name}-{backend}.jsonl"

    rows, data_sha256 = load()
    config, answers = read_run(path)
    todo = [i for i in range(len(rows)) if i not in answers]
    if todo:
        current = settings(data_sha256, backend, {"method": name, **fingerprint})
        if config is None:
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({"config": current}) + "\n", encoding="utf-8")
        elif config != current:
            sys.exit(f"{path} was made with different settings. Move it aside to start a new run.")
        print(f"{len(answers)} answers already in {path}; asking the other {len(todo)}.",
              file=sys.stderr)
        with LLM(backend, serve_args=SERVE_ARGS) as llm:
            if answers:
                check_reproducible(llm, rows, answers, answer, path)
            run(llm, rows, todo, path, answer)
        _, answers = read_run(path)

    if len(answers) < len(rows):
        sys.exit(f"Only {len(answers)}/{len(rows)} answered; rerun to finish. "
                 "No score until every question is in.")
    report(rows, answers, title, path)
    return rows, answers
