"""RAMDocs evaluation: the MADAM-RAG paper's plain-RAG baseline, scored like the paper.

Sends each question with all its documents in one prompt (the "concatenated prompt"
from arXiv:2504.13079, Appendix E) and prints exact-match accuracy over the 500
questions. The paper reports 20.6 for Qwen2.5-72B-Instruct. Every output is saved
under runs/, so it can be rescored without the GPU.

    python -m src.eval
"""
import json
import re
import string
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "RAMDocs" / "RAMDocs_test.jsonl"
DATA_URL = ("https://huggingface.co/datasets/HanNight/RAMDocs/resolve/"
            "9c041bfd158c603b615883d9a931b00cbc141494/RAMDocs_test.jsonl")  # unchanged since the paper
RUNS = ROOT / "runs"

# The paper's decoding: greedy, 128 new tokens. Their Hugging Face greedy decoding still
# applied Qwen's default repetition penalty, so we set it too.
DECODING = {"temperature": 0, "max_tokens": 128, "extra_body": {"repetition_penalty": 1.05}}

# Appendix E, with the typeset quotes straightened.
PROMPT = """You are an expert in retrieval question answering.
You will be provided a question with multiple documents. Please answer the question based on the documents.
If there are multiple answers, please provide all possible correct answers and also provide a step-by-step reasoning explanation. If there is no correct answer, please reply 'unknown'.
Please follow the format: "All Correct Answers: []. Explanation: {{}}"

The following are examples:
Question: In which year was Michael Jordan born?
Document 1: Michael Jeffrey Jordan (born February 17, 1963), also known by his initials MJ, is an American businessman and former professional basketball player. He played 15 seasons in the National Basketball Association (NBA) between 1984 and 2003, winning six NBA championships with the Chicago Bulls. He was integral in popularizing basketball and the NBA around the world in the 1980s and 1990s, becoming a global cultural icon.
Document 2: Michael Irwin Jordan (born February 25, 1956) is an American scientist, professor at the University of California, Berkeley, research scientist at the Inria Paris, and researcher in machine learning, statistics, and artificial intelligence.
Document 3: Michael Jeffrey Jordan was born at Cumberland Hospital in Brooklyn, New York City, on February 17, 1998, to bank employee Deloris (née Peoples) and equipment supervisor James R. Jordan Sr. He has two older brothers, James Jr. and Larry, as well as an older sister named Deloris and a younger sister named Roslyn. Jordan and his siblings were raised Methodist.
Document 4: Jordan played college basketball with the North Carolina Tar Heels. As a freshman, he was a member of the Tar Heels' national championship team in 1982. Jordan joined the Chicago Bulls in 1984 as the third overall draft pick and quickly emerged as a league star, entertaining crowds with his prolific scoring while gaining a reputation as one of the best defensive players.
All Correct Answers: ["1963", "1956"]. Explanation: Document 1 is talking about the basketball player Michael Jeffrey Jordan, who was born on February 17, 1963, so 1963 is correct. Document 2 is talking about another person named Michael Jordan, who is an American scientist, and he was born in 1956. Therefore, the answer 1956 from Document 2 is also correct. Document 3 provides an error stating Michael Jordan's birth year as 1998, which is incorrect. Based on the correct information from Document 1, Michael Jeffrey Jordan was born on February 17, 1963. Document 4 does not provide any useful information.

Question: {question}
{documents}"""

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


def ask(llm, row):
    docs = "\n".join(f"Document {i}: {d['text']}" for i, d in enumerate(row["documents"], 1))
    prompt = PROMPT.format(question=row["question"], documents=docs)
    return llm.chat([{"role": "user", "content": prompt}], **DECODING)


def run(llm, rows, workers=32):
    """Answer and score every row, with many requests in flight so vLLM batches them."""
    outputs = []
    with ThreadPoolExecutor(workers) as pool:
        for out in pool.map(lambda row: ask(llm, row), rows):
            outputs.append(out)
            print(f"\r{len(outputs)}/{len(rows)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return [{"question": r["question"], "gold_answers": r["gold_answers"],
             "wrong_answers": r["wrong_answers"], "output": out,
             "correct": score(out, r["gold_answers"], r["wrong_answers"])}
            for r, out in zip(rows, outputs)]


def load():
    """RAMDocs rows, downloading the file on first use."""
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA)
    return [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]


def main():
    from src.interface import LLM

    rows = load()
    with LLM() as llm:
        records = run(llm, rows)
    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"ramdocs-concat-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    k, n = sum(r["correct"] for r in records), len(records)
    p = k / n
    ci = 1.96 * (p * (1 - p) / n) ** 0.5
    print(f"RAMDocs, concatenated prompt: {k}/{n} correct = {100 * p:.1f} ± {100 * ci:.1f} (95% CI)")
    print(f"Outputs: {path}")


if __name__ == "__main__":
    main()
