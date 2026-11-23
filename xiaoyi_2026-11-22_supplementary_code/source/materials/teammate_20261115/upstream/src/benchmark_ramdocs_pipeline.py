"""RAMDocs: our pipeline (src/pipeline.py).

Each question's documents go to the pipeline as texts only, because a RAMDocs document's
other fields are its gold labels. The answers of the pipeline's entries become the
"All Correct Answers" list that the scorer reads.

`--mode` says what the pipeline does with a contradiction that nothing settles:
- forced (the default here): it picks the better-supported answer. RAMDocs wants one list
  of answers, so this is the score to compare with MADAM-RAG's.
- disagree: it reports both sides and lists neither, so RAMDocs counts the question as
  wrong. What matters then is how often the pipeline is right when it does answer.

`--memory` lets the model's own knowledge break ties, as a last resort. Off by default.

`--check N` is a quick test before a full run: N questions with the run's settings, and a
report on whether the model's replies could be read. It prints no answers and no score,
and writes nothing to the run files.

    python -m src.benchmark_ramdocs_pipeline [--mode disagree] [--memory] [--backend hf]
    python -m src.benchmark_ramdocs_pipeline --check 10 [--memory]
"""
import collections
import functools
import hashlib
import json
import statistics
import time
import traceback
from pathlib import Path

from src import pipeline, ramdocs


def answer(llm, row, mode, memory):
    documents = [{"text": d["text"]} for d in row["documents"]]  # the texts, never the gold labels
    result = pipeline.run(llm, row["question"], documents, mode, memory)
    listed = [a for e in result["entries"] for a in e["answers"]]
    output = f"All Correct Answers: {json.dumps(listed, ensure_ascii=False)}. Explanation: {result['response']}"
    cost = {k: result[k] for k in ("calls", "prompt_tokens", "completion_tokens", "capped")}
    return {"output": output, **cost, "status": result["status"], "entries": result["entries"],
            "trace": result["trace"]}


def report(rows, answers):
    """Beyond the score: how the questions ended, and how the settled ones scored."""
    def correct(ids):
        right = sum(ramdocs.score(answers[i]["output"], rows[i]["gold_answers"], rows[i]["wrong_answers"])
                    for i in ids)
        return f"{len(ids)} questions, {right} correct" + (f" = {100 * right / len(ids):.1f}" if ids else "")

    statuses = collections.Counter(a["status"] for a in answers.values())
    forced = [i for i, a in answers.items() if any(e["basis"].startswith("forced") for e in a["entries"])]
    settled = [i for i, a in answers.items() if a["status"] == "answered" and i not in forced]
    print("Status: " + ", ".join(f"{status} {n}" for status, n in statuses.most_common()))
    print(f'Settled by evidence (no forced pick, no "sources disagree"): {correct(settled)}')
    if forced:
        print(f"With a forced pick: {correct(forced)}")


def check(args, method):
    """The quick test. Questions spread over the benchmark, the same server and settings as
    a run, then one question asked again: a resumed run needs the same record back."""
    rows, _ = ramdocs.load()
    ids = list(range(0, len(rows), max(1, len(rows) // args.check)))[:args.check]
    name = f"check-pipeline-{args.mode}{'-memory' * args.memory}-{args.backend}.jsonl"
    path = ramdocs.ROOT / "runs" / name  # for finding faults, kept apart from the run files
    path.parent.mkdir(exist_ok=True)
    print(f"Quick test: {len(ids)} RAMDocs questions, mode {args.mode}, memory {'on' if args.memory else 'off'}, "
          f"backend {args.backend}.", flush=True)
    records, errors, seconds, same = [], [], [], None
    with ramdocs.LLM(args.backend, serve_args=ramdocs.SERVE_ARGS) as llm, open(path, "w", encoding="utf-8") as f:
        for n, i in enumerate(ids, 1):
            start = time.time()
            try:
                record = ramdocs.ask(llm, rows, i, method)
            except Exception:
                errors.append(i)
                print(f"[{n}/{len(ids)}] question {i}: ERROR\n{traceback.format_exc()}", flush=True)
                continue
            seconds.append(time.time() - start)
            records.append(record)
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"[{n}/{len(ids)}] question {i}: {len(rows[i]['documents'])} documents, {record['calls']} calls, "
                  f"{seconds[-1]:.0f} s", flush=True)
        if records:
            print(f"Asking question {records[0]['id']} again ...", flush=True)
            try:
                same = ramdocs.ask(llm, rows, records[0]["id"], method) == records[0]
            except Exception:
                print(traceback.format_exc(), flush=True)
                same = False

    calls = [c for r in records for c in r["trace"]["calls"]]
    steps = collections.defaultdict(collections.Counter)
    for c in calls:
        steps[c["step"]].update(calls=1, capped=c["capped"], unreadable=not c["json"])
    extract = [e for r in records for e in r["trace"]["extract"]]
    kept = sum(len(e["cards"]) for e in extract)
    dropped = sum(len(e["dropped"]) for e in extract)
    unread = sum("unread" in e for e in extract)
    close = sum(c["quote"] == "close" for e in extract for c in e["cards"])
    by_text = sum(r["trace"]["map"]["grouped_by"] == "text match" for r in records)
    no_choice = sum(e["basis"].startswith("forced: nothing favors") for r in records for e in r["entries"])
    total = collections.Counter()
    for counts in steps.values():
        total.update(counts)

    print("\nModel calls by step: calls, hit the token cap, reply had no readable JSON")
    for step, counts in steps.items():
        print(f"  {step:8} {counts['calls']:4} {counts['capped']:4} {counts['unreadable']:4}")
    print(f"  {'all':8} {total['calls']:4} {total['capped']:4} {total['unreadable']:4}")
    print(f"Documents whose cards could not be read at all: {unread} of {len(extract)}")
    print(f"Cards kept: {kept} ({close} with a nearly exact quote); dropped because the quote is not in "
          f"the document: {dropped}")
    print(f"Questions where the model's grouping could not be read (grouped by text instead): {by_text} "
          f"of {len(records)}")
    print(f"Forced picks where the model's choice could not be read: {no_choice}")
    print(f"Question asked twice, same record both times: {'yes' if same else 'NO'}")
    if seconds:
        mean = statistics.mean(seconds)
        print(f"Time: {mean:.0f} s per question, so about {mean * len(rows) / 3600:.1f} h for all {len(rows)} "
              f"(plus model startup)")
    print(f"Records, for finding faults: {path}")

    problems = [
        f"{len(errors)} questions failed with an error" if errors else "",
        "the repeated question came back different, so an interrupted run could not resume" if not same else "",
        "over 3% of replies had no readable JSON" if total["unreadable"] > 0.03 * total["calls"] else "",
        "over 3% of calls hit their token cap" if total["capped"] > 0.03 * total["calls"] else "",
        "over 15% of cards were dropped" if dropped > 0.15 * max(1, kept + dropped) else "",
        "the model's grouping could not be read in over 10% of questions" if by_text > 0.1 * len(records) else "",
    ]
    problems = [p for p in problems if p]
    if problems:
        print("\nNOT READY: " + "; ".join(problems) + ". Send this output before starting the full run.")
    else:
        print("\nLOOKS FINE: start the full run.")


if __name__ == "__main__":
    parser = ramdocs.parser("our pipeline")
    parser.add_argument("--mode", choices=pipeline.MODES, default="forced",
                        help='for a contradiction nothing settles: "forced" picks the better-supported answer '
                             '(default, the score to compare), "disagree" reports both sides')
    parser.add_argument("--memory", action="store_true",
                        help="let the model's own knowledge break ties, as a last resort")
    parser.add_argument("--check", type=int, metavar="N",
                        help="quick test before a full run: N questions, a report on whether the replies could "
                             "be read; no answers, no score, nothing written to the run files")
    args = parser.parse_args()
    method = functools.partial(answer, mode=args.mode, memory=args.memory)
    if args.check:
        check(args, method)
    else:
        code = Path(pipeline.__file__).read_bytes() + Path(__file__).read_bytes()
        rows, answers = ramdocs.main(
            f"pipeline-{args.mode}" + "-memory" * args.memory,
            f"our pipeline, {args.mode}" + ", with memory" * args.memory,
            method,
            {"code_sha256": hashlib.sha256(code).hexdigest(), "mode": args.mode, "memory": args.memory,
             "decoding": pipeline.DECODING},  # each step's token cap is in the code
            args.backend)
        report(rows, answers)
