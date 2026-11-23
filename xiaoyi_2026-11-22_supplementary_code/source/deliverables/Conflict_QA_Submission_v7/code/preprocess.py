#!/usr/bin/env python3
"""Recompute baseline NLP steps and compare with the historical records."""
import argparse
import sys
from pathlib import Path
import run

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))


def main():
    from baseline_nlp import Preprocessor, records

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["trust", "robust", "faithful"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--fresh-embeddings",
        action="store_true",
        help="Recompute SimCSE vectors instead of loading the bundled vectors",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.fresh_embeddings and args.method != "trust":
        raise ValueError("--fresh-embeddings applies to trust only")
    output = run.checked_output(args.output)
    rows = run.validate_inputs(run.read(BASE / "data/inputs.json"))
    if args.limit is not None:
        rows = rows[: args.limit]
    saved = Preprocessor(replay=True)
    fresh = Preprocessor()
    isolates = (
        {
            r["output"]["question_id"]: r["output"]
            for r in records("stages.jsonl.gz")
            if r["kind"] == "robust_isolated"
        }
        if args.method == "robust"
        else {}
    )
    mismatches, score_deltas = [], []
    for i, row in enumerate(rows, 1):
        q = row["question_id"]
        if args.method == "trust":
            import numpy as np

            embeddings = (
                None
                if args.fresh_embeddings
                else np.load(
                    BASE / "data/baselines/trust_embeddings" / (q + ".npy"),
                    allow_pickle=False,
                )
            )
            actual = fresh.filter(row, embeddings)
            reference = saved.filters[q]
            keys = ["documents", "kept_count", "original_count", "guard"]
        elif args.method == "robust":
            responses = [s["text"] for s in isolates[q]["stages"]]
            actual = fresh.keywords(row, responses)
            reference = saved.hints[q]
            keys = ["hints", "final_prompt", "abstained", "certified"]
        else:
            historical = saved.faithful[q]
            top = fresh.align(row, historical["mined_facts"])
            original = historical["selected_chunks"]
            actual, reference = {"selected_chunks": top}, {"selected_chunks": original}
            # Floating-point scores can vary across CPU builds. The final prompt
            # depends on selected chunk texts/order, which must match exactly.
            actual["prompt_chunks"] = [
                [s["chunk"] for s in t["topk_chunks"]] for t in top
            ]
            reference["prompt_chunks"] = [
                [s["chunk"] for s in t["topk_chunks"]] for t in original
            ]
            keys = ["prompt_chunks"]
            if actual["prompt_chunks"] == reference["prompt_chunks"]:
                score_deltas += [
                    abs(a["score"] - b["score"])
                    for ta, tb in zip(top, original)
                    for a, b in zip(ta["topk_chunks"], tb["topk_chunks"])
                ]
        different = [k for k in keys if actual[k] != reference[k]]
        if different:
            mismatches.append(dict(question_id=q, fields=different))
        run.write(output / "records" / (q + ".json"), actual)
        if i % 50 == 0 or i == len(rows):
            print(f"{args.method}: {i}/{len(rows)}", flush=True)
    report = dict(
        method=args.method,
        questions=len(rows),
        passed=not mismatches,
        mismatches=mismatches,
        fresh_embeddings=args.fresh_embeddings or args.method == "faithful",
        max_alignment_score_delta=max(score_deltas, default=0),
    )
    run.write(output / "validation.json", report)
    if mismatches:
        raise ValueError(
            f"{len(mismatches)} preprocessing differences; inspect validation.json"
        )


if __name__ == "__main__":
    main()
