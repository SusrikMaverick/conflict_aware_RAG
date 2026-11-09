"""Run fresh Plain RAG and Docs + saved cards through one final model.

    python -m src.compare --bundle path/to/RAMDocs_Reproducible_Package.zip --dry-run
    python -m src.compare --bundle path/to/package.zip --out runs/ramdocs-72b-paired

The existing cards were extracted with 7B. This comparison changes the final
answerer only; it does not regenerate cards or reuse historical final answers.
"""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import threading
import time
import zipfile

from src.eval import PROMPT, normalize, score, words


METHODS = ("plain", "plus_cards")
FIELDS = ("doc_id", "card_id", "entity", "scope", "claim", "answer", "quote")
SUFFIX = ("\n\nAdditional extracted evidence from the SAME documents (not independent "
          "sources and not guaranteed true). Use alongside the original documents above.\n")
PLAN = "planning/plain_plus_cards_v1"
INPUTS = "materials/prepared/v1/ramdocs.inputs.jsonl"
ANNOTATIONS = "materials/prepared/v1/ramdocs.evaluation.jsonl"


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


class Bundle:
    """Read an extracted bundle or its ZIP without extracting or executing it."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.archive = zipfile.ZipFile(self.path) if self.path.is_file() else None
        self.prefix = ""
        if self.archive is not None:
            names = self.archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("The ZIP has duplicate paths")
            if f"ramdocs_repro_bundle/{INPUTS}" in names:
                self.prefix = "ramdocs_repro_bundle/"

    def read(self, name):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError(f"Invalid bundle path: {name}")
        if self.archive is not None:
            return self.archive.read(self.prefix + str(relative))
        path = (self.path / str(relative)).resolve()
        if not path.is_relative_to(self.path):
            raise ValueError(f"Bundle path leaves the bundle: {name}")
        return path.read_bytes()

    def close(self):
        if self.archive is not None:
            self.archive.close()


def unique_rows(rows, name):
    result = {}
    for row in rows:
        qid = row["question_id"]
        if qid in result:
            raise ValueError(f"Duplicate question ID in {name}: {qid}")
        result[qid] = row
    return result


def build_prompts(case):
    """Only question, original text and projected cards enter model prompts."""
    documents = "\n".join(
        f"Document {i}: {doc['text']}" for i, doc in enumerate(case["documents"], 1)
    )
    plain = PROMPT.format(question=case["question"], documents=documents)
    return {"plain": plain, "plus_cards": plain + SUFFIX + json.dumps(case["cards"], ensure_ascii=False)}


def load_bundle(path):
    """Validate frozen data, document order and cards; return cases + provenance."""
    bundle = Bundle(path)
    hashes = {}

    def read(name):
        contents = bundle.read(name)
        hashes[name] = digest(contents)
        return contents

    try:
        protocol = json.loads(read(f"{PLAN}/protocol.json"))
        inputs = unique_rows(map(json.loads, read(INPUTS).splitlines()), INPUTS)
        annotations = unique_rows(map(json.loads, read(ANNOTATIONS).splitlines()), ANNOTATIONS)
        frozen = unique_rows(json.loads(read(f"{PLAN}/inputs.json")), "frozen inputs")
        if not inputs or set(inputs) != set(annotations) or set(inputs) != set(frozen):
            raise ValueError("Input, annotation and frozen question IDs must match exactly")
        if len(inputs) != protocol["n"]:
            raise ValueError("Question count differs from the frozen protocol")
        snapshot = ast.parse(read("conflict_aware_RAG/src/eval.py"))
        original_prompt = next(
            ast.literal_eval(node.value) for node in snapshot.body
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "PROMPT" for target in node.targets
            )
        )
        if original_prompt != PROMPT:
            raise ValueError("Current Plain prompt differs from the frozen package prompt")

        cases = []
        for qid, row in inputs.items():
            saved = frozen[qid]
            source = json.loads(read(saved["source"]))
            if source["question_id"] != qid:
                raise ValueError(f"Card source question ID differs: {qid}")
            documents = row["documents"]
            numbers = {doc["doc_id"]: i for i, doc in enumerate(documents, 1)}
            if len(numbers) != len(documents):
                raise ValueError(f"Duplicate document ID: {qid}")
            cards = [
                {**{key: card[key] for key in FIELDS if key in card},
                 "original_document_number": numbers[card["doc_id"]]}
                for card in source.get("cards", [])
            ]
            if cards != saved["cards"]:
                raise ValueError(f"Saved card fields, values or order differ: {qid}")
            gold = annotations[qid]["gold_answers"]
            wrong = annotations[qid]["wrong_answers"]
            if not all(isinstance(values, list) and all(isinstance(x, str) for x in values)
                       for values in (gold, wrong)):
                raise ValueError(f"Answer annotations must be lists of strings: {qid}")
            case = {"question_id": qid, "question": row["question"],
                    "documents": [{"doc_id": d["doc_id"], "text": d["text"]} for d in documents],
                    "cards": cards, "gold_answers": gold, "wrong_answers": wrong,
                    "source_status": saved["source_status"]}
            if build_prompts(case)["plus_cards"] != saved["prompt"]:
                raise ValueError(f"Question text, document order or frozen prompt differs: {qid}")
            cases.append(case)
        for name, observed in hashes.items():
            expected = protocol.get("hashes", {}).get(name)
            if expected is not None and observed != expected:
                raise ValueError(f"Frozen source hash differs: {name}")
        return cases, {"bundle": str(bundle.path), "source_hashes": hashes,
                       "extraction_model": protocol["model"], "available_questions": len(cases),
                       "card_policy": protocol["intervention"]}
    finally:
        bundle.close()


def generation_settings(max_tokens=128, seed=576):
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    return {"temperature": 0, "max_tokens": max_tokens, "seed": seed,
            "extra_body": {"repetition_penalty": 1.05}}


def parse_answers(text):
    """Use the archived answer parser, without repairing substantive answers."""
    match = re.search(r"All Correct Answers:\s*(\[.*?\])", text, re.S | re.I)
    if match is None:
        if text.strip().lower().strip(" .'\"") == "unknown":
            return []
        raise ValueError("missing answer list")
    try:
        answers = json.loads(match.group(1))
    except json.JSONDecodeError:
        answers = ast.literal_eval(match.group(1))
    if not isinstance(answers, list) or not all(
        isinstance(answer, str) or (type(answer) in (int, float) and math.isfinite(answer))
        for answer in answers
    ):
        raise ValueError("non-atomic answers")
    return [str(answer) for answer in answers
            if str(answer).lower().strip() not in ("unknown", "none", "n/a", "null")]


def assess(text, gold, wrong):
    """Apply identical historical scoring + strict diagnostics to either method."""
    try:
        answers = parse_answers(text)
        status = "answered" if answers else "abstain"
    except (ValueError, SyntaxError):
        answers, status = [], "parse_error"
    predicted = {normalize(answer) for answer in answers}
    gold_set = {normalize(answer) for answer in gold}
    wrong_set = {normalize(answer) for answer in wrong} - gold_set
    remaining = " ".join(sorted(predicted))
    for answer in gold_set:
        remaining = words(answer).sub("|", remaining)
    return {"status": status, "accepted_answers": answers,
            "official_correct": bool(score(text, gold, wrong)),
            "strict_correct": status != "parse_error" and predicted == gold_set,
            "wrong_answer_accepted": any(words(answer).search(remaining) for answer in wrong_set)}


def summarize(records, questions):
    summary = {"questions": questions}
    for method in METHODS:
        outputs = [record for record in records if record["method"] == method]
        correct = sum(record["official_correct"] for record in outputs)
        strict = sum(record["strict_correct"] for record in outputs)
        summary[method] = {
            "questions": questions, "completed_records": len(outputs),
            "official_correct": correct, "official_accuracy": correct / questions,
            "strict_correct": strict, "strict_accuracy": strict / questions,
            "wrong_answer_questions": sum(record["wrong_answer_accepted"] for record in outputs),
            "parse_errors": sum(record["status"] == "parse_error" for record in outputs),
            "request_errors": sum(record["status"] == "error" for record in outputs),
            "output_limit_hits": sum(record.get("finish_reason") == "length" for record in outputs),
            **{key: sum(record.get("usage", {}).get(key, 0) or 0 for record in outputs)
               for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        }
    by_question = {}
    for record in records:
        by_question.setdefault(record["question_id"], {})[record["method"]] = record
    deltas = [int(pair["plus_cards"]["official_correct"]) - int(pair["plain"]["official_correct"])
              for pair in by_question.values() if set(pair) == set(METHODS)]
    summary["paired"] = {"completed_pairs": len(deltas), "gains": sum(d > 0 for d in deltas),
                         "losses": sum(d < 0 for d in deltas), "accuracy_difference": sum(deltas) / questions}
    summary["cost_scope"] = "New final-answer calls only; saved card extraction/verification cost is not rerun or included."
    return summary


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_run(out, cases, provenance, model, max_tokens=128, seed=576, workers=8):
    """Reserve a new directory and record settings before starting the model."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {**provenance, "status": "prepared", "started_utc": datetime.now(timezone.utc).isoformat(),
                "final_model": str(model), "generation": generation_settings(max_tokens, seed),
                "workers": workers, "questions": len(cases),
                "question_ids": [case["question_id"] for case in cases],
                "subset": len(cases) != provenance["available_questions"],
                "prompt_template_sha256": digest(PROMPT.encode("utf-8")),
                "scorer": "src.eval.score (historical raw-text compatible) plus strict parsed-set diagnostics",
                "scorer_source_sha256": digest(Path(__file__).with_name("eval.py").read_bytes()),
                "runner_source_sha256": digest(Path(__file__).read_bytes()),
                "final_answers": "Both methods freshly generated with the same model and settings; no old outputs reused.",
                "scope": "Frozen 7B-extracted cards with the selected final answerer; not end-to-end re-extraction at that model size.",
                "context_policy": "No prompt truncation; request/context failures count as incorrect for that method."}
    write_json(out / "manifest.json", manifest)
    with (out / "prompts.jsonl").open("x", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps({"question_id": case["question_id"], **build_prompts(case)}, ensure_ascii=False) + "\n")
    return manifest


def run(llm, cases, *, out, workers=8, max_tokens=128, seed=576, _prepared=False):
    """Make independent paired calls; save every completion/failure immediately."""
    if not cases or workers < 1:
        raise ValueError("Need at least one case and one worker")
    settings = generation_settings(max_tokens, seed)
    out = Path(out)
    if not _prepared:
        out.mkdir(parents=True, exist_ok=False)
    lock = threading.Lock()
    records = []
    files = {}
    try:
        for method in METHODS:
            files[method] = (out / f"{method}.jsonl").open("x", encoding="utf-8")
        files["paired"] = (out / "paired.jsonl").open("x", encoding="utf-8")

        def one(case):
            pair = {"question_id": case["question_id"], "question": case["question"]}
            for method, prompt in build_prompts(case).items():
                start = time.monotonic()
                record = {"question_id": case["question_id"], "method": method,
                          "question": case["question"], "card_count": len(case["cards"]) if method == "plus_cards" else 0,
                          "source_status": case["source_status"], "prompt_sha256": digest(prompt.encode("utf-8")),
                          "gold_answers": case["gold_answers"], "wrong_answers": case["wrong_answers"]}
                try:
                    response = llm.complete([{"role": "user", "content": prompt}], **settings)
                    text = response["text"]
                    if not isinstance(text, str):
                        raise ValueError("Model response text is not a string")
                    record.update(raw_output=text, finish_reason=response.get("finish_reason"),
                                  usage=response.get("usage") or {},
                                  **assess(text, case["gold_answers"], case["wrong_answers"]))
                except Exception as exc:
                    record.update(status="error", raw_output="", accepted_answers=[],
                                  official_correct=False, strict_correct=False, wrong_answer_accepted=False,
                                  finish_reason=None, usage={}, error=f"{type(exc).__name__}: {exc}")
                record["wall_seconds"] = time.monotonic() - start
                pair[method] = {key: record[key] for key in (
                    "status", "raw_output", "accepted_answers", "official_correct",
                    "strict_correct", "finish_reason")}
                with lock:
                    files[method].write(json.dumps(record, ensure_ascii=False) + "\n")
                    files[method].flush()
                    records.append(record)
            pair["official_change"] = int(pair["plus_cards"]["official_correct"]) - int(pair["plain"]["official_correct"])
            pair["strict_change"] = int(pair["plus_cards"]["strict_correct"]) - int(pair["plain"]["strict_correct"])
            with lock:
                files["paired"].write(json.dumps(pair, ensure_ascii=False) + "\n")
                files["paired"].flush()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, case) for case in cases]
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                print(f"\r{completed}/{len(cases)} paired questions", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
    finally:
        for file in files.values():
            file.close()
    summary = summarize(records, len(cases))
    write_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Extracted bundle directory or original ZIP")
    parser.add_argument("--out", type=Path, help="New output directory; existing directories are never reused")
    parser.add_argument("--model", help="Local model path (defaults to the existing 72B AWQ setup)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Same final output cap for both methods")
    parser.add_argument("--seed", type=int, default=576)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="First N questions for a clearly labeled smoke run")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs/prompts only; no model or output writes")
    args = parser.parse_args(argv)
    if args.max_tokens < 1 or args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("max-tokens, workers and limit must be positive")
    if not args.dry_run and args.out is None:
        parser.error("Live generation requires --out pointing to a new directory")
    try:
        cases, provenance = load_bundle(args.bundle)
    except (OSError, ValueError, KeyError, StopIteration, zipfile.BadZipFile) as exc:
        parser.error(f"Cannot validate bundle: {exc}")
    if args.limit is not None:
        cases = cases[:args.limit]
    if args.dry_run:
        print(json.dumps({"validated_questions": len(cases), "available_questions": provenance["available_questions"],
                          "saved_cards": sum(len(case["cards"]) for case in cases),
                          "fresh_final_calls_if_run": 2 * len(cases), "generation": generation_settings(args.max_tokens, args.seed),
                          "extraction_model": provenance["extraction_model"],
                          "note": "Validation only. No model started and no output files written."}, indent=2))
        return 0

    from src.interface import LLM, MODEL

    model = args.model or MODEL
    try:
        manifest = prepare_run(args.out, cases, provenance, model, args.max_tokens, args.seed, args.workers)
    except OSError as exc:
        parser.error(f"Cannot create new output directory: {exc}")
    try:
        with LLM(model=model) as llm:
            summary = run(llm, cases, out=args.out, workers=args.workers,
                          max_tokens=args.max_tokens, seed=args.seed, _prepared=True)
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}", finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.out / "manifest.json", manifest)
        print(f"Run stopped; available records are preserved in {args.out}: {exc}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.out / "manifest.json", manifest)
    for method in METHODS:
        metrics = summary[method]
        print(f"{method}: {metrics['official_correct']}/{len(cases)} = {100 * metrics['official_accuracy']:.1f}% "
              f"historical-compatible; strict {metrics['strict_correct']}/{len(cases)}")
    print(f"Outputs: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
