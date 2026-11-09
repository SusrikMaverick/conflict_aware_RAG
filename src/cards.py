"""Create C cards or check them for D, with restartable per-document records."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time
import uuid

from src.compare import digest, load_bundle, unique_rows
from src.card_extraction import extract_document


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def record_path(out, case, document):
    return Path(out) / "records" / (digest([case["question_id"], document["doc_id"]]) + ".json")


def tasks(cases, parent_rows=None):
    for case in cases:
        for document in case["documents"]:
            cards = [] if parent_rows is None else [
                card for card in parent_rows[case["question_id"]]["cards"]
                if card["doc_id"] == document["doc_id"]
            ]
            yield case, document, cards


def pending_tasks(out, cases, parent_rows=None):
    pending = []
    for case, document, cards in tasks(cases, parent_rows):
        path = record_path(out, case, document)
        expected = digest([case["question"], document, cards])
        if path.exists():
            saved = read_json(path)
            if saved.get("task_sha256") != expected:
                raise ValueError("Saved document inputs changed; use a new output directory")
            if saved.get("status") == "complete":
                continue
        pending.append((case, document, cards))
    return pending


def prepare_stage(out, cases, provenance, model, stage, *, seed=576, max_tokens=None,
                  resume=False, parent=None, workers=4):
    out = Path(out)
    max_tokens = max_tokens if max_tokens is not None else (1024 if stage == "extract" else 2048)
    module = "card_extraction.py" if stage == "extract" else "card_review.py"
    code = {name: digest(Path(__file__).with_name(name).read_bytes())
            for name in ("cards.py", "card_extraction.py", module)}
    config = {"stage": stage, "model": str(model), "seed": seed, "max_tokens": max_tokens, "workers": workers,
              "question_ids": [case["question_id"] for case in cases],
              "bundle_hashes": provenance["source_hashes"], "code_hashes": code, "parent": parent}
    fingerprint = digest(config)
    if resume:
        manifest = read_json(out / "manifest.json")
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Resume inputs, model, settings or code changed; use a new output directory")
        return manifest
    out.mkdir(parents=True, exist_ok=False)
    (out / "records").mkdir()
    manifest = {**config, "fingerprint": fingerprint, "status": "prepared",
                "available_questions": provenance["available_questions"], "questions": len(cases),
                "subset": len(cases) != provenance["available_questions"],
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "extraction_model": str(model) if stage == "extract" else parent["extraction_model"],
                "source_truth_check": False,
                "policy": "All source-located C cards; MiniCheck not rerun or used for filtering."
                if stage == "extract" else "Question-fit audit of unchanged C cards; unsupported conditions reject the card."}
    atomic_json(out / "manifest.json", manifest)
    return manifest


def run_stage(llm, cases, *, out, stage, seed=576, max_tokens=None, workers=4, parent_rows=None):
    """Save each document atomically; resume retries only failed or missing tasks."""
    if not cases or workers < 1:
        raise ValueError("Need at least one question and one worker")
    out = Path(out)
    manifest = read_json(out / "manifest.json")
    max_tokens = max_tokens if max_tokens is not None else (1024 if stage == "extract" else 2048)
    if stage == "extract":
        operation = extract_document
    else:
        from src.card_review import review_document
        operation = review_document
    pending = pending_tasks(out, cases, parent_rows)
    manifest.update(status="running", pending_documents=len(pending))
    atomic_json(out / "manifest.json", manifest)
    start = time.monotonic()

    def one(task):
        case, document, cards = task
        path = record_path(out, case, document)
        previous = read_json(path) if path.exists() else {}
        requests = previous.get("requests", [])
        errors = previous.get("errors", [])
        record = {"question_id": case["question_id"], "doc_id": document["doc_id"],
                  "task_sha256": digest([case["question"], document, cards]),
                  "cards": [], "issues": [], "decisions": []}
        try:
            args = (llm, case["question"], document) + ((cards,) if stage == "review" else ())
            result = operation(*args, seed=seed, max_tokens=max_tokens)
            record.update(status="complete", cards=result["cards"], issues=result.get("issues", []),
                          decisions=result.get("decisions", []), skipped=result.get("skipped", result["status"] == "skipped"))
            request = result.get("request")
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            request = getattr(exc, "request", None)
            errors = [*errors, record["error"]]
        if request and not request.get("skipped"):
            requests = [*requests, request]
        record.update(requests=requests, errors=errors)
        atomic_json(path, record)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, task) for task in pending]
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                print(f"\r{completed}/{len(pending)} {stage} documents", end="", file=sys.stderr, flush=True)
        if pending:
            print(file=sys.stderr)
    except BaseException:
        manifest.update(status="interrupted", finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(out / "manifest.json", manifest)
        raise

    records = [read_json(record_path(out, case, document)) for case, document, _ in tasks(cases, parent_rows)]
    by_question = {case["question_id"]: [] for case in cases}
    for record in records:
        by_question[record["question_id"]].append(record)
    rows = [{"question_id": case["question_id"], "question": case["question"],
             "documents_sha256": digest(case["documents"]),
             "source_status": "error" if any(r["status"] == "error" for r in by_question[case["question_id"]]) else "complete",
             "cards": [card for record in by_question[case["question_id"]] for card in record["cards"]]}
            for case in cases]
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    temporary = out / ("cards." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(out / "cards.jsonl")
    requests = [request for record in records for request in record["requests"]]
    failures = sum(record["status"] == "error" for record in records)
    manifest.update(status="failed" if failures else "complete", cards_sha256=digest(text.encode("utf-8")),
                    cards=sum(len(row["cards"]) for row in rows), documents=len(records),
                    failed_documents=failures, issue_count=sum(len(r["issues"]) for r in records),
                    rejected_cards=sum(not d["kept"] for r in records for d in r["decisions"]),
                    cost={"requests": len(requests),
                          **{key: sum(request.get("usage", {}).get(key, 0) or 0 for request in requests)
                             for key in ("prompt_tokens", "completion_tokens", "total_tokens")}},
                    last_session_seconds=time.monotonic() - start,
                    finished_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(out / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    stages = ["extract"]
    if importlib.util.find_spec("src.card_review") is not None:
        stages.append("review")
    for stage in stages:
        command = subparsers.add_parser(stage)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--out", type=Path)
        command.add_argument("--model")
        command.add_argument("--workers", type=int, default=4)
        command.add_argument("--seed", type=int, default=576)
        command.add_argument("--max-tokens", type=int, default=1024 if stage == "extract" else 2048)
        command.add_argument("--limit", type=int)
        command.add_argument("--resume", action="store_true")
        command.add_argument("--dry-run", action="store_true")
        if stage == "review":
            command.add_argument("--cards", type=Path, required=True, help="Completed C card directory")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.max_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers, max-tokens and limit must be positive")
    if not args.dry_run and args.out is None:
        parser.error("A live stage needs --out; --resume reuses only compatible stage records")
    from src.interface import MODEL, LLM
    model = args.model or MODEL
    parent, parent_rows = None, None
    try:
        cases, provenance = load_bundle(args.bundle)
        if args.limit is not None:
            cases = cases[:args.limit]
        if args.stage == "review":
            from src.card_artifacts import load_card_set
            _, metadata = load_card_set(args.cards, cases, expected_stage="extract", expected_model=model)
            source_manifest = metadata["card_stage_manifest"]
            if source_manifest["bundle_hashes"] != provenance["source_hashes"]:
                raise ValueError("C cards belong to a different frozen bundle")
            parent_rows = unique_rows(map(json.loads, (args.cards / "cards.jsonl").read_text(encoding="utf-8").splitlines()), "C cards")
            parent = {"path": str(args.cards.resolve()), "manifest_sha256": digest((args.cards / "manifest.json").read_bytes()),
                      "cards_sha256": source_manifest["cards_sha256"], "extraction_model": source_manifest["extraction_model"]}
        work = list(tasks(cases, parent_rows))
        if args.dry_run:
            print(json.dumps({"stage": args.stage, "questions": len(cases), "documents": len(work),
                              "model": model, "max_tokens": args.max_tokens,
                              "planned_model_calls": sum(bool(document["text"].strip()) if args.stage == "extract" else bool(cards)
                                                         for _, document, cards in work),
                              "note": "Validation only; no model started or files written."}, indent=2))
            return 0
        manifest = prepare_stage(args.out, cases, provenance, model, args.stage, seed=args.seed,
                                 max_tokens=args.max_tokens, resume=args.resume, parent=parent, workers=args.workers)
        pending = pending_tasks(args.out, cases, parent_rows)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    try:
        if not pending and manifest["status"] == "complete":
            pass  # Completed artifacts remain byte-for-byte immutable for D lineage checks.
        elif pending:
            with LLM(model=model) as llm:
                manifest = run_stage(llm, cases, out=args.out, stage=args.stage, seed=args.seed,
                                     max_tokens=args.max_tokens, workers=args.workers, parent_rows=parent_rows)
        else:
            manifest = run_stage(None, cases, out=args.out, stage=args.stage, seed=args.seed,
                                 max_tokens=args.max_tokens, workers=args.workers, parent_rows=parent_rows)
    except (Exception, KeyboardInterrupt) as exc:
        manifest = read_json(args.out / "manifest.json")
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        atomic_json(args.out / "manifest.json", manifest)
        print(f"Stage stopped; records preserved. Resume with the same settings: {exc}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    print(json.dumps({key: manifest[key] for key in ("stage", "status", "questions", "cards", "failed_documents", "cost")}, indent=2))
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
