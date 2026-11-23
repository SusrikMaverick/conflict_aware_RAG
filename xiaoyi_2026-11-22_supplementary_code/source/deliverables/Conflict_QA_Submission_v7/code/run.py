#!/usr/bin/env python3
"""Verify, audit, score, or generate the six retained methods."""
import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from evaluation import METRICS, METRIC_VERSION
from methods import METHOD_REGISTRY, LEGACY_METHODS, resolve_method

BASE = Path(__file__).resolve().parent
METHODS = LEGACY_METHODS
STATUSES = {"answered", "abstain", "error", "parse_error"}
COUNTS = ["n", "answer_match", "strict_set", "labeled_wrong", "failed", "abstained"]
DEFAULT_URLS = {
    "SIX_MAIN_URL": "http://127.0.0.1:8576/v1/chat/completions",
    "SIX_AUX_URL": "http://127.0.0.1:8577/v1/chat/completions",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_inputs(data):
    if not isinstance(data, list) or not data:
        raise ValueError("Input must be a nonempty JSON list of question objects")
    seen = set()
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Question {index} must be an object")
        q = row.get("question_id")
        if (
            not isinstance(q, str)
            or not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]*", q)
            or q in seen
        ):
            raise ValueError(
                "Question IDs must be unique, nonempty, safe filename strings"
            )
        seen.add(q)
        if set(row) != {"question_id", "question", "documents"}:
            raise ValueError(
                "Only question_id, question, documents are allowed; labels must be separate"
            )
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError(f"{q}: question must be a nonempty string")
        if not isinstance(row["documents"], list) or not row["documents"]:
            raise ValueError(f"{q}: documents must be a nonempty list")
        document_ids = set()
        for doc in row["documents"]:
            if not isinstance(doc, dict) or not {"doc_id", "text"} <= set(doc):
                raise ValueError(f"{q}: each document needs doc_id and text")
            if set(doc) - {"doc_id", "text", "metadata"}:
                raise ValueError(f"{q}: unexpected document fields")
            doc_id = doc["doc_id"]
            if (
                not isinstance(doc_id, str)
                or not doc_id.strip()
                or doc_id in document_ids
            ):
                raise ValueError(f"{q}: document IDs must be unique nonempty strings")
            document_ids.add(doc_id)
            if not isinstance(doc["text"], str) or not doc["text"].strip():
                raise ValueError(f"{q}: document text must be a nonempty string")
            if "metadata" in doc and (
                not isinstance(doc["metadata"], dict) or doc["metadata"]
            ):
                raise ValueError(
                    f"{q}: document metadata must be empty; supply only doc_id and text"
                )
    return data


def checked_output(path):
    """Keep generated files outside the packaged source, data and evidence."""
    root = Path(path).resolve()
    protected = [
        "src",
        "data",
        "configs",
        "planning",
        "tokenizer",
        "provenance",
        "results",
        "tools",
        "upstream",
        "assets",
    ]
    if root == BASE or any(root.is_relative_to(BASE / name) for name in protected):
        raise ValueError(
            "Use a separate output directory outside packaged source/data/results"
        )
    if root.exists() and not root.is_dir():
        raise ValueError("Output path must be a directory")
    return root


def evaluate(
    predictions, output, metric=METRIC_VERSION, expected_ids=None, methods=None
):
    from evaluation import score

    selected_methods = METHODS if methods is None else methods

    output = checked_output(output)
    if not predictions:
        raise ValueError("No predictions to evaluate")
    gold = read(BASE / "data/evaluation.json")
    seen, totals, per_question = (set(), {}, [])
    for row in predictions:
        if not isinstance(row, dict) or not {
            "question_id",
            "method",
            "status",
            "accepted_answers",
        } <= set(row):
            raise ValueError("Malformed prediction record")
        q = row["question_id"]
        selected = resolve_method(row["method"])
        method = selected.legacy
        if (
            q not in gold
            or method not in selected_methods
            or row["status"] not in STATUSES
        ):
            raise ValueError(
                "Unknown question, method or prediction status: "
                + str((q, method, row["status"]))
            )
        answers = row["accepted_answers"]
        if not isinstance(answers, list) or any(
            (not isinstance(a, str) for a in answers)
        ):
            raise ValueError("accepted_answers must be a list of strings")
        if row["status"] in {"abstain", "error", "parse_error"} and answers:
            raise ValueError(
                "Failed/abstained prediction must have no accepted answers"
            )
        pair = (q, method)
        if pair in seen:
            raise ValueError("Duplicate prediction: " + str(pair))
        seen.add(pair)
        labels = gold[q]
        values = score(
            answers,
            row["status"],
            labels["groups"],
            labels["wrong_groups"],
            metric=metric,
        )
        total = totals.setdefault(
            (labels["dataset"], method),
            dict(
                dataset=labels["dataset"],
                method=selected.name,
                method_name=selected.title,
                legacy_method=method,
                metric=metric,
                n=0,
                **{k: 0 for k in values},
            ),
        )
        total["n"] += 1
        for key, value in values.items():
            total[key] += value
        per_question.append(
            dict(
                dataset=labels["dataset"],
                method=selected.name,
                method_name=selected.title,
                legacy_method=method,
                question_id=q,
                status=row["status"],
                metric=metric,
                **values,
                answers=json.dumps(answers, ensure_ascii=False),
            )
        )
    if expected_ids is not None:
        expected = {(q, method) for q in expected_ids for method in selected_methods}
        if seen != expected:
            raise ValueError(
                f"Prediction coverage mismatch: {len(expected - seen)} missing, {len(seen - expected)} unexpected"
            )
    for total in totals.values():
        for key in ["answer_match", "strict_set", "labeled_wrong"]:
            total[key + "_pct"] = 100 * total[key] / total["n"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("summary", list(totals.values())),
        ("per_question", per_question),
    ]:
        with (output / (name + ".csv")).open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write(output / "summary.json", list(totals.values()))
    return totals


def check_summaries(actual, expected):
    expected_map = {(r["dataset"], r["method"]): r for r in expected}
    if set(actual) != set(expected_map):
        raise ValueError("Expected summary method/dataset coverage differs")
    for pair, reference in expected_map.items():
        for key in COUNTS:
            if actual[pair][key] != reference[key]:
                raise ValueError(
                    f"Summary mismatch: {pair} {key}: {actual[pair][key]} != {reference[key]}"
                )


def generation_identity(data, replay, metric=METRIC_VERSION):
    paths = [BASE / "run.py", BASE / "evaluation.py", BASE / "methods.py"]
    for directory in ["src", "planning", "configs", "tokenizer"]:
        paths += [
            p
            for p in (BASE / directory).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and (p.suffix != ".pyc")
        ]
    paths += list(BASE.glob("requirements*.txt"))
    dependencies = {
        str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }
    tokenizer_override = os.environ.get("SIX_TOKENIZER")
    if tokenizer_override:
        tokenizer = Path(tokenizer_override).resolve()
        if not tokenizer.is_dir():
            raise ValueError(
                "SIX_TOKENIZER must name an existing local tokenizer directory"
            )
        dependencies.update(
            {
                "external_tokenizer/"
                + str(p.relative_to(tokenizer)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in sorted(tokenizer.rglob("*"))
                if p.is_file()
            }
        )
    if replay:
        paths = sorted((BASE / "data").glob("requests*.jsonl.gz"))
        paths += [BASE / "data/stages.jsonl.gz", BASE / "data/predictions.jsonl.gz"]
        dependencies.update(
            {
                str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths
            }
        )
    identity = dict(
        mode="recorded-response replay" if replay else "fresh",
        metric=metric,
        n=len(data),
        methods=METHODS,
        dependencies=dependencies,
        endpoints={k: os.environ.get(k, v) for k, v in DEFAULT_URLS.items()},
        tokenizer=(
            str(Path(tokenizer_override).resolve())
            if tokenizer_override
            else "packaged"
        ),
        input_sha256=hashlib.sha256(
            json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
    )
    identity["fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    return identity


def check_output_identity(root, identity):
    manifest = root / "generation_manifest.json"
    if manifest.exists():
        previous = read(manifest)
        if (
            previous.get("fingerprint") != identity["fingerprint"]
            or previous.get("mode") != identity["mode"]
        ):
            raise ValueError(
                "Output folder belongs to a different mode/input/code configuration; use a new folder"
            )
    elif root.exists() and any(root.iterdir()):
        raise ValueError(
            "Nonempty output folder has no generation manifest; use a new folder"
        )


def check_servers():
    import urllib.request

    for name, default in DEFAULT_URLS.items():
        endpoint = os.environ.get(name, default)
        if not endpoint.endswith("/v1/chat/completions"):
            raise ValueError(name + " must end with /v1/chat/completions")
        url = endpoint.rsplit("/chat/completions", 1)[0] + "/models"
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                models = json.load(response)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot reach {name} model service at {url}: {exc}"
            ) from exc
        if not any(
            (model.get("id") == "nlp576-qwen7b" for model in models.get("data", []))
        ):
            raise ValueError("Expected served model nlp576-qwen7b at " + url)


def compare_replay(predictions):
    expected = {
        (r["question_id"], r["method"]): r
        for r in records(BASE / "data/predictions.jsonl.gz")
    }
    mismatches = []
    for row in predictions:
        reference = expected.get(
            (row["question_id"], resolve_method(row["method"]).legacy)
        )
        if reference is None or any(
            (row[k] != reference[k] for k in ["accepted_answers", "status"])
        ):
            mismatches.append(dict(actual=row, expected=reference))
    return dict(
        predictions=len(predictions),
        exact_matches=len(predictions) - len(mismatches),
        mismatches=mismatches,
        passed=not mismatches,
    )


def generate(args):
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.input and args.dataset != "all":
        raise ValueError("--input cannot be combined with a benchmark --dataset filter")
    data = validate_inputs(
        read(args.input) if args.input else read(BASE / "data/inputs.json")
    )
    if args.dataset != "all":
        prefix = {"RAMDocs": "ramdocs-", "RGB": "rgb-", "ConflictBank-500": "cb-"}[
            args.dataset
        ]
        data = [row for row in data if row["question_id"].startswith(prefix)]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        data = data[: args.limit]
    if not data:
        raise ValueError("No input questions selected")
    root = checked_output(args.output)
    identity = generation_identity(data, args.replay, args.metric)
    check_output_identity(root, identity)
    if not args.replay:
        check_servers()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["SIX_METHOD_ROOT"] = str(root)
    sys.path.insert(0, str(BASE / "src"))
    for family in ["answer_review", "evidence_pruning"]:
        shutil.copytree(
            BASE / "planning" / family, root / "planning" / family, dirs_exist_ok=True
        )
    write(root / "generation_manifest.json", identity)
    import card_pipeline as X, answer_review as W

    if args.replay:
        from replay import ReplayClient, ReplayCompactClient

        X.Client = W.Client = ReplayClient
        X.CompactClient = W.CompactClient = ReplayCompactClient
    X.RUN = root / "runs/cross_dataset_v1"
    W.RUN = root / "runs/wrong_reduction_v1"
    W.OLD = X.RUN
    x = X.Engine()
    engines = {n: W.Engine(n) for n in range(1, 6)}
    finished = []

    def old(m, row):
        if X.path(m, row["question_id"]).exists():
            return X.read(X.path(m, row["question_id"]))
        deps = []
        if m != "extract":
            deps = ["extract"]
        if m not in ["extract", "ours_1024"]:
            deps += ["ours_1024", "D4"] if m != "D4" else ["ours_1024"]
        if m.startswith("FR_"):
            n = int(m.split("_")[1])
            stage = m.split("_")[2]
            deps += ["R5_extractive_gate", "P05", "Z_fresh"]
            if n > 1:
                deps += [f"FR_{n - 1}_M10"]
            if stage not in ["J", "J_A", "J_B"] and (not m.endswith(("J_A", "J_B"))):
                deps += [f"FR_{n}_J_A", f"FR_{n}_J_B"]
        elif m in ["P05", "Z_fresh"]:
            deps += ["R5_extractive_gate"]
        for d in dict.fromkeys(deps):
            if old(d, row)["status"] in ["error", "parse_error"]:
                break
        if args.replay:
            from replay import set_context

            set_context("old", 0, m, row["question_id"])
        return x.one(m, row)

    def wr(n, m, row):
        q = row["question_id"]
        p = W.path(0 if m == "memory" else n, m, q)
        if p.exists():
            return W.read(p)
        if m != "memory":
            wr(n, "memory", row)
        alias = {"M01": "B", "M02": "C", "M10": "F"}
        if m in alias:
            o = dict(wr(n, alias[m], row))
            o["method"] = m
            o["alias_of"] = str(W.path(n, alias[m], q).relative_to(root))
            write(p, o)
            return o
        if m in ["B", "C"]:
            available = m != "C" or old("extract", row)["status"] not in [
                "error",
                "parse_error",
            ]
            if available and n > 1 and W.cfg(n).get("use_previous"):
                wr(n - 1, W.cfg(n)["previous_method"], row)
        elif m in ["J_A", "J_B", "F"]:
            available = True
            for d in ["B", "C"]:
                if wr(n, d, row)["status"] in ["error", "parse_error"]:
                    available = False
                    break
            if available:
                available = old("R5_extractive_gate", row)["status"] not in [
                    "error",
                    "parse_error",
                ]
            if available and m == "F":
                for d in ["J_A", "J_B"]:
                    if wr(n, d, row)["status"] in ["error", "parse_error"]:
                        break
        elif m in ["M08", "M09"]:
            b = wr(n, "B", row)
            if m == "M08" and b["status"] not in ["error", "parse_error"]:
                wr(n, "C", row)
        elif m in ["M03", "M04", "M05", "M06", "M07"]:
            a = wr(n, "J_A", row)
            available = a["status"] not in ["error", "parse_error"]
            if available and m in ["M04", "M05", "M07"]:
                available = wr(n, "J_B", row)["status"] not in ["error", "parse_error"]
            if available and m in ["M03", "M04", "M05"]:
                old("R5_extractive_gate", row)
        if args.replay:
            from replay import set_context

            set_context("wr", n, m, q)
        return engines[n].make(m, row)

    def one_question(row):
        answers = []
        for m in METHODS:
            if m == "ours_1024":
                o = old(m, row)
            elif m == "FR4_M03":
                o = old("FR_4_M03", row)
            else:
                n, op = m[2:].split("_")
                o = wr(int(n), op, row)
            selected = resolve_method(m)
            answers.append(
                dict(
                    question_id=row["question_id"],
                    method=selected.name,
                    method_name=selected.title,
                    legacy_method=m,
                    status=o["status"],
                    accepted_answers=o["accepted_answers"],
                )
            )
        return answers

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1 if args.replay else args.workers) as pool:
        for i, answers in enumerate(pool.map(one_question, data), 1):
            finished.extend(answers)
            print(f"Completed {i}/{len(data)} questions", flush=True)
    write(root / "predictions.json", finished)
    if not args.input:
        evaluate(
            finished,
            root / "scores",
            args.metric,
            expected_ids=[row["question_id"] for row in data],
        )
    if args.replay:
        from replay import STATS

        write(root / "replay_stats.json", STATS)
        validation = (
            compare_replay(finished)
            if not args.input
            else dict(
                passed=STATS["missing_calls"] == 0,
                scope="custom-input request-payload replay only",
            )
        )
        write(root / "replay_validation.json", validation)
        if STATS["missing_calls"] or not validation["passed"]:
            raise RuntimeError(
                "Replay differs from the archive; inspect replay_validation.json and replay_stats.json"
            )
    if finished and all(
        (row["status"] in ["error", "parse_error"] for row in finished)
    ):
        raise RuntimeError("Every final prediction failed; inspect stage errors")
    print("Saved", root / "predictions.json")


def verify():
    for name, expected in read(BASE / "MANIFEST.json").items():
        path = BASE / name
        if not path.is_file():
            raise ValueError("Missing packaged file: " + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Hash mismatch: " + name)
    print("All packaged files verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    commands.add_parser("methods", help="List method names and command keys")
    audit = commands.add_parser(
        "audit", help="Audit question, prediction, request and archive dependencies"
    )
    audit.add_argument("--output", default=str(BASE / "audit_output"))
    scoring = commands.add_parser("score")
    scoring.add_argument("--output", default=str(BASE / "recomputed"))
    scoring.add_argument("--metric", choices=METRICS, default=METRIC_VERSION)
    generation = commands.add_parser("generate")
    generation.add_argument("--output", required=True)
    generation.add_argument(
        "--dataset",
        choices=["all", "RAMDocs", "RGB", "ConflictBank-500"],
        default="all",
    )
    generation.add_argument("--limit", type=int)
    generation.add_argument(
        "--input", help="JSON list of new unlabeled questions/documents"
    )
    generation.add_argument(
        "--replay", action="store_true", help="Use saved responses; no fresh inference"
    )
    generation.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent fresh questions; replay is serial",
    )
    generation.add_argument("--metric", choices=METRICS, default=METRIC_VERSION)
    args = parser.parse_args()
    if args.command == "verify":
        verify()
    elif args.command == "methods":
        for method in METHOD_REGISTRY:
            print(f"{method.name:18} {method.title}")
    elif args.command == "audit":
        output = checked_output(args.output)
        from tools.archive_audit import audit_package

        result = audit_package(BASE)
        write(output / "integrity.json", result)
        if not result["passed"]:
            raise ValueError("Archive dependency audit failed; inspect integrity.json")
        print(
            "Archive dependency audit passed:",
            result["predictions"],
            "predictions,",
            result["stages"],
            "stages",
        )
    elif args.command == "score":
        gold_ids = read(BASE / "data/evaluation.json")
        actual = evaluate(
            records(BASE / "data/predictions.jsonl.gz"),
            args.output,
            args.metric,
            expected_ids=gold_ids,
        )
        expected = (
            BASE
            / "results"
            / (
                "expected_summary_legacy.json"
                if args.metric == "legacy"
                else "expected_summary.json"
            )
        )
        check_summaries(actual, read(expected))
        print(
            "All 6,600 predictions rescored; all 18 summaries match. Metric:",
            args.metric,
        )
    else:
        generate(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
