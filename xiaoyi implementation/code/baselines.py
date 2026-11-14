#!/usr/bin/env python3
"""Score, replay or run the seven external baseline adaptations."""
import argparse
import hashlib
import os
import sys
from pathlib import Path
import run
from methods import BASELINE_REGISTRY, ALL_REGISTRY, resolve_method
from evaluation import METRICS, METRIC_VERSION

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
BASELINE_IDS = [m.legacy for m in BASELINE_REGISTRY]


def predictions():
    return run.records(BASE / "data/baselines/predictions.jsonl.gz")


def score(output, metric, include_ours=False):
    records = predictions()
    methods = BASELINE_IDS
    if include_ours:
        records += run.records(BASE / "data/predictions.jsonl.gz")
        methods = [m.legacy for m in ALL_REGISTRY]
    return run.evaluate(
        records,
        output,
        metric,
        expected_ids=[r["question_id"] for r in run.read(BASE / "data/inputs.json")],
        methods=methods,
    )


class ArchiveClient:
    """Replay each original request in order, matching the entire payload."""

    def __init__(self):
        from baseline_nlp import records

        self.responses = {r["key"]: r for r in records("requests.jsonl.gz")}
        self.outputs = {
            (r["kind"], r["output"]["question_id"]): r["output"]
            for r in records("stages.jsonl.gz")
        }
        self.matched = 0
        self.expected = []
        self.position = 0

    def start(self, method, question_id):
        original = self.outputs[(method, question_id)]
        self.expected = original["requests"][:]
        if method == "robust_keyword" and not question_id.startswith("ramdocs-"):
            self.expected = (
                self.outputs[("robust_isolated", question_id)]["requests"]
                + self.expected
            )
        self.position = 0
        return original

    def chat(self, prompt, max_tokens=768, json_mode=False, penalty=1.0):
        from replay import payload

        requested = payload(prompt, max_tokens, json_mode, penalty, False)
        if self.position >= len(self.expected):
            raise ValueError("Unexpected extra request")
        key = self.expected[self.position]["key"]
        response = self.responses[key]
        if requested != response["request"]:
            raise ValueError(f"Request mismatch at call {self.position}, key {key}")
        self.position += 1
        self.matched += 1
        return dict(response, cache_hit=True)


def identity(rows, methods, replay):
    paths = [
        BASE / name
        for name in [
            "baselines.py",
            "run.py",
            "methods.py",
            "evaluation.py",
            "download_model.py",
            "download_nlp_models.py",
        ]
    ]
    paths += list(BASE.glob("requirements*.txt"))
    paths.append(BASE / "data/evaluation.json")
    paths += list((BASE / "src").glob("*.py"))
    for directory in ["upstream", "assets", "data/baselines", "configs", "tokenizer"]:
        paths += [
            p
            for p in (BASE / directory).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        ]
    files = {
        str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    encoder_files = {}
    if not replay:
        model_dir = Path(os.environ.get("QA_NLP_MODELS", BASE / "models"))
        for method, folder in [
            ("trust", "sup-simcse-bert-base-uncased"),
            ("faithfulrag_adaptation", "all-MiniLM-L6-v2"),
        ]:
            if method in methods:
                for path in sorted((model_dir / folder).rglob("*")):
                    if path.is_file() and ".cache" not in path.parts:
                        encoder_files[str(path.relative_to(model_dir))] = (
                            hashlib.sha256(path.read_bytes()).hexdigest()
                        )
    return dict(
        encoder_files=encoder_files,
        inputs=rows,
        methods=methods,
        replay=replay,
        code=files,
        model_url=os.environ.get("SIX_MAIN_URL", run.DEFAULT_URLS["SIX_MAIN_URL"]),
        nlp_models=os.environ.get("QA_NLP_MODELS", str(BASE / "models")),
    )


def generate(args):
    output = run.checked_output(args.output)
    if args.input and args.dataset != "all":
        raise ValueError("Do not combine custom inputs and a dataset filter")
    rows = run.validate_inputs(run.read(args.input or BASE / "data/inputs.json"))
    if args.dataset != "all":
        prefixes = {"RAMDocs": "ramdocs-", "RGB": "rgb-", "ConflictBank-500": "cb-"}
        rows = [r for r in rows if r["question_id"].startswith(prefixes[args.dataset])]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[: args.limit]
    methods = (
        BASELINE_IDS if args.method == "all" else [resolve_method(args.method).legacy]
    )
    if any(m not in BASELINE_IDS for m in methods):
        raise ValueError("Choose a baseline method")
    stamp = identity(rows, methods, args.replay)
    manifest = output / "generation_manifest.json"
    if manifest.exists():
        if run.read(manifest) != stamp:
            raise ValueError(
                "Output belongs to a different configuration; use a new directory"
            )
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Nonempty output has no generation manifest")
    run.write(manifest, stamp)
    os.environ["SIX_METHOD_ROOT"] = str(output)
    from baseline_engine import BaselineEngine
    from baseline_nlp import Preprocessor

    if args.replay:
        client = ArchiveClient()
    else:
        from real_runtime import Client

        client = Client()
        client.cache = output / "requests"
        client.cache.mkdir(parents=True, exist_ok=True)
    engine = BaselineEngine(client, Preprocessor(replay=args.replay))
    final, mismatches = [], []
    for i, row in enumerate(rows, 1):
        for method in methods:
            expected = client.start(method, row["question_id"]) if args.replay else None
            destination = (
                output
                / "stages"
                / resolve_method(method).name
                / (row["question_id"] + ".json")
            )
            # Replay recalculates every step; fresh runs may resume completed records.
            result = (
                run.read(destination)
                if destination.exists() and not args.replay
                else engine.one(row, method)
            )
            run.write(destination, result)
            if args.replay:
                different = [
                    k
                    for k in ["status", "accepted_answers"]
                    if result[k] != expected[k]
                ]
                if client.position != len(client.expected):
                    different.append("request_count")
                if "error" in result and "error" not in expected:
                    different.append("error")
                if different:
                    mismatches.append(
                        dict(
                            question_id=row["question_id"],
                            method=method,
                            fields=different,
                            error=result.get("error"),
                        )
                    )
            named = resolve_method(method)
            final.append(
                dict(
                    question_id=row["question_id"],
                    method=named.name,
                    method_name=named.title,
                    legacy_method=method,
                    status=result["status"],
                    accepted_answers=result["accepted_answers"],
                )
            )
        if i % 25 == 0 or i == len(rows):
            print(f"Completed {i}/{len(rows)} questions", flush=True)
    run.write(output / "predictions.json", final)
    if not args.input:
        run.evaluate(
            final,
            output / "scores",
            args.metric,
            expected_ids=[r["question_id"] for r in rows],
            methods=methods,
        )
    if args.replay:
        report = dict(
            predictions=len(final),
            exact_matches=len(final) - len(mismatches),
            matched_requests=client.matched,
            mismatches=mismatches,
            passed=not mismatches,
            preprocessing="Frozen NLP outputs; recompute separately with preprocess.py",
        )
        run.write(output / "replay_validation.json", report)
        if mismatches:
            raise ValueError(
                f"{len(mismatches)} replay mismatches; inspect replay_validation.json"
            )
    if all(r["status"] in {"error", "parse_error"} for r in final):
        raise ValueError("Every prediction failed; inspect stage outputs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("methods")
    scoring = commands.add_parser("score")
    scoring.add_argument("--output", default="baseline_scores")
    scoring.add_argument("--metric", choices=METRICS, default=METRIC_VERSION)
    scoring.add_argument("--include-ours", action="store_true")
    generation = commands.add_parser("generate")
    generation.add_argument("--output", required=True)
    generation.add_argument("--replay", action="store_true")
    generation.add_argument(
        "--method", choices=["all"] + [m.name for m in BASELINE_REGISTRY], default="all"
    )
    generation.add_argument(
        "--dataset",
        choices=["all", "RAMDocs", "RGB", "ConflictBank-500"],
        default="all",
    )
    generation.add_argument("--limit", type=int)
    generation.add_argument("--input", type=Path)
    generation.add_argument("--metric", choices=METRICS, default=METRIC_VERSION)
    args = parser.parse_args()
    if args.command == "methods":
        for m in BASELINE_REGISTRY:
            print(f"{m.name:12} {m.title}")
    elif args.command == "score":
        score(args.output, args.metric, args.include_ours)
    else:
        generate(args)


if __name__ == "__main__":
    main()
