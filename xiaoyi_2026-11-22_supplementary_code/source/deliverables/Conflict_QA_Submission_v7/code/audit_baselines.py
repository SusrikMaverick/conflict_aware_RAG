"""Check baseline archive coverage, response links and copied inputs."""

from pathlib import Path
import argparse
import json
import sys
import run
from methods import BASELINE_REGISTRY

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
from real_runtime import digest, MODEL_REV


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="baseline_audit")
    args = parser.parse_args()
    output = run.checked_output(args.output)
    folder = BASE / "data/baselines"
    inputs = run.validate_inputs(run.read(BASE / "data/inputs.json"))
    ids = {r["question_id"] for r in inputs}
    predictions = run.records(folder / "predictions.jsonl.gz")
    stages = run.records(folder / "stages.jsonl.gz")
    requests = {r["key"]: r for r in run.records(folder / "requests.jsonl.gz")}
    expected = {(q, m.legacy) for q in ids for m in BASELINE_REGISTRY}
    actual = {(r["question_id"], r["method"]) for r in predictions}
    if actual != expected or len(predictions) != len(expected):
        raise ValueError("Baseline prediction coverage mismatch")
    stage_index = {r["path"]: r["output"] for r in stages}
    for row in predictions:
        original = stage_index[row["source"]]
        if any(
            original[k] != row[k] for k in ["question_id", "status", "accepted_answers"]
        ):
            raise ValueError("Prediction/source mismatch")
    links = 0
    for stage in stages:
        value = stage["output"]
        for request in value.get("requests", []):
            raw = requests[request["key"]]
            if raw["key"] != digest(
                {
                    "revision": MODEL_REV,
                    "runtime": "vllm0.8.5.post1-bf16",
                    "payload": raw["request"],
                }
            ):
                raise ValueError("Request key mismatch")
            links += 1
        for trace in value.get("stages", []):
            if "key" in trace and requests[trace["key"]]["text"] != trace["text"]:
                raise ValueError("Stage/response text mismatch")
    for filename in ["trust_filters.jsonl.gz", "robust_hints.jsonl.gz"]:
        rows = run.records(folder / filename)
        if {r["question_id"] for r in rows} != ids or len(rows) != len(ids):
            raise ValueError("Preprocessing coverage mismatch: " + filename)
    if {p.stem for p in (folder / "trust_embeddings").glob("*.npy")} != ids:
        raise ValueError("Embedding coverage mismatch")
    report = dict(
        passed=True,
        questions=len(ids),
        predictions=len(predictions),
        stages=len(stages),
        unique_requests=len(requests),
        checked_request_links=links,
        preprocessing_questions=len(ids),
    )
    run.write(output / "validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
