"""Join completed A/B, C and D answer runs without making model calls.

    python -m src.experiment_report --runs runs/ab runs/c runs/d --out runs/report

Only runs with matching questions, final-model settings and scoring provenance
are comparable. Failed answers stay in the denominator.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from src.eval import normalize


VARIANTS = ("A", "B", "C", "D")
COMPARISONS = (("A", "B"), ("B", "C"), ("C", "D"))
INPUTS = "materials/prepared/v1/ramdocs.inputs.jsonl"
ANNOTATIONS = "materials/prepared/v1/ramdocs.evaluation.jsonl"
MATCHED_FIELDS = ("final_model", "generation", "prompt_template_sha256", "scorer_source_sha256")
SCORES = ("official_correct", "strict_correct", "wrong_answer_accepted")
COST_FIELDS = ("requests", "prompt_tokens", "completion_tokens", "total_tokens")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question_ids(manifest, path):
    ids = manifest.get("question_ids")
    if (not isinstance(ids, list) or not ids or any(not isinstance(qid, str) or not qid for qid in ids)
            or len(ids) != len(set(ids))):
        raise ValueError(f"Missing, invalid or duplicate manifest question IDs: {path}")
    if manifest.get("questions") != len(ids):
        raise ValueError(f"Manifest question count differs from question IDs: {path}")
    return ids


def read_records(path, expected_ids, method):
    records = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Result row must be an object: {path}:{line_number}")
        qid = record.get("question_id")
        if not isinstance(qid, str) or not qid or qid in records:
            raise ValueError(f"Missing or duplicate result question ID: {path}:{line_number}")
        if record.get("method") != method:
            raise ValueError(f"Result method does not match its file: {path}:{qid}")
        if record.get("status") not in ("answered", "abstain", "parse_error", "error"):
            raise ValueError(f"Invalid answer status: {path}:{qid}")
        for key in SCORES:
            value = record.get(key)
            if type(value) not in (bool, int) or value not in (0, 1):
                raise ValueError(f"Missing or invalid {key}: {path}:{qid}")
        if record["status"] == "error" and any(record[key] for key in SCORES):
            raise ValueError(f"Failed request has a nonzero score: {path}:{qid}")
        for key in ("gold_answers", "wrong_answers", "accepted_answers"):
            if not isinstance(record.get(key), list) or not all(isinstance(x, str) for x in record[key]):
                raise ValueError(f"Missing or invalid {key}: {path}:{qid}")
        if not isinstance(record.get("question"), str) or not isinstance(record.get("raw_output"), str):
            raise ValueError(f"Missing question or raw output: {path}:{qid}")
        usage = record.get("usage") or {}
        if not isinstance(usage, dict):
            raise ValueError(f"Invalid token usage: {path}:{qid}")
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid {key}: {path}:{qid}")
        records[qid] = record
    if set(records) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(records))
        extra = sorted(set(records) - set(expected_ids))
        raise ValueError(f"Result question IDs differ from manifest in {path}; missing={missing[:3]}, extra={extra[:3]}")
    return records


def validate_card_lineage(c_manifest, d_manifest):
    """D must review the exact C card artifact used for C's final answers."""
    for label, manifest, stage in (("C", c_manifest, "extract"), ("D", d_manifest, "review")):
        source = manifest.get("card_source")
        card_stage = manifest.get("card_stage_manifest")
        if not isinstance(source, dict) or not isinstance(card_stage, dict):
            raise ValueError(f"{label} is missing its card-source or card-stage provenance")
        if source.get("stage") != stage or card_stage.get("stage") != stage:
            raise ValueError(f"{label} has the wrong card-source stage")
        for key in ("manifest_sha256", "cards_sha256"):
            if not isinstance(source.get(key), str) or not source[key]:
                raise ValueError(f"{label} is missing card-source {key}")
        if source["cards_sha256"] != card_stage.get("cards_sha256"):
            raise ValueError(f"{label} card-source and card-stage hashes differ")
    parent = d_manifest["card_stage_manifest"].get("parent")
    if not isinstance(parent, dict):
        raise ValueError("D is missing its C parent lineage")
    for key in ("manifest_sha256", "cards_sha256"):
        if parent.get(key) != c_manifest["card_source"][key]:
            raise ValueError(f"D parent {key} does not match the C run's card source")
    if (d_manifest.get("card_parent_cost") is not None and c_manifest.get("card_stage_cost") is not None
            and d_manifest["card_parent_cost"] != c_manifest["card_stage_cost"]):
        raise ValueError("D parent cost differs from the same C extraction artifact")


def combined_cost(extraction, review):
    """Count shared C extraction once, then D's incremental review once."""
    observed = {}
    if isinstance(extraction, dict) and isinstance(review, dict):
        for key in COST_FIELDS:
            first, second = extraction.get(key), review.get(key)
            if (type(first) in (int, float) and type(second) in (int, float)
                    and math.isfinite(first) and math.isfinite(second) and first >= 0 and second >= 0):
                observed[key] = first + second
    return {"observed_totals": observed, "all_cost_fields_present": set(observed) == set(COST_FIELDS),
            "scope": "C extraction once plus D review once; D's repeated parent-cost record is not added again."}


def load_runs(run_paths):
    """Validate three completed runs and return A-D records plus provenance."""
    if len(run_paths) != 3:
        raise ValueError("Provide exactly the A/B, C and D run directories, in that order")
    paths = [Path(path).resolve() for path in run_paths]
    if len(set(paths)) != 3:
        raise ValueError("A/B, C and D must be separate run directories")
    manifests = [read_json(path / "manifest.json") for path in paths]
    for path, manifest in zip(paths, manifests):
        if not isinstance(manifest, dict):
            raise ValueError(f"Manifest must be an object: {path}")
        if manifest.get("status") != "complete":
            raise ValueError(f"Run is not complete: {path}")
        question_ids(manifest, path)
        for key in MATCHED_FIELDS:
            if key not in manifest or manifest[key] in (None, "", {}):
                raise ValueError(f"Run is missing {key}: {path}")
        if (not isinstance(manifest["final_model"], str) or not isinstance(manifest["generation"], dict)
                or not isinstance(manifest.get("source_hashes"), dict)):
            raise ValueError(f"Invalid model, generation or source-hash provenance: {path}")
        for name in (INPUTS, ANNOTATIONS):
            if not isinstance(manifest.get("source_hashes", {}).get(name), str) or not manifest["source_hashes"][name]:
                raise ValueError(f"Run is missing dataset hash {name}: {path}")
    if manifests[0].get("card_variant") not in (None, "B"):
        raise ValueError("The first run must compare A with the original B cards, not C or D cards")
    methods = manifests[0].get("methods")
    if methods is not None and (not isinstance(methods, list) or any(not isinstance(value, str) for value in methods)
                                or not {"plain", "plus_cards"}.issubset(methods)):
        raise ValueError("The first A/B run must include both plain and plus_cards methods")
    for variant, manifest, path in zip(("C", "D"), manifests[1:], paths[1:]):
        if manifest.get("card_variant") != variant:
            raise ValueError(f"Expected card_variant={variant} in {path}")
    validate_card_lineage(manifests[1], manifests[2])

    reference = manifests[0]
    order = question_ids(reference, paths[0])
    for manifest, path in zip(manifests[1:], paths[1:]):
        if set(manifest["question_ids"]) != set(order):
            raise ValueError(f"Manifest question IDs differ across runs: {path}")
        for key in MATCHED_FIELDS:
            if manifest[key] != reference[key]:
                raise ValueError(f"Incompatible {key} across runs: {path}")
        for name in (INPUTS, ANNOTATIONS):
            if manifest["source_hashes"][name] != reference["source_hashes"][name]:
                raise ValueError(f"Incompatible dataset hash {name}: {path}")

    records = {
        "A": read_records(paths[0] / "plain.jsonl", order, "plain"),
        "B": read_records(paths[0] / "plus_cards.jsonl", order, "plus_cards"),
        "C": read_records(paths[1] / "plus_cards.jsonl", order, "plus_cards"),
        "D": read_records(paths[2] / "plus_cards.jsonl", order, "plus_cards"),
    }
    for qid in order:
        original = records["A"][qid]
        for variant in VARIANTS[1:]:
            for key in ("question", "gold_answers", "wrong_answers"):
                if records[variant][qid][key] != original[key]:
                    raise ValueError(f"Question or evaluation labels differ: {variant}:{qid}:{key}")

    provenance = {
        "question_ids": order,
        "final_model": reference["final_model"], "generation": reference["generation"],
        "prompt_template_sha256": reference["prompt_template_sha256"],
        "scorer_source_sha256": reference["scorer_source_sha256"],
        "dataset_hashes": {name: reference["source_hashes"][name] for name in (INPUTS, ANNOTATIONS)},
        "runs": {name: {"path": str(path), "manifest_sha256": fingerprint(path / "manifest.json"),
                         "subset": manifest.get("subset"), "scope": manifest.get("scope"),
                         "card_source": manifest.get("card_source")}
                 for name, path, manifest in zip(("AB", "C", "D"), paths, manifests)},
        "card_stage_cost": {
            variant: manifest.get("card_stage_cost") if manifest.get("card_stage_cost") is not None
            else {"included": False, "reason": "Card-stage cost was not supplied by this run"}
            for variant, manifest in zip(("B", "C", "D"), manifests)
        },
        "card_stage_cost_scope": {"B": "Historical card extraction, when supplied",
                                  "C": "C extraction only", "D": "Incremental D review only"},
        "card_parent_cost": {"D": manifests[2].get("card_parent_cost")
                             if manifests[2].get("card_parent_cost") is not None else {
                                 "included": False, "reason": "D did not supply a separate parent-cost record"}},
        "D_card_pipeline_cost": combined_cost(manifests[1].get("card_stage_cost"),
                                               manifests[2].get("card_stage_cost")),
    }
    return records, provenance


def metrics(records):
    count = len(records)
    correct = sum(record["official_correct"] for record in records)
    strict = sum(record["strict_correct"] for record in records)
    return {
        "questions": count, "official_correct": correct, "official_accuracy": correct / count,
        "strict_correct": strict, "strict_accuracy": strict / count,
        "wrong_answer_questions": sum(record["wrong_answer_accepted"] for record in records),
        "wrong_answer_rate": sum(record["wrong_answer_accepted"] for record in records) / count,
        "parse_errors": sum(record["status"] == "parse_error" for record in records),
        "request_errors": sum(record["status"] == "error" for record in records),
        "output_limit_hits": sum(record.get("finish_reason") == "length" for record in records),
        "records_without_total_token_usage": sum((record.get("usage") or {}).get("total_tokens") is None for record in records),
        **{key: sum((record.get("usage") or {}).get(key) or 0 for record in records)
           for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
    }


def paired_change(before, after, field, order):
    deltas = [int(after[qid][field]) - int(before[qid][field]) for qid in order]
    return {"questions": len(order), "gains": sum(delta > 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
            "unchanged_correct": sum(before[qid][field] and after[qid][field] for qid in order),
            "unchanged_incorrect": sum(not before[qid][field] and not after[qid][field] for qid in order),
            "accuracy_difference": sum(deltas) / len(order)}


def build_report(records, provenance):
    """Aggregate validated recorded scores, retaining request failures."""
    order = provenance["question_ids"]
    summary = {
        "questions": len(order), "provenance": provenance,
        "variants": {variant: metrics([records[variant][qid] for qid in order]) for variant in VARIANTS},
        "comparisons": {
            f"{before}_to_{after}": {
                "official": paired_change(records[before], records[after], "official_correct", order),
                "strict": paired_change(records[before], records[after], "strict_correct", order),
            }
            for before, after in COMPARISONS
        },
        "cost_scope": "Observed final-answer usage is summed per variant. C card-stage cost is extraction; D card-stage cost is incremental review. D's complete card-pipeline cost counts C extraction once plus D review once. Unavailable costs are never assumed zero.",
        "scoring": "Aggregates recorded score fields after verifying identical scorer and prompt provenance across runs.",
    }
    groups = {}
    for qid in order:
        number = len({normalize(value) for value in records["A"][qid]["gold_answers"]})
        group = "one_gold_answer" if number == 1 else "multiple_gold_answers" if number > 1 else "no_gold_answers"
        groups.setdefault(group, []).append(qid)
    summary["by_gold_answer_count"] = {
        group: {variant: metrics([records[variant][qid] for qid in ids]) for variant in VARIANTS}
        for group, ids in groups.items()
    }
    details = []
    fields = ("status", "raw_output", "accepted_answers", "official_correct", "strict_correct",
              "wrong_answer_accepted", "finish_reason", "usage")
    for qid in order:
        reference = records["A"][qid]
        details.append({
            "question_id": qid, "question": reference["question"],
            "gold_answers": reference["gold_answers"], "wrong_answers": reference["wrong_answers"],
            **{variant: {key: records[variant][qid].get(key) for key in fields} for variant in VARIANTS},
            "changes": {f"{before}_to_{after}": {
                "official": int(records[after][qid]["official_correct"]) - int(records[before][qid]["official_correct"]),
                "strict": int(records[after][qid]["strict_correct"]) - int(records[before][qid]["strict_correct"]),
            } for before, after in COMPARISONS},
        })
    return summary, details


def write_report(run_paths, out):
    records, provenance = load_runs(run_paths)
    summary, details = build_report(records, provenance)
    summary["created_utc"] = datetime.now(timezone.utc).isoformat()
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "summary.json").open("x", encoding="utf-8") as file:
        file.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (destination / "per_question.jsonl").open("x", encoding="utf-8") as file:
        for row in details:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs=3, required=True, metavar=("AB", "C", "D"))
    parser.add_argument("--out", type=Path, required=True, help="New report directory; existing directories are never overwritten")
    args = parser.parse_args(argv)
    try:
        summary = write_report(args.runs, args.out)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"Cannot combine runs: {exc}")
    for variant in VARIANTS:
        result = summary["variants"][variant]
        print(f"{variant}: {result['official_correct']}/{summary['questions']} historical-compatible; "
              f"strict {result['strict_correct']}/{summary['questions']}; "
              f"labeled-wrong questions {result['wrong_answer_questions']}")
    print(f"Report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
