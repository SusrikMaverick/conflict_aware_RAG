"""Bounded coverage schema and deterministic merging."""

import json
from coverage_prompts import (
    stage_spec as list_stage_spec,
    coverage_merge as list_merge,
    obj,
)


def string():
    return {"type": "string"}


def schema_for(kind, row, batch=None):
    source = {"type": "string", "enum": [""] + list(row["sources"])}
    if kind == "table":
        record = obj(
            {
                "entity": string(),
                "time_scope": string(),
                "question_conditions": string(),
                "assessment": {
                    "type": "string",
                    "enum": ["supported", "contradicted", "unknown"],
                },
                "support_1": source,
                "support_2": source,
                "conflict_1": source,
                "conflict_2": source,
            }
        )
        return obj(
            {"table": obj({c["candidate_id"]: record for c in row["candidates"]})}
        )
    if kind == "duel":
        record = obj(
            {
                "relation": {
                    "type": "string",
                    "enum": [
                        "different_entity",
                        "different_context",
                        "compatible",
                        "contradiction",
                        "unknown",
                    ],
                },
                "left_support": source,
                "right_support": source,
                "left_counter": source,
                "right_counter": source,
                "rationale": string(),
            }
        )
        return obj({"pairs": obj({p["pair_id"]: record for p in batch})})
    if kind == "coverage":
        record = obj({"answer": string(), "support_1": source, "support_2": source})
        return obj(
            {
                "additions": obj({f"A{i}": record for i in range(12)}),
                "explanation": string(),
            }
        )
    raise ValueError(kind)


def stage_spec(method, row, index=0):
    prompt, cap, schema = list_stage_spec(method, row, index)
    extra = ""
    if method == "D2_table":
        schema = schema_for("table", row)
        extra = "Serialize table as an object keyed by the supplied candidate IDs, exactly one entry per candidate. Source-reference fields hold one supplied segment ID each, or the empty string when absent. Keep entity/time/condition fields concise."
    elif method == "D3_duel":
        schema = schema_for("duel", row, row["pair_batches"][index])
        extra = "Serialize pairs as an object keyed by the requested pair IDs, exactly one entry per requested pair. Each support/counter field holds one supplied segment ID or the empty string when none is established. Keep each rationale to one short sentence."
    elif method == "D4_coverage":
        schema = schema_for("coverage", row)
        extra = "Serialize up to12 additions in fixed slots A0 through A11. For each UNUSED slot use empty strings for answer and both support fields. An occupied slot must have a short answer and supporting segment ID(s). Do not invent additions to fill unused slots. Keep explanation concise."
    elif method in ["G_notes", "G_duel"]:
        schema = obj({"notes": string()})
    if extra:
        prompt = prompt.replace(
            "\nInput: ", "\nSerialization only: " + extra + "\nInput: ", 1
        )
    return (prompt, cap, schema)


def refs(record, *keys):
    return [record[k] for k in keys if record.get(k)]


def coverage_merge(row, text):
    raw = json.loads(text)
    items = []
    for key, entry in raw["additions"].items():
        if not entry["answer"].strip():
            continue
        items.append(
            {
                "answer": entry["answer"],
                "support_ids": refs(entry, "support_1", "support_2"),
            }
        )
    return list_merge(
        row,
        json.dumps(
            {"additions": items, "explanation": raw["explanation"]}, ensure_ascii=False
        ),
    )
