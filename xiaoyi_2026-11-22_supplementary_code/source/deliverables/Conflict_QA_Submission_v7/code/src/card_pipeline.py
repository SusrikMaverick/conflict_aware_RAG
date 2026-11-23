"""Card extraction, coverage proposals and evidence audits."""

import hashlib, json
from jsonschema import Draft202012Validator
from real_runtime import ROOT, Client, write_json, load_plain_prompt
from compact_client import CompactClient
from answer_format import SCHEMA, parse
from source_cards import SourceCards, segments
from prompt_parts import docs, FIELDS, SUFFIX, CONTRACT
from coverage_format import stage_spec, coverage_merge
from answer_merge import extractive_gate, combine, key
import candidate_recovery as protection
import slot_checks as slot
import evidence_pruning as fr

RUN = ROOT / "runs/cross_dataset_v1"
read = lambda p: json.loads(p.read_text(encoding="utf-8"))


def path(m, q):
    return RUN / "outputs" / m / (q + ".json")


class Engine:

    def __init__(self):
        self.c = Client()
        self.aux = CompactClient()
        for c in [self.c, self.aux]:
            c.cache = RUN / "request_cache"
            c.cache.mkdir(parents=True, exist_ok=True)
        self.extractor = SourceCards(self.c)

    def one(self, m, row):
        q = row["question_id"]
        dest = path(m, q)
        if dest.exists():
            return read(dest)
        out = dict(
            question_id=q,
            method=m,
            status="pending",
            accepted_answers=[],
            requests=[],
            stages=[],
            upstream_outputs=[],
        )

        def dep(name):
            p = path(name, q)
            out["upstream_outputs"].append(str(p.relative_to(ROOT)))
            o = read(p)
            if o["status"] in ["error", "parse_error"]:
                raise ValueError("upstream_failure:" + name)
            return o

        def call(
            stage,
            prompt,
            cap=1024,
            schema=False,
            compact=False,
            strict=False,
            penalty=1.05,
        ):
            c = self.aux if compact else self.c
            r = c.chat(prompt, cap, schema, penalty)
            out["requests"].append(
                {
                    k: r[k]
                    for k in [
                        "key",
                        "usage",
                        "finish_reason",
                        "cache_hit",
                        "latency_seconds",
                        "attempts",
                    ]
                }
            )
            out["stages"].append(
                dict(
                    stage=stage,
                    key=r["key"],
                    text=r["text"],
                    output_cap=cap,
                    compact=compact,
                )
            )
            if strict and r["finish_reason"] == "length":
                raise ValueError("output_token_limit:" + stage)
            if schema:
                Draft202012Validator(schema).validate(json.loads(r["text"]))
            return r["text"]

        def auxcall(stage, spec):
            return call(stage, *spec, compact=True, strict=True)

        try:
            a = []
            text = None
            if m == "extract":
                cards = []
                issues = []
                for i, d in enumerate(row["documents"], 1):
                    if not d["text"].strip():
                        issues.append(
                            dict(doc_id=d["doc_id"], error="empty_document_no_evidence")
                        )
                        continue
                    try:
                        o = self.extractor.run(
                            dict(
                                question_id=q, question=row["question"], documents=[d]
                            ),
                            "C1",
                        )
                    except Exception:
                        out["requests"] += getattr(self.extractor.local, "trace", [])
                        raise
                    out["requests"] += o["requests"]
                    issues += o["extraction_issues"]
                    for raw in o["cards"]:
                        card = {k: raw[k] for k in FIELDS if k in raw}
                        card["original_document_number"] = i
                        cards.append(card)
                out.update(cards=cards, extraction_issues=issues, status="complete")
            else:
                cards = dep("extract")["cards"]
                plain = load_plain_prompt().format(
                    question=row["question"], documents=docs(row)
                )
                base = plain + SUFFIX + json.dumps(cards, ensure_ascii=False)
                if m == "ours_1024":
                    text = call("final", base + CONTRACT, 1024, SCHEMA)
                else:
                    initial = dep("ours_1024")["accepted_answers"]
                    r = make_row(row, cards, base, initial)
                    if m == "D4":
                        raw = auxcall("coverage", stage_spec("D4_coverage", r))
                        a, adds, issues, _ = coverage_merge(r, raw)
                        out.update(additions=adds, aux_issues=issues)
                    else:
                        d4 = dep("D4")
                        r["d4_additions"] = d4["additions"]
                        if m == "R5_extractive_gate":
                            adds, checks = extractive_gate(r)
                            a, selected = combine(initial, adds)
                            out.update(selected_additions=selected, diagnostics=checks)
                        else:
                            r["r5_answers"] = dep("R5_extractive_gate")[
                                "accepted_answers"
                            ]
                            initkeys = {key(v) for v in initial}
                            r["r5_candidates"] = [
                                dict(
                                    candidate_id=f"R{i}",
                                    answer=v,
                                    origin=(
                                        "initial"
                                        if key(v) in initkeys
                                        else "coverage_addition"
                                    ),
                                )
                                for i, v in enumerate(r["r5_answers"])
                            ]
                            r["requested_slot"] = slot.question_slot(row["question"])
                            if m in ["P05", "Z_fresh"]:
                                mod, name = (
                                    (protection, "P05_recover")
                                    if m == "P05"
                                    else (slot, "Z_fresh")
                                )
                                raw = auxcall(m, mod.spec(name, r))
                                _, a, adds, diag = mod.finish(name, r, raw)
                                out.update(selected_additions=adds, diagnostics=diag)
                            else:
                                p05 = dep("P05")
                                zf = dep("Z_fresh")
                                pool = []
                                seen = {key(v) for v in r["r5_answers"]}
                                for item in (
                                    r["d4_additions"]
                                    + p05["selected_additions"]
                                    + zf["selected_additions"]
                                ):
                                    if key(item["answer"]) not in seen:
                                        seen.add(key(item["answer"]))
                                        pool.append(item["answer"])
                                family, n, stage = m.split("_", 2)
                                n = int(n)
                                engine = fr
                                cfg = read(engine.plan(n) / "design.json")
                                r["round_number"] = n
                                if n > 1:
                                    prev = dep(f"{family}_{n - 1}_M10")
                                    pool += prev["accepted_answers"]
                                if cfg.get("aliases", False):
                                    pool += [
                                        v
                                        for answer in r["r5_answers"] + pool
                                        for v in engine.aliases(answer)
                                    ]
                                inv = []
                                seen = set()
                                for origin, values in [
                                    ("R5", r["r5_answers"]),
                                    ("proposal", pool),
                                ]:
                                    for v in values:
                                        if key(v) not in seen:
                                            seen.add(key(v))
                                            inv.append(
                                                dict(
                                                    id=f"C{len(inv)}",
                                                    answer=v,
                                                    origin=origin,
                                                )
                                            )
                                r["inventory"] = inv
                                out["inventory"] = inv
                                deps = (
                                    {}
                                    if stage in ["J_A", "J_B"]
                                    else {
                                        j: dep(f"{family}_{n}_{j}")
                                        for j in ["J_A", "J_B"]
                                    }
                                )
                                decision = None
                                if stage != "M03":
                                    decision = json.loads(
                                        auxcall(stage, engine.spec(n, stage, r, deps))
                                    )
                                a, diag, _, _ = engine.finish(
                                    n, stage, r, decision, deps
                                )
                                out.update(decision=decision or {}, diagnostics=diag)
                                if stage in ["J_A", "J_B"]:
                                    out["status"] = "judged"
                    out.setdefault(
                        "row_sha256",
                        hashlib.sha256(
                            json.dumps(r, ensure_ascii=False, sort_keys=True).encode()
                        ).hexdigest(),
                    )
            if text is not None:
                a = parse(text)
                out["raw_output"] = text
            if out["status"] == "pending":
                out.update(accepted_answers=a, status="answered" if a else "abstain")
        except Exception as exc:
            out.update(
                status="error",
                accepted_answers=[],
                error=type(exc).__name__ + ": " + str(exc),
            )
        write_json(dest, out)
        return out


def make_row(row, cards, base, initial):
    sources = {}
    for i, d in enumerate(row["documents"], 1):
        for j, s in enumerate(segments(d["text"]), 1):
            sources[f"D{i}S{j}"] = dict(
                document_number=i,
                doc_id=d["doc_id"],
                start=s["start"],
                end=s["end"],
                text=s["text"],
            )
    candidates = []
    lookup = {}
    for c in cards:
        k = key(c["answer"])
        if k not in lookup:
            lookup[k] = len(candidates)
            candidates.append(
                dict(
                    candidate_id=f"C{len(candidates)}", answer=c["answer"], card_ids=[]
                )
            )
        candidates[lookup[k]]["card_ids"].append(c["card_id"])
    return dict(
        question_id=row["question_id"],
        question=row["question"],
        base_prompt=base,
        cards=cards,
        sources=sources,
        candidates=candidates,
        initial_answers=initial,
    )
