"""Replay recorded model responses only when the complete request payload matches."""

import gzip, json, threading
from pathlib import Path
from real_runtime import digest
from answer_format import SCHEMA

BASE = Path(__file__).resolve().parents[1]
INDEX = None
BY_KEY = {}
STAGES = {}
EXPECTED = []
STATS = {"matched_calls": 0, "missing_calls": 0}
LOCK = threading.Lock()


def payload(prompt, max_tokens, json_mode, penalty, compact):
    p = dict(
        model="nlp576-qwen7b",
        messages=[dict(role="user", content=prompt)],
        temperature=0,
        max_tokens=max_tokens,
        seed=576,
        repetition_penalty=penalty,
        top_p=1.0,
    )
    if isinstance(json_mode, dict):
        p["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "decision", "schema": json_mode},
        }
    elif json_mode:
        p["response_format"] = {"type": "json_object"}
    if compact and json_mode != SCHEMA:
        p["guided_decoding_backend"] = "xgrammar:disable-any-whitespace"
    return p


class ReplayClient:
    compact = False

    def __init__(self):
        global INDEX
        if INDEX is None:
            INDEX = {}
            for archive in sorted((BASE / "data").glob("requests*.jsonl.gz")):
                with gzip.open(archive, "rt", encoding="utf8") as f:
                    for line in f:
                        r = json.loads(line)
                        INDEX[digest(r["request"])] = r
                        BY_KEY[r["key"]] = r
            with gzip.open(BASE / "data/stages.jsonl.gz", "rt", encoding="utf8") as f:
                for line in f:
                    o = json.loads(line)
                    STAGES[o["path"]] = o["output"]
        self.cache = BASE / "unused-cache"

    def chat(self, prompt, max_tokens=768, json_mode=False, penalty=1.0):
        p = payload(prompt, max_tokens, json_mode, penalty, self.compact)
        k = digest(p)
        for key in EXPECTED:
            r = BY_KEY.get(key)
            if r and digest(r["request"]) == k:
                with LOCK:
                    STATS["matched_calls"] += 1
                return dict(r, cache_hit=True)
        with LOCK:
            if k not in INDEX:
                STATS["missing_calls"] += 1
                raise ValueError("No recorded payload match: " + k)
            STATS["matched_calls"] += 1
        return dict(INDEX[k], cache_hit=True)


class ReplayCompactClient(ReplayClient):
    compact = True


def set_context(branch, n, m, q):
    global EXPECTED
    ram = q.startswith("ramdocs-")
    corrected = (
        f"corrected/runs/wrong_reduction_v1/round{0 if m=='memory' else n}/outputs/{m}/{q}.json"
        if branch == "wr"
        else f"corrected/runs/cross_dataset_v1/outputs/{m}/{q}.json"
    )
    if corrected in STAGES:
        EXPECTED = [r["key"] for r in STAGES[corrected].get("requests", [])]
        return
    if branch == "wr":
        path = f"runs/{'ramdocs_wrong_reduction_v1' if ram else 'wrong_reduction_v1'}/round{0 if m=='memory' else n}/outputs/{m}/{q}.json"
    elif not ram:
        path = f"runs/cross_dataset_v1/outputs/{m}/{q}.json"
    elif m.startswith("FR_"):
        _, n, stage = m.split("_", 2)
        path = f"runs/five_round_search_v1/round{n}/outputs/{stage}/{q}.json"
    else:
        folders = {
            "extract": "phase3f-transfer-v3/ramdocs/F_C1",
            "ours_1024": "two_external_v1/outputs/ours_1024",
            "D4": "five_followup_v3/outputs/D4_coverage",
            "R5_extractive_gate": "combine_improve_v1/outputs/R5_extractive_gate",
            "P05": "coverage_protect_v1/outputs/P05_recover",
            "Z_fresh": "slot_evidence_v1/outputs/Z_fresh",
        }
        path = f"runs/{folders[m]}/{q}.json"
    EXPECTED = [r["key"] for r in STAGES.get(path, {}).get("requests", [])]
