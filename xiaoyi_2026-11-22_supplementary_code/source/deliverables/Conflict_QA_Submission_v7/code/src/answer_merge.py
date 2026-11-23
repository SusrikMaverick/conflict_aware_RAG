"""Source occurrence gates and answer merging."""

import re


def key(s):
    return s.strip().casefold()


def textnorm(s):
    return re.sub("\\s+", " ", s.casefold()).strip()


def combine(answers, adds):
    result = list(answers)
    seen = {key(a) for a in answers}
    accepted = []
    for a in adds:
        if key(a["answer"]) not in seen:
            result.append(a["answer"].strip())
            seen.add(key(a["answer"]))
            accepted.append(a)
    return (result, accepted)


def extractive_gate(row):
    kept = []
    checks = []
    for a in row["d4_additions"]:
        needle = textnorm(a["answer"])
        hits = []
        if not needle:
            checks.append({"answer": a["answer"], "hits": []})
            continue
        for sid in a["support_ids"]:
            haystack = textnorm(row["sources"][sid]["text"])
            for m in re.finditer(re.escape(needle), haystack):
                left = m.start() == 0 or not (
                    needle[0].isalnum() and haystack[m.start() - 1].isalnum()
                )
                right = m.end() == len(haystack) or not (
                    needle[-1].isalnum() and haystack[m.end()].isalnum()
                )
                if left and right:
                    hits.append(
                        {
                            "source_id": sid,
                            "normalized_start": m.start(),
                            "normalized_end": m.end(),
                        }
                    )
                    break
        checks.append({"answer": a["answer"], "hits": hits})
        if hits:
            kept.append(a)
    return (kept, checks)
