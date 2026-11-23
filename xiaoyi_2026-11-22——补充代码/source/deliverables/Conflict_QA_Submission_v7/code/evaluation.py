"""Versioned evaluation shared by the CLI and compatibility import.

answer_boundary_v2 treats each accepted_answers item as a separate answer.
legacy retains the original metric exactly for historical reproduction.
"""

from src.metrics import normalize, words
from src.legacy_evaluation import score as legacy_score

METRIC_VERSION = "answer_boundary_v2"
METRICS = (METRIC_VERSION, "legacy")


def score(answers, status, groups, wrong_groups, metric=METRIC_VERSION):
    if metric == "legacy":
        return legacy_score(answers, status, groups, wrong_groups)
    if metric != METRIC_VERSION:
        raise ValueError("Unknown metric: " + str(metric))
    if not isinstance(answers, list) or any(not isinstance(a, str) for a in answers):
        raise ValueError("accepted_answers must be a list of strings")
    if not isinstance(groups, list) or not groups:
        raise ValueError("At least one nonempty gold alias group is required")
    if not isinstance(wrong_groups, list):
        raise ValueError("wrong_groups must be a list of alias groups")

    def normalized_group(group):
        if not isinstance(group, list) or any(not isinstance(a, str) for a in group):
            raise ValueError("Alias groups must be lists of strings")
        return {normalize(a) for a in group} - {""}

    gold = [normalized_group(group) for group in groups]
    if any(not group for group in gold):
        raise ValueError("Gold alias group is empty after normalization")
    all_gold = set().union(*gold)
    wrong = set().union(*(normalized_group(group) for group in wrong_groups)) - all_gold
    predicted = {normalize(a) for a in answers} - {""}
    # A name, date or wrong alternative cannot cross answer-list entries.
    covered = all(
        any(words(alias).search(text) for alias in group for text in predicted)
        for group in gold
    )
    remaining = list(predicted)
    for alias in sorted(all_gold, key=len, reverse=True):
        remaining = [words(alias).sub("|", text) for text in remaining]
    bad = any(words(alias).search(text) for alias in wrong for text in remaining)
    ok = status not in ("error", "parse_error")
    return dict(
        answer_match=int(ok and covered and not bad),
        strict_set=int(
            ok and all(predicted & group for group in gold) and predicted <= all_gold
        ),
        labeled_wrong=int(bad),
        failed=int(not ok),
        # Keep the historical explicit-empty-list definition of abstention.
        abstained=int(ok and not answers),
    )
