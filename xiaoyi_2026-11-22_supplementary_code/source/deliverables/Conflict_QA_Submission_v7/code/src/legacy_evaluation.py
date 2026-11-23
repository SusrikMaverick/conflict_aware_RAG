import json
from src.metrics import normalize, words


def score(answers, status, groups, wrong_groups):
    g = [{normalize(a) for a in group} for group in groups]
    allgold = set().union(*g)
    wrong = {normalize(a) for group in wrong_groups for a in group} - allgold
    p = {normalize(a) for a in answers}
    remaining = " ".join(sorted(p))
    text = normalize(json.dumps(answers, ensure_ascii=False))
    covered = all((any((words(a).search(text) for a in group)) for group in g))
    for a in sorted(allgold, key=len, reverse=True):
        remaining = words(a).sub("|", remaining)
        text = words(a).sub("|", text)
    bad = any((words(a).search(remaining) for a in wrong))
    ok = status not in ["error", "parse_error"]
    return dict(
        answer_match=int(
            ok and covered and (not any((words(a).search(text) for a in wrong)))
        ),
        strict_set=int(ok and all((p & group for group in g)) and (p <= allgold)),
        labeled_wrong=int(bad),
        failed=int(not ok),
        abstained=int(ok and (not answers)),
    )
