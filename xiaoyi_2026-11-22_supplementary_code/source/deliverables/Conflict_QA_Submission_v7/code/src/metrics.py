"""Shared question-answering helpers."""

import re, string

PUNCT = set(string.punctuation + "‘’“”–—")


def normalize(s):
    return " ".join(
        re.sub(
            "\\b(a|an|the)\\b", " ", "".join((c for c in s.lower() if c not in PUNCT))
        ).split()
    )


def words(s):
    return re.compile(f"(?<!\\S){re.escape(s)}(?!\\S)")
