"""Shared structured-answer schema and parser."""

import json

SCHEMA = {
    "type": "object",
    "properties": {
        "accepted_answers": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string"},
    },
    "required": ["accepted_answers", "explanation"],
    "additionalProperties": False,
}


def parse(s):
    obj = json.loads(s)
    if set(obj) != set(SCHEMA["properties"]) or not isinstance(obj["explanation"], str):
        raise ValueError("Invalid object")
    a = obj["accepted_answers"]
    if not isinstance(a, list) or any((not isinstance(x, str) for x in a)):
        raise ValueError("Non-string answers")
    return [
        x for x in a if x.strip().lower() not in ("unknown", "none", "null", "n/a", "")
    ]
