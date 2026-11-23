"""Shared document serialization, answer contract and card fields."""


def docs(row):
    return "\n".join(
        (f"Document {i}: {d['text']}" for i, d in enumerate(row["documents"], 1))
    )


CONTRACT = "\nRAMDocs final output contract (overrides example output formats): Return JSON with accepted_answers (all correct SHORT answer strings, including multiple valid answers; empty list if unsupported) and explanation (reasoning). Do not put confidence, sources or explanatory sentences in accepted_answers."


FIELDS = ("doc_id", "card_id", "entity", "scope", "claim", "answer", "quote")
SUFFIX = "\n\nAdditional extracted evidence from the SAME documents (not independent sources and not guaranteed true). Use alongside the original documents above.\n"
