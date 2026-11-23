"""Historical baseline prompts; wording and output contracts are preserved."""

import ast, json, math, re

HEADER = "You are an expert in retrieval question answering. Answer using the supplied documents. Give every correct answer when there is more than one, with a step-by-step explanation. If no answer is supported, respond unknown.\n"
REVIEW = (
    HEADER
    + "Question: {question}\n{documents}\nPrevious answer:\n{answer}\nExamine your previous answer and identify its problems."
)
REFINE = (
    REVIEW
    + "\nFeedback:\n{review}\nCorrect the problems and restate your improved answer. Begin with All Correct Answers: [a JSON list of short answers]. Then provide Explanation: followed by your reasoning."
)
MEMORY = "Write a passage with accurate information relevant to the question, using your own knowledge. Explicitly say I do not know where your knowledge is uncertain; avoid fabricating information.\nQuestion: {question}\nPassage:"
CONSOLIDATE = "Consolidate internal-memory and externally retrieved information for the question. Combine mutually consistent passages into concise source-attributed groups. Keep conflicting information in separate groups; preserve differences of entity, time and perspective. Drop irrelevant material. For each resulting group, indicate memory versus external origins and the original document numbers.\nInitial Context:\n{initial}\nLast Context:\n{last}\nQuestion: {question}\nNew Context:"
FINAL = "Use the internal and external evidence to answer the question. Consolidate consistent information while keeping conflicting groups separate and ignoring irrelevant material. Preserve source type and document numbers. Consider an answer and its confidence for each group, comparing source credibility and agreement. Select all accurate, well-supported answers, including distinct valid interpretations. Begin with All Correct Answers: [a JSON list of short answers], then Explanation: with the consolidation and confidence reasoning. Use an empty list if no answer is supported.\nInitial Context:\n{initial}\nConsolidated Context:\n{last}\nQuestion: {question}\nAnswer:"
OLD = "Begin with All Correct Answers: [a JSON list of short answers], then Explanation: with the consolidation and confidence reasoning. Use an empty list if no answer is supported."
NEW = "Return a JSON object with accepted_answers (a list of short answer strings only, without confidence, source labels or explanatory sentences) and explanation (the consolidation and confidence reasoning). Use an empty accepted_answers list if no answer is supported."


def plain_parse(text):
    m = re.search("All Correct Answers:\\s*(\\[.*?\\])", text, re.S | re.I)
    if not m:
        if text.strip().lower().strip(" .'\"") == "unknown":
            return []
        raise ValueError("missing answer list")
    try:
        out = json.loads(m.group(1))
    except json.JSONDecodeError:
        out = ast.literal_eval(m.group(1))
    if not isinstance(out, list) or not all(
        (
            isinstance(v, str) or (type(v) in (int, float) and math.isfinite(v))
            for v in out
        )
    ):
        raise ValueError("non-atomic answers")
    return [
        str(x)
        for x in out
        if str(x).lower().strip() not in ("unknown", "none", "n/a", "null")
    ]


FAITHFUL_CONTRACT = "\nRAMDocs output adaptation: Ignore the example output formatting. Return JSON with accepted_answers (all correct short answer strings, possibly multiple; empty list if unsupported) and explanation (reasoning). Do not put source labels, confidence or explanatory sentences in accepted_answers."
PATTERN = "\\d+\\.\\s([^\\d]+(?:\\s+[^\\d]+)*)"
