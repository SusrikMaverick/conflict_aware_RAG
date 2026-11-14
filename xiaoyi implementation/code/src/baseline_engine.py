"""Seven baseline workflows using the shared model client and frozen contracts."""

import json
import re
import types
from jsonschema import Draft202012Validator
from answer_format import SCHEMA, parse
from real_runtime import load_plain_prompt
from prompt_parts import CONTRACT
from baseline_prompts import (
    REVIEW,
    REFINE,
    MEMORY,
    CONSOLIDATE,
    FINAL,
    OLD,
    NEW,
    PATTERN,
    FAITHFUL_CONTRACT,
    plain_parse,
)
from baseline_sources import madam, trust_query, robust_wrapper, faithful_prompts


def documents(row):
    return "\n".join(
        f"Document {i}: {d['text']}" for i, d in enumerate(row["documents"], 1)
    )


class BaselineEngine:
    def __init__(self, client, preprocessing):
        self.client = client
        self.preprocessing = preprocessing

    def one(self, row, method):
        output = dict(
            question_id=row["question_id"],
            method=method,
            status="pending",
            accepted_answers=[],
            requests=[],
            stages=[],
        )
        ram = row["question_id"].startswith("ramdocs-")

        def call(stage, prompt, cap=1024, schema=False):
            response = self.client.chat(prompt, cap, schema, 1.05)
            output["requests"].append(
                {
                    k: response[k]
                    for k in (
                        "key",
                        "usage",
                        "finish_reason",
                        "cache_hit",
                        "latency_seconds",
                    )
                }
            )
            output["stages"].append(
                dict(
                    stage=stage,
                    text=response["text"],
                    key=response["key"],
                    output_cap=cap,
                )
            )
            # Cross-dataset runs validated structured output before parsing.
            if schema and not ram:
                Draft202012Validator(schema).validate(json.loads(response["text"]))
            return response["text"]

        try:
            if method == "plain_1024":
                prompt = load_plain_prompt().format(
                    question=row["question"], documents=documents(row)
                )
                answer = call("final", prompt + CONTRACT, 1024, SCHEMA)
            elif method == "self_reflection":
                answer = call(
                    "initial",
                    load_plain_prompt().format(
                        question=row["question"], documents=documents(row)
                    ),
                    128,
                )
                for i in range(2):
                    fields = dict(
                        question=row["question"],
                        documents=documents(row),
                        answer=answer,
                    )
                    review = call(f"review_{i + 1}", REVIEW.format(**fields), 128)
                    answer = call(
                        f"refine_{i + 1}", REFINE.format(**fields, review=review), 128
                    )
            elif method == "madam":
                debate = madam(lambda prompt, cap: call("debate", prompt, cap))
                rounds = debate(
                    row["question"], [d["text"] for d in row["documents"]], None, 3
                )
                output["rounds"] = rounds
                answer = rounds["final_aggregation"]
            elif method == "astute_structured1024":
                memory = call(
                    "internal_knowledge", MEMORY.format(question=row["question"]), 512
                )
                initial = "\n".join(
                    f"Document {i} (external retrieval): {d['text']}"
                    for i, d in enumerate(row["documents"], 1)
                )
                initial += f"\nDocument {len(row['documents']) + 1} (internal memory): {memory}"
                last = initial
                for i in range(2):
                    last = call(
                        f"consolidate_{i + 1}",
                        CONSOLIDATE.format(
                            initial=initial, last=last, question=row["question"]
                        ),
                        512,
                    )
                answer = call(
                    "final",
                    FINAL.format(
                        initial=initial, last=last, question=row["question"]
                    ).replace(OLD, NEW),
                    1024,
                    SCHEMA,
                )
            elif method == "faithfulrag_adaptation":
                prompt_type = faithful_prompts()
                pg, ex, qa = [
                    prompt_type("openai", kind)
                    for kind in ["normal", "extract", "qa-cot"]
                ]
                knowledge = call(
                    "factual_knowledge",
                    ex.system_prompt
                    + "\n"
                    + ex.generate_factual_knowledge(row["question"]),
                    1000,
                )
                context = call(
                    "self_context",
                    pg.system_prompt
                    + "\n"
                    + pg.generate_context_by_factual_knowledge(
                        row["question"], re.findall(PATTERN, knowledge)
                    ),
                    1000,
                )
                mined = call(
                    "extract_self_facts",
                    ex.system_prompt + "\n" + ex.generate_context_extract(context),
                    1000,
                )
                facts = re.findall(PATTERN, mined)
                top = self.preprocessing.align(row, facts)
                selected = (
                    " ".join(x["chunk"] for x in top[0]["topk_chunks"]) if top else None
                )
                output.update(mined_facts=facts, selected_chunks=top)
                answer = call(
                    "final",
                    qa.system_prompt
                    + "\n"
                    + qa.generate_qa_prompt_normal_cot(
                        documents(row), row["question"], facts=selected
                    )
                    + FAITHFUL_CONTRACT,
                    1024,
                    SCHEMA,
                )
            elif method == "trust":
                filtered = self.preprocessing.filter(row)

                class Pipeline:
                    def __call__(self, prompts, params):
                        if len(prompts) != 1:
                            raise ValueError("Expected one TrustRAG question")
                        i = len(output["stages"])
                        prompt = prompts[0] + (CONTRACT if i == 2 else "")
                        return [
                            types.SimpleNamespace(
                                text=call(
                                    ["internal_memory", "consolidate", "final"][i],
                                    prompt,
                                    1024,
                                    SCHEMA if i == 2 else False,
                                )
                            )
                        ]

                final, _, _ = trust_query()(
                    [filtered["documents"]], [row["question"]], Pipeline(), None
                )
                output["filter"] = filtered
                answer = final[0]
            elif method == "robust_keyword":
                prompts = robust_wrapper().wrap_prompt(
                    dict(
                        question=row["question"],
                        topk_content=[d["text"] for d in row["documents"]],
                    ),
                    as_multi_choice=False,
                    seperate=True,
                )
                isolated = [
                    call(f"document_{i}", prompt, 256)
                    for i, prompt in enumerate(prompts, 1)
                ]
                hints = self.preprocessing.keywords(row, isolated)
                output["aggregation"] = hints
                output["certified"] = False
                if hints["abstained"]:
                    explanation = (
                        "Official keyword stage abstained: no retained document answers."
                        if ram
                        else "Official keyword aggregation abstained."
                    )
                    answer = json.dumps(
                        {"accepted_answers": [], "explanation": explanation},
                        separators=(",", ":"),
                    )
                else:
                    answer = call(
                        "final", hints["final_prompt"] + CONTRACT, 1024, SCHEMA
                    )
            else:
                raise ValueError("Unknown baseline: " + method)
            output["raw_output"] = answer
            try:
                answers = (
                    plain_parse(answer)
                    if method in {"madam", "self_reflection"}
                    else parse(answer)
                )
                output.update(
                    accepted_answers=answers,
                    status="answered" if answers else "abstain",
                )
            except (ValueError, SyntaxError, TypeError):
                output["status"] = (
                    "parse_error"
                    if ram or method in {"madam", "self_reflection"}
                    else "error"
                )
        except Exception as exc:
            output.update(
                status="error",
                accepted_answers=[],
                error=type(exc).__name__ + ": " + str(exc),
            )
        return output
