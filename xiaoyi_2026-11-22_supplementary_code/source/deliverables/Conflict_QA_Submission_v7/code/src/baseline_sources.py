"""Load used upstream definitions without running model setup or entrypoints."""

import ast
import logging
import re
import string
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
UPSTREAM = BASE / "upstream"


def load(relative_path, names, namespace=None):
    path = UPSTREAM / relative_path
    tree = ast.parse(path.read_text(encoding="utf8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            selected = node.name in names
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            selected = any(
                isinstance(n, ast.Name) and n.id in names
                for t in targets
                for n in ast.walk(t)
            )
            if isinstance(node, ast.Assign) and isinstance(
                node.targets[0], ast.Subscript
            ):
                selected = (
                    isinstance(node.targets[0].value, ast.Name)
                    and node.targets[0].value.id in names
                )
        else:
            selected = False
        if selected:
            nodes.append(node)
    namespace = {} if namespace is None else namespace
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def madam(call):
    return load(
        "MADAM/run_madam_rag.py",
        {
            "normalize_answer",
            "agent_response",
            "aggregate_responses",
            "multi_agent_debate",
        },
        {
            "re": re,
            "string": string,
            "List": list,
            "call_llm": lambda prompt, generator, max_new_tokens=128: call(prompt, 128),
        },
    )["multi_agent_debate"]


def trust_query():
    return load("TrustRAG/defend_module.py", {"conflict_query"})["conflict_query"]


def faithful_prompts():
    return load("FaithfulRAG/prompts.py", {"SYSTEM_PROMPTS", "PromptGenerator"})[
        "PromptGenerator"
    ]


def robust_wrapper():
    namespace = {}
    path = UPSTREAM / "RobustRAG/prompt_template.py"
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    tree = ast.parse((UPSTREAM / "RobustRAG/models.py").read_text())
    method = next(
        n
        for c in tree.body
        if isinstance(c, ast.ClassDef)
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == "wrap_prompt"
    )
    wrapper_type = ast.ClassDef(
        name="Wrapper", bases=[], keywords=[], body=[method], decorator_list=[]
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[wrapper_type], type_ignores=[])),
            "<upstream wrapper>",
            "exec",
        ),
        namespace,
    )
    wrapper = namespace["Wrapper"]()
    wrapper.prompt_template = namespace["MISTRAL_TMPL"]
    return wrapper


def trust_filter():
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    from rouge_score import rouge_scorer

    names = {
        "calculate_similarity",
        "calculate_pairwise_rouge",
        "calculate_average_score",
        "group_n_gram_filtering",
        "k_mean_filtering",
    }
    return load(
        "TrustRAG/defend_module.py",
        names,
        dict(
            np=np,
            KMeans=KMeans,
            cosine_similarity=cosine_similarity,
            StandardScaler=StandardScaler,
            rouge_scorer=rouge_scorer,
        ),
    )["k_mean_filtering"]


def robust_keyword_type():
    import types
    import numpy as np
    import spacy
    import nltk
    from collections import defaultdict

    nltk.data.path.insert(0, str(BASE / "assets/nltk"))
    # Only the model path changes; load the bundled en_core_web_sm 3.8.0 weights.
    local_spacy = types.SimpleNamespace(
        load=lambda _: spacy.load(BASE / "assets/en_core_web_sm")
    )
    return load(
        "RobustRAG/defense.py",
        {"RRAG", "KeywordAgg", "punctuation"},
        dict(
            spacy=local_spacy,
            np=np,
            defaultdict=defaultdict,
            stopword_set=set(nltk.corpus.stopwords.words("english")),
            logger=logging.getLogger("robustrag"),
        ),
    )["KeywordAgg"]
