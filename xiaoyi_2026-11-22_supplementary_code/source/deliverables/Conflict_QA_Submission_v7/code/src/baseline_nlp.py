"""Trust filtering, Robust keyword aggregation and Faithful contextual alignment."""

import gzip
import json
import os
import re
from pathlib import Path
from baseline_sources import (
    BASE,
    load,
    trust_filter,
    robust_wrapper,
    robust_keyword_type,
)


def records(name):
    with gzip.open(BASE / "data/baselines" / name, "rt", encoding="utf8") as stream:
        return [json.loads(line) for line in stream]


def model_root():
    return Path(os.environ.get("QA_NLP_MODELS", BASE / "models"))


class Encoder:
    def __init__(self, name, cls=False):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(2)
        self.torch = torch
        self.cls = cls
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_root() / name, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_root() / name, local_files_only=True
        ).eval()

    def encode(self, texts, **kwargs):
        torch = self.torch
        if not texts:
            return torch.empty((0, 384))
        batches = []
        with torch.inference_mode():
            for i in range(0, len(texts), 64):
                settings = {} if self.cls else {"max_length": 256}
                inputs = self.tokenizer(
                    texts[i : i + 64],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    **settings,
                )
                hidden = self.model(**inputs).last_hidden_state
                if self.cls:
                    embeddings = hidden[:, 0, :]
                else:
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                batches.append(embeddings)
        return torch.cat(batches)


def faithful_alignment(encoder):
    import typing
    import nltk
    import torch

    nltk.data.path.insert(0, str(BASE / "assets/nltk"))

    class Util:
        @staticmethod
        def cos_sim(a, b):
            a = a.unsqueeze(0) if a.ndim == 1 else a
            b = b.unsqueeze(0) if b.ndim == 1 else b
            return (
                torch.nn.functional.normalize(a, p=2, dim=1)
                @ torch.nn.functional.normalize(b, p=2, dim=1).T
            )

    class Converter:
        @staticmethod
        def remove_brackets_and_content(text):
            return " ".join(
                re.sub(r"\[.*?\]", "", re.sub(r"\<.*?\>", "", text)).split()
            )

    ns = dict(vars(typing))
    ns.update(
        nltk=nltk,
        torch=torch,
        util=Util,
        FormatConverter=Converter,
        Dataset=list,
        SentenceTransformer=lambda _: encoder,
    )
    cls = load("FaithfulRAG/modules.py", {"ContextualAlignmentModule"}, ns)[
        "ContextualAlignmentModule"
    ]
    return cls("all-MiniLM-L6-v2")


class Preprocessor:
    """Fresh mode calculates NLP steps; replay checks and reuses frozen NLP outputs."""

    def __init__(self, replay=False):
        self.replay = replay
        self.encoders = {}
        self._filter = None
        self._keyword = None
        self._alignment = None
        if replay:
            self.filters = {
                r["question_id"]: r for r in records("trust_filters.jsonl.gz")
            }
            self.hints = {r["question_id"]: r for r in records("robust_hints.jsonl.gz")}
            saved_stages = records("stages.jsonl.gz")
            self.faithful = {
                r["output"]["question_id"]: r["output"]
                for r in saved_stages
                if r["kind"] == "faithfulrag_adaptation"
            }
            self.isolated = {
                r["output"]["question_id"]: [s["text"] for s in r["output"]["stages"]]
                for r in saved_stages
                if r["kind"] == "robust_isolated"
            }
            self.inputs = {
                r["question_id"]: r
                for r in json.loads((BASE / "data/inputs.json").read_text())
            }

    def check_row(self, row):
        if self.replay and self.inputs.get(row["question_id"]) != row:
            raise ValueError(
                "Frozen preprocessing requires the exact archived question and documents"
            )

    def filter(self, row, embeddings=None):
        self.check_row(row)
        if self.replay and embeddings is None:
            return self.filters[row["question_id"]]
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        docs = [d["text"] for d in row["documents"]]
        if embeddings is None:
            encoder = self.encoders.setdefault("simcse", None)
            if encoder is None:
                encoder = self.encoders["simcse"] = Encoder(
                    "sup-simcse-bert-base-uncased", cls=True
                )
            embeddings = encoder.encode(docs).numpy()
        guard = None
        if len(docs) < 2:
            kept, guard = docs, "upstream_top_k1_bypass"
        elif np.any(
            np.linalg.norm(StandardScaler().fit_transform(embeddings), axis=1) == 0
        ):
            kept, guard = docs, "zero_centered_vector_keep_all"
        else:
            if self._filter is None:
                self._filter = trust_filter()
            _, kept = self._filter(embeddings, docs, [], True)
        return dict(
            question_id=row["question_id"],
            documents=kept,
            kept_count=len(kept),
            original_count=len(docs),
            guard=guard,
        )

    def keywords(self, row, responses, recompute=False):
        self.check_row(row)
        if self.replay and not recompute:
            if self.isolated[row["question_id"]] != responses:
                raise ValueError("Document answers do not match frozen keyword input")
            return self.hints[row["question_id"]]
        wrapper = robust_wrapper()

        class Capture:
            def __init__(self):
                self.final_prompt = None
                self.hints = None

            def batch_query(self, prompts):
                if len(prompts) != len(responses):
                    raise ValueError("Document/response count mismatch")
                return responses

            def wrap_prompt(self, *args, **kwargs):
                if "hints" in kwargs:
                    self.hints = kwargs["hints"]
                return wrapper.wrap_prompt(*args, **kwargs)

            def query(self, prompt):
                self.final_prompt = prompt
                return "REPLAY_CAPTURE_ONLY"

        capture = Capture()
        if self._keyword is None:
            self._keyword = robust_keyword_type()(
                capture, relative_threshold=0.3, absolute_threshold=3, longgen=False
            )
        self._keyword.llm = capture
        self._keyword.query(
            dict(
                question=row["question"],
                topk_content=[d["text"] for d in row["documents"]],
            ),
            corruption_size=0,
        )
        return dict(
            question_id=row["question_id"],
            hints=capture.hints,
            final_prompt=capture.final_prompt,
            abstained=capture.final_prompt is None,
            certified=False,
        )

    def align(self, row, facts, recompute=False):
        self.check_row(row)
        if self.replay and not recompute:
            saved = self.faithful[row["question_id"]]
            if saved["mined_facts"] != facts:
                raise ValueError("Facts do not match frozen alignment input")
            return saved["selected_chunks"]
        if self._alignment is None:
            self._alignment = faithful_alignment(Encoder("all-MiniLM-L6-v2"))
        context = "\n".join(
            f"Document {i}: {d['text']}" for i, d in enumerate(row["documents"], 1)
        )
        chunks = self._alignment.get_contextual_chunks(
            [dict(id=row["question_id"], facts=facts)],
            [dict(id=row["question_id"], context=context)],
        )
        return self._alignment.get_topk_contextual_chunks(chunks)
