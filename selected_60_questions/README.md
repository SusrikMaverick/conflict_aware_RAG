# Selected 60 questions

A fixed subset for comparing methods and models on the same questions.

| Prepared dataset | Selected questions | Parent collection |
|---|---:|---:|
| RAMDocs | 20 | 500 |
| RGB | 20 | 100 |
| ConflictBank-500 | 20 | 500 |
| **Total** | **60** | **1,100** |

The subset includes all 221 documents supplied with these questions in the
frozen project collection. Questions, document text, document order, IDs and
evaluation labels are unchanged from that collection.

## Files

- `sample_questions.csv`: dataset, question ID and question text for browsing.
- `sample_manifest.json`: fixed IDs, seed, sampling description and SHA256 hashes
  of the parent input and evaluation files.
- `data/inputs.json`: all 60 questions and their supplied documents, without
  evaluation labels. Use this file for model input.
- `data/evaluation.json`: reference answer groups and other evaluation fields,
  keyed by question ID. Load this file only for scoring.
- `data/<dataset>/questions.json` and `evaluation.json`: the same records split
  by dataset. These are alternative views of the same 60 questions.
- `SHA256.json`: checksums for every other file in this folder.
- `RGB_SOURCE_README.md`: the original RGB README and license notice.

This is a data subset, with no model weights, inference runner or model results.
The `scope` and `order` fields in the original manifest describe a planned run;
they do not indicate completed experiments or limit reuse to three models.

## Sampling

Sampling uses one Python `random.Random(576)` instance, drawing 20 records
without replacement from each parent dataset in this order: RAMDocs, RGB,
ConflictBank-500. The selected records are stored in their parent order.
The generator is not reset between datasets. Use the supplied IDs for every
model and method; taking the first 20 parent records selects a different subset.

## Load the inputs

From this folder, using Python's standard library:

```python
import json
from pathlib import Path

questions = json.loads(Path("data/inputs.json").read_text(encoding="utf-8"))
print(len(questions))  # 60
for item in questions:
    question = item["question"]
    documents = item["documents"]
    # Pass the question and document texts to the chosen method.
```

For evaluation, join predictions to `data/evaluation.json` by `question_id`.
Report results separately for each 20-question dataset and identify them as
subset results. Do not combine these scores with full-dataset results.

Two RAMDocs records inherit overlapping correct and wrong answer aliases from
the parent labels: `ramdocs-0130-0c24f4852e8a` lists `Hindi` in both, and
`ramdocs-0172-b91092d330d1` lists `Ice hockey` / `Ice Hockey` in both after
case normalization. These labels are preserved. When comparing scores,
state how the scorer handles this overlap and apply the same rule to every
method and model.

To verify all file checksums, run this Python snippet from this folder:

```python
import hashlib
import json
from pathlib import Path

for name, expected in json.loads(Path("SHA256.json").read_text()).items():
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("Checksum mismatch: " + name)
print("All file checksums match.")
```

## Sources

- RAMDocs: the project's frozen 500-question prepared collection. Its original
  upstream revision was not recorded in the parent delivery. The manifest pins
  the exact prepared inputs used here.
- RGB: the English fact records used by the project, revision
  `65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615`. The parent preparation includes an
  original passage, an edited counterpart and a distractor per question.
  See [the paper](https://arxiv.org/abs/2309.01431) and the preserved
  [RGB README](RGB_SOURCE_README.md), including its CC BY-NC-SA 4.0 notice.
- ConflictBank: [`Warrieryes/CB_qa`](https://huggingface.co/datasets/Warrieryes/CB_qa),
  revision `b3b12e1dba7097151312c29c13e2a76f47d726c9`. The parent ConflictBank-500
  collection is a prepared short-answer adaptation with mixed documents;
  scores here are not native multiple-choice benchmark scores. No separate
  upstream license text was included in the parent delivery.

Dataset content remains subject to its respective source terms. The included
RGB README is source documentation; its setup commands are not required to
load these JSON files.
