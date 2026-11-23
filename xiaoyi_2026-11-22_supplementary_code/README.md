# Conflict QA: result-table reproduction

This package accompanies `results/Current_Results_Tables.pdf`, the three-page full-dataset table uploaded to Drive on November 19. The PDF contains RAMDocs (500 questions), RGB (100), and ConflictBank-500 (500), with Qwen2.5-7B BF16, Qwen3-32B AWQ 4-bit, and Qwen3.5-35B-A3B GPTQ Int4. It is a snapshot of those results; it does not include later development-slice experiments.

## Run

Open a terminal in the directory containing this README and `run.py`: the extracted `Conflict_QA_Results` directory, or the dated supplement directory in a GitHub checkout. Run with Python 3.10 or newer:

```bash
python run.py verify
python run.py score
```

These commands use the Python standard library and require no GPU, API key, model download, or network connection. Scoring reads saved answers and gold labels, retains failures and abstentions, and checks all 75 model/dataset/method rows against the supplied table. Results are written to `outputs/`, including a question-level CSV. The PDF displays 66 selected rows; Context Review, Answer Agreement, and Dual Review remain in the complete CSV and original implementation.

Each model/method configuration covers the same 1,100 questions. The scorer also checks question coverage, duplicate records, the metric version, and stored percentages.

To rebuild the PDF, create a separate environment and install the PDF dependency:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Then:

```bash
python -m pip install -r requirements.txt
python run.py pdf
```

The generated file is `outputs/Current_Results_Tables.pdf`, rendered from the CSV recomputed by that command. On Windows, `py` may replace `python`. Fonts and their license are included. PDF metadata may differ between builds; the table values must match.

## Metrics

- **Answer match:** all gold-answer groups are matched, with no labeled wrong answer accepted. Matching uses normalized aliases and word boundaries.
- **Strict set:** every gold-answer group has an exact normalized alias in the predicted set, with no extra answer strings.
- **Wrong accepted:** questions containing at least one labeled wrong answer, excluding aliases also marked as gold. Lower is better.
- **Failed / abstained:** recorded execution or parsing failures / successful outputs with an empty answer list. Both remain in the total question count.

The metric is `answer_boundary_v2`. Bold green entries mark the best values within each model table, with ties retained.

## Contents

| Path | Contents |
| --- | --- |
| `run.py` | Verify files, rescore saved predictions, rebuild the PDF |
| `results/` | Original PDF and all 75 summary rows |
| `source/deliverables/Conflict_QA_Submission_v7/` | Original six-method/seven-baseline implementation, complete frozen inputs and labels, response archives, tokenizer, usage instructions and attribution |
| `source/work/` | Backbone clients/runners, teammate adapter, TruthfulRAG adapter and upstream implementation, original analyses and PDF renderer |
| `source/runs/backbones_20261113/` | Qwen3/Qwen3.5 full-run predictions, per-document cards, intermediate outputs, raw request/response caches |
| `source/runs/truthful_20261112/` | TruthfulRAG question outputs, graph artifacts and compressed raw request/response files |
| `source/runs/teammate_20261115/` | Three teammate-method full runs, traces and raw requests/responses |
| `source/planning/` | Frozen experiment inputs, model revisions, execution parameters, dependency locks and TruthfulRAG source patch |
| `source/materials/` | Pinned teammate source |
| `source/models/` | Model configuration and tokenizer resources; inference weights are obtained separately |

The original relative directory layout is retained under `source/` so archived imports and file references remain traceable. Use the top-level commands for portable scoring and PDF generation. Method names in saved records are preserved to retain request identities and historical compatibility.

## Implementation and replay

The original project methods are in `source/deliverables/Conflict_QA_Submission_v7/code/run.py` and its `src/` modules, including `card_pipeline.py`, `source_cards.py`, and `answer_review.py`. The seven original baselines are documented in that package's `docs/BASELINES.md`. Commands in its nested documentation assume that package's own directory; the commands below run from this ZIP's top-level directory.

For their full workflow replay, install the lightweight replay dependency and run:

```bash
python -m pip install -r source/deliverables/Conflict_QA_Submission_v7/requirements.txt
python source/deliverables/Conflict_QA_Submission_v7/run.py baselines generate --replay --dataset all --output outputs/baseline_replay
python source/deliverables/Conflict_QA_Submission_v7/run.py ours generate --replay --dataset all --output outputs/project_replay
```

These replay commands reconstruct the original workflows from archived model responses. Top-level `score` performs scoring replay for every row, including the added backbones, teammate methods and TruthfulRAG; it does not rerun their model-generation workflows. Their raw requests and original executors are included for further replay work.

TruthfulRAG's label-free adaptation is implemented in `source/work/truthful_run.py` and `source/work/external_20261112/upstream/`. Its entropy probe uses a fixed 32-token budget instead of the reference-answer length. Gold answers are loaded only for evaluation.

Qwen3/Qwen3.5 request payloads disable thinking with `enable_thinking=False`. The model IDs and revisions are in `source/planning/backbones_20261113/protocol.json`. Token caps, parsing failures and abstentions are retained in the saved evidence. Low wrong-answer counts should be read together with failures and coverage.

Fresh GPU inference requires separate model/encoder weights and a compatible runtime; it is not provided by the two CPU reproduction commands. Runtime locks, original executor code and setup instructions are included. The frozen executors refer to their historical model/service locations; configure an isolated runtime and separate run destinations before fresh inference. Do not interpret rerunning an executor over existing outputs as fresh inference.

Results are observed exploratory comparisons with differing method budgets, not independent confirmation or original-paper scores. Source attribution and available license notices are retained with the implementations and data.
