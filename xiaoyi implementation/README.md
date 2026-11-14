# Conflict-aware question answering

Six project methods and seven external baselines, evaluated on RAMDocs (500 questions), RGB (100) and ConflictBank-500. All 1,100 frozen questions, supplied documents, labels and response archives are included. Large inference-model and encoder weights are downloaded separately.

## Start here

Open a terminal in this folder. Python 3.10 or 3.12 is supported. These commands require only Python:

```bash
python run.py verify
python run.py score --output outputs/scores
```

On Windows, `py` can be used instead of `python`. The score command reproduces the full 39-row table from 14,300 saved predictions. Add `--metric legacy` to use the historical scoring convention.

## Replay without a GPU

Create an isolated environment and install the small replay dependency:

```bash
python -m venv .venv
```

Activate it on Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Then run:

```bash
python -m pip install -r requirements.txt
python run.py test
python run.py baselines generate --replay --dataset RGB --limit 2 --output outputs/demo
python run.py baselines generate --replay --dataset all --output outputs/baselines
python run.py ours generate --replay --dataset all --output outputs/ours
```

Replay executes the workflows using saved model responses and verifies requests and final answers. It does not make new model predictions. Historical failures remain in the results.

## New inference and preprocessing

[Baseline instructions](docs/BASELINES.md) cover all seven baselines, CPU preprocessing and GPU inference. [Project-method instructions](docs/USAGE.md) cover the six project methods and custom inputs. Run their commands from this top-level folder. A compatible GPU environment and separately downloaded weights are required for new inference.

## Files

| Location | Purpose |
|---|---|
| `run.py` | Common entry point; run `python run.py --help` for commands |
| `code/` | Execution code, complete frozen data, model resources and key tests |
| `docs/` | Methods, setup instructions and source attribution |
| `results/` | Three-dataset results in PDF and CSV; all project rows marked Ours in the PDF |

The six project methods are Docs + Cards, Evidence Pruning, Memory + Cards, Context Review, Answer Agreement and Dual Review. See [methods](docs/METHODS.md) and [data and methods PDF](docs/Data_and_Methods.pdf).

## Where the project methods are implemented

The top-level `run.py` is a command wrapper. The six project workflows are implemented in `code/run.py` and the shared modules below.

| File | Role |
|---|---|
| [code/run.py](code/run.py) | Schedules prerequisite stages, runs the six methods, and saves intermediate outputs and final predictions. |
| [code/src/card_pipeline.py](code/src/card_pipeline.py) | Implements the original Docs + Cards workflow and shared card, coverage and evidence-audit stages. |
| [code/src/source_cards.py](code/src/source_cards.py) | Extracts document cards and locates/checks their quoted source spans. It does not run MiniCheck. |
| [code/src/evidence_pruning.py](code/src/evidence_pruning.py) | Defines Evidence Pruning audit requests and candidate retention/deletion rules. |
| [code/src/answer_review.py](code/src/answer_review.py) | Implements memory drafts, context reviews and answer selection used by Memory + Cards, Context Review, Answer Agreement and Dual Review. |
| [code/methods.py](code/methods.py) | Defines the six project method names and command keys. |

| Project method | Main implementation |
|---|---|
| Docs + Cards | `card_pipeline.py`, with extraction in `source_cards.py` |
| Evidence Pruning | `card_pipeline.py` and `evidence_pruning.py` |
| Memory + Cards | `answer_review.py`, with shared prerequisite stages scheduled by `code/run.py` |
| Context Review | `answer_review.py`, with shared prerequisite stages scheduled by `code/run.py` |
| Answer Agreement | `answer_review.py`, including normalized answer-list intersection |
| Dual Review | `answer_review.py`, including the two-audit candidate deletion rule |

Suggested reading order: `code/methods.py` for the names, `code/run.py` for stage scheduling, then `code/src/card_pipeline.py` for the original method. Continue with `evidence_pruning.py` and `answer_review.py` for the extensions. The methods share prerequisites rather than each occupying a standalone file. See [Methods](docs/METHODS.md) for a short explanation of each workflow.

## Interpretation

The three main metrics are answer match, strict answer-set match and questions accepting a labeled wrong answer. Lower is better for the last metric. Failed questions remain in the denominator; abstention may lower wrong-answer counts. The six methods were selected after observing the data, so these comparisons are exploratory. Baselines are local adaptations with different call budgets, not reproductions of published paper scores.

