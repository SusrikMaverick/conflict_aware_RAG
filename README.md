# conflict_aware_RAG

Runs Qwen2.5-72B-Instruct-AWQ with vLLM on one A100-80GB on ASU Sol.

## Install (once)

First open a terminal on a Sol GPU node: in Open OnDemand, start a VS Code job with
1× A100-80GB, 8 CPUs, 64 GB RAM. Then paste these commands in order.

```bash
# 1. Code
git clone https://github.com/SusrikMaverick/conflict_aware_RAG.git ~/projects/conflict_aware_RAG
cd ~/projects/conflict_aware_RAG
sed -i "s#/scratch/smukhe61/#/scratch/$USER/#g" src/interface.py pyproject.toml

# 2. Python env (8 GB, on scratch)
module load mamba/latest
mamba create -y -p /scratch/$USER/conflict_aware_RAG/envs/vllm -c conda-forge python=3.12 pip llguidance=1.7.5   # llguidance from conda: PyPI has no build for Sol's OS
source activate /scratch/$USER/conflict_aware_RAG/envs/vllm
pip install --no-cache-dir -e ".[serve]"

# 3. Model (39 GB, on scratch)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-72B-Instruct-AWQ',
                  revision='698703eae6604af048a3d2f509995dc302088217',
                  local_dir='/scratch/$USER/conflict_aware_RAG/models/Qwen2.5-72B-Instruct-AWQ')"

# 4. Benchmark
mkdir -p benchmarks/RAMDocs
curl -L -o benchmarks/RAMDocs/RAMDocs_test.jsonl \
  https://huggingface.co/datasets/HanNight/RAMDocs/resolve/9c041bfd158c603b615883d9a931b00cbc141494/RAMDocs_test.jsonl
```

## Run

Every new terminal (on a GPU node):

```bash
cd ~/projects/conflict_aware_RAG
module load mamba/latest
source activate /scratch/$USER/conflict_aware_RAG/envs/vllm

python -m src.interface   # chat in the terminal (server startup takes ~5 min)
python -m src.eval        # RAMDocs baseline eval
```

Quitting (Ctrl-C or Ctrl-D) stops the server. The server log is in `logs/vllm.log`.

## Compare Plain RAG and Docs + cards

The saved comparison on `codex/ramdocs-repro-package` already uses the same 7B
answerer, scorer, and 128-token final output limit for both methods. The new
comparison runner repeats both methods with the 72B answerer from this setup:

- Plain RAG sends the question and original documents to the answerer.
- Docs + cards sends the same input plus the existing saved cards.
- Both methods generate fresh answers using the same model, settings, output
  limit, and Python scorer. The scorer also reports strict answer-set accuracy.

The cards remain the existing **7B-extracted cards**, with all saved cards
included. This first round changes the final answerer, not card extraction or
filtering. It does not claim an entirely 72B pipeline.

Once, copy the saved package from the other branch into the ignored local
benchmark directory. This reads the branch without switching branches and
leaves an existing package unchanged:

```bash
python - <<'PY'
from pathlib import Path
import subprocess

package = Path("benchmarks/ramdocs_cards/RAMDocs_Reproducible_Package.zip")
if not package.exists():
    data = subprocess.check_output([
        "git", "show",
        "origin/codex/ramdocs-repro-package:reproductions/ramdocs_cards/RAMDocs_Reproducible_Package.zip",
    ])
    package.parent.mkdir(parents=True, exist_ok=True)
    package.write_bytes(data)
PY
```

Check the inputs first. This makes no model calls and needs no GPU:

```bash
python -m src.compare \
  --bundle benchmarks/ramdocs_cards/RAMDocs_Reproducible_Package.zip \
  --dry-run
```

Then, in the activated environment on a Sol GPU node, run the comparison:

```bash
python -m src.compare \
  --bundle benchmarks/ramdocs_cards/RAMDocs_Reproducible_Package.zip \
  --model /scratch/$USER/conflict_aware_RAG/models/Qwen2.5-72B-Instruct-AWQ \
  --max-tokens 128 \
  --out runs/ramdocs-72b-paired
```

Use a new output directory for each run; existing results are never overwritten.
An extracted `ramdocs_repro_bundle` directory also works as `--bundle`. To test a
different final output limit, change `--max-tokens`; it applies to both methods.
The run saves its configuration in `manifest.json`, prompts in `prompts.jsonl`,
raw answers and scores in `plain.jsonl` and `plus_cards.jsonl`, and a side-by-side
per-question comparison in `paired.jsonl`. Overall results are in `summary.json`.
The shared final output limit does not make total
compute equal: card extraction is additional work and its saved 7B results are
reused.

To preview a queued Sol smoke job for A/B (five questions, no submission):

```bash
bash scripts/submit_sol.sh --stage ab --dry-run \
  --run-root /scratch/$USER/conflict_aware_RAG/runs/ab-smoke
```

On Sol, use `--submit` in place of `--dry-run` when ready to queue it. After the
smoke run succeeds, use `--full` and a fresh run root for all 500 questions.
The wrapper copies the source and input package into the run root before
submission, so later branch switches do not change queued jobs. Model and
environment paths can be supplied with `--model` and `--python`.

## C: create fresh cards with 72B

C keeps the original extraction instructions and card fields. It reads each
document separately with the selected 72B model, then reconstructs quotations
from real source segments. It saves the raw extraction outputs, valid cards,
discarded-card reasons, and token usage. Evaluation labels never enter the
extractor prompt. MiniCheck is not rerun: its scores did not filter the original
all-card comparison, and are not sent to the final answerer.

Validate the full extraction plan without starting a model:

```bash
python -m src.cards extract \
  --bundle benchmarks/ramdocs_cards/RAMDocs_Reproducible_Package.zip --dry-run
```

On a Sol GPU node, create cards and then generate C's final answers:

```bash
python -m src.cards extract \
  --bundle benchmarks/ramdocs_cards/RAMDocs_Reproducible_Package.zip \
  --model /scratch/$USER/conflict_aware_RAG/models/Qwen2.5-72B-Instruct-AWQ \
  --out runs/c-cards

python -m src.compare \
  --bundle benchmarks/ramdocs_cards/RAMDocs_Reproducible_Package.zip \
  --cards runs/c-cards --only-cards \
  --model /scratch/$USER/conflict_aware_RAG/models/Qwen2.5-72B-Instruct-AWQ \
  --max-tokens 128 --out runs/c-answers
```

`--only-cards` avoids generating Plain answers again. Compare C with the B
answers from the shared A/B run, keeping the same final model and settings.
Use `--limit 5` on both commands for a smoke run. To continue interrupted card
creation, repeat its command with `--resume`; successful documents are kept and
failed or missing documents are retried. Changed inputs/settings require a new
directory. Final-answer runs always require a fresh output directory.

To preview C alone as a queued job:

```bash
bash scripts/submit_sol.sh --stage c --dry-run --run-root runs/sol-c-smoke
```

New extraction may recover the four previous source failures as well as change
card quality. This is a comparison of extractor configurations, rather than
proof that parameter count alone caused any difference.

## D: check whether C's cards answer the question

D reads the exact saved C cards, the question, and each card's original source.
The 72B reviewer checks the requested entity, attribute, time, negation,
quantity/completeness, and other explicit restrictions. Each check records a
verdict, source passage, and reason. Kept cards are unchanged; the reviewer
cannot rewrite answers or add new candidates. A source-grounded check does not
establish that the source itself is true.

```bash
python -m src.cards review \
  --bundle benchmarks/ramdocs_cards/RAMDocs_Reproducible_Package.zip \
  --cards runs/c-cards \
  --model /scratch/$USER/conflict_aware_RAG/models/Qwen2.5-72B-Instruct-AWQ \
  --out runs/d-cards

python -m src.compare \
  --bundle benchmarks/ramdocs_cards/RAMDocs_Reproducible_Package.zip \
  --cards runs/d-cards --only-cards \
  --model /scratch/$USER/conflict_aware_RAG/models/Qwen2.5-72B-Instruct-AWQ \
  --max-tokens 128 --out runs/d-answers

python -m src.experiment_report \
  --runs runs/ramdocs-72b-paired runs/c-answers runs/d-answers \
  --out runs/abcd-report
```

Use the same question selection and final-answer settings in all runs. The
report checks model/settings, scoring, benchmark hashes, and C-to-D lineage
before comparing A→B, B→C and C→D. It includes wrong-answer counts, strict
answer-set scores, single/multiple-answer groups, and stage costs. These are
development experiments on already-observed RAMDocs data, not independent
confirmation on unseen questions.

See [Sol job instructions](docs/SOL_RUNS.md) for a queued five-question smoke
test and the full dependent job chain, and [branch/push order](docs/BRANCHES.md)
for the three separate version commits.
