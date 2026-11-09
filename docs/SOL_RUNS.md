# Run the A–D experiments on Sol

The three jobs run in order:

| Job | Work |
|---|---|
| A/B | Answer from documents alone, then from documents + the old 7B cards |
| C | Create fresh cards with 72B, then answer from documents + those cards |
| D | Check C's cards against the question and source, then answer using the retained cards |

Every final answer uses the same selected 72B model, scoring rules, seed and
output limit. Card creation and checking have separate generation budgets. The
report counts their additional work; the experiments are not equal-compute
comparisons.

Use the A/B comparison branch first, then the C branch, then the D branch. The
D branch contains the earlier changes. Run the full chain from that checkout;
the submission script never switches branches. Changes have to be reviewed and
pushed separately before they are available on Sol.

## Prepare Sol once

Follow the main README's environment and pinned-model installation steps.
The scripts expect these defaults:

```text
Python: /scratch/$USER/conflict_aware_RAG/envs/vllm/bin/python
Model:  /scratch/$USER/conflict_aware_RAG/models/Qwen2.5-72B-Instruct-AWQ
Bundle: benchmarks/ramdocs_cards/RAMDocs_Reproducible_Package.zip
```

Prepare the original bundle with the command in the README. The wrapper accepts
the original ZIP; the Python stage commands also accept an extracted bundle.
No model weights or Python environments are copied into the run directory.

## Preview first

From the checkout, preview a five-question smoke run:

```bash
bash scripts/submit_sol.sh --run-root runs/sol-smoke-01 --dry-run
```

This prints the three `sbatch` commands. It creates no files, contacts no
scheduler, and does not need the Sol model or environment to exist on the
computer showing the preview. The second job waits for A/B to succeed; the
third waits for C. A failed job prevents its dependent job from starting.

For the A/B branch by itself:

```bash
bash scripts/submit_sol.sh --stage ab --run-root runs/sol-ab-smoke-01 --dry-run
```

When ready, submit the smoke run **from Sol**:

```bash
bash scripts/submit_sol.sh --run-root runs/sol-smoke-01 --submit
```

The `--submit` flag is required. An existing run root is refused; previews do
not reserve their displayed path. Use a new name for every submitted run.

## Submit all questions after the smoke check

Inspect the smoke run's error counts, generated cards, rejection reasons,
memory use and elapsed time before choosing full-run walltime. For example:

```bash
bash scripts/submit_sol.sh --run-root runs/sol-full-01 --full --submit
```

Each job requests one **A100 80 GB**, eight CPUs, and 64 GB of host RAM.
The initial four-hour walltime is a configurable request, not a measured runtime
estimate. Override your allocation settings as needed:

```bash
bash scripts/submit_sol.sh --run-root runs/sol-full-02 --full \
  --account YOUR_SOL_ACCOUNT --partition public --qos public \
  --time 04:00:00 --submit
```

Omit `--account` when your cluster default is appropriate. The corresponding
defaults are `SOL_ACCOUNT`, `SOL_PARTITION`, `SOL_QOS`, and `SOL_TIME` environment
variables. Use `--python`, `--model`, or `--bundle` for different local paths;
`--max-tokens` changes the final cap for **every** version together.

The scripts retain Slurm's GPU assignment. Each job has a separate server log
and a job-based server port, and the jobs run sequentially. No GPU work runs on
the login node: `sbatch` only queues the work.

## Saved files

At submission, the wrapper copies an explicit source allowlist and the original
ZIP into the new run root. It records the source commit, working-tree status,
and file hashes in `config.json`. Jobs read that copy, so checking out another
branch in your original directory does not change queued jobs. Do not edit a
queued run's `source/` or `inputs/` copies.

```text
config.json          Shared configuration and source/input hashes
submitted-jobs.txt   IDs returned by Slurm
source/              Copy of the code used by these jobs
inputs/              Copy of the original reproducibility ZIP
logs/                Slurm output/error files and per-stage vLLM logs
ab/                  Fresh A and B final answers
c-cards/             Fresh 72B cards and extraction records
c-answers/           C final answers
d-cards/             D's retained cards and recorded review decisions
d-answers/           D final answers
report/              Combined A–D comparison
```

The source snapshot includes `src/`, `scripts/`, `tests/`, the README,
`pyproject.toml`, and `.gitignore`. It excludes `.git`, `.env`, model weights,
other benchmark files and old outputs. Run directories under `runs/` are ignored
by Git; do not add them to the code commits.

## Interrupted jobs

Card extraction and review preserve completed records. Re-running their stage
script adds `--resume` when a card directory and its manifest already exist;
the stage validates that it is continuing the same configuration.

Final-answer directories are never overwritten or silently reused. If a job
stops after final answering has begun, inspect its manifest and preserved
records. Use the documented Python command with a **new answer output directory**
for another attempt. The submission wrapper does not automatically cancel jobs,
retry failures, or resubmit a partially submitted chain. `submitted-jobs.txt`
shows which jobs were accepted if a later submission fails.

Job monitoring uses normal Slurm commands, such as `squeue -u "$USER"` and
`sacct -j JOB_ID`. A pending dependent job is expected while its predecessor is
running.

Resource names and scheduling behavior follow the
[ASU resource guide](https://docs.rc.asu.edu/requesting-resources/) and
[Slurm job documentation](https://slurm.schedmd.com/sbatch.html).
