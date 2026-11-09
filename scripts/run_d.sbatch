#!/usr/bin/env bash
#SBATCH --job-name=rag-d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=a100:1
#SBATCH --time=04:00:00

set -euo pipefail
if [[ $# -ne 2 ]]; then
    printf 'Usage: %s RUN_ROOT PYTHON_EXECUTABLE\n' "$0" >&2
    exit 2
fi
# Slurm runs a spool copy of this script; helpers remain in the run snapshot.
source "$1/source/scripts/sol_common.sh"
sol_init d "$@"
sol_new_answers "$RUN_ROOT/d-answers"
RESUME_ARGS=()
if [[ -e "$RUN_ROOT/d-cards" ]]; then
    [[ -f "$RUN_ROOT/d-cards/manifest.json" ]] || { printf 'D card directory has no manifest.\n' >&2; exit 2; }
    RESUME_ARGS=(--resume)
fi
"$NLP_PYTHON" -m src.cards review \
    --bundle "$BUNDLE" --cards "$RUN_ROOT/c-cards" --model "$MODEL" \
    --workers "$WORKERS" --seed "$SEED" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
    --out "$RUN_ROOT/d-cards"
"$NLP_PYTHON" -m src.compare \
    --bundle "$BUNDLE" --cards "$RUN_ROOT/d-cards" --only-cards \
    --model "$MODEL" --workers "$WORKERS" --max-tokens "$MAX_TOKENS" --seed "$SEED" \
    ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} --out "$RUN_ROOT/d-answers"
"$NLP_PYTHON" -m src.experiment_report \
    --runs "$RUN_ROOT/ab" "$RUN_ROOT/c-answers" "$RUN_ROOT/d-answers" \
    --out "$RUN_ROOT/report"
