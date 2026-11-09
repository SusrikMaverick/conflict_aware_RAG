#!/usr/bin/env bash
#SBATCH --job-name=rag-c
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
source "$1/source/scripts/sol_common.sh"
sol_init c "$@"
sol_new_answers "$RUN_ROOT/c-answers"
RESUME_ARGS=()
if [[ -e "$RUN_ROOT/c-cards" ]]; then
    [[ -f "$RUN_ROOT/c-cards/manifest.json" ]] || { printf 'C card directory has no manifest.\n' >&2; exit 2; }
    RESUME_ARGS=(--resume)
fi
"$NLP_PYTHON" -m src.cards extract \
    --bundle "$BUNDLE" --model "$MODEL" --workers "$WORKERS" --seed "$SEED" \
    ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} --out "$RUN_ROOT/c-cards"
"$NLP_PYTHON" -m src.compare \
    --bundle "$BUNDLE" --cards "$RUN_ROOT/c-cards" --only-cards \
    --model "$MODEL" --workers "$WORKERS" --max-tokens "$MAX_TOKENS" --seed "$SEED" \
    ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} --out "$RUN_ROOT/c-answers"
