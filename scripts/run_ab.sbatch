#!/usr/bin/env bash
#SBATCH --job-name=rag-ab
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
sol_init ab "$@"
sol_new_answers "$RUN_ROOT/ab"
"$NLP_PYTHON" -m src.compare \
    --bundle "$BUNDLE" --model "$MODEL" --workers "$WORKERS" \
    --max-tokens "$MAX_TOKENS" --seed "$SEED" \
    ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} --out "$RUN_ROOT/ab"
