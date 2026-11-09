#!/usr/bin/env bash
# Shared setup for the batch scripts. This file does not submit jobs.

sol_init() {
    local stage="$1"
    shift
    if [[ $# -ne 2 ]]; then
        printf 'Usage: %s RUN_ROOT PYTHON_EXECUTABLE\n' "$0" >&2
        return 2
    fi
    RUN_ROOT="$1"
    NLP_PYTHON="$2"
    if [[ ! -x "$NLP_PYTHON" || ! -f "$RUN_ROOT/config.json" ]]; then
        printf 'Missing executable Python or run configuration: %s\n' "$RUN_ROOT" >&2
        return 2
    fi
    sol_config() {
        "$NLP_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$RUN_ROOT/config.json" "$1"
    }
    PROJECT_SNAPSHOT="$(sol_config snapshot)"
    BUNDLE="$(sol_config bundle)"
    MODEL="$(sol_config model)"
    WORKERS="$(sol_config workers)"
    MAX_TOKENS="$(sol_config max_tokens)"
    SEED="$(sol_config seed)"
    local limit
    limit="$(sol_config limit)"
    LIMIT_ARGS=()
    if [[ "$limit" -gt 0 ]]; then LIMIT_ARGS=(--limit "$limit"); fi
    if [[ ! -f "$BUNDLE" || ! -d "$MODEL" || ! -d "$PROJECT_SNAPSHOT/src" ]]; then
        printf 'Missing bundle, local model directory, or source snapshot.\n' >&2
        return 2
    fi
    cd -- "$PROJECT_SNAPSHOT"
    mkdir -p -- "$RUN_ROOT/logs"
    local job_id="${SLURM_JOB_ID:-manual}"
    export LLM_LOG_FILE="$RUN_ROOT/logs/${stage}-vllm-${job_id}.log"
    export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/$USER/conflict_aware_RAG/cache/vllm}"
    if [[ "$job_id" =~ ^[0-9]+$ ]]; then
        export LLM_PORT="${LLM_PORT:-$((20000 + job_id % 20000))}"
    else
        export LLM_PORT="${LLM_PORT:-8000}"
    fi
    # Keep Slurm's CUDA_VISIBLE_DEVICES; do not choose physical GPU numbers here.
}

sol_new_answers() {
    if [[ -e "$1" ]]; then
        printf 'Answer directory already exists: %s\nUse a new run root; existing answers are not overwritten or silently reused.\n' "$1" >&2
        return 2
    fi
}
