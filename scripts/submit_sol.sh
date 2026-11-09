#!/usr/bin/env bash
# Preview by default. Only --submit contacts Slurm.
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT=""
BUNDLE=""
MODEL="/scratch/$USER/conflict_aware_RAG/models/Qwen2.5-72B-Instruct-AWQ"
NLP_PYTHON="${NLP_PYTHON:-/scratch/$USER/conflict_aware_RAG/envs/vllm/bin/python}"
PARTITION="${SOL_PARTITION:-public}"
QOS="${SOL_QOS:-public}"
ACCOUNT="${SOL_ACCOUNT:-}"
WALLTIME="${SOL_TIME:-04:00:00}"
WORKERS=4
MAX_TOKENS=128
SEED=576
LIMIT=5
STAGE=all
SUBMIT=0

usage() {
    cat <<'HELP'
Usage: bash scripts/submit_sol.sh --run-root PATH [options]

Default: preview the A/B -> C -> D jobs on 5 questions; submit nothing.
  --dry-run          Print commands only (default; no files are created)
  --submit           Snapshot code/inputs and submit jobs to Slurm
  --stage ab|all     Queue A/B only, or the complete dependent chain (default all)
  --smoke N          Use the first N questions (default 5)
  --full             Use all questions; first complete a separate smoke run
  --project PATH     Project checkout (default this script's parent)
  --bundle PATH      Original reproducibility ZIP (default project's benchmark ZIP)
  --model PATH       Local 72B AWQ model directory on Sol
  --python PATH      Sol inference environment's Python executable
  --partition NAME  Slurm partition (default SOL_PARTITION or public)
  --qos NAME        Slurm QoS (default SOL_QOS or public)
  --account NAME    Optional Slurm account (default SOL_ACCOUNT or cluster default)
  --time D-HH:MM:SS  Walltime per job (default SOL_TIME or 04:00:00; not an estimate)
  --workers N       Requests in flight (default 4)
  --max-tokens N    Shared final-answer output cap (default 128)
  --help            Show this help
HELP
}

need_value() { [[ $# -ge 2 && -n "$2" ]] || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) SUBMIT=0; shift ;;
        --submit) SUBMIT=1; shift ;;
        --full) LIMIT=0; shift ;;
        --help|-h) usage; exit 0 ;;
        --run-root|--project|--bundle|--model|--python|--partition|--qos|--account|--time|--workers|--max-tokens|--smoke|--stage)
            need_value "$@"
            case "$1" in
                --run-root) RUN_ROOT="$2" ;; --project) PROJECT="$2" ;; --bundle) BUNDLE="$2" ;;
                --model) MODEL="$2" ;; --python) NLP_PYTHON="$2" ;; --partition) PARTITION="$2" ;;
                --qos) QOS="$2" ;; --account) ACCOUNT="$2" ;; --time) WALLTIME="$2" ;;
                --workers) WORKERS="$2" ;; --max-tokens) MAX_TOKENS="$2" ;;
                --smoke)
                    [[ "$2" =~ ^[0-9]+$ && "$2" -gt 0 ]] || { printf 'Smoke size must be positive.\n' >&2; exit 2; }
                    LIMIT="$2" ;;
                --stage) STAGE="$2" ;;
            esac
            shift 2 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ -n "$RUN_ROOT" ]] || { printf -- '--run-root is required.\n' >&2; exit 2; }
[[ "$STAGE" == ab || "$STAGE" == all ]] || { printf -- '--stage must be ab or all.\n' >&2; exit 2; }
for value in "$WORKERS" "$MAX_TOKENS"; do
    [[ "$value" =~ ^[0-9]+$ && "$value" -gt 0 ]] || { printf 'workers and max-tokens must be positive integers.\n' >&2; exit 2; }
done
[[ "$LIMIT" =~ ^[0-9]+$ ]] || { printf 'smoke size must be a positive integer; use --full for all questions.\n' >&2; exit 2; }
PROJECT="$(cd -- "$PROJECT" && pwd)"
[[ -n "$BUNDLE" ]] || BUNDLE="$PROJECT/benchmarks/ramdocs_cards/RAMDocs_Reproducible_Package.zip"
# Canonicalize data paths without requiring model/environment files on this host.
RUN_ROOT="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$RUN_ROOT")"
BUNDLE="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$BUNDLE")"
MODEL="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$MODEL")"
NLP_PYTHON="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$NLP_PYTHON")"
SNAPSHOT="$RUN_ROOT/source"
FROZEN_BUNDLE="$RUN_ROOT/inputs/RAMDocs_Reproducible_Package.zip"
STAGES=(ab)
if [[ "$STAGE" == all ]]; then STAGES+=(c d); fi
for stage in "${STAGES[@]}"; do
    [[ -f "$PROJECT/scripts/run_${stage}.sbatch" ]] || { printf 'Missing stage script: %s\n' "$stage" >&2; exit 2; }
done

print_command() { printf '%q ' "$@"; printf '\n'; }
make_command() {
    local stage="$1" dependency="$2"
    COMMAND=(sbatch --parsable --job-name="rag-$stage" --partition="$PARTITION" --qos="$QOS"
        --time="$WALLTIME" --chdir="$SNAPSHOT"
        --output="$RUN_ROOT/logs/${stage}-%j.out" --error="$RUN_ROOT/logs/${stage}-%j.err")
    if [[ -n "$ACCOUNT" ]]; then COMMAND+=(--account="$ACCOUNT"); fi
    if [[ -n "$dependency" ]]; then COMMAND+=(--dependency="afterok:$dependency"); fi
    COMMAND+=("$SNAPSHOT/scripts/run_${stage}.sbatch" "$RUN_ROOT" "$NLP_PYTHON")
}

if [[ "$SUBMIT" -eq 0 ]]; then
    printf 'Preview only: no files created and no jobs submitted.\n'
    question_count="$LIMIT"
    if [[ "$LIMIT" -eq 0 ]]; then question_count=all; fi
    printf 'Questions: %s; source snapshot: %s\n' "$question_count" "$SNAPSHOT"
    dependency=""
    for stage in "${STAGES[@]}"; do
        make_command "$stage" "$dependency"
        print_command "${COMMAND[@]}"
        dependency="${stage}_JOB_ID"
    done
    exit 0
fi

command -v sbatch >/dev/null || { printf 'sbatch is unavailable; submit from Sol.\n' >&2; exit 2; }
[[ -x "$NLP_PYTHON" && -d "$MODEL" && -f "$BUNDLE" ]] || {
    printf 'Check --python, --model and --bundle paths on Sol.\n' >&2; exit 2;
}
[[ ! -e "$RUN_ROOT" ]] || { printf 'Run root already exists; choose a new one: %s\n' "$RUN_ROOT" >&2; exit 2; }
python3 - "$PROJECT" "$RUN_ROOT" "$BUNDLE" "$MODEL" "$NLP_PYTHON" "$WORKERS" "$MAX_TOKENS" "$LIMIT" "$SEED" <<'PY'
from pathlib import Path
import hashlib, json, shutil, subprocess, sys
project, root, bundle, model, interpreter = map(Path, sys.argv[1:6])
workers, max_tokens, limit, seed = map(int, sys.argv[6:])
for name in ("src", "scripts", "tests"):
    source = project / name
    if root.is_relative_to(source):
        raise ValueError("Run root must be outside src, scripts and tests")
    if source.exists() and any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Source snapshots must not contain symlinks")
root.mkdir(parents=True, exist_ok=False)
snapshot = root / "source"
snapshot.mkdir()
for name in ("src", "scripts", "tests"):
    source = project / name
    if source.exists():
        shutil.copytree(source, snapshot / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
for name in ("README.md", "pyproject.toml", ".gitignore"):
    source = project / name
    if source.is_file():
        shutil.copyfile(source, snapshot / name)
(root / "inputs").mkdir()
frozen = root / "inputs/RAMDocs_Reproducible_Package.zip"
shutil.copyfile(bundle, frozen)
(root / "logs").mkdir()
def git(*args):
    result = subprocess.run(["git", "-C", str(project), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None
config = {"snapshot": str(snapshot), "bundle": str(frozen), "model": str(model),
          "python": str(interpreter), "workers": workers, "max_tokens": max_tokens,
          "limit": limit, "seed": seed, "source_commit": git("rev-parse", "HEAD"),
          "source_branch": git("branch", "--show-current"), "source_status": git("status", "--porcelain"),
          "source_hashes": {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(snapshot.rglob("*")) if p.is_file()},
          "bundle_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest()}
(root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
PY
dependency=""
for stage in "${STAGES[@]}"; do
    make_command "$stage" "$dependency"
    job_id="$("${COMMAND[@]}")"
    job_id="${job_id%%;*}"
    [[ "$job_id" =~ ^[0-9]+$ ]] || { printf 'Unexpected Slurm job ID; inspect %s before retrying.\n' "$RUN_ROOT" >&2; exit 1; }
    printf '%s %s\n' "$stage" "$job_id" >> "$RUN_ROOT/submitted-jobs.txt"
    printf 'Submitted %s: %s\n' "$stage" "$job_id"
    dependency="$job_id"
done
