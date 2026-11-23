"""Run scoring, archived replay, inference, preprocessing or key tests."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
HELP = """Usage: python run.py COMMAND [arguments]

  verify             Check all packaged files (Python only)
  score              Reproduce the 13-method table (Python only)
  test               Run the 39 key behavior tests
  ours               Six project methods: methods, score, audit, generate
  baselines          Seven baselines: methods, score, generate
  audit-baselines    Check baseline archive dependencies
  preprocess         Recompute baseline NLP preprocessing
  download-model     Download the pinned language-model weights
  download-encoders  Download the pinned NLP encoders

Examples:
  python run.py score --output outputs/scores
  python run.py baselines generate --replay --output outputs/baselines
  python run.py ours generate --replay --dataset all --output outputs/ours

For command options: python run.py baselines --help
"""


def protect_output(arguments):
    for index, argument in enumerate(arguments):
        if argument == "--output" and index + 1 < len(arguments):
            target = arguments[index + 1]
        elif argument.startswith("--output="):
            target = argument.split("=", 1)[1]
        else:
            continue
        path = Path(target).resolve()
        if path == ROOT or any(
            path.is_relative_to(ROOT / folder) for folder in ("code", "docs", "results")
        ):
            raise ValueError("Use a separate output folder, such as outputs/run1")


def main():
    arguments = sys.argv[1:]
    if not arguments or arguments[0] in ("--help", "-h"):
        print(HELP)
        return 0
    command, arguments = arguments[0], arguments[1:]
    if command in ("verify", "test"):
        if arguments:
            raise ValueError(command + " takes no arguments")
        if command == "verify":
            for name, expected in json.loads((ROOT / ".manifest.json").read_text()).items():
                path = ROOT / name
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError("Missing or changed packaged file: " + name)
            print("All packaged files verified")
            return 0
        for name in ("tests.py", "tests_baselines.py"):
            result = subprocess.run([sys.executable, str(CODE / name)])
            if result.returncode:
                return result.returncode
        return 0
    routes = {
        "score": ("baselines.py", ["score", "--include-ours"]),
        "ours": ("run.py", []),
        "baselines": ("baselines.py", []),
        "audit-baselines": ("audit_baselines.py", []),
        "preprocess": ("preprocess.py", []),
        "download-model": ("download_model.py", []),
        "download-encoders": ("download_nlp_models.py", []),
    }
    if command not in routes:
        raise ValueError("Unknown command: " + command + "\n" + HELP)
    protect_output(arguments)
    script, prefix = routes[command]
    has_output = any(a == "--output" or a.startswith("--output=") for a in arguments)
    if not has_output:
        if command == "score":
            arguments += ["--output", "outputs/scores"]
        elif command in ("ours", "baselines") and arguments and arguments[0] in ("score", "audit"):
            arguments += ["--output", "outputs/" + command + "_" + arguments[0]]
        elif command == "audit-baselines":
            arguments += ["--output", "outputs/baseline_audit"]
    return subprocess.run([sys.executable, str(CODE / script), *prefix, *arguments]).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
