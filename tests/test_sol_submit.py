"""Exercise Sol submission setup without contacting a scheduler."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SolSubmitTests(unittest.TestCase):
    def run_wrapper(self, directory, mode):
        project = directory / "project with spaces"
        (project / "scripts").mkdir(parents=True)
        (project / "src").mkdir()
        (project / "src/__init__.py").write_text("# snapshot marker\n")
        shutil.copyfile(ROOT / "scripts/run_ab.sbatch", project / "scripts/run_ab.sbatch")
        model = directory / "model"
        model.mkdir()
        bundle = directory / "input.zip"
        bundle.write_bytes(b"snapshot fixture")
        run = directory / "run with spaces"

        # Simulate the older login-node interpreter: simple path normalization
        # works, but the modern snapshot program cannot run through it.
        binary = directory / "bin"
        binary.mkdir()
        system_python = binary / "python3"
        system_python.write_text(
            '#!/bin/sh\n'
            'if [ "$1" != "-c" ]; then\n'
            '    printf "System Python cannot run the snapshot program\\n" >&2\n'
            '    exit 97\n'
            'fi\n'
            'exec "$SOL_TEST_PYTHON" "$@"\n'
        )
        fake_sbatch = binary / "sbatch"
        fake_sbatch.write_text(
            '#!/bin/sh\n'
            'printf "called\\n" >> "$SOL_TEST_SBATCH_LOG"\n'
            'printf "12345\\n"\n'
        )
        system_python.chmod(0o755)
        fake_sbatch.chmod(0o755)
        log = directory / "sbatch.log"
        environment = {
            **os.environ,
            "PATH": str(binary) + os.pathsep + os.environ["PATH"],
            "SOL_TEST_PYTHON": sys.executable,
            "SOL_TEST_SBATCH_LOG": str(log),
        }
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/submit_sol.sh"), "--stage", "ab",
             "--project", str(project), "--run-root", str(run), "--model", str(model),
             "--bundle", str(bundle), "--python", sys.executable, mode],
            env=environment, capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result, run, log

    def test_submit_uses_configured_python_for_snapshot(self):
        with tempfile.TemporaryDirectory(prefix="sol-submit-") as temporary:
            _, run, log = self.run_wrapper(Path(temporary), "--submit")
            self.assertEqual(log.read_text(), "called\n")
            self.assertEqual((run / "submitted-jobs.txt").read_text(), "ab 12345\n")
            self.assertEqual((run / "source/src/__init__.py").read_text(), "# snapshot marker\n")
            self.assertEqual((run / "inputs/RAMDocs_Reproducible_Package.zip").read_bytes(),
                             b"snapshot fixture")
            self.assertEqual(json.loads((run / "config.json").read_text())["python"], sys.executable)

    def test_dry_run_never_creates_snapshot_or_calls_sbatch(self):
        with tempfile.TemporaryDirectory(prefix="sol-preview-") as temporary:
            result, run, log = self.run_wrapper(Path(temporary), "--dry-run")
            self.assertIn("Preview only", result.stdout)
            self.assertFalse(run.exists())
            self.assertFalse(log.exists())


if __name__ == "__main__":
    unittest.main()
