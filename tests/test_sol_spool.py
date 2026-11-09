"""SBATCH scripts must find their helper when Slurm moves them to its spool."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SlurmSpoolTests(unittest.TestCase):
    def check_spooled_stage(self, stage, expected_calls):
        with tempfile.TemporaryDirectory(prefix="sol-spool-") as temporary:
            temporary = Path(temporary)
            run = temporary / "run with spaces"
            snapshot = run / "source"
            (snapshot / "scripts").mkdir(parents=True)
            (snapshot / "src").mkdir()
            shutil.copyfile(ROOT / "scripts/sol_common.sh", snapshot / "scripts/sol_common.sh")
            model = temporary / "model"
            model.mkdir()
            bundle = temporary / "bundle.zip"
            bundle.touch()
            (run / "config.json").write_text(json.dumps({
                "snapshot": str(snapshot), "bundle": str(bundle), "model": str(model),
                "workers": 4, "max_tokens": 128, "seed": 576, "limit": 5,
            }))

            # Slurm copies only the entry script. There is deliberately no helper
            # alongside it and the current directory is outside the snapshot.
            spool = temporary / "spool"
            spool.mkdir()
            entry = spool / "slurm_script"
            shutil.copyfile(ROOT / f"scripts/run_{stage}.sbatch", entry)
            outside = temporary / "unrelated-directory"
            outside.mkdir()
            log = temporary / "calls.jsonl"
            interpreter = temporary / "python-shim"
            interpreter.write_text(
                "#!/usr/bin/env python3\n"
                "import json,os,subprocess,sys\n"
                "from pathlib import Path\n"
                "if sys.argv[1:2] == ['-c']:\n"
                "    sys.exit(subprocess.call([os.environ['REAL_PYTHON'], *sys.argv[1:]]))\n"
                "with Path(os.environ['CALL_LOG']).open('a') as file:\n"
                "    file.write(json.dumps({'args':sys.argv[1:],"
                "'cuda':os.environ.get('CUDA_VISIBLE_DEVICES'),"
                "'port':os.environ.get('LLM_PORT'),"
                "'log':os.environ.get('LLM_LOG_FILE')}) + '\\n')\n"
            )
            interpreter.chmod(0o755)
            environment = {**os.environ, "REAL_PYTHON": sys.executable, "CALL_LOG": str(log),
                           "CUDA_VISIBLE_DEVICES": "3", "SLURM_JOB_ID": "112"}
            environment.pop("LLM_PORT", None)
            result = subprocess.run(
                ["bash", str(entry), str(run), str(interpreter)], cwd=outside,
                env=environment, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), expected_calls)
            self.assertTrue(all(call["cuda"] == "3" and call["port"] == "20112" for call in calls))
            self.assertTrue(all(call["log"] == str(run / f"logs/{stage}-vllm-112.log") for call in calls))
            self.assertTrue(all("" not in call["args"] for call in calls))
            self.assertEqual(calls[0]["args"][:2], ["-m", "src.compare" if stage == "ab" else "src.cards"])

    def test_ab_entry_runs_from_slurm_spool(self):
        self.check_spooled_stage("ab", 1)

    def test_c_entry_runs_from_slurm_spool(self):
        self.check_spooled_stage("c", 2)

    def test_d_entry_runs_from_slurm_spool(self):
        self.check_spooled_stage("d", 3)


if __name__ == "__main__":
    unittest.main()
