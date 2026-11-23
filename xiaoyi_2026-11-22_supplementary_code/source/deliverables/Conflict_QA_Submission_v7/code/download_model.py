"""Download the pinned public model into a user-selected local directory."""

import argparse
from huggingface_hub import snapshot_download

p = argparse.ArgumentParser()
p.add_argument("--output", default="model")
args = p.parse_args()
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    revision="a09a35458c702b33eeacc393d103063234e8bc28",
    local_dir=args.output,
)
