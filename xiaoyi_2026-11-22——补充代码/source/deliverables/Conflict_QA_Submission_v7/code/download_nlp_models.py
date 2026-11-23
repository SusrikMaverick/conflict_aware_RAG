"""Download the two pinned CPU encoders used by TrustRAG and FaithfulRAG."""

import argparse
from pathlib import Path

MODELS = {
    "all-MiniLM-L6-v2": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    ),
    "sup-simcse-bert-base-uncased": (
        "princeton-nlp/sup-simcse-bert-base-uncased",
        "2d82fab19ac3a73a20dd20333d27eb8a52d6e97f",
    ),
}


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "models"
    )
    args = parser.parse_args()
    for name, (repo, revision) in MODELS.items():
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=args.output / name,
            allow_patterns=[
                "*.json",
                "*.txt",
                "model.safetensors",
                "pytorch_model.bin",
                "README.md",
            ],
        )
        print("Downloaded", name)


if __name__ == "__main__":
    main()
