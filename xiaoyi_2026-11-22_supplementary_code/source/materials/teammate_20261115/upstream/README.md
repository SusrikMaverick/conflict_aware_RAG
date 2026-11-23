# conflict_aware_RAG

Runs Qwen2.5-72B-Instruct-AWQ with vLLM on one A100-80GB on ASU Sol.

## Install (once)

First open a terminal on a Sol GPU node: in Open OnDemand, start a VS Code job with
1× A100-80GB, 8 CPUs, 64 GB RAM. Then paste these commands in order.

```bash
# 1. Code
git clone https://github.com/SusrikMaverick/conflict_aware_RAG.git ~/projects/conflict_aware_RAG
cd ~/projects/conflict_aware_RAG
sed -i "s#/scratch/smukhe61/#/scratch/$USER/#g" src/interface.py pyproject.toml

# 2. Python env (8 GB, on scratch)
module load mamba/latest
mamba create -y -p /scratch/$USER/conflict_aware_RAG/envs/vllm -c conda-forge python=3.12 pip llguidance=1.7.5   # llguidance from conda: PyPI has no build for Sol's OS
source activate /scratch/$USER/conflict_aware_RAG/envs/vllm
pip install --no-cache-dir -e ".[serve]"

# 3. Model (39 GB, on scratch)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-72B-Instruct-AWQ',
                  revision='698703eae6604af048a3d2f509995dc302088217',
                  local_dir='/scratch/$USER/conflict_aware_RAG/models/Qwen2.5-72B-Instruct-AWQ')"

# 4. Benchmark (the MADAM-RAG script downloads itself on first use, at a pinned commit)
mkdir -p benchmarks/RAMDocs
curl -L -o benchmarks/RAMDocs/RAMDocs_test.jsonl \
  https://huggingface.co/datasets/HanNight/RAMDocs/resolve/9c041bfd158c603b615883d9a931b00cbc141494/RAMDocs_test.jsonl
```

## Run

Every new terminal (on a GPU node):

```bash
cd ~/projects/conflict_aware_RAG
module load mamba/latest
source activate /scratch/$USER/conflict_aware_RAG/envs/vllm

python -m src.interface   # chat in the terminal (server startup takes 5-15 min)
python -m src.pipeline --trace             # our pipeline on one question (the plan's Avatar example)
python -m src.benchmark_ramdocs_baseline   # RAMDocs plain-RAG baseline (ours: 26.4, paper: 20.6)
python -m src.benchmark_ramdocs_madam      # RAMDocs MADAM-RAG, the authors' own code (ours: 27.8, paper: 26.4)
python -m src.benchmark_ramdocs_pipeline   # RAMDocs, our pipeline (ours: 38.8)
```

Our pipeline (`src/pipeline.py`) takes `--mode`. It decides what happens to a contradiction
that nothing in the documents settles: `disagree` reports both sides, `forced` picks the
better-supported one. The RAMDocs benchmark defaults to `forced`, because RAMDocs wants one
list of answers. `--memory` lets the model's own knowledge break ties.

Quitting (Ctrl-C or Ctrl-D) stops the server. The server log is in `logs/vllm.log`.
Every command and its arguments, including how to watch a run: `commands.md`.

The benchmarks are resumable without changing the result. Answers go to
`runs/ramdocs-<method>-<backend>.jsonl` as they arrive, and rerunning the same command asks
only the missing questions. They run one request at a time so answers don't depend on
batching, and a resume first re-asks a few answered questions and stops if any differ. To
start over, move that file aside.

MADAM-RAG makes about 18 calls per question, so it takes about 8 hours. To run it (or the
pipeline benchmark) as a batch job that outlives the terminal, from the same activated terminal:

```bash
sbatch -p public -q public --gres=gpu:a100:1 -c 8 --mem=64G -t 12:00:00 -o runs/madam-%j.log \
  --wrap "python -m src.benchmark_ramdocs_madam"
```

## Paper setup (optional, much slower)

The MADAM-RAG paper ran Hugging Face transformers with NF4 4-bit weights, not vLLM with
AWQ. `--backend hf` on any command above does the same. It needs two more installs:

```bash
pip install --no-cache-dir -e ".[hf]"

# Full weights (136 GB, on scratch); they are quantized to NF4 on load
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-72B-Instruct',
                  revision='495f39366efef23836d0cfae4fbe635880d2be31',
                  local_dir='/scratch/$USER/conflict_aware_RAG/models/Qwen2.5-72B-Instruct')"
```
