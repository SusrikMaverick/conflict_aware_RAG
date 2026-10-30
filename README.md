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

# 4. Benchmark
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

python -m src.interface   # chat in the terminal (server startup takes ~5 min)
python -m src.eval        # RAMDocs baseline eval
```

Quitting (Ctrl-C or Ctrl-D) stops the server. The server log is in `logs/vllm.log`.
