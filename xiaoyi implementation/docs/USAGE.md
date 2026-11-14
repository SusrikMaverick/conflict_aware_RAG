# New model inference and custom inputs

## GPU setup

Download the model weights separately. Download the pinned public model revision and run two vLLM services for the **same** model: an ordinary JSON endpoint and a compact auxiliary JSON endpoint. This requires Linux/CUDA, a compatible NVIDIA driver, about 15 GB of model weights and sufficient GPU memory. Direct dependency versions are pinned; the files are not a complete transitive environment lock.

In a dedicated inference environment, install the server dependencies and download the pinned model revision:

```bash
python3.12 -m venv .venv-server
source .venv-server/bin/activate
python -m pip install -r code/requirements-inference.txt
python run.py download-model --output model
```

Terminal 1 (set GPU index and memory budget for your machine):

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model ./model --served-model-name nlp576-qwen7b \
  --host 127.0.0.1 --port 8576 --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.9 \
  --max-num-seqs 24 --max-num-batched-tokens 8192 --seed 576
```

Terminal 2 (activate the same `.venv-server` environment first):

```bash
CUDA_VISIBLE_DEVICES=1 python code/serve_aux.py \
  --model ./model --served-model-name nlp576-qwen7b \
  --host 127.0.0.1 --port 8577 --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.9 \
  --max-num-seqs 24 --max-num-batched-tokens 8192 --seed 576 \
  --guided-decoding-backend xgrammar:disable-any-whitespace
```

The small `serve_aux.py` exposes an option already implemented by the pinned vLLM backend; it does not modify the installed environment. Wait until both servers report startup complete.

Terminal 3, in a separate CPU client environment if desired:

```bash
python -m venv .venv-client
source .venv-client/bin/activate
# Windows client: .venv-client\Scripts\Activate.ps1
python -m pip install -r code/requirements-client.txt
python run.py ours generate --dataset RGB --limit 1 --output outputs/fresh_smoke
# All 1,100 questions, six methods, after inspecting smoke outputs:
python run.py ours generate --dataset all --output outputs/fresh_all
```

The client requires jsonschema, transformers, Jinja2 and huggingface_hub. vLLM is needed on the servers only. Keep the served model name `nlp576-qwen7b`. The included tokenizer is pinned to the downloaded revision. Remote endpoints must end in `/v1/chat/completions`:

```bash
# Linux/macOS examples; replace addresses with your actual servers.
export SIX_MAIN_URL=http://your-host:8576/v1/chat/completions
export SIX_AUX_URL=http://your-host:8577/v1/chat/completions
# PowerShell: $env:SIX_MAIN_URL = "http://your-host:8576/v1/chat/completions"
# PowerShell: $env:SIX_AUX_URL = "http://your-host:8577/v1/chat/completions"
```

`--workers 8` is the fresh default; use `--workers 1` for serial execution or to reduce GPU pressure. Final output goes to `predictions.json`, stage files to `runs/`, scores to `scores/`, and the input/code identity to `generation_manifest.json`. Raw successful model requests and responses are retained in the run's request caches. Inspect nonzero `failed` counts even when the batch finishes. An all-failed batch, unreachable server, invalid input or inconsistent directory identity causes nonzero exit. Some archived ConflictBank predictions already have failure status; exact replay preserves these historical failures.

Fresh generation runs all required preliminary steps automatically. Labels are read for benchmark scoring only after all answers have been generated. Token caps and sampling settings are retained. Hardware/batching differences can change new outputs even at temperature 0; archived replay is the exact historical route.

## Custom questions

Create `your_inputs.json` as a UTF-8 JSON list, for example:

```json
[
  {
    "question_id": "example-1",
    "question": "Which city is mentioned?",
    "documents": [
      {"doc_id": "d1", "text": "The city mentioned is Tempe."}
    ]
  }
]
```

```bash
python run.py ours generate --input your_inputs.json --output outputs/custom_run
```

Question IDs must be unique safe filename strings starting with a letter or digit. Document IDs must be unique nonempty strings within a question. Document text/question text must be nonempty strings. Only empty `metadata` is accepted; other metadata must be removed before generation. Keep labels and correctness flags in evaluation files. Do not combine `--input` with a benchmark dataset filter. Custom questions are generated without benchmark scoring. Custom replay is only possible when every requested payload already exists in the archive; it is not a general offline model.

## Output folders

Use a new output folder for each input, code or endpoint change. `score`, `audit` and `generate` reject packaged source, data, result and provenance directories. Examples in README use separate output folders. Interrupted runs can resume with the same configuration; saved errors remain errors. To retry a failed experiment, use a new folder and keep the previous output as evidence.
