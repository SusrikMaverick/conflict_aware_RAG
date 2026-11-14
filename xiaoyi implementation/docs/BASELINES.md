# Seven baselines

All seven use the pinned Qwen2.5-7B-Instruct BF16 backbone. None requires a project-specific trained adapter. They share inputs and scoring with our six methods, but retain their historical call budgets and output contracts.

| Command name | Steps | Output caps |
|---|---|---|
| `plain` | Question + all documents -> answer | 1,024 |
| `madam` | One agent per document -> up to three debate rounds with prior-round agreement checks -> aggregate | 128 per call |
| `reflection` | Initial answer -> critique/revise -> critique/revise | 128 per call |
| `astute` | Memory passage -> two context consolidations -> structured final answer | 512 / 512 / 512 / 1,024 |
| `faithful` | Memory facts -> self-context -> mined facts -> align to document chunks -> final answer with full documents and selected chunks | 1,000 / 1,000 / 1,000 / 1,024 |
| `trust` | SimCSE embeddings -> K-means/ROUGE filtering -> memory -> consolidation -> final answer | 1,024 per generation call |
| `robust` | Answer each document separately -> aggregate keywords -> final answer, or abstain | 256 per document, 1,024 final |

MADAM-RAG uses the upstream convergence and execution order. Self-reflection uses an archived Plain RAG response at 128 tokens for its RAMDocs initial answer. Astute uses three memory/consolidation stages followed by a structured JSON answer. FaithfulRAG retains its original fact regex and alignment defaults. TrustRAG retains thresholds 0.25/0.88 and the single-document/zero-vector guards. RobustRAG uses KeywordAgg with alpha 0.3, beta 3 and corruption size 0; no robustness certificate is claimed.

## CPU preprocessing

Use a separate Python 3.12 environment. The following is the tested Linux CPU installation route:

```bash
python3.12 -m venv .venv-nlp
source .venv-nlp/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r code/requirements-baseline-nlp.txt
```

Recompute filtering from included SimCSE vectors, and recompute keyword aggregation from the isolated document answers:

```bash
python run.py preprocess trust --output outputs/trust_check
python run.py preprocess robust --output outputs/robust_check
```

To recompute neural embeddings and FaithfulRAG's alignment, download the two pinned encoders (about 530 MB combined):

```bash
python run.py download-encoders
python run.py preprocess trust --fresh-embeddings --output outputs/trust_embeddings_check
python run.py preprocess faithful --output outputs/faithful_check
```

`preprocess.py` recomputes the selected step and compares it against the archive. A changed filtering decision, keyword prompt or selected chunk order exits with an error and writes the differences. For FaithfulRAG, floating-point similarity differences are recorded separately from the exact chunk texts/order used in the final prompt. Its audit reuses the archived LLM-mined facts; new model inference below regenerates those facts too.

The encoder directory defaults to `code/models/` from the project root. Set `QA_NLP_MODELS` to an absolute directory containing the same two model folders if you have already downloaded them. NLP data is loaded from `code/assets/`; no runtime resource download is hidden in the code.

## New model inference

Download Qwen using the included script:

```bash
python run.py download-model --output model
```

In a new server terminal, create and activate a separate environment:

```bash
python3.12 -m venv .venv-server
source .venv-server/bin/activate
python -m pip install -r code/requirements-inference.txt
```

Start one service in that terminal (replace the GPU index and memory budget for the machine):

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model ./model --served-model-name nlp576-qwen7b \
  --host 127.0.0.1 --port 8576 --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.9 \
  --max-num-seqs 4 --max-num-batched-tokens 8192 --seed 576
```

After startup completes, activate `.venv-nlp` in a different terminal and download both NLP encoders if not already present. Use it as the client:

```bash
source .venv-nlp/bin/activate
python run.py download-encoders
python run.py baselines generate --dataset RGB --limit 1 --output outputs/fresh_demo
# After checking the smoke outputs:
python run.py baselines generate --dataset all --output outputs/fresh_baselines
```

Only one model service is needed for the seven baselines. Use `SIX_MAIN_URL` for a different endpoint; it must end in `/v1/chat/completions`. Methods run serially to keep dependency and request order explicit. GPU inference can produce different responses across hardware/batching even with temperature 0. Historical scores are reproduced with replay, not assumed for a new run.

For a custom question, use the input format in `USAGE.md`:

```bash
python run.py baselines generate --input your_inputs.json --output outputs/custom_baselines
```

Gold answers and correctness flags do not belong in generation inputs. Custom questions are not scored against benchmark labels. Frozen replay/preprocessing accepts only the exact archived question/document content.

## Sources and adaptations

| Method | Source used |
|---|---|
| Plain RAG | Frozen Plain RAG prompt in `code/configs/plain_prompt.txt` |
| MADAM-RAG | [HanNight/RAMDocs](https://github.com/HanNight/RAMDocs), `run_madam_rag.py`; copied upstream functions, local transport |
| Self-reflection | RAMDocs paper comparison structure; two critique/revision rounds with local prompts |
| Astute RAG | Astute/RAMDocs paper structures; local memory/consolidation prompts and repaired final JSON contract |
| FaithfulRAG | [XMUDeepLIT/Faithful-RAG](https://github.com/XMUDeepLIT/Faithful-RAG), original prompts and contextual alignment class |
| TrustRAG | [HuichiZhou/TrustRAG](https://github.com/HuichiZhou/TrustRAG), original filtering and `conflict_query` |
| RobustRAG | [inspire-group/RobustRAG](https://github.com/inspire-group/RobustRAG), original prompt wrapper and KeywordAgg |

The bundled archives contain all records needed for replay.

The executed upstream files are bundled in `code/upstream/`; adapters load only the used definitions, avoiding upstream training/attack entrypoints. Source hashes and Git blob IDs are in `docs/sources/`. MIT notices available from the MADAM-RAG and TrustRAG repositories are included. FaithfulRAG and RobustRAG snapshots retain source attribution; no top-level license file is bundled for those two repositories.

The main adaptations are the 7B BF16/vLLM runtime, multi-answer output schemas, documented token caps and CPU embedding implementations. Final truncated outputs remain accepted only when the historical parser accepts them. No failed question is dropped or retried to improve its score. The raw-to-frozen dataset preparation pipeline and full upstream benchmarks are outside this delivery.
