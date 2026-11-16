# Commands

Every command the project uses, with its arguments. Install steps are in `README.md`.

1. [Set up a terminal](#1-set-up-a-terminal)
2. [Chat with the model](#2-chat-with-the-model)
3. [Try the pipeline on one question](#3-try-the-pipeline-on-one-question)
4. [Run a benchmark](#4-run-a-benchmark)
5. [Watch a run](#5-watch-a-run)
6. [Batch jobs](#6-batch-jobs)
7. [Results](#7-results)

## 1. Set up a terminal

Every new terminal, on a GPU node. Without it, `python` is not found.

```bash
cd ~/projects/conflict_aware_RAG
module load mamba/latest
source activate /scratch/$USER/conflict_aware_RAG/envs/vllm
```

## 2. Chat with the model

```bash
python -m src.interface
```

`/reset` clears the history. Ctrl-C or Ctrl-D quits and stops the model.

## 3. Try the pipeline on one question

```bash
python -m src.pipeline --trace
```

Runs our pipeline on the plan's example ("When was Avatar released?", six documents) and
prints the answer, its status and what it cost: about 11 model calls, a minute or two once
the model has started. Do this before a full benchmark, to see that the model follows the
prompts.

| Argument | Meaning |
|---|---|
| `file` | Your own question, as a JSON file: `{"question": "...", "documents": [{"text": "..."}]}`. Without it, the Avatar example |
| `--mode disagree` (default) | A contradiction that nothing settles is reported as "sources disagree", with both sides |
| `--mode forced` | The pipeline picks the better-supported side instead, and marks the pick as forced |
| `--memory` | The model's own knowledge may break ties, as a last resort. Off by default |
| `--trace` | Also prints what every step recorded |
| `--backend` | As in section 4 |

## 4. Run a benchmark

| Command | Method | Calls per question | Time (vllm) | Our score (vllm) | Paper |
|---|---|---|---|---|---|
| `python -m src.benchmark_ramdocs_baseline` | Plain RAG, one prompt with all documents | 1 | about 40 min | 26.4 ± 3.9 | 20.6 |
| `python -m src.benchmark_ramdocs_madam` | MADAM-RAG, the authors' own code | 18 | about 8 h | 27.8 ± 3.9 | 26.4 |
| `python -m src.benchmark_ramdocs_pipeline` | Our pipeline (`src/pipeline.py`), forced | 8.4 (9.4 with `--memory`) | about 3.5 h | 38.8 ± 4.3 (same with `--memory`) | |

Add 5 to 25 minutes for the model to start. Rerunning a command resumes an interrupted run.

### Arguments

`-h` prints them.

| Argument | Value | Commands | Meaning |
|---|---|---|---|
| `--backend` | `vllm` (default) | all | vLLM server, AWQ 4-bit weights. Fast. For development |
| `--backend` | `hf` | all | Hugging Face transformers, NF4 4-bit weights, as in the MADAM-RAG paper. Much slower. For final numbers |
| `--mode` | `forced` (default here) | pipeline benchmark | A contradiction that nothing settles: pick the better-supported answer. The score to compare with MADAM-RAG |
| `--mode` | `disagree` | pipeline benchmark | Report both sides and list neither. RAMDocs then counts the question as wrong; the report shows how often the pipeline was right when it did answer |
| `--memory` | (flag) | pipeline benchmark | The model's own knowledge may break ties, as a last resort. Off by default |

`--backend hf` needs the extra installs under "Paper setup" in `README.md`.

```bash
python -m src.benchmark_ramdocs_pipeline                   # forced, the score to compare
python -m src.benchmark_ramdocs_pipeline --memory          # forced, memory may break ties
python -m src.benchmark_ramdocs_pipeline --mode disagree   # may answer "sources disagree"
python -m src.benchmark_ramdocs_madam --backend hf
```

Each combination writes its own file (section 7), so they don't mix.

### Quick test before a full run

```bash
python -u -m src.benchmark_ramdocs_pipeline --check 10 --memory 2>&1 | tee runs/check.log
```

Starts the model, runs 10 questions spread over RAMDocs with the run's settings, asks one
again, and ends with `LOOKS FINE` or `NOT READY`. Prints no answers and no score, and writes
nothing to the run files. About 20 to 35 min, mostly model startup.

### Keep a run going after the terminal closes

A run started the plain way stops when its terminal closes. Started like this, it keeps
going until it finishes or the session's time runs out:

```bash
setsid nohup python -m src.benchmark_ramdocs_pipeline > runs/pipeline.log 2>&1 &
```

| What | Command |
|---|---|
| Watch it (Ctrl-C stops only the viewer) | `tail -f runs/pipeline.log` |
| Is it still running? | `pgrep -af benchmark_ramdocs` |
| Stop it | `pkill -f benchmark_ramdocs_pipeline` |

For another benchmark, change the module name and the log name. Start it once only: two
runs of the same command write to the same file.

Several runs, one after another, each with its own log. This one runs RAMDocs without and
with memory:

```bash
setsid nohup sh -c 'python -m src.benchmark_ramdocs_pipeline > runs/pipeline.log 2>&1; python -m src.benchmark_ramdocs_pipeline --memory > runs/pipeline-memory.log 2>&1' > /dev/null 2>&1 &
```

| What | Command |
|---|---|
| Watch the current step | `tail -f runs/pipeline.log`, then `runs/pipeline-memory.log` |
| Stop all of it | `pkill -f "python -m src"` |

Don't edit `src/pipeline.py` or `src/benchmark_ramdocs_pipeline.py` while it runs: the later
steps would run changed code.

### Fixed in the code

These are not arguments. Changing one makes an existing run file refuse to resume.

| Setting | Value | Where |
|---|---|---|
| Decoding, baseline and MADAM-RAG | Greedy, 128 new tokens, repetition penalty 1.05 | `DECODING` in `src/ramdocs.py` |
| Decoding, our pipeline | Greedy, repetition penalty 1.05; 60 to 1024 new tokens, by step | `DECODING` and the `s.ask(...)` calls in `src/pipeline.py` |
| Pipeline thresholds (first values, not tuned) | Quote match 0.8, copy 0.8, majority on, clearly ahead 2 times, memory confidence 90 | Top of `src/pipeline.py` |
| vLLM server flags | One request at a time, no prefix caching | `SERVE_ARGS` in `src/ramdocs.py` |
| MADAM-RAG rounds | 3 | `ROUNDS` in `src/benchmark_ramdocs_madam.py` |
| Model folders | One per backend | `MODELS` in `src/interface.py` |

Any edit to `src/pipeline.py` or `src/benchmark_ramdocs_pipeline.py` counts as a change for
an unfinished pipeline run.

## 5. Watch a run

The run's own terminal shows one progress line: questions answered, minutes elapsed,
minutes left. For more, open a second terminal (section 1) and use any of these. Ctrl-C
stops only the viewer.

| What | Command |
|---|---|
| Server log: loading during startup, then tokens per second | `tail -f logs/vllm.log` |
| GPU memory and load, every 5 s | `watch -n 5 nvidia-smi` |
| How many questions are saved | `wc -l runs/ramdocs-pipeline-forced-vllm.jsonl` (minus 1 for the settings line) |

Each finished question as it is saved (number of calls, question, final answer). Start it
after the run prints `vLLM ready`, with the run's own file name:

```bash
tail -n +2 -f runs/ramdocs-pipeline-forced-vllm.jsonl | python -c "
import sys, json
for line in sys.stdin:
    a = json.loads(line)
    print(f\"q{a['id']} | {a['calls']} calls | {a['question']}\n   {a['output'][:300]}\n\", flush=True)"
```

## 6. Batch jobs

A batch job runs on its own GPU node without a terminal. Submit from a set-up terminal
(section 1).

```bash
sbatch -p public -q public --gres=gpu:a100:1 -c 8 --mem=64G -t 10:00:00 -o runs/pipeline-%j.log \
  --wrap "python -m src.benchmark_ramdocs_pipeline"
```

| Argument | Meaning |
|---|---|
| `-p public -q public` | Partition and queue |
| `--gres=gpu:a100:1` | One A100 |
| `-c 8 --mem=64G` | 8 CPUs, 64 GB of RAM |
| `-t 10:00:00` | Time limit, 10 hours (use 12 for MADAM-RAG, which takes about 8) |
| `-o runs/pipeline-%j.log` | Log file; `%j` becomes the job number |
| `--wrap "..."` | The command to run |
| `--begin=02:30` | Optional: don't start before 02:30 |
| `--test-only` | Optional: check the command and print the expected start, without submitting |

Never let two runs of the same command overlap: they write to the same file. To continue a
terminal run after its session ends, submit with `--begin` set after that time.

| What | Command |
|---|---|
| My jobs: waiting (`PD`) or running (`R`), time used and left | `squeue -u $USER -o "%.10i %.20j %.3t %.10M %.10L %R"` |
| Progress, and the score at the end | `tail -c 600 runs/pipeline-<job>.log` |
| Cancel | `scancel <job>` |

## 7. Results

Results are in `runs/ramdocs-<method>-<backend>.jsonl`: one settings line, then one line
per question.

| Method in the file name | Command |
|---|---|
| `baseline` | `src.benchmark_ramdocs_baseline` |
| `madam` | `src.benchmark_ramdocs_madam` |
| `pipeline-forced` | `src.benchmark_ramdocs_pipeline` |
| `pipeline-forced-memory` | `src.benchmark_ramdocs_pipeline --memory` |
| `pipeline-disagree` | `src.benchmark_ramdocs_pipeline --mode disagree` |
| `pipeline-disagree-memory` | `src.benchmark_ramdocs_pipeline --mode disagree --memory` |

A pipeline line also holds the question's status, its entries (each answer with its
sources and basis) and the trace of every step.

| What | How |
|---|---|
| Print the score of a finished run again (no GPU) | Run its benchmark command again |
| Finish an interrupted run | Run its benchmark command again |
| Start a run over | Move its file aside first: `mv runs/ramdocs-pipeline-forced-vllm.jsonl runs/old-pipeline-forced.jsonl` |
