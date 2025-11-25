# Sudoku Reasoning Lab

An end-to-end playground for Sudoku research that combines:

- classical exact solving (`sudoku_solver.py`)
- dataset generation/export (`sokudu_dataset.py`)
- LLM-based interactive solving (`gpt_sudoku_session.py`, `gpt_sudoku_session_16.py`)
- automated pass@k benchmarking (`sudoku_passk_eval.py`)

All components speak plain text and store exhaustive logs so you can reproduce experiments, diff prompts, and run local or cloud models interchangeably.

---

## 1. Classical Solver CLI

```bash
python sudoku_solver.py
```

Features:

- Backtracking solver for 9×9 grids
- Interactive or sample puzzle input (0 = empty)
- Pretty board rendering + validation

Use it to sanity-check puzzles or export fully solved boards.

---

## 2. Dataset Generation (`sokudu_dataset.py`)

Create large corpora with unique solutions and human-readable JSON formatting.

```bash
# 1000 puzzles for each size, stored under sokudu_dataset/
python sokudu_dataset.py

# Only 4×4 (1000 easy puzzles)
python sokudu_dataset.py --num-4x4 1000 --num-9x9 0 --num-16x16 0

# Verify uniqueness or check duplicates
python sokudu_dataset.py --verify-unique sokudu_dataset/sudoku_4x4.json
python sokudu_dataset.py --check-file sokudu_dataset/sudoku_9x9.json
```

Outputs live under `sokudu_dataset/` as `sudoku_<size>x<size>.json` with both puzzles and solutions serialized as 2D arrays.

---

## 3. LLM Interactive Sessions (`gpt_sudoku_session.py`)

Run a single puzzle or sweep through a dataset while automatically:

- building rich prompts (size-aware, tool-use bans, strict formatting)
- tracking conversation history + reasoning
- validating each answer and generating cumulative feedback
- logging every attempt under an organized directory tree

### Example: dataset sweep with remote Qwen

```bash
export DASHSCOPE_API_KEY=sk-xxxx

python gpt_sudoku_session.py \
  --provider qwen \
  --model qwen3-max \
  --dataset sokudu_dataset/sudoku_9x9.json \
  --dataset-limit 20 \
  --max-rounds 10 \
  --retry-attempts 3 \
  --history-dir gpt_sudoku_histories
```

Logs are placed under `gpt_sudoku_histories/dataset_run_<timestamp>/puzzle_XXXX/attempt_YY/` with:

- `conversation.json` – full chat with reasoning fields
- `summary.json` – success flag, rounds, final issues
- `rounds.txt` – number of dialogue rounds used

### Local/Remote Ollama (gpt-oss:20b)

```bash
# optional: point to remote host
export OLLAMA_BASE_URL="http://192.168.3.103:11434/v1"

python gpt_sudoku_session.py \
  --use-ollama \
  --dataset sokudu_dataset/sudoku_9x9.json \
  --dataset-limit 20 \
  --max-rounds 10 \
  --history-dir gpt_sudoku_histories
```

`--use-ollama` switches provider to `ollama` and defaults to `gpt-oss:20b`.

---

## 4. Pass@k Benchmarking (`sudoku_passk_eval.py`)

Benchmark any provider/model across a dataset with multiple independent samples per puzzle.

```bash
# Qwen cloud
python sudoku_passk_eval.py \
  --dataset sokudu_dataset/sudoku_9x9.json \
  --provider qwen \
  --model qwen3-max \
  --num-samples 10 \
  --limit 20 \
  --log-level INFO

# Local/Remote Ollama shortcut
python sudoku_passk_eval.py \
  --dataset sokudu_dataset/sudoku_9x9.json \
  --use-ollama \
  --num-samples 5 \
  --limit 20
```

Outputs saved under `eval_results/run_<timestamp>/`:

- `summary.json` – pass@1/3/5/10, majority@pass, latency stats
- `llm_calls.jsonl` – each sample’s prompt/response/parsed board
- `puzzle_XXXX.json` – per-puzzle records with readable 2D arrays

---

## 5. LLM Providers & Environment Variables

Supported providers (`llm_client.py`):

| Provider | Base URL | Env Var |
|----------|----------|---------|
| openai   | https://api.openai.com/v1 | `OPENAI_API_KEY` |
| deepseek | https://api.deepseek.com/v1 | `DEEPSEEK_API_KEY` |
| qwen     | https://dashscope.aliyuncs.com/compatible-mode/v1 | `DASHSCOPE_API_KEY` |
| glm      | https://open.bigmodel.cn/api/paas/v4 | `GLM_API_KEY` |
| ollama   | http://127.0.0.1:11434/v1 (override via `OLLAMA_BASE_URL`) | *none* |

Ollama runs without API keys; just ensure `ollama serve` is listening on the desired host/port.

---

## 6. Requirements

```bash
pip install -r requirements.txt
```

(Currently only `requests` + `loguru`, but keep requirements pinned here.)

---

## 7. Repository Layout

```
sudoku_solver.py           # CLI backtracking solver
sokudu_dataset.py          # dataset generation & verification
gpt_sudoku_session.py      # 9×9 LLM loop with feedback
gpt_sudoku_session_16.py   # 16×16 variant
sudoku_passk_eval.py       # pass@k + majority benchmarking
gpt_sudoku_histories/      # interactive session logs
eval_results/              # benchmarking outputs
sokudu_dataset/            # generated puzzle corpora
```

---

Happy experimenting! Contributions and prompt ideas are welcome. Feel free to open issues describing new providers, dataset formats, or evaluation metrics you’d like to see. 👋

