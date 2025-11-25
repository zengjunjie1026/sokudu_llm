"""
Pass@k evaluation script for Sudoku-solving LLMs.

Given a Sudoku dataset (e.g. `sokudu_dataset/sudoku_9x9.json`), this script
queries an LLM multiple times per puzzle and reports pass@k metrics along with
the majority-vote accuracy.

Usage examples
--------------

Remote GLM (BigModel Cloud):

    python sudoku_passk_eval.py \
        --dataset /home/andrew/sokudo_llm/sokudu_dataset/sudoku_9x9.json \
        --provider glm \
        --model glm-4 \
        --num-samples 10 \
        --temperature 0.7 \
        --limit 100

Local Ollama (e.g., gpt-oss served via `ollama serve`):

    python sudoku_passk_eval.py \
        --dataset /home/andrew/sokudo_llm/sokudu_dataset/sudoku_9x9.json \
        --provider ollama \
        --model gpt-oss \
        --num-samples 5 \
        --temperature 0.7 \
        --limit 20

Environment variables `GLM_API_KEY` (for provider `glm`) or `DASHSCOPE_API_KEY`
(for provider `qwen`) must be set before running the script. Provider `ollama`
talks to the local server at http://127.0.0.1:11434 and does not require an API
key. See `llm_client.py` for more details about provider configuration.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from llm_client import LLMClientError, PROVIDERS, chat_completion
from loguru import logger

BOARD_ROWS_PATTERN = re.compile(r"\d+")


def format_puzzle(puzzle: Sequence[Sequence[int]]) -> str:
    lines = []
    for row in puzzle:
        line = " ".join(str(cell) if cell else "." for cell in row)
        lines.append(line)
    return "\n".join(lines)


def build_prompt(puzzle: Sequence[Sequence[int]]) -> str:
    return (
        
       """You are a reasoning-only assistant operating in a plain text environment. Your task is to solve the given Sudoku puzzle using logical deduction only—no guessing, no external tools, and no reliance on precomputed solutions.

Instructions:
1. The puzzle is a {size}×{size} grid, where {size} is typically 9 (standard Sudoku), but may vary (e.g., 4 or 6).
2. Each row, each column, and each designated subgrid (box) must contain all digits from 1 to {size} exactly once.
3. Empty cells in the puzzle are represented by '0' or '.' — treat them as unknowns to be filled.
4. Use step-by-step deductive reasoning to determine the correct digit for each empty cell.
5. Do not output any explanations, comments, thought processes, or formatting beyond the final answer.
6. Return exactly {size} lines of output.
7. Each line must contain exactly {size} digits (from 1 to {size}), separated by single spaces.
8. Ensure the completed grid satisfies all Sudoku rules.

Puzzle Input Format:
- The puzzle is provided below under "Puzzle:".
- Each line represents a row of the grid.
- Digits are separated by spaces; empty cells are marked as '0' or '.'.

Your Output Format:
- Only the solved grid.
- {size} lines.
- Each line: {size} numbers separated by single spaces.
- No extra text before or after.

Now solve the following Sudoku puzzle:

Puzzle:
{puzzle}"""
    ).format(size=len(puzzle), puzzle=format_puzzle(puzzle))


def parse_board_from_response(
    response: str,
    size: int,
) -> Optional[List[List[int]]]:
    candidate_rows: List[List[int]] = []
    for line in response.splitlines():
        digits = BOARD_ROWS_PATTERN.findall(line)
        if len(digits) != size:
            continue
        try:
            row = [int(token) for token in digits]
        except ValueError:
            continue
        candidate_rows.append(row)

    if len(candidate_rows) < size:
        return None

    board = candidate_rows[:size]
    if any(len(row) != size for row in board):
        return None

    return board


def board_to_key(board: Sequence[Sequence[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(row) for row in board)


def is_valid_solution(
    candidate: Sequence[Sequence[int]],
    puzzle: Sequence[Sequence[int]],
    solution: Sequence[Sequence[int]],
) -> bool:
    size = len(puzzle)
    if len(candidate) != size or any(len(row) != size for row in candidate):
        return False

    for r in range(size):
        for c in range(size):
            puzzle_value = puzzle[r][c]
            candidate_value = candidate[r][c]
            if puzzle_value and candidate_value != puzzle_value:
                return False
            if candidate_value != solution[r][c]:
                return False
    return True


def normalize_temperature(provider: str, model: str, temperature: float) -> float:
    """
    Some providers/models (e.g., OpenAI GPT-5) do not allow overriding temperature.
    Force their supported default and warn the user when necessary.
    """

    if provider == "openai" and model and model.lower().startswith("gpt-5"):
        if abs(temperature - 1.0) > 1e-6:
            logger.warning(
                "Model %s does not allow temperature=%.2f; forcing temperature=1.0",
                model,
                temperature,
            )
        return 1.0
    return temperature


@dataclass
class SampleResult:
    raw_answer: str
    parsed_board: Optional[List[List[int]]]
    is_correct: bool
    latency: float
    prompt: str
    error: Optional[str] = None


def evaluate_puzzle(
    puzzle: Sequence[Sequence[int]],
    solution: Sequence[Sequence[int]],
    *,
    provider: str,
    model: str,
    temperature: float,
    num_samples: int,
    seed: Optional[int] = None,
) -> List[SampleResult]:
    prompt = build_prompt(puzzle)
    results: List[SampleResult] = []
    rng = random.Random(seed)
    logger.debug("Evaluating puzzle: {}", puzzle)
    logger.debug("Solution: {}", solution)
    logger.debug("Provider: {}", provider)
    logger.debug("Model: {}", model)
    logger.debug("Temperature: {}", temperature)
    logger.debug("Num samples: {}", num_samples)
    logger.debug("Seed: {}", seed)
    effective_temperature = normalize_temperature(provider, model, temperature)
    for sample_index in range(num_samples):
        time.sleep(rng.uniform(0.0, 0.2))
        start = time.time()
        try:
            logger.info("Prompt: {}", prompt)
            answer, _ = chat_completion(
                provider=provider,
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You must respond in pure text and obey all user instructions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=effective_temperature,
            )
            logger.info("Answer: {}", answer)
        except LLMClientError as exc:
            sample_result = SampleResult(
                raw_answer=f"[ERROR] {exc}",
                parsed_board=None,
                is_correct=False,
                latency=time.time() - start,
                prompt=prompt,
                error=str(exc),
            )
            results.append(sample_result)
            continue
        parsed_board = parse_board_from_response(answer, len(puzzle))
        is_correct = (
            parsed_board is not None
            and is_valid_solution(parsed_board, puzzle, solution)
        )
        logger.info("Parsed board: {}", parsed_board)
        logger.info("Is correct: {}", is_correct)
        logger.info("Latency: {}", time.time() - start)
        logger.info("Prompt: {}", prompt)
        logger.info("Answer: {}", answer)
        sample_result = SampleResult(
            raw_answer=answer,
            parsed_board=parsed_board,
            is_correct=is_correct,
            latency=time.time() - start,
            prompt=prompt,
        )
        results.append(sample_result)

    return results


def compute_pass_at_k(results: Sequence[List[SampleResult]], ks: Iterable[int]) -> Dict[int, float]:
    total = len(results)
    metrics: Dict[int, float] = {}

    for k in ks:
        success = 0
        for samples in results:
            window = samples[:k]
            if any(sample.is_correct for sample in window):
                success += 1
        metrics[k] = success / total if total else 0.0
    return metrics


def compute_majority_accuracy(results: Sequence[List[SampleResult]]) -> float:
    total = len(results)
    correct_majorities = 0

    for samples in results:
        counter: Counter[Tuple[Tuple[int, ...], ...]] = Counter()
        for sample in samples:
            if sample.parsed_board is None:
                continue
            counter[board_to_key(sample.parsed_board)] += 1
        if not counter:
            continue

        top_key, _ = counter.most_common(1)[0]
        if any(sample.is_correct and sample.parsed_board and board_to_key(sample.parsed_board) == top_key for sample in samples):
            correct_majorities += 1

    return correct_majorities / total if total else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pass@k evaluation for Sudoku datasets using an LLM."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(
            (Path(__file__).resolve().parent / "sokudu_dataset" / "sudoku_9x9.json")
        ),
        help="Path to the Sudoku dataset JSON file.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="glm",
        help="LLM provider name (default: glm).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model identifier (defaults to provider's default model).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the LLM.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of independent samples per puzzle (max k).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit the number of puzzles evaluated.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for per-call jitter.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle dataset before evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "eval_results"),
        help="Directory to write evaluation summaries and logs.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for console and file outputs (default: INFO).",
    )
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Shortcut to run against the local/remote Ollama server using gpt-oss:20b.",
    )

    args = parser.parse_args()
    if args.use_ollama:
        args.provider = "ollama"
        if args.model is None:
            args.model = "gpt-oss:20b"

    if args.model is None:
        args.model = PROVIDERS[args.provider].default_model

    return args


def load_dataset(path: str) -> List[Dict[str, List[List[int]]]]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    puzzles = payload.get("puzzles")
    if not isinstance(puzzles, list):
        raise ValueError("Invalid dataset: missing 'puzzles' list.")

    return puzzles


def _write_board(fp, board: Sequence[Sequence[int]], indent: int) -> None:
    indent_str = " " * indent
    fp.write("[\n")
    row_indent = indent_str + "  "
    for idx, row in enumerate(board):
        row_text = "[" + ", ".join(str(cell) for cell in row) + "]"
        trailing = "," if idx < len(board) - 1 else ""
        fp.write(f"{row_indent}{row_text}{trailing}\n")
    fp.write(f"{indent_str}]" if indent else "]")


def write_puzzle_file(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        fp.write("{\n")
        fp.write(f'  "index": {payload["index"]},\n')

        fp.write('  "puzzle": ')
        _write_board(fp, payload["puzzle"], indent=2)
        fp.write(",\n")

        fp.write('  "solution": ')
        _write_board(fp, payload["solution"], indent=2)
        fp.write(",\n")

        fp.write('  "samples": [\n')
        samples = payload.get("samples", [])
        for sample_idx, sample in enumerate(samples):
            fp.write("    {\n")
            fp.write(f'      "sample_index": {sample["sample_index"]},\n')
            fp.write(f'      "prompt": {json.dumps(sample["prompt"], ensure_ascii=False)},\n')
            fp.write(f'      "raw_answer": {json.dumps(sample["raw_answer"], ensure_ascii=False)},\n')

            fp.write('      "parsed_board": ')
            parsed = sample.get("parsed_board")
            if parsed is None:
                fp.write("null,\n")
            else:
                _write_board(fp, parsed, indent=6)
                fp.write(",\n")

            fp.write(f'      "is_correct": {"true" if sample["is_correct"] else "false"},\n')
            fp.write(f'      "latency": {sample["latency"]},\n')
            error = sample.get("error")
            if error is None:
                fp.write('      "error": null\n')
            else:
                fp.write(f'      "error": {json.dumps(error, ensure_ascii=False)}\n')

            fp.write("    }")
            if sample_idx < len(samples) - 1:
                fp.write(",")
            fp.write("\n")

        fp.write("  ]\n")
        fp.write("}\n")


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    if args.shuffle:
        random.Random(args.seed).shuffle(dataset)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level, enqueue=True)
    logger.add(
        run_dir / "run.log",
        level="DEBUG",
        encoding="utf-8",
        enqueue=True,
    )

    logger.info(
        "Evaluating {} puzzle(s) from {} using provider={}, model={}, temperature={}, samples={}",
        len(dataset),
        args.dataset,
        args.provider,
        args.model,
        args.temperature,
        args.num_samples,
    )

    all_results: List[List[SampleResult]] = []
    latencies: List[float] = []
    llm_logs: List[Dict[str, object]] = []

    for index, entry in enumerate(dataset, start=1):
        puzzle = entry["puzzle"]
        solution = entry["solution"]

        logger.info("\n--- Puzzle {}/{} ---", index, len(dataset))

        samples = evaluate_puzzle(
            puzzle,
            solution,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        all_results.append(samples)
        latencies.extend(sample.latency for sample in samples if sample.latency >= 0)

        puzzle_payload = {
            "index": index - 1,
            "puzzle": puzzle,
            "solution": solution,
            "samples": [],
        }

        for sample_index, sample in enumerate(samples):
            sample_dict = {
                "sample_index": sample_index,
                "prompt": sample.prompt,
                "raw_answer": sample.raw_answer,
                "parsed_board": sample.parsed_board,
                "is_correct": sample.is_correct,
                "latency": sample.latency,
                "error": sample.error,
            }
            puzzle_payload["samples"].append(sample_dict)
            llm_logs.append(
                {
                    "puzzle_index": index - 1,
                    "sample_index": sample_index,
                    "prompt": sample.prompt,
                    "response": sample.raw_answer,
                    "parsed_board": sample.parsed_board,
                    "is_correct": sample.is_correct,
                    "latency": sample.latency,
                    "error": sample.error,
                }
            )

        puzzle_path = run_dir / f"puzzle_{index:04d}.json"
        write_puzzle_file(puzzle_path, puzzle_payload)

        success_flag = "✓" if any(sample.is_correct for sample in samples) else "✗"
        correct_count = sum(sample.is_correct for sample in samples)
        logger.info(
            "[{}/{}] {} correct_samples={}/{}",
            index,
            len(dataset),
            success_flag,
            correct_count,
            args.num_samples,
        )

    ks = [1, 3, 5, 10]
    selected_ks = [k for k in ks if k <= args.num_samples]
    pass_metrics = compute_pass_at_k(all_results, selected_ks)
    majority_accuracy = compute_majority_accuracy(all_results)

    logger.info("\n=== Evaluation Summary ===")
    for k in selected_ks:
        logger.info("pass@{}: {:.3%}", k, pass_metrics[k])
    logger.info("majority@pass: {:.3%}", majority_accuracy)
    if latencies:
        logger.info("Average latency per call: {:.2f}s", statistics.mean(latencies))
        logger.info("Median latency per call: {:.2f}s", statistics.median(latencies))

    summary_payload = {
        "dataset": args.dataset,
        "provider": args.provider,
        "model": args.model,
        "temperature": args.temperature,
        "num_samples": args.num_samples,
        "puzzle_count": len(all_results),
        "pass_metrics": {f"pass@{k}": pass_metrics[k] for k in selected_ks},
        "majority_at_pass": majority_accuracy,
        "average_latency": statistics.mean(latencies) if latencies else None,
        "median_latency": statistics.median(latencies) if latencies else None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, ensure_ascii=False, indent=2)

    log_path = run_dir / "llm_calls.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for entry in llm_logs:
            fh.write(json.dumps(entry, ensure_ascii=False))
            fh.write("\n")

    logger.info("\nDetailed results written to {}", run_dir)


if __name__ == "__main__":
    main()


