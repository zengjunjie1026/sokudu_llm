"""
Pass@k evaluation script for Sudoku-solving LLMs.

Given a Sudoku dataset (e.g. `sokudu_dataset/sudoku_9x9.json`), this script
queries an LLM multiple times per puzzle and reports pass@k metrics along with
the majority-vote accuracy.

Usage example
-------------

    python sudoku_passk_eval.py \
        --dataset /home/andrew/sokudo_llm/sokudu_dataset/sudoku_9x9.json \
        --provider qwen \
        --model qwen3-max \
        --num-samples 10 \
        --temperature 0.8 \
        --limit 100

Environment variable `DASHSCOPE_API_KEY` must be set when using the `qwen`
provider. See `llm_client.py` for more details about provider configuration.
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
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from llm_client import LLMClientError, chat_completion
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
        "You are a reasoning-only assistant working in plain text. "
        "Solve the following Sudoku puzzle without using any external tools. "
        "Return the completed grid as {size} lines, each with {size} numbers "
        "separated by spaces.\n\n"
        "Puzzle:\n{puzzle}\n"
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
    progress_callback: Optional[Callable[[int, SampleResult], None]] = None,
) -> List[SampleResult]:
    prompt = build_prompt(puzzle)
    results: List[SampleResult] = []

    rng = random.Random(seed)
    for sample_index in range(num_samples):
        time.sleep(rng.uniform(0.0, 0.2))
        start = time.time()
        try:
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
                temperature=temperature,
            )
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
            if progress_callback:
                progress_callback(sample_index, sample_result)
            continue
        parsed_board = parse_board_from_response(answer, len(puzzle))
        is_correct = (
            parsed_board is not None
            and is_valid_solution(parsed_board, puzzle, solution)
        )
        sample_result = SampleResult(
            raw_answer=answer,
            parsed_board=parsed_board,
            is_correct=is_correct,
            latency=time.time() - start,
            prompt=prompt,
        )
        results.append(sample_result)
        if progress_callback:
            progress_callback(sample_index, sample_result)

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


def _truncate_preview(text: str, limit: int = 120) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    first_line = stripped.splitlines()[0]
    if len(first_line) <= limit:
        return first_line
    return first_line[: limit - 3] + "..."


def _print_sample_progress(
    *,
    puzzle_idx: int,
    total_puzzles: int,
    sample_idx: int,
    total_samples: int,
    result: SampleResult,
) -> None:
    status = "✓" if result.is_correct else ("!" if result.error else "…")
    latency_text = f"{result.latency:.2f}s" if result.latency >= 0 else "n/a"
    logger.info(
        "[Puzzle {}/{}] Sample {}/{} {} latency={}",
        puzzle_idx,
        total_puzzles,
        sample_idx + 1,
        total_samples,
        status,
        latency_text,
    )

    if result.error:
        logger.error("  error: {}", result.error)
        return

    preview = _truncate_preview(result.raw_answer)
    if preview:
        logger.info("  output: {}", preview)


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
        default="qwen",
        help="LLM provider name (default: qwen).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-max",
        help="LLM model identifier (default: qwen3-max).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
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
    return parser.parse_args()


def load_dataset(path: str) -> List[Dict[str, List[List[int]]]]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    puzzles = payload.get("puzzles")
    if not isinstance(puzzles, list):
        raise ValueError("Invalid dataset: missing 'puzzles' list.")

    return puzzles


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
    logger.add(sys.stdout, level="INFO", enqueue=True)
    logger.add(run_dir / "run.log", level="DEBUG", encoding="utf-8", enqueue=True)

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
            progress_callback=lambda sample_idx, sample_result, _index=index, _total=len(dataset): _print_sample_progress(
                puzzle_idx=_index,
                total_puzzles=_total,
                sample_idx=sample_idx,
                total_samples=args.num_samples,
                result=sample_result,
            ),
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
        with puzzle_path.open("w", encoding="utf-8") as fh:
            json.dump(puzzle_payload, fh, ensure_ascii=False, indent=2)

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


