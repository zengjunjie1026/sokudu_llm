"""
Sudoku dataset generation utilities.

Generates batches of 9x9 and 16x16 Sudoku puzzles with unique solutions and
saves them, together with their canonical solutions, under the `sokudu_dataset`
directory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

Board = List[List[int]]


class SudokuSolver:
    """Backtracking Sudoku solver supporting square board sizes (e.g. 9, 16)."""

    def __init__(
        self,
        board: Sequence[Sequence[int]],
        *,
        randomizer: Optional[random.Random] = None,
    ) -> None:
        if not board:
            raise ValueError("Sudoku board must not be empty.")

        self.board: Board = [list(row) for row in board]
        self.size = len(self.board)

        if any(len(row) != self.size for row in self.board):
            raise ValueError("Sudoku board must be square.")

        self.box_size = int(math.isqrt(self.size))
        if self.box_size * self.box_size != self.size:
            raise ValueError(
                f"Board size {self.size} is not a perfect square; cannot form boxes."
            )

        self.random = randomizer
        self.digits = tuple(range(1, self.size + 1))

        self.rows = [set() for _ in range(self.size)]
        self.cols = [set() for _ in range(self.size)]
        self.boxes = [set() for _ in range(self.size)]

        for r in range(self.size):
            for c in range(self.size):
                value = self.board[r][c]
                if value == 0:
                    continue
                if value not in self.digits:
                    raise ValueError(
                        f"Cell ({r}, {c}) contains invalid value {value} for size {self.size}."
                    )
                box_index = self._box_index(r, c)
                if (
                    value in self.rows[r]
                    or value in self.cols[c]
                    or value in self.boxes[box_index]
                ):
                    raise ValueError(
                        f"Duplicate value {value} detected at cell ({r}, {c})."
                    )
                self.rows[r].add(value)
                self.cols[c].add(value)
                self.boxes[box_index].add(value)

        self.solution_counter = 0
        self.first_solution: Optional[Board] = None
        self._solution_limit = 1

    def solve(self, limit: int = 1):
        """
        Solve the Sudoku puzzle.

        Args:
            limit: Maximum number of solutions to search for. Use 1 to find a
                single solution; use 2 to test uniqueness.

        Returns:
            bool: True if at least one solution is found when limit == 1.
            int:  Number of solutions found (up to the limit) when limit > 1.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1.")

        self.solution_counter = 0
        self.first_solution = None
        self._solution_limit = limit
        self._backtrack()

        if limit == 1:
            if self.solution_counter >= 1 and self.first_solution is not None:
                self.board = [row[:] for row in self.first_solution]
                return True
            return False

        return self.solution_counter

    def get_solution(self) -> Board:
        """Return a deep copy of the current board state."""
        return [row[:] for row in self.board]

    # Internal helpers -----------------------------------------------------

    def _box_index(self, row: int, col: int) -> int:
        return (row // self.box_size) * self.box_size + (col // self.box_size)

    def _candidate_list(self, row: int, col: int) -> List[int]:
        used = (
            self.rows[row]
            | self.cols[col]
            | self.boxes[self._box_index(row, col)]
        )
        candidates = [digit for digit in self.digits if digit not in used]
        if self.random:
            self.random.shuffle(candidates)
        return candidates

    def _select_unassigned_cell(self) -> Optional[Tuple[int, int, List[int]]]:
        best_cell: Optional[Tuple[int, int]] = None
        best_candidates: Optional[List[int]] = None
        min_candidate_count = self.size + 1

        for r in range(self.size):
            row = self.board[r]
            for c in range(self.size):
                if row[c] != 0:
                    continue
                candidates = self._candidate_list(r, c)
                candidate_count = len(candidates)

                if candidate_count == 0:
                    return (r, c, [])

                if candidate_count < min_candidate_count:
                    min_candidate_count = candidate_count
                    best_cell = (r, c)
                    best_candidates = candidates
                    if candidate_count == 1:
                        return (r, c, candidates)

        if best_cell is None or best_candidates is None:
            return None

        return (best_cell[0], best_cell[1], best_candidates)

    def _place_value(self, row: int, col: int, value: int) -> None:
        self.board[row][col] = value
        self.rows[row].add(value)
        self.cols[col].add(value)
        self.boxes[self._box_index(row, col)].add(value)

    def _remove_value(self, row: int, col: int, value: int) -> None:
        self.board[row][col] = 0
        self.rows[row].remove(value)
        self.cols[col].remove(value)
        self.boxes[self._box_index(row, col)].remove(value)

    def _backtrack(self) -> bool:
        selection = self._select_unassigned_cell()
        if selection is None:
            self.solution_counter += 1
            if self.first_solution is None:
                self.first_solution = [row[:] for row in self.board]
            return self.solution_counter >= self._solution_limit

        row, col, candidates = selection
        if not candidates:
            return False

        for value in candidates:
            self._place_value(row, col, value)
            should_stop = self._backtrack()
            self._remove_value(row, col, value)
            if should_stop:
                return True

        return False


class SudokuDatasetGenerator:
    """Generate Sudoku puzzles with guaranteed unique solutions."""

    def __init__(
        self,
        size: int,
        *,
        min_clues: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.size = size
        self.box_size = int(math.isqrt(size))
        if self.box_size * self.box_size != size:
            raise ValueError(
                f"Board size {size} is not a perfect square; cannot form boxes."
            )

        self.random = random.Random(seed)
        total_cells = size * size

        if min_clues is None:
            if size == 9:
                min_clues = 30
            elif size == 16:
                min_clues = 90
            else:
                min_clues = max(int(total_cells * 0.35), size * 2)

        self.min_clues = max(min_clues, size * 2)

    def generate_full_solution(self) -> Board:
        empty_board = [[0] * self.size for _ in range(self.size)]
        solver = SudokuSolver(empty_board, randomizer=self.random)
        if not solver.solve(limit=1):
            raise RuntimeError("Failed to generate a complete Sudoku solution.")
        return solver.get_solution()

    def create_puzzle_with_solution(self) -> Tuple[Board, Board]:
        solution = self.generate_full_solution()
        puzzle = self._carve_puzzle(solution)

        verifier = SudokuSolver(puzzle)
        if not verifier.solve(limit=1):
            raise RuntimeError("Generated puzzle is not solvable.")

        return puzzle, verifier.get_solution()

    def _carve_puzzle(self, solution: Board) -> Board:
        puzzle = [row[:] for row in solution]
        filled_cells = self.size * self.size
        cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        self.random.shuffle(cells)

        for row, col in cells:
            if filled_cells <= self.min_clues:
                break

            backup = puzzle[row][col]
            puzzle[row][col] = 0

            if self._has_unique_solution(puzzle):
                filled_cells -= 1
            else:
                puzzle[row][col] = backup

        return puzzle

    def _has_unique_solution(self, puzzle: Board) -> bool:
        solver = SudokuSolver(puzzle)
        solution_count = solver.solve(limit=2)
        return isinstance(solution_count, int) and solution_count == 1


def generate_dataset(
    size: int,
    count: int,
    *,
    min_clues: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[List[Dict[str, Board]], Dict[str, Optional[float]]]:
    generator = SudokuDatasetGenerator(size, min_clues=min_clues, seed=seed)
    puzzles: List[Dict[str, Board]] = []
    start_time = time.time()

    progress_step = max(1, count // 20)

    for index in range(count):
        puzzle, solution = generator.create_puzzle_with_solution()
        puzzles.append({"puzzle": puzzle, "solution": solution})

        if (index + 1) % progress_step == 0 or index == count - 1:
            print(f"[{size}x{size}] Generated {index + 1}/{count} puzzles.")

    elapsed = time.time() - start_time
    metadata = {
        "min_clues": generator.min_clues,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 2),
    }
    print(f"[{size}x{size}] Dataset ready in {elapsed:.1f}s.")

    return puzzles, metadata


def save_dataset(
    entries: List[Dict[str, Board]],
    output_dir: Path,
    size: int,
    metadata: Optional[Dict[str, Optional[float]]] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / f"sudoku_{size}x{size}.json"

    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    count = len(entries)

    with dataset_path.open("w", encoding="utf-8") as fp:
        fp.write("{\n")
        fp.write(f'  "size": {size},\n')
        fp.write(f'  "count": {count},\n')
        fp.write(f'  "generated_at": "{generated_at}",\n')

        if metadata:
            metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
            metadata_lines = metadata_json.splitlines()
            fp.write('  "metadata": ')
            if metadata_lines:
                fp.write(metadata_lines[0] + "\n")
                for line in metadata_lines[1:]:
                    fp.write("  " + line + "\n")
                fp.write(",\n")
            else:
                fp.write("{} ,\n")

        fp.write('  "puzzles": [\n')
        for entry_index, entry in enumerate(entries):
            fp.write("    {\n")

            fp.write('      "puzzle": [\n')
            for row_index, row in enumerate(entry["puzzle"]):
                row_json = json.dumps(row, ensure_ascii=False)
                row_trailing = "," if row_index < size - 1 else ""
                fp.write(f"        {row_json}{row_trailing}\n")
            fp.write("      ],\n")

            fp.write('      "solution": [\n')
            for row_index, row in enumerate(entry["solution"]):
                row_json = json.dumps(row, ensure_ascii=False)
                row_trailing = "," if row_index < size - 1 else ""
                fp.write(f"        {row_json}{row_trailing}\n")
            fp.write("      ]\n")

            entry_trailing = "," if entry_index < count - 1 else ""
            fp.write(f"    }}{entry_trailing}\n")
        fp.write("  ]\n")
        fp.write("}\n")

    return dataset_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Sudoku datasets with unique solutions."
    )
    parser.add_argument(
        "--num-9x9",
        type=int,
        default=1000,
        help="Number of 9x9 puzzles to generate (default: 1000).",
    )
    parser.add_argument(
        "--num-16x16",
        type=int,
        default=1000,
        help="Number of 16x16 puzzles to generate (default: 1000).",
    )
    parser.add_argument(
        "--min-clues-9x9",
        type=int,
        default=None,
        help="Minimum number of clues retained in 9x9 puzzles.",
    )
    parser.add_argument(
        "--min-clues-16x16",
        type=int,
        default=None,
        help="Minimum number of clues retained in 16x16 puzzles.",
    )
    parser.add_argument(
        "--seed-9x9",
        type=int,
        default=2024,
        help="Random seed for 9x9 puzzle generation.",
    )
    parser.add_argument(
        "--seed-16x16",
        type=int,
        default=2025,
        help="Random seed for 16x16 puzzle generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "sokudu_dataset",
        help="Directory to store generated datasets.",
    )
    parser.add_argument(
        "--check-file",
        type=Path,
        default=None,
        help="Check the specified dataset file for duplicate puzzles.",
    )
    return parser.parse_args()


def check_dataset_duplicates(dataset_path: Path) -> None:
    dataset_path = dataset_path.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    puzzles = payload.get("puzzles")
    if not isinstance(puzzles, list):
        raise ValueError("Dataset payload missing 'puzzles' list.")

    seen: Dict[Tuple[Tuple[int, ...], ...], int] = {}
    duplicates: List[Tuple[int, int]] = []

    for index, entry in enumerate(puzzles):
        puzzle = entry.get("puzzle")
        if not isinstance(puzzle, list):
            raise ValueError(f"Entry {index} missing 'puzzle' grid.")

        key = tuple(tuple(int(cell) for cell in row) for row in puzzle)

        if key in seen:
            duplicates.append((seen[key], index))
        else:
            seen[key] = index

    if duplicates:
        print(f"✗ Found {len(duplicates)} duplicate puzzle(s) in {dataset_path}:")
        for first_index, dup_index in duplicates:
            print(f"  - Duplicate puzzle at indices {first_index} and {dup_index}")
    else:
        print(f"✓ No duplicate puzzles found in {dataset_path}.")


def main() -> None:
    args = parse_args()
    if args.check_file:
        check_dataset_duplicates(args.check_file)
        return

    output_dir = args.output_dir.resolve()

    tasks = [
        (9, args.num_9x9, args.min_clues_9x9, args.seed_9x9),
        (16, args.num_16x16, args.min_clues_16x16, args.seed_16x16),
    ]

    overall_start = time.time()

    for size, count, min_clues, seed in tasks:
        if count <= 0:
            print(f"[{size}x{size}] Skipping generation (count <= 0).")
            continue

        print(
            f"[{size}x{size}] Starting generation of {count} puzzles "
            f"(min_clues={min_clues}, seed={seed})."
        )
        entries, metadata = generate_dataset(
            size,
            count,
            min_clues=min_clues,
            seed=seed,
        )
        metadata.update(
            {
                "requested_count": count,
                "actual_count": len(entries),
            }
        )
        path = save_dataset(entries, output_dir, size, metadata=metadata)
        print(f"[{size}x{size}] Dataset saved to {path}.")

    overall_elapsed = time.time() - overall_start
    print(f"All tasks completed in {overall_elapsed:.1f}s.")


if __name__ == "__main__":
    main()


