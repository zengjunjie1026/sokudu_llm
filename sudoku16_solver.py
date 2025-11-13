"""
16x16 数独求解器与棋盘生成工具。

该模块提供：
- Sudoku16Solver：使用回溯与 MRV 策略的解题器
- 生成完整 16x16 棋盘与挖空的工具函数
- 创建一个固定的示例题目（使用确定性随机种子）
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

BASE = 4  # 16x16 数独是 4x4 的小宫格
SIZE = BASE * BASE
DIGITS: Set[int] = set(range(1, SIZE + 1))


def board_to_text(board: Sequence[Sequence[int]]) -> str:
    """将棋盘转换为文本，0 以 '.' 表示。"""

    return "\n".join(
        " ".join(str(value) if value else "." for value in row) for row in board
    )


@dataclass
class Cell:
    row: int
    col: int
    candidates: Set[int]


class Sudoku16Solver:
    """16x16 数独求解器（使用回溯 + 最小剩余值 MRV 选择）。"""

    def __init__(self, board: Sequence[Sequence[int]]) -> None:
        self.board: List[List[int]] = [list(row) for row in board]
        if any(len(row) != SIZE for row in self.board) or len(self.board) != SIZE:
            raise ValueError("16x16 棋盘必须是 16 行 16 列。")

    # ------------------------------------------------------------------
    # 求解相关工具
    # ------------------------------------------------------------------
    def solve(self) -> bool:
        """解决数独，如果找到解返回 True。"""

        cell = self._select_cell_with_least_candidates()
        if cell is None:
            return True  # 已无空格

        r, c, candidates = cell.row, cell.col, cell.candidates
        for num in sorted(candidates):
            if self._place_number(r, c, num):
                if self.solve():
                    return True
                self._remove_number(r, c)

        return False

    def _select_cell_with_least_candidates(self) -> Optional[Cell]:
        """选择候选数最少的空格（MRV）。"""

        min_cell: Optional[Cell] = None
        min_count = SIZE + 1

        for r in range(SIZE):
            for c in range(SIZE):
                if self.board[r][c] != 0:
                    continue
                candidates = self._get_candidates(r, c)
                count = len(candidates)
                if count == 0:
                    return Cell(r, c, set())
                if count < min_count:
                    min_cell = Cell(r, c, candidates)
                    min_count = count
                    if min_count == 1:
                        return min_cell

        return min_cell

    def _get_candidates(self, row: int, col: int) -> Set[int]:
        used = set(self.board[row])  # 行
        used.update(self.board[r][col] for r in range(SIZE))  # 列

        start_row = (row // BASE) * BASE
        start_col = (col // BASE) * BASE
        for r in range(start_row, start_row + BASE):
            used.update(self.board[r][start_col:start_col + BASE])

        return {num for num in DIGITS if num not in used}

    def _place_number(self, row: int, col: int, num: int) -> bool:
        if num not in self._get_candidates(row, col):
            return False
        self.board[row][col] = num
        return True

    def _remove_number(self, row: int, col: int) -> None:
        self.board[row][col] = 0

    def get_solution(self) -> List[List[int]]:
        return [row[:] for row in self.board]


# ----------------------------------------------------------------------
# 棋盘生成工具
# ----------------------------------------------------------------------
def _pattern(row: int, col: int) -> int:
    """使用标准 pattern 公式生成完整棋盘。"""
    return (BASE * (row % BASE) + row // BASE + col) % SIZE


def _shuffled(seq, rng: random.Random):
    items = list(seq)
    rng.shuffle(items)
    return items


def generate_complete_board(rng: Optional[random.Random] = None) -> List[List[int]]:
    rng = rng or random.Random()
    rows = [g * BASE + r for g in _shuffled(range(BASE), rng) for r in _shuffled(range(BASE), rng)]
    cols = [g * BASE + c for g in _shuffled(range(BASE), rng) for c in _shuffled(range(BASE), rng)]
    nums = _shuffled(range(1, SIZE + 1), rng)

    return [[nums[_pattern(r, c)] for c in cols] for r in rows]


def carve_puzzle(board: Sequence[Sequence[int]], holes: int, rng: Optional[random.Random] = None) -> List[List[int]]:
    rng = rng or random.Random()
    puzzle = [list(row) for row in board]
    cells = [(r, c) for r in range(SIZE) for c in range(SIZE)]
    rng.shuffle(cells)

    for r, c in cells[: min(len(cells), holes)]:
        puzzle[r][c] = 0

    return puzzle


def create_example_board() -> Tuple[List[List[int]], List[List[int]]]:
    """
    创建一个固定示例题目，并返回 (题目, 标准解)。
    使用固定随机种子保证每次生成一致。
    """

    rng = random.Random(2025)
    solution = generate_complete_board(rng)
    puzzle = carve_puzzle(solution, holes=140, rng=rng)  # 挖掉约 55% 的格子

    solver = Sudoku16Solver(puzzle)
    assert solver.solve(), "示例题目应当可解。"
    resolved = solver.get_solution()

    return puzzle, resolved


__all__ = [
    "Sudoku16Solver",
    "create_example_board",
    "generate_complete_board",
    "carve_puzzle",
    "board_to_text",
    "SIZE",
]

