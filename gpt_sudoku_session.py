"""调用 LLM（OpenAI/DeepSeek/Qwen 等）解答 9x9 数独题目的脚本。

该脚本会：
- 构造限制模型不能调用任何工具的提示词
- 将之前的提示词与回复作为上下文继续对话
- 调用 LLM 获取数独解答
- 解析并校验解答的正确性，反馈问题
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sudoku_solver import SudokuSolver

from llm_client import LLMClientError, PROVIDERS, chat_completion, get_provider


SYSTEM_PROMPT = (
    "You are a reasoning-only assistant working in a plain text environment. "
    "You must not invoke, simulate, or reference any external tools, code execution, "
    "or calculators. Solve the Sudoku puzzle strictly by mental reasoning and provide "
    "your final answer clearly."
)

USER_PROMPT_TEMPLATE = (
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
   
)

FEEDBACK_PROMPT_TEMPLATE = """Your previous solutions are still incorrect. Review every past attempt and fix all listed issues.

Instructions (apply to a {size}×{size} Sudoku grid):
1. You must NOT use or mention any external tools, code, or solvers—only mental reasoning.
2. Produce exactly {size} lines of output; each line must contain {size} digits (1–{size}) separated by single spaces.
3. Do not add commentary before or after the grid. If you need to explain, do it only after the grid on a new paragraph.
4. Every row, column, and subgrid must satisfy Sudoku rules and must respect the givens from the puzzle.
5. Address every issue listed below before submitting a new answer.

History of answers and detected problems:
{history}

Puzzle (repeated for convenience):
{puzzle}
"""

def board_to_text(board: Sequence[Sequence[int]]) -> str:
    """将 9x9 数独棋盘转换为字符串表示，空格使用句点表示。"""

    lines: List[str] = []
    for row in board:
        line = " ".join(str(value) if value else "." for value in row)
        lines.append(line)
    return "\n".join(lines)


def slice_rows_with_digits(text: str, expected: int = 9) -> List[List[int]]:
    """从模型回复中截取包含 9 个数字的行。"""

    rows: List[List[int]] = []
    for line in text.splitlines():
        digits = [int(ch) for ch in re.findall(r"[0-9]", line)]
        if len(digits) == expected:
            rows.append(digits)
    return rows


def first_mismatch(
    board_a: Sequence[Sequence[int]],
    board_b: Sequence[Sequence[int]],
) -> Optional[Tuple[int, int]]:
    """返回两个棋盘第一个不同的坐标。"""

    for r in range(9):
        for c in range(9):
            if board_a[r][c] != board_b[r][c]:
                return r, c
    return None


@dataclass
class SudokuCheckResult:
    """封装校验结果。"""

    is_correct: bool
    issues: List[str]
    parsed_board: Optional[List[List[int]]] = None


class SudokuChatSession:
    """与 LLM 进行数独对话的会话管理器。"""

    def __init__(
        self,
        puzzle: Sequence[Sequence[int]],
        model: str = "gpt-5",
        temperature: float = 1,
        provider: str = "openai",
        history_dir: Optional[Path] = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.puzzle = [list(row) for row in puzzle]
        self.model = model
        self.temperature = temperature
        self.provider = provider
        self.session_dir = history_dir or self._default_session_dir()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.session_dir / "conversation.json"
        self.system_prompt = system_prompt
        self.provider_config = get_provider(provider)
        self.correct_solution = self._solve_baseline()
        self.initial_prompt = USER_PROMPT_TEMPLATE.format(
            puzzle=board_to_text(self.puzzle),
            size=len(self.puzzle),
        )
        self.messages: List[dict] = []
        self.round_records: List[dict] = []
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @staticmethod
    def _default_session_dir() -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return Path(__file__).resolve().with_name(f"gpt_sudoku_session_{timestamp}")

    def _save_history(self) -> None:
        data = {
            "system_prompt": self.system_prompt,
            "puzzle": board_to_text(self.puzzle),
            "created_at": self.created_at,
            "rounds": self.round_records,
            "messages": self.messages,
        }
        with self.history_file.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 业务逻辑
    # ------------------------------------------------------------------
    def _solve_baseline(self) -> List[List[int]]:
        solver = SudokuSolver(self.puzzle)
        if not solver.solve():
            raise RuntimeError("示例数独无法被内部求解器解决，请检查题目数据。")
        return solver.get_solution()

    def build_initial_prompt(self) -> str:
        return self.initial_prompt

    def build_feedback_prompt(
        self,
        history: Sequence[Tuple[str, Sequence[str]]],
    ) -> str:
        sections: List[str] = []
        for idx, (answer, issues) in enumerate(history, start=1):
            cleaned_answer = answer.strip() or "(未识别到有效答案)"
            if issues:
                issues_text = "\n".join(f"    - {issue}" for issue in issues)
            else:
                issues_text = "    - 未提供问题详情"
            sections.append(
                f"回答 {idx}：\n{cleaned_answer}\n存在的问题：\n{issues_text}"
            )

        history_text = "\n\n".join(sections) if sections else "(暂无历史记录)"
        return FEEDBACK_PROMPT_TEMPLATE.format(
            size=len(self.puzzle),
            history=history_text,
            puzzle=board_to_text(self.puzzle),
        )

    def request_solution(self, user_content: str) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        """向 LLM 发送请求并返回文本回复。"""

        user_message = {"role": "user", "content": user_content}
        messages = [{"role": "system", "content": self.system_prompt}] + self.messages + [user_message]

        assistant_message, reasoning_text, usage = chat_completion(
            provider=self.provider,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        # 更新历史
        assistant_entry = {"role": "assistant", "content": assistant_message}
        if reasoning_text:
            assistant_entry["reasoning"] = reasoning_text
        self.messages.extend([user_message, assistant_entry])
        self._save_history()

        if usage:
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if isinstance(prompt_tokens, int):
                self.prompt_tokens += prompt_tokens
            if isinstance(completion_tokens, int):
                self.completion_tokens += completion_tokens

        return assistant_message, reasoning_text, usage

    # ------------------------------------------------------------------
    # 校验逻辑
    # ------------------------------------------------------------------
    def evaluate_answer(self, raw_answer: str) -> SudokuCheckResult:
        rows = slice_rows_with_digits(raw_answer)
        if len(rows) < 9:
            return SudokuCheckResult(
                is_correct=False,
                issues=[
                    "Failed to detect 9 valid lines. Please output 9 lines, each containing exactly 9 digits."
                ],
                parsed_board=None,
            )

        candidate = rows[:9]
        issues: List[str] = []

        # 长度与范围检查
        for r_idx, row in enumerate(candidate, start=1):
            if len(row) != 9:
                issues.append(f"Row {r_idx} does not contain exactly 9 digits.")
            out_of_range = [num for num in row if num < 1 or num > 9]
            if out_of_range:
                issues.append(f"Row {r_idx} contains digits outside 1-9: {sorted(out_of_range)}.")

        # 线索一致性
        for r in range(9):
            for c in range(9):
                clue = self.puzzle[r][c]
                if clue and candidate[r][c] != clue:
                    issues.append(
                        f"Cell ({r + 1}, {c + 1}) must be {clue} per the puzzle, but your answer uses {candidate[r][c]}."
                    )

        # 行、列、宫格检测
        expected_set = set(range(1, 10))

        for idx, row in enumerate(candidate, start=1):
            row_set = set(row)
            if row_set != expected_set:
                missing = expected_set - row_set
                duplicates = [num for num in row if row.count(num) > 1]
                issue_parts = []
                if missing:
                    issue_parts.append(f"missing {sorted(missing)}")
                if duplicates:
                    issue_parts.append(f"duplicate {sorted(set(duplicates))}")
                issues.append(f"Row {idx} violates Sudoku rules: {'; '.join(issue_parts)}.")

        for col in range(9):
            column = [candidate[row][col] for row in range(9)]
            col_set = set(column)
            if col_set != expected_set:
                missing = expected_set - col_set
                duplicates = [num for num in column if column.count(num) > 1]
                issue_parts = []
                if missing:
                    issue_parts.append(f"missing {sorted(missing)}")
                if duplicates:
                    issue_parts.append(f"duplicate {sorted(set(duplicates))}")
                issues.append(f"Column {col + 1} violates Sudoku rules: {'; '.join(issue_parts)}.")

        for box_row in range(3):
            for box_col in range(3):
                cells = [
                    candidate[r][c]
                    for r in range(box_row * 3, box_row * 3 + 3)
                    for c in range(box_col * 3, box_col * 3 + 3)
                ]
                cell_set = set(cells)
                if cell_set != expected_set:
                    missing = expected_set - cell_set
                    duplicates = [num for num in cells if cells.count(num) > 1]
                    issue_parts = []
                    if missing:
                            issue_parts.append(f"missing {sorted(missing)}")
                    if duplicates:
                            issue_parts.append(f"duplicate {sorted(set(duplicates))}")
                    issues.append(
                            f"Subgrid ({box_row + 1}, {box_col + 1}) violates Sudoku rules: {'; '.join(issue_parts)}."
                    )

        is_correct = not issues
        return SudokuCheckResult(is_correct=is_correct, issues=issues, parsed_board=candidate)

    def record_round(
        self,
        round_index: int,
        user_message: str,
        assistant_message: str,
        result: SudokuCheckResult,
        reasoning_log: Optional[str] = None,
        token_usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "round": round_index,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "assistant_reasoning": reasoning_log,
            "validation": {
                "is_correct": result.is_correct,
                "issues": result.issues,
            },
            "parsed_board": board_to_text(result.parsed_board) if result.parsed_board else None,
        }
        if token_usage:
            record["token_usage"] = token_usage
        self.round_records.append(record)
        self._save_history()


def pattern(base: int, row: int, col: int) -> int:
    side = base * base
    return (base * (row % base) + row // base + col) % side


def shuffled(sequence):
    seq = list(sequence)
    random.shuffle(seq)
    return seq


def generate_complete_board(base: int = 3) -> List[List[int]]:
    side = base * base
    rows = [g * base + r for g in shuffled(range(base)) for r in shuffled(range(base))]
    cols = [g * base + c for g in shuffled(range(base)) for c in shuffled(range(base))]
    nums = shuffled(range(1, side + 1))
    return [[nums[pattern(base, r, c)] for c in cols] for r in rows]


def carve_puzzle(board: Sequence[Sequence[int]], holes: int) -> List[List[int]]:
    puzzle = [row[:] for row in board]
    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)
    for r, c in cells[:holes]:
        puzzle[r][c] = 0
    return puzzle


def generate_random_puzzle(holes: int = 45) -> List[List[int]]:
    holes = max(0, min(81, holes))
    full_board = generate_complete_board()
    return carve_puzzle(full_board, holes)


def run_session(
    model: str,
    temperature: float,
    provider: str,
    reset: bool,
    history_dir: Path,
    holes: int,
    max_rounds: int,
    puzzle_override: Optional[Sequence[Sequence[int]]] = None,
    session_dir_override: Optional[Path] = None,
) -> Dict[str, Any]:
    history_dir = history_dir.resolve()
    history_dir.mkdir(parents=True, exist_ok=True)

    if reset:
        removed = 0
        for path in history_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
            elif path.is_file():
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    continue
        print(f"🧹 已清空历史记录目录，删除 {removed} 个历史条目。")

    if puzzle_override is not None:
        puzzle = [list(row) for row in puzzle_override]
    else:
        puzzle = generate_random_puzzle(holes=holes)
    session_ts = int(time.time() * 1000)
    if session_dir_override is not None:
        session_dir = session_dir_override.resolve()
        session_dir.mkdir(parents=True, exist_ok=True)
    else:
        session_dir = history_dir / f"session_{session_ts}"
        session_dir.mkdir(parents=True, exist_ok=True)

    session = SudokuChatSession(
        puzzle=puzzle,
        model=model,
        temperature=temperature,
        provider=provider,
        history_dir=session_dir,
    )

    print(f"📨 发送给 {provider} 的题目：")
    print(board_to_text(puzzle))

    last_answer_text = ""
    last_result: Optional[SudokuCheckResult] = None
    attempt_history: List[Tuple[str, List[str]]] = []
    round_count = 0
    final_result: Optional[SudokuCheckResult] = None
    success = False

    for round_index in range(1, max_rounds + 1):
        print(f"\n🔁 开始第 {round_index} 轮对话")

        if round_index == 1:
            user_prompt = session.build_initial_prompt()
        else:
            answer_snapshot = (
                board_to_text(last_result.parsed_board)
                if last_result and last_result.parsed_board
                else last_answer_text
            )
            user_prompt = session.build_feedback_prompt(history=attempt_history)

        try:
            assistant_reply, reasoning_log, usage = session.request_solution(user_prompt)
        except LLMClientError as exc:  # pragma: no cover - 运行期容错
            print(f"❌ 调用 {provider} 接口失败：{exc}")
            return {
                "model": model,
                "temperature": temperature,
                "provider": provider,
                "timestamp": session.created_at,
                "rounds": round_count,
                "max_rounds": max_rounds,
                "success": False,
                "puzzle": board_to_text(puzzle),
                "conversation_file": str(session.history_file.name),
                "error": str(exc),
            }
        except Exception as exc:  # pragma: no cover - 运行期容错
            print(f"❌ 调用 {provider} 接口出现未知错误：{exc}")
            return {
                "model": model,
                "temperature": temperature,
                "provider": provider,
                "timestamp": session.created_at,
                "rounds": round_count,
                "max_rounds": max_rounds,
                "success": False,
                "puzzle": board_to_text(puzzle),
                "conversation_file": str(session.history_file.name),
                "error": str(exc),
            }

        print(f"\n🤖 {provider} 的完整回复：\n")
        print(assistant_reply or "(未识别到任何回答内容)")

        result = session.evaluate_answer(assistant_reply)
        round_count = round_index
        final_result = result
        session.record_round(
            round_index,
            user_prompt,
            assistant_reply,
            result,
            reasoning_log=reasoning_log,
            token_usage=usage,
        )

        if result.is_correct:
            print(f"\n✅ {provider} 在本轮提供了正确的数独解答。")
            if result.parsed_board:
                print("\n🧾 最终解析的 9x9 解答：")
                print(board_to_text(result.parsed_board))
            success = True
            break

        print(f"\n❌ {provider} 的解答仍存在问题：")
        for issue in result.issues:
            print(f"- {issue}")

        if result.parsed_board:
            print("\n🧾 本轮解析出的 9x9 解答：")
            print(board_to_text(result.parsed_board))

        last_result = result
        last_answer_text = assistant_reply
        attempt_history.append(
            (
                assistant_reply.strip()
                or (board_to_text(result.parsed_board) if result.parsed_board else "(未识别到有效答案)"),
                list(result.issues),
            )
        )
    else:
        print(f"\n⚠️ 已进行 {max_rounds} 轮对话，仍未获得正确解答，请稍后重试或调整提示。")

    summary = {
        "model": model,
        "temperature": temperature,
        "provider": provider,
        "timestamp": session.created_at,
        "rounds": round_count,
        "max_rounds": max_rounds,
        "success": success,
        "puzzle": board_to_text(puzzle),
        "conversation_file": str(session.history_file.name),
        "error": None,
        "token_usage": {
            "prompt_tokens": session.prompt_tokens,
            "completion_tokens": session.completion_tokens,
            "total_tokens": session.prompt_tokens + session.completion_tokens,
        },
    }
    if final_result:
        summary["final_issues"] = final_result.issues

    summary_path = session.session_dir / "summary.json"
    rounds_path = session.session_dir / "rounds.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    rounds_path.write_text(f"{round_count}\n", encoding="utf-8")

    print("\n📁 会话记录目录:", session.session_dir)
    print("   - 对话文件:", session.history_file)
    print("   - 概要文件:", summary_path)

    return summary


def load_dataset_puzzles(dataset_path: Path) -> List[List[List[int]]]:
    dataset_path = dataset_path.resolve()
    with dataset_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    puzzles = payload.get("puzzles")
    if not isinstance(puzzles, list):
        raise ValueError("数据集缺少 puzzles 列表。")

    extracted: List[List[List[int]]] = []
    for entry in puzzles:
        puzzle = entry.get("puzzle")
        if not isinstance(puzzle, list):
            continue
        extracted.append(puzzle)
    return extracted


def run_dataset_benchmark(
    dataset_path: Path,
    limit: int,
    model: str,
    temperature: float,
    provider: str,
    reset: bool,
    history_dir: Path,
    max_rounds: int,
    retry_attempts: int,
) -> None:
    puzzles = load_dataset_puzzles(dataset_path)
    limit = max(0, min(limit, len(puzzles)))
    if limit == 0:
        print(f"⚠️ 数据集为空或 limit=0：{dataset_path}")
        return

    dataset_history_root = history_dir.resolve() / f"dataset_run_{time.strftime('%Y%m%d_%H%M%S')}"
    dataset_history_root.mkdir(parents=True, exist_ok=True)
    print(f"📁 数据集日志目录: {dataset_history_root}")

    print(
        f"📚 使用数据集 {dataset_path} 的前 {limit} 道题，评估模型 {provider}:{model} "
        f"(temperature={temperature}, max_rounds={max_rounds})"
    )

    success_count = 0
    total_rounds = 0
    per_puzzle_rounds: List[int] = []
    per_puzzle_success: List[bool] = []
    skipped_puzzles: List[Dict[str, Any]] = []

    for idx in range(limit):
        print(f"\n=== 数据集题目 {idx + 1}/{limit} ===")
        summary: Optional[Dict[str, Any]] = None
        last_error: Optional[str] = None

        for attempt in range(1, max(1, retry_attempts) + 1):
            attempt_dir = dataset_history_root / f"puzzle_{idx + 1:04d}" / f"attempt_{attempt:02d}"
            summary = run_session(
                model=model,
                temperature=temperature,
            provider=provider,
            reset=reset and idx == 0 and attempt == 1,
                history_dir=history_dir,
                holes=0,
                max_rounds=max_rounds,
                puzzle_override=puzzles[idx],
                session_dir_override=attempt_dir,
            )
            if summary is None:
                last_error = "unknown failure"
                print(
                    f"⚠️ 题目 {idx + 1} 第 {attempt}/{retry_attempts} 次尝试失败：未知原因"
                )
                time.sleep(1)
                continue
            if summary.get("error"):
                last_error = summary["error"]
                print(
                    f"⚠️ 题目 {idx + 1} 第 {attempt}/{retry_attempts} 次尝试失败：{last_error}"
                )
                time.sleep(1)
                continue
            break

        if summary is None or summary.get("error"):
            print(f"🚫 题目 {idx + 1} 多次重试失败，跳过。")
            skipped_puzzles.append(
                {
                    "index": idx,
                    "error": last_error or "unknown failure",
                    "attempts": retry_attempts,
                }
            )
            per_puzzle_success.append(False)
            per_puzzle_rounds.append(0)
            continue

        per_puzzle_success.append(summary.get("success", False))
        per_puzzle_rounds.append(summary.get("rounds", max_rounds))
        if summary.get("success"):
            success_count += 1
            total_rounds += summary.get("rounds", 0)

    print("\n=== 数据集评估总结 ===")
    success_rate = success_count / limit
    avg_rounds = total_rounds / success_count if success_count else None
    print(f"总题目数: {limit}")
    print(f"成功题目数: {success_count} ({success_rate:.1%})")
    if avg_rounds is not None:
        print(f"平均成功轮数: {avg_rounds:.2f}")
    else:
        print("平均成功轮数: 无成功题目")

    if skipped_puzzles:
        skipped_path = history_dir.resolve() / "dataset_skipped.json"
        record = {
            "dataset": str(dataset_path),
            "model": model,
            "provider": provider,
            "temperature": temperature,
            "max_rounds": max_rounds,
            "retry_attempts": retry_attempts,
            "failed_puzzles": skipped_puzzles,
        }
        with skipped_path.open("w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
        print(f"⚠️ 有 {len(skipped_puzzles)} 道题未完成，已记录在 {skipped_path}")
    else:
        print("所有题目均已尝试完成。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用 LLM 解答数独并验证结果的脚本")
    parser.add_argument(
        "--model",
        default=None,
        help="调用的模型名称；默认为所选 provider 的默认模型",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="生成温度，默认为 1",
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        default="openai",
        help="选择调用的 LLM 供应商（openai/deepseek/qwen），默认为 openai",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="在查询前清除历史对话记录",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path(__file__).resolve().with_name("gpt_sudoku_histories"),
        help="保存会话记录的目录，默认与脚本同级的 gpt_sudoku_histories",
    )
    parser.add_argument(
        "--holes",
        type=int,
        default=45,
        help="移除的格子数量，范围 0-81，默认为 45 (中等难度)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="允许与模型进行的最大轮数，默认为 10",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="指定数据集 JSON（如 sokudu_dataset/sudoku_9x9.json）时，将按顺序使用题目，而非随机生成。",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=100,
        help="使用数据集模式时，读取的题目数量（默认 100）。",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=10,
        help="调用失败时的最大重试次数（默认 10 次）。",
    )
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="快捷方式：使用本地 ollama 提供的 gpt-oss 模型（将 provider 设置为 ollama）。",
    )

    args = parser.parse_args()
    if args.use_ollama:
        args.provider = "ollama"
        if args.model is None:
            args.model = "gpt-oss:20b"

    if args.model is None:
        args.model = PROVIDERS[args.provider].default_model

    return args


if __name__ == "__main__":
    args = parse_args()
    effective_temp = (
        1.0 if args.provider == "openai" and args.model.lower().startswith("gpt-5") else args.temperature
    )
    if args.dataset:
        run_dataset_benchmark(
            dataset_path=args.dataset,
            limit=args.dataset_limit,
            model=args.model,
            temperature=effective_temp,
            provider=args.provider,
            reset=args.reset,
            history_dir=args.history_dir,
            max_rounds=max(args.max_rounds, 1),
            retry_attempts=max(1, args.retry_attempts),
        )
    else:
        run_session(
            model=args.model,
            temperature=effective_temp,
            provider=args.provider,
            reset=args.reset,
            history_dir=args.history_dir,
            holes=args.holes,
            max_rounds=max(args.max_rounds, 1),
        )
