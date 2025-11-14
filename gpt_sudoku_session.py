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
from typing import Any, List, Optional, Sequence, Tuple

from sudoku_solver import SudokuSolver

from llm_client import LLMClientError, PROVIDERS, chat_completion, get_provider


SYSTEM_PROMPT = (
    "You are a reasoning-only assistant working in a plain text environment. "
    "You must not invoke, simulate, or reference any external tools, code execution, "
    "or calculators. Solve the Sudoku puzzle strictly by mental reasoning and provide "
    "your final answer clearly."
)

USER_PROMPT_TEMPLATE = (
    "请在纯文本环境中解答下面的 9x9 数独题目。\n"
    "- 禁止使用任何外部工具、程序或求解器，也不要声称使用了工具。\n"
    "- 请在推理后给出最终解答，格式为 9 行，每行 9 个数字，使用空格分隔。\n"
    "- 如需解释，请放在解答之后。\n\n"
    "题目：\n{puzzle}\n"
)

FEEDBACK_PROMPT_TEMPLATE = (
    "上一轮你的答案存在错误，请在严格遵守以下规则的前提下重新解答：\n"
    "- 仍然禁止使用任何外部工具或程序，也不要声称使用了工具。\n"
    "- 输出 9 行，每行 9 个数字（空格分隔），在答案之后再给出必要的说明。\n"
    "- 必须修正列出的所有问题，确保与题面给出的已知数字完全一致。\n\n"
    "上一轮的答案：\n{last_answer}\n\n"
    "发现的问题：\n{issues}\n\n"
    "请重新给出完整解答。题目再次提供如下：\n{puzzle}\n"
)

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
        self.initial_prompt = USER_PROMPT_TEMPLATE.format(puzzle=board_to_text(self.puzzle))
        self.messages: List[dict] = []
        self.round_records: List[dict] = []
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

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
        last_answer: str,
        last_issues: Sequence[str],
    ) -> str:
        issues_text = "\n".join(f"- {issue}" for issue in last_issues) if last_issues else "- 未提供问题详情"
        return FEEDBACK_PROMPT_TEMPLATE.format(
            last_answer=last_answer.strip() or "(上一轮没有识别出有效答案)",
            issues=issues_text,
            puzzle=board_to_text(self.puzzle),
        )

    def request_solution(self, user_content: str) -> Tuple[str, Optional[str]]:
        """向 LLM 发送请求并返回文本回复。"""

        user_message = {"role": "user", "content": user_content}
        messages = [{"role": "system", "content": self.system_prompt}] + self.messages + [user_message]

        assistant_message, reasoning_text = chat_completion(
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

        return assistant_message, reasoning_text

    # ------------------------------------------------------------------
    # 校验逻辑
    # ------------------------------------------------------------------
    def evaluate_answer(self, raw_answer: str) -> SudokuCheckResult:
        rows = slice_rows_with_digits(raw_answer)
        if len(rows) < 9:
            return SudokuCheckResult(
                is_correct=False,
                issues=["未能识别出完整的 9 行解答，请确保输出 9 行、每行 9 个数字。"],
                parsed_board=None,
            )

        candidate = rows[:9]
        issues: List[str] = []

        # 长度与范围检查
        for r_idx, row in enumerate(candidate, start=1):
            if len(row) != 9:
                issues.append(f"第 {r_idx} 行不是 9 个数字。")
            out_of_range = [num for num in row if num < 1 or num > 9]
            if out_of_range:
                issues.append(
                    f"第 {r_idx} 行存在非 1-9 的数字：{', '.join(map(str, out_of_range))}。"
                )

        # 线索一致性
        for r in range(9):
            for c in range(9):
                clue = self.puzzle[r][c]
                if clue and candidate[r][c] != clue:
                    issues.append(
                        f"原题第 {r + 1} 行第 {c + 1} 列应为 {clue}，但回答为 {candidate[r][c]}。"
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
                    issue_parts.append(f"缺少 {sorted(missing)}")
                if duplicates:
                    issue_parts.append(f"存在重复 {sorted(set(duplicates))}")
                issues.append(f"第 {idx} 行不符合数独规则：{'; '.join(issue_parts)}。")

        for col in range(9):
            column = [candidate[row][col] for row in range(9)]
            col_set = set(column)
            if col_set != expected_set:
                missing = expected_set - col_set
                duplicates = [num for num in column if column.count(num) > 1]
                issue_parts = []
                if missing:
                    issue_parts.append(f"缺少 {sorted(missing)}")
                if duplicates:
                    issue_parts.append(f"存在重复 {sorted(set(duplicates))}")
                issues.append(f"第 {col + 1} 列不符合数独规则：{'; '.join(issue_parts)}。")

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
                        issue_parts.append(f"缺少 {sorted(missing)}")
                    if duplicates:
                        issue_parts.append(f"存在重复 {sorted(set(duplicates))}")
                    issues.append(
                        f"第 {box_row + 1} 行第 {box_col + 1} 宫不符合数独规则：{'; '.join(issue_parts)}。"
                    )

        # 最终与基准解比较
        mismatch = first_mismatch(candidate, self.correct_solution)
        if mismatch is not None:
            r, c = mismatch
            issues.append(
                "与内部验证解不同："
                f"第 {r + 1} 行第 {c + 1} 列回答为 {candidate[r][c]}，"
                f"而内部解为 {self.correct_solution[r][c]}。"
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
) -> None:
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

    puzzle = generate_random_puzzle(holes=holes)
    session_ts = int(time.time() * 1000)
    session_dir = history_dir / f"session_{session_ts}"

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
            user_prompt = session.build_feedback_prompt(
                last_answer=answer_snapshot,
                last_issues=last_result.issues if last_result else [],
            )

        try:
            assistant_reply, reasoning_log = session.request_solution(user_prompt)
        except LLMClientError as exc:  # pragma: no cover - 运行期容错
            print(f"❌ 调用 {provider} 接口失败：{exc}")
            return
        except Exception as exc:  # pragma: no cover - 运行期容错
            print(f"❌ 调用 {provider} 接口出现未知错误：{exc}")
            return

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用 LLM 解答数独并验证结果的脚本")
    parser.add_argument("--model", default="gpt-5", help="调用的 OpenAI 模型名称")
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_session(
        model=args.model,
        temperature=args.temperature,
        provider=args.provider,
        reset=args.reset,
        history_dir=args.history_dir,
        holes=args.holes,
        max_rounds=max(args.max_rounds, 1),
    )
