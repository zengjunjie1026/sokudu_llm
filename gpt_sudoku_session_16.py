"""
调用 LLM（OpenAI/DeepSeek/Qwen 等）解答 16x16 数独题目的脚本。

流程：
1. 随机生成一个 16x16 数独题目（支持自定义挖空数量）。
2. 与模型进行多轮对话，严格禁止使用任何外部工具。
3. 每轮回复都会解析、校验，并把问题反馈给模型重新作答。
4. 所有提示词、回复、校验结果与（若存在）思考摘要都会写入 JSON 历史文件。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from sudoku16_solver import (
    SIZE,
    Sudoku16Solver,
    board_to_text,
    carve_puzzle,
    generate_complete_board,
)

from llm_client import LLMClientError, PROVIDERS, chat_completion, get_provider

SYSTEM_PROMPT = (
    "You are a reasoning-only assistant working in a plain text environment. "
    "You must not invoke, simulate, or reference any external tools, code execution, "
    "or calculators. Solve the Sudoku puzzle strictly by mental reasoning and provide "
    "your final answer clearly."
)

USER_PROMPT_TEMPLATE = (
    "请在纯文本环境中解答下面的 16x16 数独题目。\n"
    "- 禁止使用任何外部工具、程序或求解器，也不要声称使用了工具。\n"
    "- 输出 16 行，每行 16 个数字（范围 1-16），使用空格分隔。\n"
    "- 如果需要解释，请在答案之后附加。\n\n"
    "题目：\n{puzzle}\n"
)

FEEDBACK_PROMPT_TEMPLATE = (
    "上一轮你的答案存在问题，请在严格遵守以下规则的前提下重新解答：\n"
    "- 仍然禁止使用任何外部工具或程序，也不要声称使用了工具。\n"
    "- 输出 16 行，每行 16 个数字（范围 1-16），使用空格分隔。\n"
    "- 必须修正列出的所有问题，确保与题面给出的已知数字完全一致。\n\n"
    "上一轮的答案：\n{last_answer}\n\n"
    "发现的问题：\n{issues}\n\n"
    "请重新给出完整解答。题目再次提供如下：\n{puzzle}\n"
)

EXPECTED_SET = set(range(1, SIZE + 1))


def board_to_display_text(board: Sequence[Sequence[int]]) -> str:
    """展示用文本，0 使用 '.'。"""
    return board_to_text(board)


def parse_rows_from_text(text: str, expected: int = SIZE) -> List[List[int]]:
    """
    从模型回复中提取每行的 16 个数字。
    支持识别 0（视为留空）与 1-16。
    """

    pattern = re.compile(r"\b(?:1[0-6]|[1-9]|0)\b")
    rows: List[List[int]] = []

    for line in text.splitlines():
        values = [int(token) for token in pattern.findall(line)]
        if len(values) == expected:
            rows.append(values)

    return rows


def first_mismatch(
    board_a: Sequence[Sequence[int]],
    board_b: Sequence[Sequence[int]],
) -> Optional[Tuple[int, int]]:
    """返回两个棋盘第一个不同的坐标。"""

    for r in range(SIZE):
        for c in range(SIZE):
            if board_a[r][c] != board_b[r][c]:
                return r, c
    return None


# ----------------------------------------------------------------------
# 校验结果
# ----------------------------------------------------------------------
class SudokuCheckResult:
    def __init__(
        self,
        is_correct: bool,
        issues: List[str],
        parsed_board: Optional[List[List[int]]] = None,
    ) -> None:
        self.is_correct = is_correct
        self.issues = issues
        self.parsed_board = parsed_board


# ----------------------------------------------------------------------
# LLM 会话管理
# ----------------------------------------------------------------------
class Sudoku16ChatSession:
    def __init__(
        self,
        puzzle: Sequence[Sequence[int]],
        model: str = "gpt-5",
        temperature: float = 1.0,
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
        self.initial_prompt = USER_PROMPT_TEMPLATE.format(
            puzzle=board_to_display_text(self.puzzle)
        )
        self.messages: List[dict] = []
        self.round_records: List[dict] = []
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # 通过内部求解器验证题目可解，并记录一个参考解
        solver = Sudoku16Solver(self.puzzle)
        if not solver.solve():
            raise RuntimeError("生成的 16x16 数独题目无法被内部求解器解决。")
        self.reference_solution = solver.get_solution()

    @staticmethod
    def _default_session_dir() -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return Path(__file__).resolve().with_name(f"gpt_sudoku16_session_{timestamp}")

    def _save_history(self) -> None:
        data = {
            "system_prompt": self.system_prompt,
            "puzzle": board_to_display_text(self.puzzle),
            "created_at": self.created_at,
            "rounds": self.round_records,
            "messages": self.messages,
        }
        with self.history_file.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 构造提示
    # ------------------------------------------------------------------
    def build_initial_prompt(self) -> str:
        return self.initial_prompt

    def build_feedback_prompt(
        self,
        last_answer: str,
        last_issues: Sequence[str],
    ) -> str:
        issues_text = (
            "\n".join(f"- {issue}" for issue in last_issues)
            if last_issues
            else "- 未提供问题详情"
        )
        return FEEDBACK_PROMPT_TEMPLATE.format(
            last_answer=last_answer.strip() or "(上一轮没有识别出有效答案)",
            issues=issues_text,
            puzzle=board_to_display_text(self.puzzle),
        )

    # ------------------------------------------------------------------
    # 与 LLM 交互
    # ------------------------------------------------------------------
    def request_solution(self, user_content: str) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        user_message = {"role": "user", "content": user_content}
        messages = [{"role": "system", "content": self.system_prompt}] + self.messages + [user_message]
        assistant_message, reasoning_text, usage = chat_completion(
            provider=self.provider,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        self.messages.extend(
            [
                user_message,
                {"role": "assistant", "content": assistant_message, "reasoning": reasoning_text},
            ]
        )
        self._save_history()

        return assistant_message, reasoning_text, usage

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def evaluate_answer(self, raw_answer: str) -> SudokuCheckResult:
        rows = parse_rows_from_text(raw_answer, expected=SIZE)
        if len(rows) < SIZE:
            return SudokuCheckResult(
                is_correct=False,
                issues=["未能识别出完整的 16 行解答，请确保输出 16 行、每行 16 个数字。"],
                parsed_board=None,
            )

        candidate = rows[:SIZE]
        issues: List[str] = []

        # 线索一致性
        for r in range(SIZE):
            for c in range(SIZE):
                clue = self.puzzle[r][c]
                if clue and candidate[r][c] != clue:
                    issues.append(
                        f"原题第 {r + 1} 行第 {c + 1} 列应为 {clue}，但回答为 {candidate[r][c]}。"
                    )

        # 行规则
        for idx, row in enumerate(candidate, start=1):
            row_set = set(row)
            if row_set != EXPECTED_SET:
                missing = EXPECTED_SET - row_set
                duplicates = [num for num in row if row.count(num) > 1]
                details = []
                if missing:
                    details.append(f"缺少 {sorted(missing)}")
                if duplicates:
                    details.append(f"存在重复 {sorted(set(duplicates))}")
                issues.append(f"第 {idx} 行不符合数独规则：{'; '.join(details)}。")

        # 列规则
        for col in range(SIZE):
            column = [candidate[row][col] for row in range(SIZE)]
            col_set = set(column)
            if col_set != EXPECTED_SET:
                missing = EXPECTED_SET - col_set
                duplicates = [num for num in column if column.count(num) > 1]
                details = []
                if missing:
                    details.append(f"缺少 {sorted(missing)}")
                if duplicates:
                    details.append(f"存在重复 {sorted(set(duplicates))}")
                issues.append(f"第 {col + 1} 列不符合数独规则：{'; '.join(details)}。")

        # 宫格规则
        for box_row in range(BASE):
            for box_col in range(BASE):
                cells = [
                    candidate[r][c]
                    for r in range(box_row * BASE, box_row * BASE + BASE)
                    for c in range(box_col * BASE, box_col * BASE + BASE)
                ]
                cell_set = set(cells)
                if cell_set != EXPECTED_SET:
                    missing = EXPECTED_SET - cell_set
                    duplicates = [num for num in cells if cells.count(num) > 1]
                    details = []
                    if missing:
                        details.append(f"缺少 {sorted(missing)}")
                    if duplicates:
                        details.append(f"存在重复 {sorted(set(duplicates))}")
                    issues.append(
                        f"宫格 (行 {box_row + 1}, 列 {box_col + 1}) 不符合数独规则：{'; '.join(details)}。"
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
            "parsed_board": board_to_display_text(result.parsed_board)
            if result.parsed_board
            else None,
        }
        self.round_records.append(record)
        self._save_history()


# ----------------------------------------------------------------------
# 题目生成
# ----------------------------------------------------------------------
def generate_random_puzzle(holes: int = 180) -> List[List[int]]:
    rng = random.Random()  # 使用系统随机
    solution = generate_complete_board(rng)
    puzzle = carve_puzzle(solution, holes=holes, rng=rng)
    return puzzle


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
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
        print(f"🧹 已清空 16x16 历史记录目录，删除 {removed} 个历史条目。")

    if holes < 0 or holes > SIZE * SIZE:
        raise ValueError(f"holes 参数应在 0~{SIZE * SIZE} 范围内。")

    puzzle = generate_random_puzzle(holes=holes)
    session_ts = int(time.time() * 1000)
    session_dir = history_dir / f"session_{session_ts}"

    session = Sudoku16ChatSession(
        puzzle=puzzle,
        model=model,
        temperature=temperature,
        provider=provider,
        history_dir=session_dir,
    )

    print(f"📨 发送给 {provider} 的 16x16 题目：")
    print(board_to_display_text(puzzle))

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
                board_to_display_text(last_result.parsed_board)
                if last_result and last_result.parsed_board
                else last_answer_text
            )
            user_prompt = session.build_feedback_prompt(
                last_answer=answer_snapshot,
                last_issues=last_result.issues if last_result else [],
            )

        try:
            assistant_reply, reasoning_log, usage = session.request_solution(user_prompt)
        except LLMClientError as exc:  # pragma: no cover
            print(f"❌ 调用 {provider} 接口失败：{exc}")
            return
        except Exception as exc:  # pragma: no cover
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
            print(f"\n✅ {provider} 在本轮提供了正确的 16x16 数独解答。")
            if result.parsed_board:
                print("\n🧾 最终解析的 16x16 解答：")
                print(board_to_display_text(result.parsed_board))
            success = True
            break

        print(f"\n❌ {provider} 的解答仍存在问题：")
        for issue in result.issues:
            print(f"- {issue}")

        if result.parsed_board:
            print("\n🧾 本轮解析出的 16x16 解答：")
            print(board_to_display_text(result.parsed_board))

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
        "puzzle": board_to_display_text(puzzle),
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
    parser = argparse.ArgumentParser(description="调用 LLM 解答 16x16 数独并验证结果的脚本")
    parser.add_argument("--model", default="gpt-5", help="调用的模型名称")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="生成温度，默认为 1.0",
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        default="openai",
        help="选择调用的 LLM 供应商（openai/deepseek/qwen），默认为 openai",
    )
    parser.add_argument(
        "--holes",
        type=int,
        default=180,
        help="挖空数量，范围 0-256，默认 180（较高难度）",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path(__file__).resolve().with_name("gpt_sudoku16_histories"),
        help="保存 16x16 会话记录的目录，默认与脚本同级的 gpt_sudoku16_histories",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="在查询前清除历史对话记录目录",
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

