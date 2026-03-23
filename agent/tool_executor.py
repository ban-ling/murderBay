"""
<tool_code> 解析与安全执行模块。

从模型输出中提取 <tool_code> 标签内的代码，
在仅包含预注册工具函数的沙箱中执行，防止任意代码注入。
"""

import re
from typing import Optional
from tools import TOOL_REGISTRY


# ──────────────────────────────────────────────
# 解析
# ──────────────────────────────────────────────

_TOOL_CODE_RE = re.compile(r"<tool_code>(.*?)</tool_code>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_tool_code(raw_output: str) -> Optional[str]:
    """从模型原始输出中提取 <tool_code> 内的代码字符串。"""
    match = _TOOL_CODE_RE.search(raw_output)
    return match.group(1).strip() if match else None


def extract_think(raw_output: str) -> Optional[str]:
    """从模型原始输出中提取 <think> 内的推理文本。"""
    match = _THINK_RE.search(raw_output)
    return match.group(1).strip() if match else None


def extract_narrative(raw_output: str) -> str:
    """
    从模型原始输出中提取面向玩家的叙事文本。
    移除 <think>...</think> 和 <tool_code>...</tool_code> 两段。
    """
    text = _THINK_RE.sub("", raw_output)
    text = _TOOL_CODE_RE.sub("", text)
    return text.strip()


# ──────────────────────────────────────────────
# 安全执行
# ──────────────────────────────────────────────

# 沙箱命名空间：只暴露注册的工具函数，禁用所有内置函数
_SANDBOX_GLOBALS: dict = {
    "__builtins__": {},   # 禁用 import、open、eval 等危险内置
    **TOOL_REGISTRY,
}


def execute_tool_code(raw_output: str) -> list[str]:
    """
    从模型输出中提取并执行 <tool_code>。

    返回执行过程中产生的错误列表（空列表表示全部成功）。
    模型有时会输出多条以分号分隔的调用，逐条执行。
    """
    code = extract_tool_code(raw_output)
    if not code:
        return []

    errors: list[str] = []

    # 按分号或换行拆分为独立语句，逐条执行
    statements = [s.strip() for s in re.split(r";|\n", code) if s.strip()]

    for stmt in statements:
        try:
            exec(stmt, _SANDBOX_GLOBALS.copy())  # copy 防止状态污染
        except Exception as e:
            errors.append(f"执行失败 [{stmt[:60]}]: {e}")

    return errors


# ──────────────────────────────────────────────
# 一步式解析输出（供 game_loop 直接调用）
# ──────────────────────────────────────────────

class ParsedOutput:
    """封装模型输出的三个部分。"""
    __slots__ = ("think", "tool_code", "narrative", "errors")

    def __init__(
        self,
        think: Optional[str],
        tool_code: Optional[str],
        narrative: str,
        errors: list[str],
    ):
        self.think = think
        self.tool_code = tool_code
        self.narrative = narrative
        self.errors = errors


def parse_and_execute(raw_output: str) -> ParsedOutput:
    """
    一步完成：解析模型输出，执行工具调用，返回结构化结果。

    典型用法：
        result = parse_and_execute(model_response)
        print(result.narrative)   # 给玩家看
        if result.think:
            print(result.think)   # GM 视角
    """
    think = extract_think(raw_output)
    tool_code = extract_tool_code(raw_output)
    narrative = extract_narrative(raw_output)
    errors = execute_tool_code(raw_output)

    return ParsedOutput(
        think=think,
        tool_code=tool_code,
        narrative=narrative,
        errors=errors,
    )
