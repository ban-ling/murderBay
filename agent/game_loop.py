"""
谋杀湾·叙事引擎主程序入口。

用法：
    python game_loop.py
    python game_loop.py --backend api
    python game_loop.py --load autosave
    python game_loop.py --base-model /path/to/model --lora /path/to/lora

内置指令（游戏中输入）：
    /status   — 显示当前玩家与世界状态（GM视角）
    /gm       — 切换 GM 视角模式（显示/隐藏 <think> 内容）
    /save     — 保存进度
    /load     — 加载上次自动存档
    /saves    — 列出所有存档
    /history  — 显示本局事件日志
    /quit     — 退出游戏
    /help     — 显示帮助
"""

import argparse
import sys
import os

# 确保从 agent/ 目录下运行时能正确找到同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import world_state as ws
from prompt_builder import build_messages
from tool_executor import parse_and_execute
from model_client import get_client


# ──────────────────────────────────────────────
# 终端渲染辅助
# ──────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[96m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
GREEN   = "\033[92m"
MAGENTA = "\033[95m"
GRAY    = "\033[90m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def _print_divider(char: str = "─", width: int = 60) -> None:
    print(_c(char * width, DIM))


def _print_banner() -> None:
    banner = r"""
  ███╗   ███╗██╗   ██╗██████╗ ██████╗ ███████╗██████╗
  ████╗ ████║██║   ██║██╔══██╗██╔══██╗██╔════╝██╔══██╗
  ██╔████╔██║██║   ██║██████╔╝██║  ██║█████╗  ██████╔╝
  ██║╚██╔╝██║██║   ██║██╔══██╗██║  ██║██╔══╝  ██╔══██╗
  ██║ ╚═╝ ██║╚██████╔╝██║  ██║██████╔╝███████╗██║  ██║
  ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝
           B A Y  ·  谋 杀 湾  ·  岁 月 史 书
"""
    print(_c(banner, CYAN))
    print(_c("  输入 /help 查看指令列表。每一个选择，都将被永久铭记。\n", DIM))


def _print_narrative(text: str) -> None:
    """格式化叙事文本输出。"""
    _print_divider()
    print()
    # 高亮【选项】格式
    for line in text.splitlines():
        if line.strip().startswith(("1.", "2.", "3.", "4.")):
            print(_c(f"  {line.strip()}", YELLOW))
        elif line.strip().startswith("【") or line.strip().startswith("<|"):
            print(_c(f"  {line.strip()}", YELLOW))
        else:
            print(f"  {line}")
    print()


def _print_think(text: str) -> None:
    """GM 视角：显示模型推理内容。"""
    _print_divider("·")
    print(_c("  ◆ GM视角 / 岁月史书推理", MAGENTA))
    for line in text.splitlines():
        print(_c(f"    {line}", GRAY))
    print()


def _print_tool(code: str) -> None:
    """GM 视角：显示工具调用代码。"""
    print(_c("  ◆ 工具调用", MAGENTA))
    for line in code.splitlines():
        print(_c(f"    {line}", GRAY))
    _print_divider("·")
    print()


def _print_errors(errors: list[str]) -> None:
    for err in errors:
        print(_c(f"  [执行错误] {err}", RED))


def _print_status() -> None:
    print()
    print(_c(ws.build_status_summary(), CYAN))
    print()


def _print_event_log() -> None:
    world = ws.get_world()
    log = world["event_log"]
    if not log:
        print(_c("  （本局暂无事件记录）", DIM))
        return
    print(_c(f"\n  ═══ 事件日志（共 {len(log)} 条）═══", CYAN))
    for entry in log[-20:]:
        print(_c(f"    · {entry}", GRAY))
    print()


# ──────────────────────────────────────────────
# 内置指令处理
# ──────────────────────────────────────────────

HELP_TEXT = """
  /status   显示玩家与世界状态（GM视角）
  /gm       切换 GM 视角（显示/隐藏推理过程和工具调用）
  /save     保存当前进度到 autosave
  /save <槽名>  保存到指定槽
  /load     加载 autosave
  /load <槽名>  加载指定存档
  /saves    列出所有存档
  /history  显示本局事件日志
  /quit     退出游戏
  /help     显示此帮助
"""


def _handle_command(cmd: str, gm_mode: bool) -> tuple[bool, bool]:
    """
    处理内置指令。
    返回 (是否为内置指令, 新的 gm_mode)
    """
    parts = cmd.strip().split(maxsplit=1)
    verb = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else "autosave"

    if verb == "/help":
        print(HELP_TEXT)
        return True, gm_mode

    if verb == "/status":
        _print_status()
        return True, gm_mode

    if verb == "/gm":
        gm_mode = not gm_mode
        state = "开启" if gm_mode else "关闭"
        print(_c(f"\n  [GM视角已{state}]\n", MAGENTA))
        return True, gm_mode

    if verb == "/save":
        path = ws.save_game(arg)
        print(_c(f"\n  [存档成功] → {path}\n", GREEN))
        return True, gm_mode

    if verb == "/load":
        ok = ws.load_game(arg)
        if ok:
            print(_c(f"\n  [读档成功] 槽: {arg}\n", GREEN))
        else:
            print(_c(f"\n  [读档失败] 找不到存档: {arg}\n", RED))
        return True, gm_mode

    if verb == "/saves":
        saves = ws.list_saves()
        if saves:
            print(_c(f"\n  存档列表: {', '.join(saves)}\n", CYAN))
        else:
            print(_c("\n  （暂无存档）\n", DIM))
        return True, gm_mode

    if verb == "/history":
        _print_event_log()
        return True, gm_mode

    if verb in ("/quit", "/exit", "/q"):
        print(_c("\n  「历史永远不会遗忘你的选择。」\n", DIM))
        sys.exit(0)

    return False, gm_mode


# ──────────────────────────────────────────────
# 主游戏循环
# ──────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    _print_banner()

    # 初始化世界状态
    if args.load:
        loaded = ws.load_game(args.load)
        if loaded:
            print(_c(f"[存档已加载] {args.load}\n", GREEN))
        else:
            print(_c(f"[未找到存档 '{args.load}'，开始新游戏]\n", YELLOW))
            ws.init_world()
    else:
        ws.init_world()

    # 加载模型
    client = get_client(
        backend=args.backend,
        base_model=args.base_model,
        lora_adapter=args.lora,
    )

    history: list[dict] = []
    gm_mode: bool = args.gm

    print(_c("谋杀湾。一座被遗忘的港口城市，每一块石板下都埋着秘密。", CYAN))
    print(_c("你，带着一个没有过去的名字，踏上了这片潮湿的土地。\n", CYAN))

    turn = 0
    while True:
        turn += 1
        ws.increment_turn()

        # 每10回合自动存档
        if turn % 10 == 0:
            ws.save_game("autosave")
            print(_c("  [自动存档]\n", DIM))

        # 获取输入
        try:
            raw_input = input(_c(f"[第{ws.get_world()['turn_count']}回合] 你的行动 > ", BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            ws.save_game("autosave")
            print(_c("\n  [已自动存档并退出]\n", DIM))
            break

        if not raw_input:
            continue

        # 内置指令
        if raw_input.startswith("/"):
            is_cmd, gm_mode = _handle_command(raw_input, gm_mode)
            if is_cmd:
                continue

        # 构建消息并调用模型
        messages = build_messages(history, raw_input)

        print(_c("\n  [岁月史书运转中...]\n", DIM))

        try:
            raw_output = client.chat(messages)
        except Exception as e:
            print(_c(f"\n  [模型调用失败] {e}\n", RED))
            continue

        # 解析输出并执行工具调用
        result = parse_and_execute(raw_output)

        # GM 视角
        if gm_mode:
            if result.think:
                _print_think(result.think)
            if result.tool_code:
                _print_tool(result.tool_code)

        # 执行错误提示（GM模式下显示）
        if result.errors and gm_mode:
            _print_errors(result.errors)

        # 输出叙事文本
        _print_narrative(result.narrative)

        # 更新历史（存储原始完整输出，下轮上下文用）
        history.append({"role": "user", "content": raw_input})
        history.append({"role": "assistant", "content": raw_output})

        # SAN 值归零提示
        player = ws.get_player()
        if player.get("san", 80) <= 0:
            print(_c(
                "\n  ════════════════════════════════════════\n"
                "  你的理性已彻底崩溃。\n"
                "  谋杀湾将你吞噬，连同你的名字。\n"
                "  ════════════════════════════════════════\n",
                RED
            ))
            ws.save_game("final_" + str(ws.get_world()["turn_count"]))
            break

        # HP 归零提示
        if player.get("hp", 100) <= 0:
            print(_c(
                "\n  ════════════════════════════════════════\n"
                "  你死了。\n"
                "  但岁月史书会将这一刻永远铭记。\n"
                "  ════════════════════════════════════════\n",
                RED
            ))
            ws.save_game("death_" + str(ws.get_world()["turn_count"]))
            break


# ──────────────────────────────────────────────
# 命令行参数解析
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="谋杀湾·叙事 Agent 引擎",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=["local", "api"],
        default="local",
        help="推理后端：local=本地模型，api=HTTP API",
    )
    parser.add_argument(
        "--base-model",
        default="/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct",
        help="基座模型路径（local 模式）",
    )
    parser.add_argument(
        "--lora",
        default="/root/output/murder_bay_lora/v2-20260322-002516/checkpoint-207",
        help="LoRA 适配器路径（local 模式，留空则使用纯基座）",
    )
    parser.add_argument(
        "--load",
        default=None,
        metavar="SLOT",
        help="启动时加载指定存档槽（如 autosave）",
    )
    parser.add_argument(
        "--gm",
        action="store_true",
        default=False,
        help="启动时开启 GM 视角（显示推理和工具调用）",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
