"""
System Prompt 构建模块。

每一轮对话前，将当前世界状态摘要注入 system prompt，
让模型能够"记住"之前所有行动的后果。
"""

from world_state import get_world, get_player

# 与训练数据完全一致的基础 system prompt
_BASE_SYSTEM = (
    "你是「谋杀湾」世界的AI叙事引擎，代号「岁月史书」。"
    "你的职责是响应玩家的每一个行为指令，并按照以下固定流程输出：\n"
    "1. 在<think>标签内进行内部推理：评估NPC身份、决定世界状态变化；\n"
    "2. 在<tool_code>标签内输出工具调用代码：更新NPC属性、重写历史、触发事件；\n"
    "3. 标签结束后，直接输出面向玩家的叙事文本，风格冷酷、悬疑、充满不确定性。\n"
    "禁止解释你的推理过程，禁止打破第四堵墙，禁止输出与叙事无关的内容。"
)

# 历史对话最多保留的轮数（user+assistant 各算一条）
MAX_HISTORY_TURNS = 8


def _build_world_state_section() -> str:
    """将当前世界状态格式化为可注入 system prompt 的文本段。"""
    world = get_world()
    p = get_player()
    lines: list[str] = []

    # 玩家状态
    player_line = (
        f"[玩家状态] HP={p.get('hp', 100)} SAN={p.get('san', 80)} "
        f"金币={p.get('gold', 50)} 暴露度={p.get('visibility', 0)} "
        f"隐藏关注={p.get('hidden_attention', 0)}"
    )
    if p.get("infected_memes"):
        player_line += f" 精神污染={','.join(p['infected_memes'][-3:])}"
    if p.get("inventory"):
        player_line += f" 背包=[{','.join(p['inventory'][-5:])}]"
    lines.append(player_line)

    # 活跃 NPC（只显示有意义身份的）
    active_npcs = {
        k: v for k, v in world["npcs"].items()
        if v.get("active", True) and v.get("identity", "unknown") != "unknown"
    }
    if active_npcs:
        npc_parts = []
        for nid, ndata in list(active_npcs.items())[-8:]:   # 最多显示8个
            identity = ndata.get("identity", "?")
            threat = ndata.get("threat", 0)
            loyalty = ndata.get("loyalty", 0)
            hatred = ndata.get("hatred_towards_player", ndata.get("grudge_against_player", 0))
            tag = f"{nid}[{identity}]"
            if threat > 50:
                tag += f"⚠威胁{int(threat)}"
            if loyalty > 60:
                tag += f"★忠诚{int(loyalty)}"
            if hatred > 60:
                tag += f"✗仇恨{int(hatred)}"
            npc_parts.append(tag)
        lines.append(f"[已知NPC] {'; '.join(npc_parts)}")

    # 世界旗标（只显示非零/非False的）
    flags = world["world_flags"]
    notable_flags = {
        k: v for k, v in flags.items()
        if v and v != 0 and v is not False
    }
    if notable_flags:
        flag_str = " ".join(f"{k}={v}" for k, v in list(notable_flags.items())[:6])
        lines.append(f"[世界状态] {flag_str}")

    # 最近的历史篡改（最多2条）
    if world["history_rewrites"]:
        hw_lines = []
        for hw in world["history_rewrites"][-2:]:
            hw_lines.append(f"「{hw[:50]}{'...' if len(hw) > 50 else ''}」")
        lines.append(f"[已篡改历史] {' / '.join(hw_lines)}")

    # 活跃任务
    if world["active_quests"]:
        lines.append(f"[活跃伏笔] {', '.join(world['active_quests'][-5:])}")

    # 世界事件
    if world["world_events"]:
        latest = world["world_events"][-1]
        lines.append(f"[最近世界事件] {latest['id']}")

    return "\n".join(lines)


def build_messages(
    history: list[dict],
    current_input: str,
) -> list[dict]:
    """
    构建完整的消息列表，注入当前世界状态到 system prompt。

    参数：
        history       - 历史对话（[{role, content}, ...]），不含 system 消息
        current_input - 当前玩家输入

    返回：
        符合 OpenAI 格式的 messages 列表
    """
    world_state_text = _build_world_state_section()
    system_content = _BASE_SYSTEM + "\n\n[当前世界状态]\n" + world_state_text

    # 截断历史，避免超长上下文
    trimmed_history = _trim_history(history, MAX_HISTORY_TURNS)

    messages = [{"role": "system", "content": system_content}]
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": current_input})

    return messages


def _trim_history(history: list[dict], max_turns: int) -> list[dict]:
    """
    保留最近 max_turns 轮的对话历史。
    一轮 = user 消息 + assistant 消息（共2条）。
    """
    max_messages = max_turns * 2
    if len(history) > max_messages:
        return history[-max_messages:]
    return history
