"""
工具函数库 —— 对应模型输出 <tool_code> 中调用的所有函数。

每个函数都直接修改全局世界状态，并在 event_log 中留下记录。
函数签名与训练数据中的调用格式完全一致。
"""

from __future__ import annotations
from typing import Any
from world_state import get_world, get_npc, get_player, get_zone, get_item


# ──────────────────────────────────────────────
# 内部辅助
# ──────────────────────────────────────────────

def _log(msg: str) -> None:
    get_world()["event_log"].append(msg)


def _add_delta(d: dict, key: str, delta: Any) -> None:
    """对字典中的数值字段执行加减操作，支持 int/float 和百分比字符串。"""
    if isinstance(delta, str):
        delta = delta.strip()
        if delta.endswith("%"):
            current = d.get(key, 0)
            delta = current * float(delta[:-1]) / 100
        else:
            delta = float(delta)
    current = d.get(key, 0)
    if isinstance(current, (int, float)):
        d[key] = current + delta
    else:
        d[key] = delta


# ══════════════════════════════════════════════
# NPC 操作
# ══════════════════════════════════════════════

def set_npc_identity(npc_id: str, identity: str) -> None:
    """设置 NPC 的隐藏身份（里身份）。"""
    npc = get_npc(npc_id)
    old = npc.get("identity", "unknown")
    npc["identity"] = identity
    _log(f"[身份变更] {npc_id}: {old} → {identity}")


def update_npc_stat(npc_id: str, stat: str, delta: Any) -> None:
    """更新 NPC（或 player）的某个属性值。"""
    if npc_id == "player":
        update_player_stat("player", stat, delta)
        return
    npc = get_npc(npc_id)
    _add_delta(npc, stat, delta)
    _log(f"[NPC更新] {npc_id}.{stat} += {delta} → {npc[stat]}")


def set_npc_trait(npc_id: str, trait: str) -> None:
    """为 NPC 添加特质标签。"""
    npc = get_npc(npc_id)
    traits = npc.setdefault("traits", [])
    if trait not in traits:
        traits.append(trait)
    _log(f"[特质] {npc_id} 获得特质: {trait}")


def set_npc_goal(npc_id: str, goal: str) -> None:
    """设置 NPC 的行动目标。"""
    npc = get_npc(npc_id)
    npc["goal"] = goal
    _log(f"[目标] {npc_id} 目标设为: {goal}")


def deactivate_npc(npc_id: str) -> None:
    """将 NPC 标记为不活跃（消失/死亡）。"""
    npc = get_npc(npc_id)
    npc["active"] = False
    _log(f"[NPC失活] {npc_id} 已从世界中移除")


def spawn_hidden_npc(npc_id: str, location: str) -> None:
    """在指定位置生成一个隐藏 NPC。"""
    npc = get_npc(npc_id)
    npc["location"] = location
    npc["hidden"] = True
    npc["active"] = True
    _log(f"[NPC生成] {npc_id} 在 {location} 隐藏出现")


# ══════════════════════════════════════════════
# 玩家操作
# ══════════════════════════════════════════════

def update_player_stat(player_id: str, stat: str, delta: Any) -> None:
    """更新玩家属性（player_id 参数仅兼容训练数据格式，始终操作当前玩家）。"""
    player = get_player()

    if stat == "infected_memes":
        # delta 可以是字符串列表或单个字符串
        memes = player.setdefault("infected_memes", [])
        if isinstance(delta, list):
            for m in delta:
                if m not in memes:
                    memes.append(m)
        elif isinstance(delta, str) and delta not in memes:
            memes.append(delta)
        _log(f"[精神污染] 玩家感染: {delta}")
        return

    if stat == "inventory":
        inv = player.setdefault("inventory", [])
        if isinstance(delta, str) and delta not in inv:
            inv.append(delta)
        return

    _add_delta(player, stat, delta)
    _log(f"[玩家更新] {stat} += {delta} → {player.get(stat)}")


# ══════════════════════════════════════════════
# 世界状态操作
# ══════════════════════════════════════════════

def update_world_stat(stat: str, delta: Any) -> None:
    """更新全局世界属性。"""
    flags = get_world()["world_flags"]
    _add_delta(flags, stat, delta)
    _log(f"[世界更新] {stat} += {delta} → {flags.get(stat)}")


def set_world_flag(flag: str, value: Any) -> None:
    """直接设置世界标志为指定值（布尔/字符串均可）。"""
    get_world()["world_flags"][flag] = value
    _log(f"[世界标志] {flag} = {value}")


def rewrite_history(new_fact: str) -> None:
    """岁月史书篡改历史记录。"""
    get_world()["history_rewrites"].append(new_fact)
    _log(f"[历史篡改] {new_fact[:60]}{'...' if len(new_fact) > 60 else ''}")


def trigger_world_event(event_id: str, *args, **kwargs) -> None:
    """触发一个全局事件（记录存档，可在游戏逻辑中响应）。"""
    event = {"id": event_id, "args": args, "kwargs": kwargs}
    get_world()["world_events"].append(event)
    _log(f"[世界事件] {event_id} 触发，参数: {args} {kwargs}")


# ══════════════════════════════════════════════
# 区域操作
# ══════════════════════════════════════════════

def update_zone_stat(zone_id: str, stat: str, delta: Any) -> None:
    """更新某个区域的属性（如 tension、public_order）。"""
    zone = get_zone(zone_id)
    _add_delta(zone, stat, delta)
    _log(f"[区域更新] {zone_id}.{stat} += {delta} → {zone.get(stat)}")


# ══════════════════════════════════════════════
# 物品操作
# ══════════════════════════════════════════════

def create_item(item_id: str, **kwargs) -> None:
    """创建一个新物品并放入世界（可通过 location='player' 直接给玩家）。"""
    item = get_item(item_id)
    item.update(kwargs)
    location = kwargs.get("location", "world")
    if location == "player" or location == "grave":
        player = get_player()
        inv = player.setdefault("inventory", [])
        if item_id not in inv:
            inv.append(item_id)
    _log(f"[物品创建] {item_id}，位置: {location}")


def add_item_effect(item_id: str, effect_type: str, effect_value: str, chance: Any = 1.0) -> None:
    """为物品添加隐藏效果。"""
    item = get_item(item_id)
    effects = item.setdefault("effects", [])
    effects.append({"type": effect_type, "value": effect_value, "chance": chance})
    _log(f"[物品效果] {item_id} 添加效果: {effect_type}={effect_value}")


def trigger_item_event(item_id: str, event_type: str) -> None:
    """触发物品相关的特殊事件。"""
    get_world()["event_log"].append(f"[物品事件] {item_id}: {event_type}")
    item = get_item(item_id)
    item.setdefault("triggered_events", []).append(event_type)


# ══════════════════════════════════════════════
# 任务与杂项
# ══════════════════════════════════════════════

def add_quest_seed(quest_id: str) -> None:
    """在任务列表中种下一个任务伏笔。"""
    quests = get_world()["active_quests"]
    if quest_id not in quests:
        quests.append(quest_id)
    _log(f"[任务] 新任务伏笔: {quest_id}")


def log_event(event_id: str) -> None:
    """记录一个具名事件到事件日志。"""
    get_world()["event_log"].append(f"[事件] {event_id}")


def log_rumor_content(content: str) -> None:
    """记录散布的谣言内容。"""
    get_world()["rumors"].append(content)
    _log(f"[谣言] 散布: {content[:40]}...")


# ══════════════════════════════════════════════
# 导出：供 tool_executor 使用的函数注册表
# ══════════════════════════════════════════════

TOOL_REGISTRY: dict[str, Any] = {
    # NPC
    "set_npc_identity": set_npc_identity,
    "update_npc_stat": update_npc_stat,
    "set_npc_trait": set_npc_trait,
    "set_npc_goal": set_npc_goal,
    "deactivate_npc": deactivate_npc,
    "spawn_hidden_npc": spawn_hidden_npc,
    # 玩家
    "update_player_stat": update_player_stat,
    # 世界
    "update_world_stat": update_world_stat,
    "set_world_flag": set_world_flag,
    "rewrite_history": rewrite_history,
    "trigger_world_event": trigger_world_event,
    # 区域
    "update_zone_stat": update_zone_stat,
    # 物品
    "create_item": create_item,
    "add_item_effect": add_item_effect,
    "trigger_item_event": trigger_item_event,
    # 任务/日志
    "add_quest_seed": add_quest_seed,
    "log_event": log_event,
    "log_rumor_content": log_rumor_content,
}
