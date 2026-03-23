"""
世界状态管理模块。

维护谋杀湾世界中所有可变数据：
- 玩家属性
- NPC 状态与身份
- 区域状态
- 物品系统
- 历史记录与事件日志
"""

import json
import os
import copy
from datetime import datetime
from typing import Any

# ──────────────────────────────────────────────
# 默认初始世界状态
# ──────────────────────────────────────────────

DEFAULT_WORLD: dict = {
    "player": {
        "hp": 100,
        "san": 80,
        "gold": 50,
        "visibility": 0,          # 在地下世界的暴露程度
        "guilt": 0,               # 道德愧疚值
        "hidden_attention": 0,    # 被未知力量关注程度
        "memories_lost": 0,       # 被夺取的记忆数量
        "infected_memes": [],     # 植入的精神污染概念
        "lore": 0,                # 禁忌知识量
        "profile_in_underworld": 0,
        "unease": 0,
        "inventory": [],          # 物品列表
        "traits": [],             # 特质列表
    },
    "npcs": {
        # npc_id -> 属性字典，动态生成，无需预定义
    },
    "zones": {
        # zone_id -> 区域属性
        "harbor": {"tension": 10, "public_order": 60, "supernatural_activity": 0},
        "slums":  {"tension": 20, "public_order": 30, "supernatural_activity": 5},
        "market": {"tension": 5,  "public_order": 70, "supernatural_activity": 0},
    },
    "world_flags": {
        "chaos_index": 10,
        "mystery_level": 0,
        "rebellion_chance": 0,
        "supernatural_activity": 0,
        "corruption_network_alerted": False,
        "plot_twist_readiness": 0,
        "faction_abstraction_level": 0,
        "player_marked_by_deep": 0,
        "haunting_attention": 0,
    },
    "items": {
        # item_id -> 属性与状态
    },
    "history_rewrites": [],       # rewrite_history() 的所有记录
    "active_quests": [],          # 活跃任务列表
    "event_log": [],              # 已触发的全局事件
    "rumors": [],                 # 散布的谣言记录
    "world_events": [],           # 触发的世界级事件
    "turn_count": 0,              # 当前回合数
    "session_start": "",          # 本次游戏开始时间
}


# ──────────────────────────────────────────────
# 全局世界状态实例（单例模式）
# ──────────────────────────────────────────────

_world: dict = {}


def init_world() -> None:
    """初始化世界状态为默认值（新游戏）。"""
    global _world
    _world = copy.deepcopy(DEFAULT_WORLD)
    _world["session_start"] = datetime.now().isoformat()


def get_world() -> dict:
    """获取当前世界状态字典的引用。"""
    if not _world:
        init_world()
    return _world


# ──────────────────────────────────────────────
# 存档 / 读档
# ──────────────────────────────────────────────

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saves")


def save_game(slot: str = "autosave") -> str:
    """保存当前世界状态到 JSON 文件，返回存档路径。"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{slot}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_world, f, ensure_ascii=False, indent=2)
    return path


def load_game(slot: str = "autosave") -> bool:
    """从 JSON 文件加载世界状态，返回是否成功。"""
    global _world
    path = os.path.join(SAVE_DIR, f"{slot}.json")
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        _world = json.load(f)
    return True


def list_saves() -> list[str]:
    """列出所有存档槽名称。"""
    if not os.path.exists(SAVE_DIR):
        return []
    return [f[:-5] for f in os.listdir(SAVE_DIR) if f.endswith(".json")]


# ──────────────────────────────────────────────
# 状态访问辅助函数
# ──────────────────────────────────────────────

def get_npc(npc_id: str) -> dict:
    """获取 NPC 属性字典，不存在则自动创建。"""
    world = get_world()
    if npc_id not in world["npcs"]:
        world["npcs"][npc_id] = {
            "identity": "unknown",
            "active": True,
            "disposition_towards_player": 0,
        }
    return world["npcs"][npc_id]


def get_player() -> dict:
    return get_world()["player"]


def get_zone(zone_id: str) -> dict:
    world = get_world()
    if zone_id not in world["zones"]:
        world["zones"][zone_id] = {"tension": 0, "public_order": 50}
    return world["zones"][zone_id]


def get_item(item_id: str) -> dict:
    world = get_world()
    if item_id not in world["items"]:
        world["items"][item_id] = {"effects": [], "active": True}
    return world["items"][item_id]


def increment_turn() -> int:
    world = get_world()
    world["turn_count"] += 1
    return world["turn_count"]


def build_status_summary() -> str:
    """构建面向 GM 视角的世界状态摘要文字。"""
    world = get_world()
    p = world["player"]

    lines = [
        f"═══ 谋杀湾·世界状态 [第 {world['turn_count']} 回合] ═══",
        f"玩家: HP={p['hp']}  SAN={p['san']}  金币={p['gold']}",
        f"      暴露度={p['visibility']}  关注度={p['hidden_attention']}  愧疚={p['guilt']}",
    ]

    if p["infected_memes"]:
        lines.append(f"      精神污染: {', '.join(p['infected_memes'])}")

    if p["inventory"]:
        lines.append(f"      背包: {', '.join(p['inventory'])}")

    active_npcs = {k: v for k, v in world["npcs"].items() if v.get("active", True)}
    if active_npcs:
        lines.append("NPC:")
        for nid, ndata in active_npcs.items():
            identity = ndata.get("identity", "unknown")
            threat = ndata.get("threat", 0)
            loyalty = ndata.get("loyalty", 0)
            tag = f"[{identity}]"
            if threat > 0:
                tag += f" 威胁={threat}"
            if loyalty > 0:
                tag += f" 忠诚={loyalty}"
            lines.append(f"  {nid}: {tag}")

    flags = world["world_flags"]
    lines.append(
        f"世界: 混沌={flags['chaos_index']}  神秘={flags['mystery_level']}"
        f"  超自然={flags['supernatural_activity']}"
    )

    if world["history_rewrites"]:
        lines.append(f"历史篡改记录 ({len(world['history_rewrites'])} 条):")
        for hw in world["history_rewrites"][-3:]:
            lines.append(f"  · {hw[:60]}{'...' if len(hw) > 60 else ''}")

    if world["active_quests"]:
        lines.append(f"活跃任务: {', '.join(world['active_quests'])}")

    return "\n".join(lines)
