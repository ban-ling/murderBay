"""
谋杀湾·迭代训练数据构建器
============================

从玩家行为数据库中提取高质量对局，转化为两种训练格式：
  1. SFT 数据   — 高质量对局直接加入训练集
  2. DPO 数据   — 利用玩家选择信号构造 chosen/rejected 偏好对

行为质量信号：
  ├─ 阅读时长（engagement_ms）      高 = 叙事吸引人
  ├─ 选项选择速度                   快 = 内容清晰/有代入感
  ├─ 会话长度（session_length）     长 = 整体体验好
  └─ 是否重试（is_retry）          True = 负反馈

DPO 构造策略：
  同一 instruction 的多条输出中：
    chosen   = 阅读时长最长 / 会话长度最长 的输出（玩家最投入）
    rejected = 阅读时长最短 / is_retry=True 的输出（玩家最不投入）

用法：
  # 查看当前数据库统计
  python scripts/build_iter_data.py --mode stats

  # 导出 SFT 数据（合并进现有训练集）
  python scripts/build_iter_data.py --mode sft --merge

  # 导出 DPO 偏好对
  python scripts/build_iter_data.py --mode dpo

  # 全量导出并重新运行完整流水线
  python scripts/build_iter_data.py --mode all --retrain
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "agent"))

from feedback_store import get_all_turns, get_stats, init_db

# ══════════════════════════════════════════════
# 质量过滤阈值
# ══════════════════════════════════════════════

# SFT：最低阅读时长（毫秒）
SFT_MIN_ENGAGEMENT_MS   = 3000   # 至少读了 3 秒
# SFT：最短会话（回合数）
SFT_MIN_SESSION_LENGTH  = 2
# DPO：同一 instruction 需要至少几条不同输出才构造偏好对
DPO_MIN_CANDIDATES      = 2
# DPO：chosen/rejected 阅读时长差异至少要有多少倍
DPO_ENGAGEMENT_RATIO    = 1.5

SYSTEM_PROMPT = (
    "你是「谋杀湾」世界的AI叙事引擎，代号「岁月史书」。"
    "你的职责是响应玩家的每一个行为指令，并按照以下固定流程输出：\n"
    "1. 在<think>标签内进行内部推理：评估NPC身份、决定世界状态变化；\n"
    "2. 在<tool_code>标签内输出工具调用代码：更新NPC属性、重写历史、触发事件；\n"
    "3. 标签结束后，直接输出面向玩家的叙事文本，风格冷酷、悬疑、充满不确定性。\n"
    "禁止解释你的推理过程，禁止打破第四堵墙，禁止输出与叙事无关的内容。"
)


# ══════════════════════════════════════════════
# 质量评分（用于排序和筛选）
# ══════════════════════════════════════════════

def quality_score(turn: Dict) -> float:
    """
    综合质量分：
      - 阅读时长权重 0.5（越长越好，但上限 60s）
      - 会话长度权重 0.3（越长越好，但上限 20 轮）
      - 重试惩罚    权重 -0.3
      - 有明确选择奖励        +0.1
    """
    eng  = min(turn.get("engagement_ms") or 0, 60_000) / 60_000
    sess = min(turn.get("session_length") or 0, 20) / 20
    retry = 1 if turn.get("is_retry") else 0
    chose = 0.1 if turn.get("player_choice") else 0

    return eng * 0.5 + sess * 0.3 - retry * 0.3 + chose


# ══════════════════════════════════════════════
# SFT 数据导出
# ══════════════════════════════════════════════

def build_sft(
    turns: List[Dict],
    min_engagement_ms: int = SFT_MIN_ENGAGEMENT_MS,
    min_session: int       = SFT_MIN_SESSION_LENGTH,
) -> List[Dict]:
    """
    筛选高质量 turn → SFT 格式。
    过滤规则：
      - 非重试
      - 阅读时长 >= min_engagement_ms（或无阅读记录但会话够长）
      - 会话长度 >= min_session
      - raw_output 包含完整三段式结构
    """
    import re
    think_re = re.compile(r"<think>.*?</think>", re.DOTALL)
    tool_re  = re.compile(r"<tool_code>.*?</tool_code>", re.DOTALL)

    results: List[Dict] = []
    for t in turns:
        if t.get("is_retry"):
            continue
        if t.get("session_length", 0) < min_session:
            continue
        eng = t.get("engagement_ms")
        if eng is not None and eng < min_engagement_ms:
            continue
        raw = t.get("raw_output", "")
        if not think_re.search(raw) or not tool_re.search(raw):
            continue
        results.append({
            "system":      SYSTEM_PROMPT,
            "instruction": t["instruction"],
            "output":      raw,
        })
    return results


# ══════════════════════════════════════════════
# DPO 数据构造
# ══════════════════════════════════════════════

def build_dpo(turns: List[Dict]) -> List[Dict]:
    """
    利用玩家行为信号构造 DPO 偏好对。

    分组策略：对同一 instruction 的多次游玩（不同会话或重试），
    按质量分排序，取最高分作为 chosen，最低分作为 rejected。
    """
    # 按 instruction 分组
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for t in turns:
        inst = t.get("instruction", "").strip()
        if inst:
            groups[inst].append(t)

    pairs: List[Dict] = []
    for inst, group in groups.items():
        if len(group) < DPO_MIN_CANDIDATES:
            continue

        # 按质量分排序
        group.sort(key=quality_score, reverse=True)
        best  = group[0]
        worst = group[-1]

        # 检查差异是否足够
        eng_best  = best.get("engagement_ms") or 0
        eng_worst = worst.get("engagement_ms") or 0

        # 需要两者质量分有明显差异
        score_best  = quality_score(best)
        score_worst = quality_score(worst)
        if score_best - score_worst < 0.15:
            continue

        # 不能是同一条输出
        if best.get("raw_output") == worst.get("raw_output"):
            continue

        pairs.append({
            "system":      SYSTEM_PROMPT,
            "instruction": inst,
            "chosen":      best["raw_output"],
            "rejected":    worst["raw_output"],
            "_meta": {
                "chosen_session":      best["session_id"],
                "chosen_engagement":   eng_best,
                "chosen_score":        round(score_best, 3),
                "rejected_session":    worst["session_id"],
                "rejected_engagement": eng_worst,
                "rejected_score":      round(score_worst, 3),
            },
        })

    return pairs


# ══════════════════════════════════════════════
# 与现有数据集合并
# ══════════════════════════════════════════════

def merge_with_existing(
    new_items: List[Dict],
    existing_path: str,
    output_path: str,
) -> int:
    """
    将新 SFT 数据与现有训练集合并，自动去重（按 instruction 精确匹配）。
    返回合并后总条数。
    """
    existing: List[Dict] = []
    ep = Path(existing_path)
    if ep.exists():
        with open(ep, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))

    existing_insts = {it.get("instruction", "") for it in existing}

    added = 0
    for item in new_items:
        if item.get("instruction", "") not in existing_insts:
            existing.append(item)
            existing_insts.add(item["instruction"])
            added += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in existing:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(existing)


# ══════════════════════════════════════════════
# 统计报告
# ══════════════════════════════════════════════

def print_stats() -> None:
    stats = get_stats()
    print("\n═══ 玩家行为数据库统计 ═══")
    print(f"  总会话数:         {stats['total_sessions']}")
    print(f"  总回合数:         {stats['total_turns']}")
    print(f"  有明确选择回合:   {stats['turns_with_choice']}")
    print(f"  平均阅读时长:     {stats['avg_engagement_ms']} ms")
    print(f"  平均会话长度:     {stats['avg_session_length']} 回合")
    print()


# ══════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════

def run(args: argparse.Namespace) -> None:
    os.chdir(ROOT)
    init_db()

    if args.mode == "stats":
        print_stats()
        return

    print_stats()
    all_turns = get_all_turns()
    print(f"[数据] 加载 {len(all_turns)} 条行为记录")

    if args.mode in ("sft", "all"):
        print("\n[SFT] 构建监督微调数据...")
        sft_items = build_sft(all_turns)
        print(f"  筛选出 {len(sft_items)} 条高质量样本")

        if len(sft_items) == 0:
            print("  (当前数据不足，需要更多玩家对局积累)")
        else:
            out_path = args.sft_output
            if args.merge:
                total = merge_with_existing(sft_items, "dataset/train.jsonl", out_path)
                print(f"  合并后训练集: {total} 条 → {out_path}")
            else:
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    for item in sft_items:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"  SFT 数据 → {out_path}")

    if args.mode in ("dpo", "all"):
        print("\n[DPO] 构建偏好对数据...")
        dpo_items = build_dpo(all_turns)
        print(f"  构造出 {len(dpo_items)} 对 chosen/rejected")

        if len(dpo_items) == 0:
            print("  (需要同一 instruction 出现 ≥2 次、质量分差异 ≥0.15)")
        else:
            # 去掉 _meta 字段再写出（_meta 只用于调试）
            clean_items = [{k: v for k, v in it.items() if k != "_meta"} for it in dpo_items]
            Path(args.dpo_output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.dpo_output, "w", encoding="utf-8") as f:
                for item in clean_items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  DPO 数据 → {args.dpo_output}")

            # 打印几个样例的元信息
            for pair in dpo_items[:3]:
                m = pair["_meta"]
                print(f"    · chosen(score={m['chosen_score']},eng={m['chosen_engagement']}ms)"
                      f" vs rejected(score={m['rejected_score']},eng={m['rejected_engagement']}ms)")

    # 触发重新训练
    if args.retrain and args.mode in ("sft", "all"):
        sft_path = args.sft_output
        print(f"\n[重训练] 使用 {sft_path} 启动 ms-swift SFT...")
        cmd = [
            "swift", "sft",
            "--model",       os.getenv("MURDER_BAY_BASE_MODEL", "/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct"),
            "--dataset",     sft_path,
            "--train_type",  "lora",
            "--lora_rank",   "16",
            "--lora_alpha",  "32",
            "--output_dir",  f"output/murder_bay_iter_{datetime.now().strftime('%m%d_%H%M')}",
            "--num_train_epochs", "3",
            "--per_device_train_batch_size", "2",
            "--gradient_accumulation_steps", "8",
        ]
        print("  命令:", " ".join(cmd))
        subprocess.run(cmd)


# ══════════════════════════════════════════════
# 命令行
# ══════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="谋杀湾·迭代训练数据构建器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["stats", "sft", "dpo", "all"], default="stats")
    parser.add_argument("--sft-output", default="dataset/iter_sft.jsonl",  help="SFT 输出路径")
    parser.add_argument("--dpo-output", default="dataset/iter_dpo.jsonl",  help="DPO 输出路径")
    parser.add_argument("--merge",   action="store_true", help="SFT 数据与现有 train.jsonl 合并")
    parser.add_argument("--retrain", action="store_true", help="导出后自动触发 swift sft 训练")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
