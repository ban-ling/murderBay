"""
谋杀湾·数据处理流水线
=====================

功能模块：
  1. 结构完整性校验  — 验证 <think>/<tool_code>/叙事三段式格式
  2. 内容质量过滤    — 过滤过短、缺字段、含截断标记的样本
  3. 精确去重        — 基于 instruction 字段的完全匹配去重
  4. 模糊去重        — 基于字符 4-gram Jaccard 相似度的近似重复检测
  5. 动词语义分类    — 将 230+ 种动词归并为 8 大行为类别，用于分层采样
  6. 分层验证集划分  — 按行为类别分层采样，保证训练/验证集分布一致
  7. 数据质量报告    — 输出 JSON 格式的完整统计报告

用法：
  # 使用默认路径
  python scripts/data_pipeline.py

  # 自定义参数
  python scripts/data_pipeline.py \\
      --input  dataset/murder_bay_swift.jsonl \\
      --train  dataset/train.jsonl \\
      --val    dataset/val.jsonl \\
      --report dataset/data_report.json \\
      --val-ratio 0.1 \\
      --fuzzy-threshold 0.75
"""

from __future__ import annotations

import argparse
import json
import math
import re
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ══════════════════════════════════════════════
# 1. 动词语义分类表
# ══════════════════════════════════════════════

# 将训练数据中出现的所有动词归并为 8 大行为类别。
# 未命中的动词默认归入 OTHER 类。
VERB_TAXONOMY: dict[str, str] = {
    # ── 社交互动 ──
    "闲聊": "SOCIAL", "对话": "SOCIAL", "搭话": "SOCIAL", "交流": "SOCIAL",
    "谈判": "SOCIAL", "求情": "SOCIAL", "示好": "SOCIAL", "道谢": "SOCIAL",
    "安抚": "SOCIAL", "恭维": "SOCIAL", "挑衅": "SOCIAL", "辱骂": "SOCIAL",
    "斥责": "SOCIAL", "敷衍": "SOCIAL", "嘲讽": "SOCIAL",
    # ── 信息获取 ──
    "询问": "INFO", "打听": "INFO", "调查": "INFO", "审问": "INFO",
    "侦查": "INFO", "观察": "INFO", "监视": "INFO", "窃听": "INFO",
    "查阅": "INFO", "阅读": "INFO", "分析": "INFO", "检查": "INFO",
    "探查": "INFO", "探索": "INFO", "追查": "INFO", "追踪": "INFO",
    "寻找": "INFO", "搜寻": "INFO", "盘问": "INFO",
    # ── 交易/经济 ──
    "购买": "TRADE", "交易": "TRADE", "贿赂": "TRADE", "施舍": "TRADE",
    "出售": "TRADE", "拍卖": "TRADE", "兑换": "TRADE", "赠礼": "TRADE",
    "偷窃": "TRADE", "抢劫": "TRADE", "勒索": "TRADE",
    # ── 任务/委托 ──
    "委托": "TASK", "发布任务": "TASK", "雇佣": "TASK", "招募": "TASK",
    "合作": "TASK", "谋划": "TASK", "部署": "TASK", "分配": "TASK",
    "解雇": "TASK", "背叛": "TASK",
    # ── 暴力/对抗 ──
    "攻击": "VIOLENCE", "战斗": "VIOLENCE", "击杀": "VIOLENCE", "暗杀": "VIOLENCE",
    "审讯": "VIOLENCE", "拷打": "VIOLENCE", "威胁": "VIOLENCE", "胁迫": "VIOLENCE",
    "恐吓": "VIOLENCE", "惩罚": "VIOLENCE", "对抗": "VIOLENCE", "突袭": "VIOLENCE",
    "压制": "VIOLENCE", "格斗": "VIOLENCE",
    # ── 指挥/命令 ──
    "命令": "COMMAND", "下令": "COMMAND", "指挥": "COMMAND", "召唤": "COMMAND",
    "调遣": "COMMAND", "禁止": "COMMAND", "驱逐": "COMMAND", "处决": "COMMAND",
    "审判": "COMMAND", "判决": "COMMAND",
    # ── 求助/合作 ──
    "求助": "COOP", "求救": "COOP", "救助": "COOP", "保护": "COOP",
    "帮助": "COOP", "支援": "COOP", "治疗": "COOP", "埋葬": "COOP",
    "拯救": "COOP", "援救": "COOP",
}

OTHER_CATEGORY = "OTHER"

_KNOWN_VERBS = set(VERB_TAXONOMY.keys())


def classify_verb(instruction: str) -> str:
    """从 instruction 提取动词并归类。"""
    m = re.match(r"\[([^\]]+)\]", instruction)
    if not m:
        return OTHER_CATEGORY
    verb = m.group(1).strip()
    return VERB_TAXONOMY.get(verb, OTHER_CATEGORY)


def extract_verb(instruction: str) -> str:
    m = re.match(r"\[([^\]]+)\]", instruction)
    return m.group(1).strip() if m else ""


# ══════════════════════════════════════════════
# 2. 结构完整性校验
# ══════════════════════════════════════════════

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CODE_RE = re.compile(r"<tool_code>(.*?)</tool_code>", re.DOTALL)

# 含截断/无意义特殊标记（这些是数据采集时的噪声）
_NOISE_TOKENS = re.compile(r"<\|(?!im_start|im_end)[^|]+\|>")

# 最短叙事文本长度（标签外的纯叙事部分）
MIN_NARRATIVE_LEN = 30
# 输出总长度上下限
MIN_OUTPUT_LEN = 150
MAX_OUTPUT_LEN = 2000


class QualityFlag:
    OK = "ok"
    NO_THINK = "no_think"
    NO_TOOL_CODE = "no_tool_code"
    NARRATIVE_TOO_SHORT = "narrative_too_short"
    OUTPUT_TOO_SHORT = "output_too_short"
    OUTPUT_TOO_LONG = "output_too_long"
    MISSING_INSTRUCTION = "missing_instruction"
    EMPTY_TOOL_CODE = "empty_tool_code"
    NOISE_TOKEN = "noise_token"


def check_quality(item: dict) -> list[str]:
    """
    对单条样本进行质量校验，返回问题标记列表。
    空列表表示通过全部校验。
    """
    flags: list[str] = []
    instruction = item.get("instruction", "").strip()
    output = item.get("output", "").strip()

    if not instruction:
        flags.append(QualityFlag.MISSING_INSTRUCTION)
        return flags

    if len(output) < MIN_OUTPUT_LEN:
        flags.append(QualityFlag.OUTPUT_TOO_SHORT)
    if len(output) > MAX_OUTPUT_LEN:
        flags.append(QualityFlag.OUTPUT_TOO_LONG)

    think_match = _THINK_RE.search(output)
    if not think_match:
        flags.append(QualityFlag.NO_THINK)

    tool_match = _TOOL_CODE_RE.search(output)
    if not tool_match:
        flags.append(QualityFlag.NO_TOOL_CODE)
    elif not tool_match.group(1).strip():
        flags.append(QualityFlag.EMPTY_TOOL_CODE)

    # 计算纯叙事部分长度（去掉两个标签块）
    narrative = _THINK_RE.sub("", output)
    narrative = _TOOL_CODE_RE.sub("", narrative).strip()
    if len(narrative) < MIN_NARRATIVE_LEN:
        flags.append(QualityFlag.NARRATIVE_TOO_SHORT)

    # 噪声 token 检测（如 <|item_hint|> 等训练数据遗留标记）
    if _NOISE_TOKEN_PRESENT(output):
        flags.append(QualityFlag.NOISE_TOKEN)

    return flags


def _NOISE_TOKEN_PRESENT(text: str) -> bool:
    return bool(_NOISE_TOKENS.search(text))


def clean_noise_tokens(output: str) -> str:
    """清除噪声特殊标记，保留文本内容。"""
    return _NOISE_TOKENS.sub("", output).strip()


# ══════════════════════════════════════════════
# 3. 去重：精确 + 模糊
# ══════════════════════════════════════════════

def _char_ngrams(text: str, n: int = 4) -> set[str]:
    """生成字符 n-gram 集合。"""
    text = re.sub(r"\s+", " ", text).lower()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def deduplicate(
    items: list[dict],
    fuzzy_threshold: float = 0.75,
    ngram_size: int = 4,
) -> tuple[list[dict], dict]:
    """
    两阶段去重：
    1. 精确去重：对 instruction 字段做完全匹配
    2. 模糊去重：对 instruction 字段做字符 n-gram Jaccard 相似度比较，
                 超过阈值的后出现样本被丢弃

    返回：(去重后列表, 统计信息)
    """
    stats: dict = {
        "before": len(items),
        "exact_removed": 0,
        "fuzzy_removed": 0,
        "after": 0,
    }

    # ── 阶段1：精确去重 ──
    seen_instructions: set[str] = set()
    after_exact: list[dict] = []
    for item in items:
        key = item.get("instruction", "").strip()
        if key in seen_instructions:
            stats["exact_removed"] += 1
        else:
            seen_instructions.add(key)
            after_exact.append(item)

    # ── 阶段2：模糊去重 ──
    # 为每个样本预计算 n-gram 集合
    ngram_sets: list[set[str]] = [
        _char_ngrams(it["instruction"], ngram_size) for it in after_exact
    ]

    keep_mask = [True] * len(after_exact)
    fuzzy_removed = 0

    for i in range(len(after_exact)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(after_exact)):
            if not keep_mask[j]:
                continue
            sim = jaccard(ngram_sets[i], ngram_sets[j])
            if sim >= fuzzy_threshold:
                keep_mask[j] = False
                fuzzy_removed += 1

    after_fuzzy = [it for it, keep in zip(after_exact, keep_mask) if keep]
    stats["fuzzy_removed"] = fuzzy_removed
    stats["after"] = len(after_fuzzy)

    return after_fuzzy, stats


# ══════════════════════════════════════════════
# 4. 分层验证集划分
# ══════════════════════════════════════════════

def stratified_split(
    items: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    按动词行为类别（8类）分层采样验证集。

    策略：
    - 每个类别按 val_ratio 比例抽取，至少保留 1 条进验证集
    - OTHER 类别（长尾稀有动词）整体按同比例采样
    - 类别样本数过少（≤3）时全部归入训练集

    返回：(train_items, val_items)
    """
    rng = random.Random(seed)

    # 按类别分组
    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        cat = classify_verb(item["instruction"])
        buckets[cat].append(item)

    train_list: list[dict] = []
    val_list: list[dict] = []

    for cat, bucket in sorted(buckets.items()):
        rng.shuffle(bucket)
        n = len(bucket)
        if n <= 3:
            # 样本太少，全给训练集
            train_list.extend(bucket)
            continue
        n_val = max(1, math.floor(n * val_ratio))
        val_list.extend(bucket[:n_val])
        train_list.extend(bucket[n_val:])

    # 最终再打乱
    rng.shuffle(train_list)
    rng.shuffle(val_list)

    return train_list, val_list


# ══════════════════════════════════════════════
# 5. 数据质量报告
# ══════════════════════════════════════════════

def generate_report(
    raw_items: list[dict],
    cleaned_items: list[dict],
    dedup_stats: dict,
    train_items: list[dict],
    val_items: list[dict],
    noise_fixed: int,
) -> dict:
    """生成完整的数据质量报告（JSON 可序列化）。"""

    def _verb_dist(items: list[dict]) -> dict[str, int]:
        return dict(Counter(extract_verb(it["instruction"]) for it in items).most_common(30))

    def _cat_dist(items: list[dict]) -> dict[str, int]:
        return dict(Counter(classify_verb(it["instruction"]) for it in items))

    def _len_stats(items: list[dict], field: str = "output") -> dict:
        lengths = [len(it.get(field, "")) for it in items]
        if not lengths:
            return {}
        lengths.sort()
        n = len(lengths)
        return {
            "count": n,
            "min": lengths[0],
            "max": lengths[-1],
            "mean": round(sum(lengths) / n, 1),
            "median": lengths[n // 2],
            "p10": lengths[max(0, n // 10)],
            "p90": lengths[min(n - 1, 9 * n // 10)],
        }

    def _tool_func_dist(items: list[dict]) -> dict[str, int]:
        funcs: list[str] = []
        for it in items:
            tc = _TOOL_CODE_RE.search(it.get("output", ""))
            if tc:
                funcs.extend(re.findall(r"(\w+)\(", tc.group(1)))
        return dict(Counter(funcs).most_common(20))

    train_cat = _cat_dist(train_items)
    val_cat = _cat_dist(val_items)
    all_cats = set(train_cat) | set(val_cat)
    cat_balance = {
        c: {
            "train": train_cat.get(c, 0),
            "val": val_cat.get(c, 0),
            "val_ratio": round(
                val_cat.get(c, 0) / (train_cat.get(c, 0) + val_cat.get(c, 0)), 3
            ) if (train_cat.get(c, 0) + val_cat.get(c, 0)) > 0 else 0,
        }
        for c in sorted(all_cats)
    }

    return {
        "pipeline_summary": {
            "raw_total": len(raw_items),
            "quality_filter_removed": len(raw_items) - len(cleaned_items) - dedup_stats["exact_removed"] - dedup_stats["fuzzy_removed"],
            "noise_token_fixed": noise_fixed,
            "exact_dedup_removed": dedup_stats["exact_removed"],
            "fuzzy_dedup_removed": dedup_stats["fuzzy_removed"],
            "after_dedup": dedup_stats["after"],
            "train_size": len(train_items),
            "val_size": len(val_items),
        },
        "verb_distribution_top30": _verb_dist(train_items + val_items),
        "category_distribution": _cat_dist(train_items + val_items),
        "train_val_category_balance": cat_balance,
        "output_length_stats": _len_stats(train_items + val_items, "output"),
        "instruction_length_stats": _len_stats(train_items + val_items, "instruction"),
        "tool_function_distribution": _tool_func_dist(train_items + val_items),
    }


# ══════════════════════════════════════════════
# 6. 主流程
# ══════════════════════════════════════════════

def run_pipeline(
    input_path: str,
    train_path: str,
    val_path: str,
    report_path: str,
    val_ratio: float,
    fuzzy_threshold: float,
    seed: int,
) -> None:
    print("=" * 60)
    print("谋杀湾·数据处理流水线")
    print("=" * 60)

    # ── 读取原始数据 ──
    print(f"\n[1/6] 读取数据: {input_path}")
    raw_items: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [警告] 第 {i} 行 JSON 解析失败: {e}")
    print(f"  读取 {len(raw_items)} 条原始数据")

    # ── 质量校验 + 清洗 ──
    print("\n[2/6] 质量校验与清洗")
    flag_counter: Counter = Counter()
    cleaned_items: list[dict] = []
    noise_fixed = 0

    for item in raw_items:
        flags = check_quality(item)

        # 噪声 token 可自动修复，不计入过滤
        if QualityFlag.NOISE_TOKEN in flags:
            item = dict(item)
            item["output"] = clean_noise_tokens(item["output"])
            flags.remove(QualityFlag.NOISE_TOKEN)
            noise_fixed += 1

        fatal_flags = [f for f in flags if f != QualityFlag.NOISE_TOKEN]

        for f in flags:
            flag_counter[f] += 1

        if not fatal_flags:
            cleaned_items.append(item)

    removed_by_quality = len(raw_items) - len(cleaned_items)
    print(f"  噪声标记自动修复: {noise_fixed} 条")
    print(f"  质量过滤移除:      {removed_by_quality} 条")
    if flag_counter:
        for flag, count in flag_counter.most_common():
            print(f"    · {flag}: {count} 条")
    print(f"  清洗后保留:        {len(cleaned_items)} 条")

    # ── 去重 ──
    print(f"\n[3/6] 去重 (模糊阈值={fuzzy_threshold})")
    deduped_items, dedup_stats = deduplicate(
        cleaned_items,
        fuzzy_threshold=fuzzy_threshold,
    )
    print(f"  精确重复移除: {dedup_stats['exact_removed']} 条")
    print(f"  模糊重复移除: {dedup_stats['fuzzy_removed']} 条")
    print(f"  去重后保留:   {dedup_stats['after']} 条")

    # ── 动词分布分析 ──
    print("\n[4/6] 动词行为类别分布")
    cat_dist = Counter(classify_verb(it["instruction"]) for it in deduped_items)
    for cat, count in sorted(cat_dist.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 5)
        print(f"  {cat:<12} {count:>4}  {bar}")

    # ── 分层划分 ──
    print(f"\n[5/6] 分层划分 (val_ratio={val_ratio}, seed={seed})")
    train_items, val_items = stratified_split(
        deduped_items,
        val_ratio=val_ratio,
        seed=seed,
    )
    print(f"  训练集: {len(train_items)} 条")
    print(f"  验证集: {len(val_items)} 条")
    actual_ratio = len(val_items) / (len(train_items) + len(val_items))
    print(f"  实际验证比例: {actual_ratio:.3f}")

    # ── 输出文件 ──
    print("\n[6/6] 写出文件")
    for path, items, label in [
        (train_path, train_items, "训练集"),
        (val_path, val_items, "验证集"),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {label} → {path} ({len(items)} 条)")

    # ── 质量报告 ──
    report = generate_report(
        raw_items=raw_items,
        cleaned_items=cleaned_items,
        dedup_stats=dedup_stats,
        train_items=train_items,
        val_items=val_items,
        noise_fixed=noise_fixed,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  质量报告 → {report_path}")

    # ── 最终摘要 ──
    print("\n" + "=" * 60)
    s = report["pipeline_summary"]
    print(f"  原始数据:   {s['raw_total']:>5} 条")
    print(f"  质量过滤:  -{s['quality_filter_removed']:>4} 条")
    print(f"  噪声修复:  +{s['noise_token_fixed']:>4} 条（自动修复，不过滤）")
    print(f"  精确去重:  -{s['exact_dedup_removed']:>4} 条")
    print(f"  模糊去重:  -{s['fuzzy_dedup_removed']:>4} 条")
    print(f"  ─────────────────")
    print(f"  最终数据:   {s['after_dedup']:>5} 条  (训练 {s['train_size']} / 验证 {s['val_size']})")
    print("=" * 60)


# ══════════════════════════════════════════════
# 7. 命令行入口
# ══════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="谋杀湾数据处理流水线：清洗·去重·分层划分·报告",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  default="dataset/murder_bay_swift.jsonl", help="输入 JSONL 路径")
    parser.add_argument("--train",  default="dataset/train.jsonl",            help="训练集输出路径")
    parser.add_argument("--val",    default="dataset/val.jsonl",              help="验证集输出路径")
    parser.add_argument("--report", default="dataset/data_report.json",       help="质量报告输出路径")
    parser.add_argument("--val-ratio",        type=float, default=0.10,  help="验证集比例")
    parser.add_argument("--fuzzy-threshold",  type=float, default=0.75,  help="模糊去重 Jaccard 阈值")
    parser.add_argument("--seed",             type=int,   default=42,    help="随机种子")
    args = parser.parse_args()

    # 兼容从项目根或 scripts/ 目录运行
    from pathlib import Path as _P
    import os
    root = _P(__file__).parent.parent
    os.chdir(root)

    run_pipeline(
        input_path=args.input,
        train_path=args.train,
        val_path=args.val,
        report_path=args.report,
        val_ratio=args.val_ratio,
        fuzzy_threshold=args.fuzzy_threshold,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
