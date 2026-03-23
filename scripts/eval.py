"""
谋杀湾·模型评估脚本
====================

在验证集上量化评估微调效果，输出多维度对比报告。

评估维度：
  1. 结构有效率    — 输出是否包含完整的 <think>/<tool_code>/叙事三段式
  2. 工具调用有效率— <tool_code> 内的代码是否能被 Python 语法解析
  3. 风格一致性得分— 叙事部分是否包含悬疑/克苏鲁风格关键词
  4. ROUGE-L      — 与参考输出的最长公共子序列相似度（不依赖外部库）
  5. 输出长度分布  — 生成长度是否合理（避免过短/截断）

支持两种模式：
  --mode offline  仅统计验证集参考答案质量，无需加载模型（快速）
  --mode online   加载模型推理，对比微调前后指标（需要 GPU）

用法：
  # 离线模式（统计验证集本身）
  python scripts/eval.py --mode offline

  # 在线模式（评估微调后模型）
  python scripts/eval.py --mode online \\
      --base-model /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \\
      --lora /root/output/murder_bay_lora/v2-20260322-002516/checkpoint-207

  # 对比模式（微调前 vs 微调后）
  python scripts/eval.py --mode compare \\
      --base-model /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \\
      --lora /root/output/murder_bay_lora/v2-20260322-002516/checkpoint-207
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 确保从项目根运行
sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))


# ══════════════════════════════════════════════
# 风格关键词词典（谋杀湾·克苏鲁悬疑风格）
# ══════════════════════════════════════════════

STYLE_KEYWORDS = {
    # 感官描写
    "冰冷", "黏腻", "腥臭", "腐烂", "锈蚀", "潮湿", "刺骨", "阴冷",
    # 心理状态
    "不安", "恐惧", "惊恐", "疑惑", "迷惘", "战栗", "颤抖", "绝望",
    # 超自然
    "诡异", "扭曲", "幻觉", "虚无", "深渊", "旧神", "召唤", "预言",
    "符文", "仪式", "低语", "窃语", "阴影", "怨灵",
    # 叙事风格
    "意味深长", "莫名", "不自然", "令人不安", "无法解释",
    "死死盯着", "冷若冰霜", "空洞", "如刀", "无波澜",
    # 悬念收尾
    "也许", "或许", "……", "某种", "不知为何",
}

# 风格关键词覆盖比例达到该阈值视为"风格一致"
STYLE_THRESHOLD = 0.04  # 每百字 4 个关键词


# ══════════════════════════════════════════════
# 工具函数正则
# ══════════════════════════════════════════════

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CODE_RE = re.compile(r"<tool_code>(.*?)</tool_code>", re.DOTALL)


# ══════════════════════════════════════════════
# 评估指标计算
# ══════════════════════════════════════════════

def check_structure(output: str) -> Dict[str, bool]:
    """检验输出的三段式结构完整性。"""
    think_m = _THINK_RE.search(output)
    tool_m = _TOOL_CODE_RE.search(output)
    narrative = _THINK_RE.sub("", output)
    narrative = _TOOL_CODE_RE.sub("", narrative).strip()
    return {
        "has_think": bool(think_m and think_m.group(1).strip()),
        "has_tool_code": bool(tool_m and tool_m.group(1).strip()),
        "has_narrative": len(narrative) >= 30,
        "is_fully_valid": bool(
            think_m and think_m.group(1).strip()
            and tool_m and tool_m.group(1).strip()
            and len(narrative) >= 30
        ),
    }


def check_tool_syntax(output: str) -> Dict[str, object]:
    """检验 <tool_code> 内代码的 Python 语法合法性。"""
    tool_m = _TOOL_CODE_RE.search(output)
    if not tool_m:
        return {"has_tool_code": False, "syntax_valid": False, "call_count": 0}

    code = tool_m.group(1).strip()
    call_count = len(re.findall(r"\w+\s*\(", code))

    # 逐语句尝试解析
    stmts = [s.strip() for s in re.split(r";|\n", code) if s.strip()]
    invalid = 0
    for stmt in stmts:
        try:
            ast.parse(stmt)
        except SyntaxError:
            invalid += 1

    return {
        "has_tool_code": True,
        "syntax_valid": invalid == 0,
        "invalid_stmts": invalid,
        "call_count": call_count,
    }


def style_score(output: str) -> Dict[str, float]:
    """计算叙事部分的风格一致性得分。"""
    # 只分析叙事部分（去掉 think 和 tool_code）
    text = _THINK_RE.sub("", output)
    text = _TOOL_CODE_RE.sub("", text)

    total_chars = max(len(text), 1)
    hit_count = sum(1 for kw in STYLE_KEYWORDS if kw in text)
    # 每百字关键词密度
    density = hit_count / (total_chars / 100)
    score = min(1.0, density / STYLE_THRESHOLD)

    return {
        "keyword_hits": hit_count,
        "density_per_100": round(density, 3),
        "style_score": round(score, 3),
        "passes_threshold": score >= 1.0,
    }


def rouge_l(prediction: str, reference: str) -> float:
    """
    计算两个字符串之间的字符级 ROUGE-L（最长公共子序列 / 参考长度）。
    不依赖任何外部库，使用动态规划。
    """
    # 在中文文本上以字符为单位更合适
    pred = list(prediction.strip())
    ref = list(reference.strip())
    if not ref:
        return 0.0

    # LCS 长度（空间优化版 DP）
    m, n = len(pred), len(ref)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if pred[i - 1] == ref[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr

    lcs_len = prev[n]
    precision = lcs_len / max(len(pred), 1)
    recall = lcs_len / n
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


# ══════════════════════════════════════════════
# 批量评估（给定输出列表）
# ══════════════════════════════════════════════

def evaluate_outputs(
    predictions: List[str],
    references: Optional[List[str]] = None,
    label: str = "model",
) -> Dict:
    """
    对一批输出进行全维度评估，返回汇总统计字典。
    references 为空时跳过 ROUGE-L 计算。
    """
    n = len(predictions)
    if n == 0:
        return {}

    struct_results = [check_structure(p) for p in predictions]
    tool_results = [check_tool_syntax(p) for p in predictions]
    style_results = [style_score(p) for p in predictions]

    lengths = [len(p) for p in predictions]

    rouge_scores: List[float] = []
    if references and len(references) == n:
        rouge_scores = [rouge_l(pred, ref) for pred, ref in zip(predictions, references)]

    def _rate(lst: List[bool]) -> float:
        return round(sum(lst) / len(lst) * 100, 1) if lst else 0.0

    result = {
        "label": label,
        "sample_count": n,
        "structural_validity": {
            "has_think_rate":     _rate([r["has_think"] for r in struct_results]),
            "has_tool_code_rate": _rate([r["has_tool_code"] for r in struct_results]),
            "has_narrative_rate": _rate([r["has_narrative"] for r in struct_results]),
            "fully_valid_rate":   _rate([r["is_fully_valid"] for r in struct_results]),
        },
        "tool_call_validity": {
            "has_tool_code_rate": _rate([r["has_tool_code"] for r in tool_results]),
            "syntax_valid_rate":  _rate([
                r["syntax_valid"] for r in tool_results if r["has_tool_code"]
            ]),
            "avg_call_count": round(
                sum(r["call_count"] for r in tool_results) / n, 2
            ),
        },
        "style_consistency": {
            "passes_threshold_rate": _rate([r["passes_threshold"] for r in style_results]),
            "avg_keyword_density":   round(
                sum(r["density_per_100"] for r in style_results) / n, 3
            ),
            "avg_style_score":       round(
                sum(r["style_score"] for r in style_results) / n, 3
            ),
        },
        "output_length": {
            "mean":   round(sum(lengths) / n, 1),
            "median": sorted(lengths)[n // 2],
            "min":    min(lengths),
            "max":    max(lengths),
        },
    }

    if rouge_scores:
        result["rouge_l"] = {
            "mean":   round(sum(rouge_scores) / len(rouge_scores), 4),
            "median": sorted(rouge_scores)[len(rouge_scores) // 2],
        }

    return result


# ══════════════════════════════════════════════
# 模型推理
# ══════════════════════════════════════════════

SYSTEM_PROMPT = (
    "你是「谋杀湾」世界的AI叙事引擎，代号「岁月史书」。"
    "你的职责是响应玩家的每一个行为指令，并按照以下固定流程输出：\n"
    "1. 在<think>标签内进行内部推理：评估NPC身份、决定世界状态变化；\n"
    "2. 在<tool_code>标签内输出工具调用代码：更新NPC属性、重写历史、触发事件；\n"
    "3. 标签结束后，直接输出面向玩家的叙事文本，风格冷酷、悬疑、充满不确定性。\n"
    "禁止解释你的推理过程，禁止打破第四堵墙，禁止输出与叙事无关的内容。"
)


def load_model(base_model: str, lora: Optional[str] = None):
    """加载模型和 tokenizer，返回 (model, tokenizer)。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  加载 tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"  加载基座模型: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    if lora and os.path.exists(lora):
        print(f"  加载 LoRA: {lora}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    instructions: List[str],
    max_new_tokens: int = 512,
    batch_size: int = 1,
) -> List[str]:
    """批量推理，返回生成文本列表。"""
    import torch
    results: List[str] = []
    total = len(instructions)

    for i, instruction in enumerate(instructions):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  推理进度: {i+1}/{total}", end="\r", flush=True)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        results.append(tokenizer.decode(new_ids, skip_special_tokens=True))

    print(f"  推理完成: {total}/{total}           ")
    return results


# ══════════════════════════════════════════════
# 报告打印
# ══════════════════════════════════════════════

def print_report(results: List[Dict], save_path: Optional[str] = None) -> None:
    """打印对比报告，可选保存到 JSON。"""
    print("\n" + "═" * 70)
    print("谋杀湾·模型评估报告")
    print("═" * 70)

    headers = ["指标", ] + [r["label"] for r in results]
    col_w = 28

    def _row(name: str, *vals) -> str:
        row = f"  {name:<{col_w}}"
        for v in vals:
            row += f"  {str(v):>12}"
        return row

    print(_row(*headers))
    print("  " + "─" * (col_w + 16 * len(results)))

    # 结构有效率
    print("  [结构有效率]")
    for key, label in [
        ("has_think_rate",     "  含<think>"),
        ("has_tool_code_rate", "  含<tool_code>"),
        ("has_narrative_rate", "  含叙事文本"),
        ("fully_valid_rate",   "  三段式完整率 ★"),
    ]:
        vals = [f"{r['structural_validity'][key]}%" for r in results]
        print(_row(label, *vals))

    print("  [工具调用有效率]")
    for key, label in [
        ("has_tool_code_rate", "  含工具调用"),
        ("syntax_valid_rate",  "  语法合法率 ★"),
        ("avg_call_count",     "  平均调用次数"),
    ]:
        vals = [
            f"{r['tool_call_validity'][key]}{'%' if 'rate' in key else ''}"
            for r in results
        ]
        print(_row(label, *vals))

    print("  [风格一致性]")
    for key, label in [
        ("passes_threshold_rate", "  风格达标率 ★"),
        ("avg_keyword_density",   "  关键词密度/百字"),
        ("avg_style_score",       "  综合风格得分"),
    ]:
        vals = [
            f"{r['style_consistency'][key]}{'%' if 'rate' in key else ''}"
            for r in results
        ]
        print(_row(label, *vals))

    print("  [输出长度]")
    for key, label in [
        ("mean",   "  平均长度"),
        ("median", "  中位长度"),
    ]:
        vals = [str(r["output_length"][key]) for r in results]
        print(_row(label, *vals))

    if any("rouge_l" in r for r in results):
        print("  [语义相似度]")
        for key, label in [("mean", "  ROUGE-L (mean)"), ("median", "  ROUGE-L (median)")]:
            vals = [
                str(r["rouge_l"][key]) if "rouge_l" in r else "N/A"
                for r in results
            ]
            print(_row(label, *vals))

    print("═" * 70)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n评估结果已保存: {save_path}")


# ══════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="谋杀湾·模型评估脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--val",        default="dataset/val.jsonl",
                        help="验证集路径")
    parser.add_argument("--mode",       choices=["offline", "online", "compare"],
                        default="offline",
                        help="offline=仅统计参考答案; online=评估微调后模型; compare=微调前后对比")
    parser.add_argument("--base-model", default="/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora",       default="/root/output/murder_bay_lora/v2-20260322-002516/checkpoint-207")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="online/compare 模式下最多评估的样本数（节省时间）")
    parser.add_argument("--output",     default="dataset/eval_report.json",
                        help="评估报告输出路径")
    args = parser.parse_args()

    # 定位到项目根
    root = Path(__file__).parent.parent
    os.chdir(root)

    # ── 加载验证集 ──
    print(f"[加载验证集] {args.val}")
    val_items: List[Dict] = []
    with open(args.val, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                val_items.append(json.loads(line))

    # 截取样本数
    sample_items = val_items[: args.max_samples]
    instructions = [it["instruction"] for it in sample_items]
    references   = [it["output"]      for it in sample_items]

    print(f"验证集总量: {len(val_items)} 条，本次评估: {len(sample_items)} 条")
    print(f"评估模式: {args.mode}\n")

    all_results: List[Dict] = []

    # ── 模式1：offline ──
    if args.mode == "offline":
        print("[offline] 统计参考答案质量指标...")
        ref_result = evaluate_outputs(references, label="参考答案(验证集)")
        all_results.append(ref_result)

    # ── 模式2：online ──
    elif args.mode == "online":
        print("[online] 加载微调后模型...")
        model, tokenizer = load_model(args.base_model, args.lora)
        t0 = time.time()
        predictions = run_inference(model, tokenizer, instructions)
        elapsed = time.time() - t0
        print(f"  推理耗时: {elapsed:.1f}s ({elapsed/len(instructions):.1f}s/条)")

        ft_result = evaluate_outputs(predictions, references, label="微调后模型")
        all_results.append(ft_result)

    # ── 模式3：compare ──
    elif args.mode == "compare":
        # 先跑参考答案
        ref_result = evaluate_outputs(references, label="参考答案")
        all_results.append(ref_result)

        # 纯基座模型
        print("[compare] 加载纯基座模型（无LoRA）...")
        model_base, tokenizer = load_model(args.base_model, lora=None)
        base_preds = run_inference(model_base, tokenizer, instructions)
        base_result = evaluate_outputs(base_preds, references, label="基座模型(无微调)")
        all_results.append(base_result)
        del model_base

        # 微调后模型
        print("[compare] 加载微调后模型...")
        import torch; torch.cuda.empty_cache()
        model_ft, _ = load_model(args.base_model, lora=args.lora)
        ft_preds = run_inference(model_ft, tokenizer, instructions)
        ft_result = evaluate_outputs(ft_preds, references, label="微调后模型(+LoRA)")
        all_results.append(ft_result)

    print_report(all_results, save_path=args.output)


if __name__ == "__main__":
    main()
