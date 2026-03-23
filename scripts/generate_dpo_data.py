"""
谋杀湾·DPO 偏好对数据生成器
==============================

策略：
  chosen  = 训练集中已有的高质量输出（三段式完整、风格强烈、工具调用丰富）
  rejected = 调用 DeepSeek API，用"降质 prompt"生成同一 instruction 的劣质回复：
             · 无 <think> 深度推理（直接拍板）
             · <tool_code> 极简（只用一个最普通的调用）
             · 叙事平淡，缺乏悬疑感，不体现岁月史书特色

DPO 数据格式（ms-swift 兼容）：
  {"system": "...", "instruction": "...", "chosen": "...", "rejected": "..."}

用法：
  python scripts/generate_dpo_data.py \\
      --api-key YOUR_DEEPSEEK_KEY \\
      --n 200 \\
      --output dataset/dpo_data.jsonl

  # 断点续传（输出文件已存在时自动跳过已完成的条目）
  python scripts/generate_dpo_data.py --api-key KEY --resume
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL   = "deepseek-chat"

SYSTEM_PROMPT = (
    "你是「谋杀湾」世界的AI叙事引擎，代号「岁月史书」。"
    "你的职责是响应玩家的每一个行为指令，并按照以下固定流程输出：\n"
    "1. 在<think>标签内进行内部推理：评估NPC身份、决定世界状态变化；\n"
    "2. 在<tool_code>标签内输出工具调用代码：更新NPC属性、重写历史、触发事件；\n"
    "3. 标签结束后，直接输出面向玩家的叙事文本，风格冷酷、悬疑、充满不确定性。\n"
    "禁止解释你的推理过程，禁止打破第四堵墙，禁止输出与叙事无关的内容。"
)

# 降质 prompt：让 API 生成"差"的回复用作 rejected
_REJECTED_SYSTEM = (
    "你是一个普通的文字游戏叙述者，简单直接地回应玩家行为。"
    "不需要深度分析，也不需要特殊格式标签。"
    "用平实、常规的方式描述结果即可，保持简短。"
)

_REJECTED_USER_TEMPLATE = """\
玩家行动：{instruction}

请给出一个简短、平淡的游戏叙述回应。
要求：
- 使用 <think> 标签，但里面只写一句话的简单判断
- 使用 <tool_code> 标签，但只调用一个最简单的函数
- 叙事部分控制在 2-3 句话，不需要悬疑感，正常描述结果即可
"""

# 重试配置
MAX_RETRIES = 3
RETRY_DELAY = 3.0
REQUEST_INTERVAL = 1.2   # 请求间隔（秒），避免触发速率限制


# ══════════════════════════════════════════════
# API 调用
# ══════════════════════════════════════════════

def call_api(
    api_key: str,
    messages: List[Dict],
    temperature: float = 0.85,
    max_tokens: int = 512,
) -> Optional[str]:
    """调用 DeepSeek API，返回文本内容，失败返回 None。"""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model":       DEEPSEEK_MODEL,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream":      False,
    }).encode("utf-8")

    req = urllib.request.Request(
        DEEPSEEK_API_URL,
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            if e.code == 429:
                wait = RETRY_DELAY * attempt * 2
                print(f"    [限速] 等待 {wait:.0f}s 后重试...")
                time.sleep(wait)
            else:
                print(f"    [HTTP错误 {e.code}] {body[:120]}")
                return None
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"    [重试 {attempt}/{MAX_RETRIES}] {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    [失败] {e}")
                return None
    return None


def generate_rejected(api_key: str, instruction: str) -> Optional[str]:
    """为给定 instruction 生成 rejected（劣质）回复。"""
    messages = [
        {"role": "system", "content": _REJECTED_SYSTEM},
        {"role": "user",   "content": _REJECTED_USER_TEMPLATE.format(instruction=instruction)},
    ]
    return call_api(api_key, messages, temperature=0.6, max_tokens=400)


# ══════════════════════════════════════════════
# 质量过滤（对 rejected 的最低要求）
# ══════════════════════════════════════════════

_THINK_RE    = re.compile(r"<think>(.*?)</think>",       re.DOTALL)
_TOOL_RE     = re.compile(r"<tool_code>(.*?)</tool_code>", re.DOTALL)


def is_valid_rejected(text: str, chosen: str) -> Tuple[bool, str]:
    """
    验证 rejected 回复是否符合要求：
    1. 包含基本结构（否则无法对比）
    2. 不能和 chosen 过于相似（否则无学习价值）
    3. 叙事部分不能过短（说明 API 生成失败）
    """
    if not text:
        return False, "空回复"

    # 必须有 <think> 和 <tool_code>（否则格式差异太极端，训练不稳定）
    if not _THINK_RE.search(text):
        return False, "缺少<think>"
    if not _TOOL_RE.search(text):
        return False, "缺少<tool_code>"

    # 叙事部分长度校验
    narrative = _THINK_RE.sub("", text)
    narrative = _TOOL_RE.sub("", narrative).strip()
    if len(narrative) < 20:
        return False, f"叙事过短({len(narrative)}字)"

    # 和 chosen 不能过于相似（基于简单字符串重叠率）
    overlap = len(set(text) & set(chosen)) / max(len(set(chosen)), 1)
    if overlap > 0.95:
        return False, "与chosen过于相似"

    # rejected 的叙事不能比 chosen 更长（说明质量没有降低）
    chosen_narrative = _THINK_RE.sub("", chosen)
    chosen_narrative = _TOOL_RE.sub("", chosen_narrative).strip()
    if len(narrative) > len(chosen_narrative) * 1.2:
        return False, "rejected比chosen更长，质量对比不明确"

    return True, "ok"


# ══════════════════════════════════════════════
# 断点续传：加载已完成的 instruction 集合
# ══════════════════════════════════════════════

def load_done_instructions(output_path: str) -> set:
    done = set()
    p = Path(output_path)
    if not p.exists():
        return done
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    done.add(item.get("instruction", ""))
                except json.JSONDecodeError:
                    pass
    return done


# ══════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════

def run(
    api_key: str,
    train_path: str,
    output_path: str,
    n: int,
    seed: int,
    resume: bool,
) -> None:
    print("=" * 60)
    print("谋杀湾·DPO 偏好对数据生成器")
    print("=" * 60)

    # 加载训练集
    print(f"\n[1/4] 读取训练集: {train_path}")
    train_items: List[Dict] = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                train_items.append(json.loads(line))
    print(f"  共 {len(train_items)} 条训练样本")

    # 断点续传
    done_instructions: set = set()
    if resume:
        done_instructions = load_done_instructions(output_path)
        print(f"  断点续传：已完成 {len(done_instructions)} 条，跳过")

    # 随机采样（排除已完成）
    candidates = [
        it for it in train_items
        if it["instruction"] not in done_instructions
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    to_process = candidates[:n]

    print(f"\n[2/4] 本次生成目标: {len(to_process)} 条 DPO 偏好对")
    print(f"  策略: chosen=训练集原始输出  rejected=API生成劣质版本\n")

    # 准备输出文件
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_file = open(output_path, "a", encoding="utf-8")

    success = 0
    skipped = 0
    failed  = 0

    try:
        for i, item in enumerate(to_process):
            instruction = item["instruction"]
            chosen      = item["output"]

            print(f"  [{i+1:>4}/{len(to_process)}] {instruction[:45]}...", end="", flush=True)

            rejected = generate_rejected(api_key, instruction)
            time.sleep(REQUEST_INTERVAL)

            if rejected is None:
                print(" ✗ API失败")
                failed += 1
                continue

            valid, reason = is_valid_rejected(rejected, chosen)
            if not valid:
                print(f" ⚠ 跳过({reason})")
                skipped += 1
                continue

            dpo_item = {
                "system":      SYSTEM_PROMPT,
                "instruction": instruction,
                "chosen":      chosen,
                "rejected":    rejected,
            }
            out_file.write(json.dumps(dpo_item, ensure_ascii=False) + "\n")
            out_file.flush()
            success += 1
            print(f" ✓ (chosen={len(chosen)}字 | rejected={len(rejected)}字)")

    finally:
        out_file.close()

    print(f"\n[3/4] 完成统计")
    print(f"  成功生成: {success} 条")
    print(f"  质量跳过: {skipped} 条")
    print(f"  API 失败: {failed} 条")
    print(f"  输出文件: {output_path}")

    # 输出格式样例
    if success > 0:
        print(f"\n[4/4] DPO 数据格式示例（ms-swift 兼容）：")
        print("""  {
    "system":      "你是「谋杀湾」世界的AI叙事引擎...",
    "instruction": "[调查] ...",
    "chosen":      "<think>\\n深度推理...\\n</think>\\n<tool_code>\\n...\\n</tool_code>\\n强烈叙事...",
    "rejected":    "<think>\\n简单判断\\n</think>\\n<tool_code>\\n一行代码\\n</tool_code>\\n平淡叙述"
  }""")

        print(f"\n下一步（ms-swift DPO 训练命令）：")
        print(f"""  swift dpo \\
    --model /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \\
    --dataset {output_path} \\
    --train_type lora \\
    --lora_rank 16 \\
    --lora_alpha 32 \\
    --output_dir output/murder_bay_dpo \\
    --num_train_epochs 2 \\
    --per_device_train_batch_size 1 \\
    --gradient_accumulation_steps 8""")


# ══════════════════════════════════════════════
# 命令行入口
# ══════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="调用 DeepSeek API 为谋杀湾数据集生成 DPO 偏好对",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-key",  required=True,
                        help="DeepSeek API Key（或设置环境变量 DEEPSEEK_API_KEY）")
    parser.add_argument("--train",    default="dataset/train.jsonl",
                        help="训练集路径（chosen 来源）")
    parser.add_argument("--output",   default="dataset/dpo_data.jsonl",
                        help="DPO 数据输出路径")
    parser.add_argument("--n",        type=int, default=200,
                        help="生成条数（建议与训练集规模 1:5~1:3 比例）")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--resume",   action="store_true",
                        help="断点续传：跳过已写入的 instruction")
    args = parser.parse_args()

    # 支持环境变量
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key or api_key == "你的DEEPSEEK_API_KEY":
        print("[错误] 请提供有效的 DeepSeek API Key：")
        print("       --api-key YOUR_KEY")
        print("       或设置环境变量 DEEPSEEK_API_KEY=YOUR_KEY")
        return

    root = Path(__file__).parent.parent
    os.chdir(root)

    run(
        api_key=api_key,
        train_path=args.train,
        output_path=args.output,
        n=args.n,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
