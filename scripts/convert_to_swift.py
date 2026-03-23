"""
将 murder_bay_data.jsonl 转换为 ms-swift 兼容的 alpaca 格式。

原始格式：
  {"instruction": "...", "thought": "...", "tool_code": "...", "output": "..."}

目标格式（alpaca）：
  {"system": "...", "instruction": "...", "output": "<think>...</think>\n<tool_code>...</tool_code>\n..."}

用法：
  python scripts/convert_to_swift.py
  python scripts/convert_to_swift.py --input dataset/murder_bay_data_cleaned.jsonl --output dataset/murder_bay_swift.jsonl
"""

import json
import argparse
import os
from pathlib import Path

SYSTEM_PROMPT = (
    "你是「谋杀湾」世界的AI叙事引擎，代号「岁月史书」。"
    "你的职责是响应玩家的每一个行为指令，并按照以下固定流程输出：\n"
    "1. 在<think>标签内进行内部推理：评估NPC身份、决定世界状态变化；\n"
    "2. 在<tool_code>标签内输出工具调用代码：更新NPC属性、重写历史、触发事件；\n"
    "3. 标签结束后，直接输出面向玩家的叙事文本，风格冷酷、悬疑、充满不确定性。\n"
    "禁止解释你的推理过程，禁止打破第四堵墙，禁止输出与叙事无关的内容。"
)


def convert_record(record: dict) -> dict | None:
    """将单条原始记录转换为 ms-swift alpaca 格式。"""
    instruction = record.get("instruction", "").strip()
    thought = record.get("thought", "").strip()
    tool_code = record.get("tool_code", "").strip()
    output = record.get("output", "").strip()

    # 跳过字段不完整的记录
    if not instruction or not output:
        return None

    parts = []
    if thought:
        parts.append(f"<think>\n{thought}\n</think>")
    if tool_code:
        parts.append(f"<tool_code>\n{tool_code}\n</tool_code>")
    parts.append(output)

    merged_output = "\n".join(parts)

    return {
        "system": SYSTEM_PROMPT,
        "instruction": instruction,
        "output": merged_output,
    }


def convert_file(input_path: str, output_path: str) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = skipped = converted = 0

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line_num, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [警告] 第 {line_num} 行 JSON 解析失败，已跳过：{e}")
                skipped += 1
                continue

            new_record = convert_record(record)
            if new_record is None:
                print(f"  [警告] 第 {line_num} 行字段不完整，已跳过。")
                skipped += 1
                continue

            f_out.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            converted += 1

    print(f"\n转换完成：")
    print(f"  输入文件：{input_path}")
    print(f"  输出文件：{output_path}")
    print(f"  总计读取：{total} 条")
    print(f"  成功转换：{converted} 条")
    print(f"  跳过记录：{skipped} 条")


def main():
    parser = argparse.ArgumentParser(description="将原始数据集转换为 ms-swift 兼容的 alpaca 格式")
    parser.add_argument(
        "--input",
        default="dataset/murder_bay_data_cleaned.jsonl",
        help="输入 JSONL 文件路径（默认：dataset/murder_bay_data_cleaned.jsonl）",
    )
    parser.add_argument(
        "--output",
        default="dataset/murder_bay_swift.jsonl",
        help="输出 JSONL 文件路径（默认：dataset/murder_bay_swift.jsonl）",
    )
    args = parser.parse_args()

    # 兼容从项目根目录或 scripts/ 目录运行
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    input_path = args.input if os.path.isabs(args.input) else project_root / args.input
    output_path = args.output if os.path.isabs(args.output) else project_root / args.output

    convert_file(str(input_path), str(output_path))

    print("\n下一步（在 AutoDL 服务器上执行）：")
    print("  swift sft \\")
    print("    --model_type qwen2_5-7b-instruct \\")
    print("    --dataset dataset/murder_bay_swift.jsonl \\")
    print("    --template_type qwen \\")
    print("    --sft_type lora \\")
    print("    --output_dir output/murder_bay_lora \\")
    print("    --num_train_epochs 3 \\")
    print("    --lora_rank 16 \\")
    print("    --lora_alpha 32 \\")
    print("    --batch_size 2 \\")
    print("    --gradient_accumulation_steps 8")


if __name__ == "__main__":
    main()
