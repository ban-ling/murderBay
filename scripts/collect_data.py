import requests
import json
import time

# 配置参数
API_KEY = "你的DEEPSEEK_API_KEY"  # 请替换为你的实际 API Key
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"  # DeepSeek-V3 的模型标识符通常是 deepseek-chat
OUTPUT_FILE = "a.txt"

# 固定的 Prompt
PROMPT = """
请按以下要求生成 1 条 JSONL 数据：
# Role: 极简主义架构师 & 混沌叙事专家
任务：为 MiniMind (0.02B) 生成具有“角色不确定性”的微调数据。
包含字段：instruction, thought, tool_code, output。
要求：保持冷酷、悬疑，包含数值操纵或岁月史书逻辑。
"""

def call_deepseek_api(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "stream": False
    }
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            print(f"API 请求失败，状态码: {response.status_code}, 错误信息: {response.text}")
            return None
    except Exception as e:
        print(f"发生异常: {e}")
        return None

def main(iterations=10):
    print(f"开始调用 API，预计执行 {iterations} 次...")
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i in range(iterations):
            print(f"正在执行第 {i+1} 次请求...")
            content = call_deepseek_api(PROMPT)
            
            if content:
                # 写入文件并添加分隔符
                f.write(content + "\n")
                # 强制刷新缓冲区，确保实时写入
                f.flush()
                print(f"第 {i+1} 次请求成功并已写入文件。")
            else:
                print(f"第 {i+1} 次请求失败。")
            
            # 适当休眠，避免触发速率限制
            time.sleep(1)

if __name__ == "__main__":
    # 你可以根据需要修改循环次数
    main(iterations=20)


