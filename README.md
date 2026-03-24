# 谋杀湾 · Murder Bay

> 一座被遗忘的港口城市，每一块石板下都埋着秘密。

基于 **Qwen2.5-7B-Instruct + LoRA** 微调的克苏鲁悬疑风格 AI 叙事游戏框架。玩家的每一次行动都触发真实的工具调用，永久改变世界状态——由「岁月史书」记录在案，不可撤销。

---

## 项目简介

「谋杀湾」是一个完整的大模型微调与 Agent 游戏工程，涵盖从**数据采集 → 清洗 → 训练 → 评估 → 游戏运行**的全流程。

模型以三段式格式响应玩家指令：

```
<think>   内部推理（GM视角可见）    </think>
<tool_code>   世界状态工具调用       </tool_code>
叙事文本（玩家可见，风格冷酷悬疑）
```

---

## 目录结构

```
murderBay/
├── agent/                  # 游戏引擎
│   ├── game_loop.py        # 主程序入口
│   ├── world_state.py      # 世界状态管理（单例，支持存/读档）
│   ├── tools.py            # 工具函数定义
│   ├── tool_executor.py    # 解析并安全执行 <tool_code>
│   ├── model_client.py     # 本地模型 / API 推理客户端
│   ├── prompt_builder.py   # 注入世界状态的 system prompt 构建
│   └── saves/              # 存档目录（自动创建）
├── dataset/                # 数据集
│   ├── murder_bay_data.jsonl       # 原始训练数据
│   ├── murder_bay_swift.jsonl      # Swift 格式训练数据
│   ├── train.jsonl / val.jsonl     # 处理后的训练/验证集
│   ├── dpo_data.jsonl              # DPO 偏好数据
│   └── data_report.json            # 数据质量报告
├── scripts/                # 工具脚本
│   ├── collect_data.py     # 调用 DeepSeek API 批量采集数据
│   ├── data_pipeline.py    # 数据清洗·去重·分层划分流水线
│   ├── build_iter_data.py  # 迭代数据构建
│   ├── convert_to_swift.py # 转换为 ms-swift 训练格式
│   ├── generate_dpo_data.py# 生成 DPO 对比数据
│   ├── eval.py             # 模型量化评估脚本
│   └── visualize_report.py # 可视化数据报告
├── trainer/
│   └── full_sft_murder_bay.py  # MiniMind 全量 SFT 训练脚本
└── eval_llm_murder.py      # 通用 LLM 评估入口
```

---

## 快速开始

### 环境要求

- Python 3.10+
- CUDA GPU（推荐 RTX 4090 / A100，Qwen2.5-7B bfloat16 约占 15~16 GB 显存）
- `transformers`, `peft`, `torch`

### 运行游戏

```bash
cd agent

# 使用默认路径（本地 Qwen2.5-7B + LoRA）
python game_loop.py

# 指定自定义模型路径
python game_loop.py \
  --base-model /path/to/Qwen2.5-7B-Instruct \
  --lora /path/to/lora/checkpoint

# 开启 GM 视角（显示推理过程和工具调用）
python game_loop.py --gm

# 加载存档继续游戏
python game_loop.py --load autosave
```

### API 模式（使用 ms-swift 部署服务）

```bash
# 终端1：启动推理服务
swift deploy \
  --model /path/to/Qwen2.5-7B-Instruct \
  --adapters /path/to/lora/checkpoint \
  --port 8000

# 终端2：启动游戏
python game_loop.py --backend api
```

---

## 游戏内指令

| 指令 | 说明 |
|------|------|
| `/status` | 显示玩家属性与世界状态（GM视角） |
| `/gm` | 切换 GM 视角（显示/隐藏推理过程） |
| `/save` | 保存进度到自动存档槽 |
| `/save <槽名>` | 保存到指定槽，如 `/save chapter1` |
| `/load` | 加载自动存档 |
| `/load <槽名>` | 加载指定存档 |
| `/saves` | 列出所有存档 |
| `/history` | 查看本局事件日志 |
| `/quit` | 退出游戏（自动存档） |
| `/help` | 显示帮助 |

### 输入格式

推荐使用 `[动词] 描述` 格式以获得最佳效果：

```
[购买] 从码头的走私商人那里买一瓶廉价麦芽酒
[调查] 追踪那个总在深夜出没的独眼修士
[询问] 向酒保打听关于失踪船长的消息
[攻击] 教训那个在市场上欺负老人的混混
[观察] 仔细打量站在灯塔旁的神秘女人
```

也支持自然语言输入，模型同样能理解。

---

## 完整训练流程

### 第一步：数据采集

```bash
# 编辑 scripts/collect_data.py，填入你的 DeepSeek API Key
python scripts/collect_data.py
```

### 第二步：数据清洗与划分

```bash
python scripts/data_pipeline.py \
  --input  dataset/murder_bay_swift.jsonl \
  --train  dataset/train.jsonl \
  --val    dataset/val.jsonl \
  --report dataset/data_report.json \
  --val-ratio 0.1 \
  --fuzzy-threshold 0.75
```

数据流水线包含：
- 结构完整性校验（三段式格式）
- 内容质量过滤（长度、缺字段、截断标记）
- 精确去重 + 模糊去重（字符 4-gram Jaccard）
- 动词语义分类（8 大行为类别）
- 分层验证集划分

### 第三步：LoRA 微调（推荐，使用 ms-swift）

```bash
swift sft \
  --model /path/to/Qwen2.5-7B-Instruct \
  --dataset dataset/murder_bay_swift.jsonl \
  --train_type lora \
  --output_dir /path/to/output \
  --num_train_epochs 3
```

### 第四步（可选）：MiniMind 全量 SFT

```bash
python trainer/full_sft_murder_bay.py \
  --data_path dataset/murder_bay_data.jsonl \
  --save_dir out \
  --epochs 5
```

### 第五步：模型评估

```bash
# 离线模式（统计验证集质量，无需 GPU）
python scripts/eval.py --mode offline

# 在线模式（评估微调后模型）
python scripts/eval.py --mode online \
  --base-model /path/to/Qwen2.5-7B-Instruct \
  --lora /path/to/lora/checkpoint

# 对比模式（基座 vs 微调后）
python scripts/eval.py --mode compare \
  --base-model /path/to/Qwen2.5-7B-Instruct \
  --lora /path/to/lora/checkpoint
```

---

## 评估维度

| 维度 | 说明 |
|------|------|
| 三段式完整率 ★ | 输出是否包含完整的 `<think>` / `<tool_code>` / 叙事三段结构 |
| 工具调用语法合法率 ★ | `<tool_code>` 内代码能否通过 Python AST 解析 |
| 风格达标率 ★ | 叙事部分是否包含悬疑/克苏鲁风格关键词 |
| ROUGE-L | 与参考输出的最长公共子序列相似度 |
| 输出长度分布 | 生成长度是否合理（避免过短/截断） |

---

## 世界状态

### 玩家属性

| 属性 | 含义 | 初始值 |
|------|------|--------|
| `hp` | 生命值，归零则死亡结局 | 100 |
| `san` | 理性值，归零则疯狂结局 | 80 |
| `gold` | 金币 | 50 |
| `visibility` | 在地下世界的暴露程度 | 0 |
| `hidden_attention` | 被未知力量关注程度 | 0 |
| `guilt` | 道德愧疚值 | 0 |
| `infected_memes` | 植入的精神污染概念列表 | [] |
| `inventory` | 物品栏 | [] |

### 工具函数

模型输出的 `<tool_code>` 中调用这些函数（全部真实执行，永久改变世界状态）：

| 函数 | 作用 |
|------|------|
| `set_npc_identity(id, identity)` | 揭示/改变 NPC 的隐藏身份 |
| `update_npc_stat(id, stat, delta)` | 修改 NPC 或玩家的属性值 |
| `update_world_stat(stat, delta)` | 修改全局世界属性 |
| `rewrite_history(fact)` | 岁月史书：永久篡改历史记录 |
| `set_world_flag(flag, value)` | 设置世界事件标志 |
| `trigger_world_event(id, ...)` | 触发全局事件 |
| `update_zone_stat(zone, stat, delta)` | 修改特定区域状态 |
| `spawn_hidden_npc(id, location)` | 在世界中生成隐藏 NPC |
| `add_quest_seed(quest_id)` | 埋下任务伏笔 |
| `create_item(id, ...)` | 创建并放置物品 |

---

## 技术架构

```
玩家输入 "[调查] 追踪独眼修士"
    ↓
prompt_builder.py
  将当前世界状态注入 system prompt
    ↓
model_client.py
  调用 Qwen2.5-7B + LoRA 生成三段式输出
    ↓
tool_executor.py
  解析 <tool_code>，在安全沙箱中执行工具调用
  → NPC 身份真正被修改
  → 玩家 SAN 值真正减少
  → 下一轮世界状态摘要反映变化
    ↓
玩家看到叙事文本（<think> 和 <tool_code> 默认隐藏）
```

---

## 注意事项

- **显存需求**：Qwen2.5-7B bfloat16 约占 15~16 GB，RTX 4090（24GB）可正常运行
- **首次加载**：模型 LoRA 合并约需 15~30 秒
- **历史截断**：默认保留最近 8 轮对话历史，防止 OOM
- **执行错误**：若模型输出的工具调用有语法错误，游戏不会崩溃，GM 模式下可查看报错
- **自动存档**：每 10 回合自动存档一次，退出时也会自动存档

---

## 数据格式（JSONL）

每条训练样本包含以下字段：

```json
{
  "instruction": "[调查] 追踪那个总在深夜出没的独眼修士",
  "thought": "修士（monk_blind）的隐藏身份可能是...",
  "tool_code": "set_npc_identity('monk_blind', '深潮祭司')\nupdate_npc_stat('player', 'san', -10)",
  "output": "修士停下脚步，缓缓转身。他的独眼中映出一种令人不安的光芒……"
}
```

---

## License

MIT
