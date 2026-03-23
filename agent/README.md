# 谋杀湾·叙事 Agent 引擎

基于微调 Qwen2.5-7B-Instruct + LoRA 的真实 Agent 冒险游戏框架。
每一次玩家行动都会触发真实的工具调用，永久改变世界状态。

---

## 文件结构

```
agent/
├── game_loop.py        # 主程序入口，运行这个文件开始游戏
├── world_state.py      # 世界状态数据结构、存档/读档
├── tools.py            # 所有工具函数（set_npc_identity 等）
├── tool_executor.py    # 解析 <tool_code> 并安全执行
├── model_client.py     # 本地模型/API 推理客户端
├── prompt_builder.py   # 构建注入世界状态的 system prompt
├── saves/              # 存档目录（自动创建）
└── README.md
```

---

## 快速开始（AutoDL 服务器）

### 1. 确认模型路径

训练完成后的默认路径：
- 基座模型：`/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct`
- LoRA 适配器：`/root/output/murder_bay_lora/v2-20260322-002516/checkpoint-207`

如果路径不同，通过 `--base-model` 和 `--lora` 参数指定（见下方）。

### 2. 进入 agent 目录并运行

```bash
cd /root/autodl-tmp/project/agent
python game_loop.py
```

### 3. 自定义路径

```bash
python game_loop.py \
  --base-model /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \
  --lora /root/output/murder_bay_lora/v2-20260322-002516/checkpoint-207
```

### 4. 开启 GM 视角（显示模型推理和工具调用）

```bash
python game_loop.py --gm
```

### 5. 加载存档继续游戏

```bash
python game_loop.py --load autosave
```

---

## API 模式（先用 swift deploy 启动服务）

```bash
# 终端1：启动推理服务
swift deploy \
  --model /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \
  --adapters /root/output/murder_bay_lora/v2-20260322-002516/checkpoint-207 \
  --port 8000

# 终端2：启动游戏
python game_loop.py --backend api
```

---

## 游戏内指令

| 指令 | 说明 |
|------|------|
| `/status` | 显示当前玩家属性与世界状态 |
| `/gm` | 切换 GM 视角（显示/隐藏推理过程） |
| `/save` | 保存进度（自动存档槽） |
| `/save 槽名` | 保存到指定槽，如 `/save chapter1` |
| `/load` | 加载自动存档 |
| `/load 槽名` | 加载指定存档 |
| `/saves` | 列出所有存档 |
| `/history` | 查看本局事件日志 |
| `/quit` | 退出游戏（自动存档） |
| `/help` | 显示帮助 |

---

## 玩家输入格式

模型训练时使用 `[动词] 描述` 格式，建议按此输入以获得最好效果：

```
[购买] 从码头的走私商人那里买一瓶廉价麦芽酒
[调查] 追踪那个总在深夜出没的独眼修士
[询问] 向酒保打听关于失踪船长的消息
[攻击] 教训那个在市场上欺负老人的混混
[观察] 仔细打量站在灯塔旁的神秘女人
```

也可以使用自然语言，模型同样能理解。

---

## 工作原理

```
玩家输入 "[调查] 追踪独眼修士"
    ↓
prompt_builder.py
  将当前世界状态（NPC身份、玩家属性、历史篡改记录...）
  注入 system prompt，让模型"记住"之前所有行动后果
    ↓
model_client.py
  调用本地 Qwen2.5-7B + LoRA 模型生成输出：
  ┌─────────────────────────────────────────┐
  │ <think>                                 │
  │   修士（monk_blind）的隐藏身份是...      │  ← GM 视角可见
  │ </think>                                │
  │ <tool_code>                             │
  │   set_npc_identity('monk_blind', '...') │  ← 真实执行
  │   update_npc_stat('player', 'san', -10) │
  │ </tool_code>                            │
  │ 修士停下脚步，缓缓转身...               │  ← 玩家看到
  └─────────────────────────────────────────┘
    ↓
tool_executor.py
  解析 <tool_code>，在安全沙箱中执行：
  - NPC 身份真正被修改（world_state.py 中的字典）
  - 玩家 SAN 值真正减少
  - 下一轮对话时世界状态摘要会反映这些变化
    ↓
玩家看到叙事文本（<think> 和 <tool_code> 被隐藏）
```

---

## 世界状态说明

所有状态都保存在内存字典中，`/save` 命令序列化为 JSON。

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

### 工具函数说明

模型输出的 `<tool_code>` 中会调用这些函数（全部真实执行）：

| 函数 | 作用 |
|------|------|
| `set_npc_identity(id, identity)` | 揭示/改变 NPC 的隐藏身份 |
| `update_npc_stat(id, stat, delta)` | 修改 NPC 或玩家的属性值 |
| `update_world_stat(stat, delta)` | 修改全局世界属性 |
| `rewrite_history(fact)` | 岁月史书篡改历史 |
| `set_world_flag(flag, value)` | 设置世界事件标志 |
| `trigger_world_event(id, ...)` | 触发全局事件 |
| `update_zone_stat(zone, stat, delta)` | 修改特定区域状态 |
| `spawn_hidden_npc(id, location)` | 在世界中生成隐藏 NPC |
| `add_quest_seed(quest_id)` | 埋下任务伏笔 |
| `create_item(id, ...)` | 创建并放置物品 |

---

## 注意事项

- **显存需求**：Qwen2.5-7B bfloat16 约占用 15~16 GB 显存，RTX 4090（24GB）可正常运行
- **首次加载**：模型 LoRA 合并约需 15~30 秒，之后推理正常
- **历史截断**：默认保留最近 8 轮对话历史，超出则截断（防止 OOM）
- **执行错误**：若模型输出的工具调用有语法错误，游戏不会崩溃，错误会在 GM 模式下显示
