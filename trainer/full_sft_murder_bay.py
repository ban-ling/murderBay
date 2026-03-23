import os
import sys
import json
import torch
import argparse
import time
import warnings
from contextlib import nullcontext
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

# 假设你的项目结构中包含这些模块
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, \
    init_model, SkipBatchSampler

warnings.filterwarnings('ignore')

# 1. 定义特殊 Token
SPECIAL_TOKENS = {
    "thought": "<|thought|>",
    "tool": "<|tool|>",
    "response": "<|response|>",
    "action": "<|silent_action|>"
}


# 2. 定制化数据集，支持 Loss Masking
class MurderBayDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        # 构造训练字符串
        # 格式：[玩家指令] {instruction} <|thought|> {thought} <|tool|> {tool_code} <|response|> {output}
        prompt = f"[玩家指令] {item['instruction']}"
        # 这里的 key 对应你 JSONL 中的字段
        answer = f"{SPECIAL_TOKENS['thought']}{item.get('thought', '')}{SPECIAL_TOKENS['tool']}{item.get('tool_code', '')}{SPECIAL_TOKENS['response']}{item.get('output', '')}"

        full_text = prompt + answer + self.tokenizer.eos_token

        enc = self.tokenizer(full_text, add_special_tokens=False)
        input_ids = enc['input_ids']

        # 关键步骤：计算 Prompt 的长度，将其对应的 Label 设置为 -100
        prompt_enc = self.tokenizer(prompt, add_special_tokens=False)
        prompt_len = len(prompt_enc['input_ids'])

        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # 截断与填充
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]

        # Padding
        padding_len = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_len
        labels += [-100] * padding_len

        return torch.tensor(input_ids), torch.tensor(labels)


def train_epoch(epoch, loader, iters, optimizer, model, scaler, autocast_ctx, args):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader):
        input_ids, labels = input_ids.to(args.device), labels.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {loss.item() * args.accumulation_steps:.4f}, lr: {lr:.8f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--learning_rate", type=float, default=1e-4)  # 调高 LR
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=512)  # 增加长度
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--save_weight", default='murder_bay_demo', type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/murder_bay_data.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # 初始化分布式与种子
    local_rank = init_distributed_mode()
    setup_seed(42)

    # 加载模型与分词器
    # 注意：这里需要确保你的 init_model 能返回原生 tokenizer
    lm_config = MiniMindConfig(hidden_size=512, num_hidden_layers=8)
    model, tokenizer = init_model(lm_config, 'pretrain', device=args.device)

    # 【核心修改】添加特殊 Token 并扩充 Embedding 层
    new_tokens = list(SPECIAL_TOKENS.values())
    num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
    model.resize_token_embeddings(len(tokenizer))
    Logger(f"词表已扩充，加入：{new_tokens}")

    # 数据准备
    train_ds = MurderBayDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)

    # 开始训练
    model.train()
    for epoch in range(args.epochs):
        train_epoch(epoch, loader, len(loader), optimizer, model, scaler, autocast_ctx, args)

        # 每个 Epoch 保存一次
        if is_main_process():
            torch.save(model.state_dict(), f"{args.save_dir}/{args.save_weight}_epoch_{epoch}.pth")

    if dist.is_initialized(): dist.destroy_process_group()