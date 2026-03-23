import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params

warnings.filterwarnings('ignore')

# 1. 定义与训练脚本一致的特殊 Token
SPECIAL_TOKENS = {
    "thought": "<|thought|>",
    "tool": "<|tool|>",
    "response": "<|response|>",
    "action": "<|silent_action|>"
}

def init_model(args):
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    
    # 核心：必须扩充词表以匹配训练时的权重维度
    new_tokens = list(SPECIAL_TOKENS.values())
    tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
    
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        
        # 调整 Embedding 层大小
        model.resize_token_embeddings(len(tokenizer))
        
        moe_suffix = '_moe' if args.use_moe else ''
        # 默认加载 murder_bay_demo 相关的权重
        ckp = f'./{args.save_dir}/{args.weight}.pth'
        print(f"正在加载权重: {ckp}")
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        model.resize_token_embeddings(len(tokenizer)) # Transformers 格式也需要调整

    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind 谋杀湾专用推理脚本")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='murder_bay_demo_epoch_4', type=str, help="权重名称（例如 murder_bay_demo_epoch_0）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推")
    parser.add_argument('--max_new_tokens', default=1024, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.7, type=float, help="生成温度")
    parser.add_argument('--top_p', default=0.9, type=float, help="采样阈值")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    
    model, tokenizer = init_model(args)
    
    print("\n" + "="*50)
    print("💀 欢迎来到谋杀湾 (Murder Bay) 💀")
    print("系统已就绪，请输入指令开启你的混沌叙事。")
    print("="*50 + "\n")

    # 自定义 Streamer 以处理特殊标签的显示（可选）
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    while True:
        prompt = input('💬 [玩家指令]: ')
        if prompt.lower() in ['exit', 'quit', '退出']:
            break
        
        setup_seed(random.randint(0, 2048))
        
        # 构造与训练一致的输入格式
        # 格式：[玩家指令] {instruction} <|thought|>
        full_prompt = f"[玩家指令] {prompt}{SPECIAL_TOKENS['thought']}"
        
        inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to(args.device)

        print('🤖 [Agent 思考与行动]: \n', end='')
        st = time.time()
        
        # 开始生成
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=1.1 # 稍微增加惩罚防止复读
            )
        
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n')

if __name__ == "__main__":
    main()


