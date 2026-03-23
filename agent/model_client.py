"""
模型推理客户端。

支持两种后端：
1. local  — 从本地路径加载基座模型 + LoRA 适配器（AutoDL 上使用）
2. api    — 通过 OpenAI 兼容 API 调用远程模型（测试用，如 swift deploy 后）

通过环境变量或构造参数控制后端选择。
"""

import os
from typing import Optional

# 默认路径（与训练脚本保持一致，可通过环境变量覆盖）
DEFAULT_BASE_MODEL = os.getenv(
    "MURDER_BAY_BASE_MODEL",
    "/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct",
)
DEFAULT_LORA_ADAPTER = os.getenv(
    "MURDER_BAY_LORA",
    "/root/output/murder_bay_lora/v2-20260322-002516/checkpoint-207",
)
DEFAULT_API_BASE = os.getenv("MURDER_BAY_API_BASE", "http://localhost:8000/v1")
DEFAULT_API_KEY = os.getenv("MURDER_BAY_API_KEY", "EMPTY")


class ModelClient:
    """
    统一的模型推理接口。

    用法示例（本地）：
        client = ModelClient(backend="local")
        response = client.chat(messages)

    用法示例（API）：
        client = ModelClient(backend="api")
        response = client.chat(messages)
    """

    def __init__(
        self,
        backend: str = "local",
        base_model: str = DEFAULT_BASE_MODEL,
        lora_adapter: str = DEFAULT_LORA_ADAPTER,
        api_base: str = DEFAULT_API_BASE,
        api_key: str = DEFAULT_API_KEY,
        max_new_tokens: int = 512,
        temperature: float = 0.85,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
    ):
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        if backend == "local":
            self._init_local(base_model, lora_adapter)
        elif backend == "api":
            self._init_api(api_base, api_key)
        else:
            raise ValueError(f"不支持的 backend: {backend}，请选择 'local' 或 'api'")

    # ──────────────────────────────────────────
    # 本地模式初始化
    # ──────────────────────────────────────────

    def _init_local(self, base_model: str, lora_adapter: str) -> None:
        print(f"[模型] 加载基座模型: {base_model}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )

        if lora_adapter and os.path.exists(lora_adapter):
            print(f"[模型] 加载 LoRA 适配器: {lora_adapter}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_adapter)
            model = model.merge_and_unload()   # 合并权重，推理更快
            print("[模型] LoRA 已合并")
        else:
            print(f"[警告] LoRA 路径不存在或未指定，使用纯基座模型: {lora_adapter}")

        model.eval()
        self.model = model
        print("[模型] 就绪 ✓")

    # ──────────────────────────────────────────
    # API 模式初始化（swift deploy 或 vllm 后端）
    # ──────────────────────────────────────────

    def _init_api(self, api_base: str, api_key: str) -> None:
        print(f"[模型] API 模式，endpoint: {api_base}")
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("API 模式需要安装 openai 包：pip install openai")
        self._api_client = OpenAI(base_url=api_base, api_key=api_key)
        self._api_model_name = "default"
        print("[模型] API 客户端就绪 ✓")

    # ──────────────────────────────────────────
    # 统一推理接口
    # ──────────────────────────────────────────

    def chat(self, messages: list[dict]) -> str:
        """
        接收 OpenAI 格式的消息列表，返回模型回复字符串。

        messages 格式：
            [
                {"role": "system", "content": "..."},
                {"role": "user",   "content": "..."},
                {"role": "assistant", "content": "..."},  # 历史轮次
                {"role": "user",   "content": "当前输入"},
            ]
        """
        if self.backend == "local":
            return self._chat_local(messages)
        else:
            return self._chat_api(messages)

    def _chat_local(self, messages: list[dict]) -> str:
        import torch

        # 使用 tokenizer 的 apply_chat_template 构建输入
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 只取新生成的 token
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def _chat_api(self, messages: list[dict]) -> str:
        response = self._api_client.chat.completions.create(
            model=self._api_model_name,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return response.choices[0].message.content


# ──────────────────────────────────────────────
# 单例懒加载（供其他模块直接 import 使用）
# ──────────────────────────────────────────────

_client: Optional[ModelClient] = None


def get_client(
    backend: str = "local",
    base_model: str = DEFAULT_BASE_MODEL,
    lora_adapter: str = DEFAULT_LORA_ADAPTER,
    **kwargs,
) -> ModelClient:
    """获取全局模型客户端，首次调用时初始化。"""
    global _client
    if _client is None:
        _client = ModelClient(
            backend=backend,
            base_model=base_model,
            lora_adapter=lora_adapter,
            **kwargs,
        )
    return _client
