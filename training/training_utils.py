"""
训练辅助工具模块

包含 LoRA 配置函数，常用训练工具等
"""

from peft import LoraConfig, TaskType

def get_lora_config():
    """
    返回 LoRA 配置对象
    """
    return LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
