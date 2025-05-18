"""
QLoRA 微调训练主程序模板

功能：
- 加载预训练 LLaMA3 模型和分词器
- 使用 QLoRA 进行低秩适配微调
- 训练循环与保存模型
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader

def main():
    # 模型名称或路径（LLaMA3 1.5B）
    base_model_name = "meta-llama/Llama-2-7b-hf"  # 替换为实际模型路径
    # 数据路径
    dataset_path = "data/processed/train_data.jsonl"
    # 输出目录
    output_dir = "output"

    # 1. 加载分词器和模型
    print("加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=True,          # 4bit 量化
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 2. 配置 LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # 3. 加载数据集
    print("加载数据集...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def preprocess_function(examples):
        # 把 instruction 和 output 拼成模型输入格式
        inputs = []
        for ins, out in zip(examples["instruction"], examples["output"]):
            prompt = f"Instruction: {ins}\nAnswer: {out}"
            inputs.append(prompt)
        tokenized = tokenizer(inputs, padding=True, truncation=True, max_length=512)
        return tokenized

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    train_dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)

    # 4. 训练参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # 5. 训练循环（简化示例）
    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item():.4f}")

    # 6. 保存微调后的模型
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"模型保存至：{output_dir}")

if __name__ == "__main__":
    main()
