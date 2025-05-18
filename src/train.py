# ✅ 第一部分：安装依赖，确保版本兼容（transformers >=4.41.0）
!pip install -q \
  transformers==4.41.1 \
  peft==0.10.0 \
  accelerate==0.27.2 \
  bitsandbytes==0.42.0 \
  datasets \
  sentence-transformers \
  huggingface_hub \
  fsspec==2025.3.2

# ✅ 第二部分：导入所需库
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
import os, json

# ✅ 第三部分：登录 Hugging Face（将你自己的 token 替换到下面这个字符串中）
# ⚠️ 请将 YOUR_HF_TOKEN 替换为你自己的 Token（设置为私密模式）
login("YOUR_HF_TOKEN")

# ✅ 第四部分：加载数据集（来自你的 Hugging Face 仓库）
data_path = "whoc666/nlp_llama3_project"
file_name = "1_data_en_wiki_arxiv.jsonl"
dataset = load_dataset(data_path, data_files=file_name, split="train")

# ✅ 第五部分：加载 LLaMA 3 1.5B 模型和分词器（4bit 量化）
model_id = "meta-llama/Meta-Llama-3-1.5B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# ✅ 第六部分：准备模型用于 LoRA 训练
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# ✅ 第七部分：定义 LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ✅ 第八部分：分词 + 格式化数据
def tokenize_function(example):
    outputs = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    outputs["labels"] = outputs["input_ids"]
    return outputs

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ✅ 第九部分：训练参数设置
training_args = TrainingArguments(
    output_dir="llama3-1.5b-nlp",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    push_to_hub=True,
    hub_model_id="whoc666/nlp_llama3_project",
    hub_strategy="every_save",
    hub_token="YOUR_HF_TOKEN"
)

# ✅ 第十部分：训练模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

# ✅ 第十一部分：推送模型到 Hugging Face Hub
trainer.push_to_hub()
