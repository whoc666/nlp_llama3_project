"""
数据清洗与准备脚本

功能：
- 读取原始数据
- 清洗并格式化数据
- 输出成 jsonl 文件供训练使用
"""

import json
import os
from datasets import load_dataset

def load_dataset_from_hf(dataset_name="whoc666/nlp_llama3_project_dataset", split="train"):
    """
    从 Hugging Face Hub 加载数据集。

    参数:
    - dataset_name: str，Hugging Face 上数据集的路径
    - split: str，加载数据集的拆分（train/test/validation）

    返回:
    - datasets.Dataset 对象
    """
    print(f"开始加载数据集: {dataset_name}, 拆分: {split}")
    dataset = load_dataset(dataset_name, split=split)
    print(f"加载完成，数据集大小: {len(dataset)}")
    return dataset


def load_raw_data(raw_dir="data/raw"):
    """
    读取原始数据文件
    参数:
        raw_dir: 原始数据文件夹路径
    返回:
        raw_data: list，包含所有原始数据条目
    """
    raw_data = []
    file_path = os.path.join(raw_dir, "raw_data.txt")  # 示例文件名
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_data.append(line.strip())
    return raw_data

def clean_data(raw_data):
    """
    简单数据清洗示例
    参数:
        raw_data: list，原始数据
    返回:
        cleaned: list，清洗后数据
    """
    cleaned = []
    for item in raw_data:
        if item:  # 去除空行
            cleaned.append(item)
    return cleaned

def save_to_jsonl(cleaned_data, save_path="data/processed/train_data.jsonl"):
    """
    保存数据为jsonl格式，每行为一个json对象
    参数:
        cleaned_data: list，清洗后的数据
        save_path: str，保存文件路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for text in cleaned_data:
            example = {
                "instruction": text,
                "output": "这是示范回答"  # 训练目标输出
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    raw = load_raw_data()
    print(f"读取原始数据条数：{len(raw)}")
    cleaned = clean_data(raw)
    print(f"清洗后数据条数：{len(cleaned)}")
    save_to_jsonl(cleaned)
    print("数据处理完成，保存路径：data/processed/train_data.jsonl")


