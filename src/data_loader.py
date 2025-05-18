import os
import json
import re
import gc
import shutil
import time
import requests
from datasets import load_dataset, Dataset
from tqdm import tqdm

# 清理缓存函数（仅在最后可选调用）
def clean_cache():
    try:
        shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)
        shutil.rmtree(os.path.expanduser("~/.cache/datasets"), ignore_errors=True)
        gc.collect()
        print("✅ 缓存已清理")
    except OSError as e:
        print(f"❌ 清理缓存失败: {e}")

# 检查磁盘空间
def check_disk_space():
    stat = os.statvfs('/')
    free_space = stat.f_bavail * stat.f_frsize / (1024 ** 3)  # GB
    print(f"可用磁盘空间：{free_space:.2f} GB")
    if free_space < 5:
        print("⚠️ 磁盘空间不足，建议至少 5GB")

# try_except 函数：加载和过滤数据集
def try_except_load_dataset(dataset_name, config_name=None, split="train", map_fn=None, filter_fn=None, limit=None, streaming=False):
    try:
        print(f"▶ 正在加载 {dataset_name} 数据集...")
        # 检查网络
        try:
            response = requests.get("https://api.hf.co", timeout=5)
            print(f"网络连接正常 (状态码: {response.status_code})")
        except requests.RequestException as e:
            print(f"⚠️ 网络连接问题: {e}")

        # 加载数据集
        start_time = time.time()
        dataset = load_dataset(dataset_name, config_name, split=split, streaming=streaming) if config_name else load_dataset(dataset_name, split=split, streaming=streaming)
        print(f"数据集加载耗时：{time.time() - start_time:.2f} 秒")

        # 非流式加载
        print(f"{dataset_name} 原始数据量：{len(dataset)}")
        if map_fn:
            print(f"应用映射函数...")
            dataset = dataset.map(map_fn, num_proc=2)
        if filter_fn:
            print(f"应用过滤函数...")
            dataset = dataset.filter(filter_fn, num_proc=2)
        if limit and len(dataset) > limit:
            print(f"限制数据量到 {limit} 条...")
            dataset = dataset.select(range(limit))
        print(f"过滤后的 {dataset_name} 数据量：{len(dataset)}")
        if len(dataset) > 0:
            print(f"{dataset_name} 示例：", dataset[0])
        return dataset
    except Exception as e:
        print(f"❌ 加载或处理 {dataset_name} 数据失败: {e}")
        return None

# 创建保存路径
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
output_dir = os.path.join(script_dir, "..", "data")
os.makedirs(output_dir, exist_ok=True)

print(f"当前工作目录：{os.getcwd()}")
print(f"脚本目录：{script_dir}")
print(f"输出目录：{output_dir}")
check_disk_space()

# 关键词（简化以加速）
keywords = ["machine learning", "artificial intelligence"]

# 清洗文本函数
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# 筛选函数
def filter_ai_articles(example):
    content = clean_text(
        example.get("text", "") or
        example.get("content", "") or
        example.get("abstract", "") or
        example.get("summary", "") or
        (example.get("question", "") + example.get("answer", "") if example.get("question") or example.get("answer") else "")
    )
    for kw in keywords:
        if kw.lower() in content.lower():
            return True
    return False

# 主流程
# 1. 加载数据集
filtered_wiki = try_except_load_dataset(
    dataset_name="wikimedia/wikipedia",
    config_name="20231101.en",
    split="train[:1000]",  # 限制前 1000 条
    filter_fn=filter_ai_articles,
    limit=1000,
    streaming=False
)

filtered_arxiv = try_except_load_dataset(
    dataset_name="ccdv/arxiv-summarization",
    split="train[:1000]",
    map_fn=lambda x: {"text": x["abstract"]},
    filter_fn=filter_ai_articles,
    limit=1000,
    streaming=False
)

filtered_scitldr = try_except_load_dataset(
    dataset_name="allenai/scitldr",
    split="train",
    map_fn=lambda x: {"text": x["summary"]},
    filter_fn=filter_ai_articles,
    streaming=False
)

# 2. 合并数据集
merged_dataset = None
try:
    print("▶ 合并数据...")
    datasets_to_merge = [ds for ds in [filtered_wiki, filtered_arxiv, filtered_scitldr] if ds is not None and len(ds) > 0]
    if not datasets_to_merge:
        raise ValueError("没有可合并的数据集")
    merged_dataset = datasets_to_merge[0]
    for ds in datasets_to_merge[1:]:
        merged_dataset = merged_dataset.concatenate(ds)
    MAX_SAMPLES = 2000
    if len(merged_dataset) > MAX_SAMPLES:
        merged_dataset = merged_dataset.select(range(MAX_SAMPLES))
    print(f"✅ 最终数据条数：{len(merged_dataset)}")
    print("合并数据集示例：", merged_dataset[0] if len(merged_dataset) > 0 else "空数据集")
except Exception as e:
    print(f"❌ 合并数据失败: {e}")

# 3. 保存到文件
output_file = os.path.join(output_dir, '1_data_en_wiki_arxiv_scitldr.jsonl')
if merged_dataset:
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in merged_dataset:
                content = clean_text(example.get("text", ""))
                if content:
                    json.dump({"text": content}, f, ensure_ascii=False)
                    f.write('\n')
        print(f"✅ 数据保存完成，文件位置：{output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"输出文件行数：{len(lines)}")
            if lines:
                print("输出文件示例：", lines[0])
    except Exception as e:
        print(f"❌ 数据保存失败: {e}")
else:
    print("❌ 无数据可保存，跳过保存步骤")

# 可选：清理缓存
# clean_cache()