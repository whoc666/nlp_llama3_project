# 分词器加载工具

"""
分词器加载模块

统一管理分词器加载，方便推理和训练调用
"""

from transformers import AutoTokenizer

def load_tokenizer(model_path):
    """
    加载分词器
    参数:
        model_path: str，模型目录或名称
    返回:
        tokenizer 对象
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer
