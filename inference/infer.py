"""
命令行推理程序

加载训练好的模型，进行文本生成推理
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from inference.tokenizer_loader import load_tokenizer

def main():
    model_path = "./output"  # 模型保存路径

    tokenizer = load_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    print("欢迎使用NLP学习助手，输入 exit 退出")
    while True:
        user_input = input("输入：")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("退出程序")
            break

        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=512, do_sample=True, temperature=0.7)
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("模型回复：", reply)

if __name__ == "__main__":
    main()
