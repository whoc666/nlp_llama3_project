"""
Gradio Web UI程序

提供网页交互界面，用户输入问题，模型生成回答
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "../output"  # 模型目录

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
model.eval()

def respond(text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=512, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

with gr.Blocks() as demo:
    gr.Markdown("# NLP 学习助手")
    txt = gr.Textbox(lines=2, placeholder="请输入问题...")
    out = gr.Textbox(label="模型回答", lines=4)
    btn = gr.Button("生成回答")
    btn.click(fn=respond, inputs=txt, outputs=out)

if __name__ == "__main__":
    demo.launch()
