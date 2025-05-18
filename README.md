
# 🤖 NLP 学习助手项目（基于 LLaMA3 1.5B + QLoRA 微调）

本项目旨在构建一个中文 NLP 学习助手，基于 Meta 发布的 LLaMA3 1.5B 模型，通过 QLoRA 进行高效微调，结合 Gradio 构建用户交互界面，并部署到 Hugging Face Spaces。适用于 NLP 入门学习、术语查询、问答推理等场景。

---

## 📁 项目结构

```
nlp_llama3_project/
├── data/                  # 数据集原始数据与清洗代码
│   ├── raw/               # 原始数据文件
│   ├── processed/         # 清洗后的数据
│   └── prepare_data.py    # 数据清洗脚本
├── training/              # 模型训练相关代码
│   ├── train_qlora.py     # 微调主程序
│   └── training_utils.py  # LoRA配置与训练辅助
├── inference/             # 推理测试代码
│   ├── infer.py           # CLI 推理
│   └── tokenizer_loader.py# 分词器加载模块
├── gradio_app/            # Gradio 可视化界面与部署
│   ├── app.py             # Gradio 主程序
│   └── requirements.txt   # HF Spaces 所需依赖
├── scripts/               # 工具脚本
│   └── hf_upload_model.py # 上传模型到 Hugging Face
├── README.md              # 项目说明文件
├── LICENSE                # 开源协议（MIT）
└── .gitignore             # 忽略项配置
```

---

## 📊 技术栈

- Python
- Transformers（LLaMA3）
- PEFT（LoRA / QLoRA）
- BitsAndBytes（4bit 量化）
- Hugging Face Hub（模型、数据、Space 托管）
- Gradio（Web UI）
- Google Colab / VSCode（开发与训练）

---

## 🚀 项目目标

- [x] 构建中文 AI/NLP 学习语料数据集
- [x] 使用 QLoRA 微调 LLaMA3-1.5B 模型
- [ ] 上传微调权重到 Hugging Face Model Hub
- [ ] 构建 Gradio 网页助手界面
- [ ] 部署到 Hugging Face Spaces

---

## 📦 仓库分工（Hugging Face）

| 仓库类型       | 仓库名                                | 用途说明                                  |
| -------------- | ------------------------------------- | ----------------------------------------- |
| 模型仓库       | `whoc666/nlp_llama3_project-model`     | 存放微调后的模型权重                       |
| 数据仓库       | `whoc666/nlp_llama3_project-data`      | 存放训练语料（如 `*.jsonl` 文件）         |
| Space 展示仓库 | `whoc666/nlp_llama3_project-app`       | Gradio 网页问答助手，部署到 Spaces        |

---

## 📬 联系作者

如果你有任何建议或合作意向，请通过 [GitHub Issues](https://github.com/whoc666/nlp-llama3-project/issues) 与我联系。
