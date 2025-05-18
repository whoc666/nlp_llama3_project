# 上传模型到 Hugging Face 的脚本

"""
上传模型到 Hugging Face Hub 脚本

使用 huggingface_hub 库将本地模型上传到你创建的模型仓库
"""

from huggingface_hub import HfApi, Repository
import os

def upload_model(model_dir, repo_id, token):
    """
    上传模型目录到 Hugging Face
    参数:
        model_dir: 本地模型目录
        repo_id: 仓库名称，例如 "whoc666/nlp-llama3-project-model"
        token: HF 访问令牌
    """
    repo = Repository(local_dir=model_dir, clone_from=repo_id, use_auth_token=token)
    repo.git_add(auto_lfs_track=True)
    repo.git_commit("Upload fine-tuned model")
    repo.git_push()

if __name__ == "__main__":
    # TODO: 替换成实际路径和仓库名，HF_TOKEN需提前设置为环境变量或替换为字符串
    MODEL_DIR = "./output"
    REPO_ID = "whoc666/nlp-llama3-project-model"
    HF_TOKEN = os.getenv("HF_TOKEN")  # 你需要先设置环境变量：export HF_TOKEN="你的token"

    if HF_TOKEN is None:
        print("请先设置环境变量 HF_TOKEN")
    else:
        upload_model(MODEL_DIR, REPO_ID, HF_TOKEN)
