import gc
import logging
import subprocess
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.model import get_conversation_template


def load_model_and_tokenizer(model_path, FP16 = True, tokenizer_path=None, device='cuda:0', **kwargs):
    """
    加载预训练的语言模型和对应的分词器

    参数:
        model_path (str): 预训练模型在 Hugging Face 上的名称或本地路径
                          - 示例: "lmsys/vicuna-7b-v1.5", "meta-llama/Llama-2-7b-chat-hf"
                          - 类型: str

        FP16 (bool, 可选): 是否使用半精度浮点数(float16)加载模型，可减少内存占用
                          - 默认: True
                          - 类型: bool

        tokenizer_path (str, 可选): 分词器的路径，如果为 None 则使用 model_path
                          - 默认: None
                          - 类型: str

        device (str, 可选): 模型加载的设备
                          - 默认: 'cuda:0' (第一个GPU)
                          - 类型: str
                          - 示例: 'cuda:0', 'cpu', 'cuda:1'

        **kwargs: 传递给 transformers.AutoModelForCausalLM.from_pretrained 的额外参数
                  - 常用参数:
                    - low_cpu_mem_usage (bool): 减少CPU内存使用
                    - use_cache (bool): 是否使用KV缓存
                    - torch_dtype (torch.dtype): 指定张量数据类型

    返回:
        tuple: 包含两个元素的元组
            - model (transformers.PreTrainedModel): 加载的因果语言模型
                - 设备: 已移动到指定device
                - 模式: 设置为eval模式(不训练)

            - tokenizer (transformers.PreTrainedTokenizer): 对应的分词器
                - 类型: 具体分词器类根据模型类型决定
                - 功能: 文本编码/解码，特殊token处理
    """
    if FP16:
        model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,   # 使用半精度浮点数
                trust_remote_code=True,      # 信任远程代码(如自定义模型)
                **kwargs                     # 传递额外参数
            ).to(device).eval()              # 移动到指定设备并设为评估模式
    else:
        # 使用默认精度(通常为float32)加载模型
        model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                **kwargs
            ).to(device).eval()

    # 特殊处理: guanaco-13b-merged模型使用llama的分词器
    if model_path == "timdettmers/guanaco-13b-merged":
        tokenizer_path = "huggyllama/llama-7b"

    # 如果 tokenizer_path 未指定，使用 model_path
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False             # 不使用快速分词器(某些模型需要)
    )


    if 'oasst-sft-6-llama-30b' in tokenizer_path:    # 1. OASST-Llama模型: 设置特殊token的ID
        tokenizer.bos_token_id = 1   # 设置 开始符 ID
        tokenizer.unk_token_id = 0   # 设置 未知符 ID
    if 'guanaco' in tokenizer_path:                  # 2. Guanaco模型: 设置特殊token的I
        tokenizer.eos_token_id = 2   # 设置 结束符 ID
        tokenizer.unk_token_id = 0   # 设置 未知符 ID
    if 'llama-2' in tokenizer_path:                  # 3. Llama-2模型: 设置填充token和填充方向
        tokenizer.pad_token = tokenizer.unk_token # 使用 未知token 作为 填充token
        tokenizer.padding_side = 'left'           # 在 左侧 填充 (适合自回归生成)
    if 'falcon' in tokenizer_path:                   # 4. Falcon模型: 设置填充方向
        tokenizer.padding_side = 'left' # 在左侧填充
    if not tokenizer.pad_token:                      # 5. 通用处理: 如果分词器没有填充token，使用结束token作为填充token
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def get_latest_commit_info():
    try:
        # Get the latest commit hash
        commit_hash = subprocess.run(["git", "log", "-1", "--format=%H"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Get the latest commit date
        commit_date = subprocess.run(["git", "log", "-1", "--format=%cd"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check if both commands were executed successfully
        if commit_hash.returncode == 0 and commit_date.returncode == 0:
            return commit_hash.stdout.strip(), commit_date.stdout.strip()
        else:
            error_message = commit_hash.stderr if commit_hash.returncode != 0 else commit_date.stderr
            return "Error fetching commit information:", error_message
    except FileNotFoundError:
        # Git not installed or not found in the path
        return "Git is not installed or not found in the path.", ""