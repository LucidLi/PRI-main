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
    Load a pretrained causal language model and its tokenizer.

    Args:
        model_path (str): Hugging Face model name or local model path.
        FP16 (bool): Whether to load the model with float16 weights.
        tokenizer_path (str or None): Tokenizer path; defaults to model_path.
        device (str): Device used to load the model.
        **kwargs: Additional arguments passed to
            ``AutoModelForCausalLM.from_pretrained``.

    Returns:
        tuple: The loaded model and tokenizer.
    """
    if FP16:
        model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,   # Load weights in half precision.
                trust_remote_code=True,      # Allow model-specific remote code.
                **kwargs                     # Forward additional arguments.
            ).to(device).eval()              # Move to the target device and evaluation mode.
    else:
        # Load with the default precision, usually float32.
        model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                **kwargs
            ).to(device).eval()

    # Guanaco-13b-merged requires the Llama tokenizer.
    if model_path == "timdettmers/guanaco-13b-merged":
        tokenizer_path = "huggyllama/llama-7b"

    # Use model_path when no tokenizer path is provided.
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False             # Some models require the non-fast tokenizer.
    )


    if 'oasst-sft-6-llama-30b' in tokenizer_path:    # 1. Set special-token IDs for OASST-Llama.
        tokenizer.bos_token_id = 1   # Beginning-of-sequence token.
        tokenizer.unk_token_id = 0   # Unknown token.
    if 'guanaco' in tokenizer_path:                  # 2. Set special-token IDs for Guanaco.
        tokenizer.eos_token_id = 2   # End-of-sequence token.
        tokenizer.unk_token_id = 0   # Unknown token.
    if 'llama-2' in tokenizer_path:                  # 3. Configure Llama-2 padding.
        tokenizer.pad_token = tokenizer.unk_token # Use the unknown token for padding.
        tokenizer.padding_side = 'left'           # Left-pad for autoregressive generation.
    if 'falcon' in tokenizer_path:                   # 4. Configure Falcon padding.
        tokenizer.padding_side = 'left' # Left-pad the input.
    if not tokenizer.pad_token:                      # 5. Use EOS as padding when no pad token exists.
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
