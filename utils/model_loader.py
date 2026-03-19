# ============================================================================
# MODEL LOADING UTILITIES
# Helper functions to load the base model and tokenizer.
# ============================================================================

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path

# MODEL_ID = "openlm-research/open_llama_7b"  # Target model to quantize
# OUTPUT_ROOT = Path("./quantized_models")  # Directory for saving quantized models

MODEL_ID = "openlm-research/open_llama_3b"  # Target model to quantize
OUTPUT_ROOT = Path("./quantized_models_3b")  # Directory for saving quantized models


def load_tokenizer(model_id: str = MODEL_ID):
    """
    Load tokenizer for the specified model.

    Args:
        model_id: HuggingFace model ID or local path

    Returns:
        AutoTokenizer instance
    """
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    # Ensure pad token is set (required for batched inference)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return tok


def load_base_model(model_id: str = MODEL_ID):
    """
    Load base model in FP16 precision.

    Args:
        model_id: HuggingFace model ID or local path

    Returns:
        AutoModelForCausalLM instance loaded in FP16
    """
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use FP16 to save memory
        device_map="auto",  # Automatically distribute across available GPUs
        low_cpu_mem_usage=True,  # Optimize CPU memory during loading
        use_safetensors=True,
    )
