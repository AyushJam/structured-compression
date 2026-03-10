# ============================================================================
# RTN (ROUND-TO-NEAREST) QUANTIZATION IMPLEMENTATION
# ============================================================================

import torch
import torch.nn as nn
from pathlib import Path
from typing import Iterable, List
import utils.model_loader as model_loader

def rtn_quantize_tensor(w: torch.Tensor, bits: int, group_size: int = 128) -> torch.Tensor:
    """
    Quantize a weight tensor using Round-To-Nearest method with group-wise scaling.
    
    Args:
        w: Weight tensor to quantize
        bits: Target bit width (e.g., 4 for 4-bit, 8 for 8-bit)
        group_size: Size of groups for group-wise quantization (smaller = more accurate but larger overhead)
    
    Returns:
        Quantized and dequantized weight tensor (still in original dtype)
    """
    # Skip quantization for 16-bit or higher
    if bits >= 16:
        return w

    # Calculate quantization range for signed integers
    qmin = -(2 ** (bits - 1))  # e.g., -8 for 4-bit
    qmax = (2 ** (bits - 1)) - 1  # e.g., 7 for 4-bit

    orig_shape = w.shape
    w_flat = w.reshape(-1, w.shape[-1])  # Flatten to 2D for processing

    # Per-channel quantization (when group_size is too large or disabled)
    if group_size <= 0 or group_size >= w_flat.shape[1]:
        # Find max absolute value per channel
        max_abs = w_flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = max_abs / qmax
        
        # Quantize and dequantize
        q = torch.round(w_flat / scale).clamp(qmin, qmax)
        w_q = q * scale
    
    # Group-wise quantization (more accurate)
    else:
        n_cols = w_flat.shape[1]
        
        # Pad if necessary to make columns divisible by group_size
        pad = (group_size - (n_cols % group_size)) % group_size
        if pad > 0:
            w_flat = torch.cat([
                w_flat, 
                torch.zeros(w_flat.size(0), pad, device=w.device, dtype=w.dtype)
            ], dim=1)

        # Reshape into groups
        w_grp = w_flat.view(w_flat.size(0), -1, group_size)
        
        # Find max absolute value per group
        max_abs = w_grp.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
        scale = max_abs / qmax
        
        # Quantize and dequantize
        q = torch.round(w_grp / scale).clamp(qmin, qmax)
        w_q = (q * scale).view(w_flat.size(0), -1)

        # Remove padding
        if pad > 0:
            w_q = w_q[:, :n_cols]

    return w_q.reshape(orig_shape).to(w.dtype)


@torch.no_grad()
def quantize_rtn_inplace(model: nn.Module, bits: int, group_size: int = 128):
    """
    Apply RTN quantization to all Linear layers in the model (in-place).
    
    Args:
        model: Model to quantize
        bits: Target bit width
        group_size: Group size for quantization
    
    Returns:
        Modified model (also modifies in-place)
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Quantize the weight matrix
            module.weight.data = rtn_quantize_tensor(
                module.weight.data, 
                bits=bits, 
                group_size=group_size
            )
    return model


def quantize_and_save_rtn(
    model_id: str,
    bits_list: Iterable[int],
    out_root: Path = model_loader.OUTPUT_ROOT,
    group_size: int = 128,
):
    """
    Quantize model using RTN at multiple bit widths and save results.
    
    Args:
        model_id: HuggingFace model ID or path
        bits_list: List of bit widths to quantize to (e.g., [4, 8])
        out_root: Root directory for saving quantized models
        group_size: Group size for quantization
    """
    # Load tokenizer once (shared across all bit widths)
    tok = model_loader.load_tokenizer(model_id)
    
    # Quantize at each bit width
    for bits in bits_list:
        print(f"\n{'='*60}")
        print(f"Quantizing with RTN at {bits}-bit...")
        print(f"{'='*60}")
        
        # Load fresh model for each quantization
        model = model_loader.load_base_model(model_id)
        
        # Apply RTN quantization
        quantize_rtn_inplace(model, bits=bits, group_size=group_size)

        # Save quantized model
        out_dir = out_root / f"rtn_w{bits}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        
        print(f"✓ Saved RTN {bits}-bit model to {out_dir}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()