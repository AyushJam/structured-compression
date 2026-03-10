# ============================================================================
# CALIBRATION DATA LOADING
# ============================================================================

from typing import Iterable, List
from datasets import load_dataset

def load_wikitext_calibration(num_samples: int = 128, min_chars: int = 100) -> List[str]:
    """
    Load calibration texts from WikiText-2 dataset.
    
    Calibration data is used by quantization algorithms to determine optimal
    quantization parameters that minimize accuracy loss.

    Args:
        num_samples: Number of text samples to load
        min_chars: Minimum character length for valid samples (filters headers/empty lines)
    
    Returns:
        List of text strings for calibration
    """
    print(f"Loading WikiText-2 dataset for calibration...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    texts: List[str] = []
    for item in dataset:
        text = item["text"].strip()
        
        # Filter out very short texts (headers, empty lines, etc.)
        if len(text) >= min_chars:
            texts.append(text)
            if len(texts) >= num_samples:
                break
    
    # If we didn't find enough samples, repeat existing ones
    if len(texts) < num_samples:
        print(f"Warning: Only found {len(texts)} samples, repeating to reach {num_samples}")
        repeats = (num_samples + len(texts) - 1) // len(texts)
        texts = (texts * repeats)[:num_samples]
    
    print(f"✓ Loaded {len(texts)} calibration samples from WikiText-2")
    return texts

