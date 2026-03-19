"""Shared evaluation utilities for quantized model benchmarking.

This module is designed to be imported by multiple evaluation notebooks.
It contains repeatable logic for computing perplexity, model size/bandwidth,
and saving standardized evaluation outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, List

import math
import torch
import pandas as pd
from tqdm import tqdm
import sys


def evaluate_perplexity(
    model: torch.nn.Module,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_samples: int = 100,
    max_length: int = 512,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate model perplexity on a dataset.

    Args:
        model: The language model to evaluate.
        tokenizer: The tokenizer instance.
        dataset_name: Dataset name (e.g. "wikitext").
        dataset_config: Dataset configuration (e.g. "wikitext-2-raw-v1").
        split: Dataset split to use.
        max_samples: Max number of samples to evaluate.
        max_length: Max sequence length.
        device: Device string, e.g. "cuda" or "cpu".

    Returns:
        Dict with perplexity, loss, and token counts.
    """

    from datasets import load_dataset

    # if device is not None:
    #     model = model.to(device)

    print(
        f"Evaluating perplexity on {dataset_name} {split} split ({max_samples} samples)..."
    )

    dataset = load_dataset(dataset_name, dataset_config, split=split)

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, item in enumerate(
            tqdm(dataset, desc="Evaluating", total=min(max_samples, len(dataset)))
        ):
            if i >= max_samples:
                break

            text = item["text"].strip()
            if len(text) < 50:
                continue

            encodings = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            )
            input_ids = encodings["input_ids"].to("cuda")
            # input_ids = encodings["input_ids"]

            if input_ids.shape[1] < 2:
                continue

            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            total_loss += loss.item() * input_ids.shape[1]
            total_tokens += input_ids.shape[1]

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "total_tokens": total_tokens,
    }


def calculate_model_size_and_bandwidth(
    model: torch.nn.Module,
    bits: int = 16,
    batch_size: int = 1,
    seq_length: int = 512,
    include_activations: bool = True,
) -> Dict[str, Any]:
    """Estimate model size and memory bandwidth.

    Args:
        model: The model to analyze.
        bits: Quantization bits (16 for FP16, 8 for INT8, 4 for INT4).
        batch_size: Batch size for inference.
        seq_length: Sequence length.
        include_activations: Whether to include activation storage estimate.

    Returns:
        Dictionary containing size/bandwidth metrics.
    """

    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = bits / 8
    weight_memory_bytes = total_params * bytes_per_param
    weight_memory_mb = weight_memory_bytes / (1024**2)
    weight_memory_gb = weight_memory_bytes / (1024**3)

    activation_memory_mb = 0
    if include_activations and hasattr(model, "config"):
        config = model.config
        hidden_size = getattr(config, "hidden_size", None)
        num_layers = getattr(config, "num_hidden_layers", None)
        if hidden_size is not None and num_layers is not None:
            activation_memory_bytes = (
                batch_size * seq_length * hidden_size * num_layers * 4 * 2
            )
            activation_memory_mb = activation_memory_bytes / (1024**2)

    total_memory_mb = weight_memory_mb + activation_memory_mb
    total_memory_gb = total_memory_mb / 1024

    # Simplified bandwidth estimate (worst-case: load all weights per token)
    bytes_per_token = weight_memory_bytes
    bandwidth_mb_per_token = bytes_per_token / (1024**2)

    return {
        "total_params": total_params,
        "total_params_millions": total_params / 1e6,
        "total_params_billions": total_params / 1e9,
        "weight_memory_mb": weight_memory_mb,
        "weight_memory_gb": weight_memory_gb,
        "activation_memory_mb": activation_memory_mb,
        "total_memory_mb": total_memory_mb,
        "total_memory_gb": total_memory_gb,
        "bits": bits,
        "bandwidth_mb_per_token": bandwidth_mb_per_token,
        "bandwidth_gb_per_token": bandwidth_mb_per_token / 1024,
    }


def evaluate_lm_harness(
    model_obj,      
    tokenizer_obj,  
    tasks: list = None,
    num_fewshot: int = 0,
    limit: int = None,
):
    from lm_eval.models.huggingface import HFLM
    from lm_eval import simple_evaluate

    # Wrap your existing model in the HFLM class
    lm = HFLM(
        pretrained=model_obj,
        tokenizer=tokenizer_obj,
        device="cuda"
    )

    # gen_kwargs are passed here to control "generate_until" tasks (GSM8K)
    # 'stop' sequences prevent the model from hallucinating a second question
    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
    )

    # FILTERING LOGIC: Keep only the high-level group results
    # MMLU groups usually appear in the results dict as the task name you passed (e.g., 'mmlu_stem')
    summary_results = {}
    for task_name in tasks:
        if task_name in results['results']:
            summary_results[task_name] = results['results'][task_name]
    
    # Return a simplified dict that the plotter can handle easily
    return {"results": summary_results}


def _flatten_record(
    record: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """Flatten a nested dict into a single-level dict using dot notation.

    Lists are serialized as JSON strings so they remain in a single CSV cell.
    """

    flat: Dict[str, Any] = {}
    for k, v in record.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flat.update(_flatten_record(v, new_key, sep=sep))
        elif isinstance(v, list):
            try:
                flat[new_key] = json.dumps(v)
            except Exception:
                flat[new_key] = str(v)
        else:
            flat[new_key] = v
    return flat


def _dict_to_dataframe(record: Dict[str, Any]) -> "pd.DataFrame":
    """Convert a single-record dict to a pandas DataFrame (one row)."""
    return pd.DataFrame([_flatten_record(record)])


def save_eval_result(
    result: Any,
    output_dir: Path,
    file_stem: str = "evaluation",
    save_csv: bool = True,
    save_json: bool = True,
) -> None:
    """Save evaluation results to a directory in JSON and/or CSV format.

    Args:
        result: Evaluation results (dict, list of dicts, or DataFrame).
        output_dir: Directory to save the result files.
        file_stem: Base filename (without extension).
        save_csv: Whether to write a CSV file.
        save_json: Whether to write a JSON file.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_json:
        json_path = output_dir / f"{file_stem}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if save_csv:
        if isinstance(result, pd.DataFrame):
            df = result
        elif isinstance(result, list):
            # Flatten any dict records in a list
            df = pd.DataFrame(
                [_flatten_record(r) if isinstance(r, dict) else r for r in result]
            )
        elif isinstance(result, dict):
            df = _dict_to_dataframe(result)
        else:
            df = pd.DataFrame([{"value": result}])

        csv_path = output_dir / f"{file_stem}.csv"
        df.to_csv(csv_path, index=False)


def load_eval_results(path: Path) -> Optional[Dict[str, Any]]:
    """Load evaluation results from a JSON file (preferred)."""
    path = Path(path)
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def measure_inference_time(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    num_new_tokens: int = 100,
    batch_size: int = 1,
    device: str = "cuda",
    num_runs: int = 5,
) -> Dict[str, Any]:
    """Measure inference time and throughput for text generation.

    Args:
        model: The language model to evaluate.
        tokenizer: The tokenizer instance.
        prompts: List of input prompts for generation.
        num_new_tokens: Number of new tokens to generate per prompt.
        batch_size: Batch size (currently supports 1).
        device: Device string, e.g. "cuda" or "cpu".
        num_runs: Number of runs to average over for stability.

    Returns:
        Dict with average time per prompt, throughput, and generated responses.
    """
    if batch_size != 1:
        raise NotImplementedError("Batch size > 1 not yet supported.")

    model.eval()
    model.to(device)

    total_times = []
    total_tokens_generated = 0
    generated_responses = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Warm-up run
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)

        times = []
        response = None
        for run in range(num_runs):
            torch.cuda.empty_cache() if device == "cuda" else None

            start_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
            end_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

            if device == "cuda":
                start_event.record()
            else:
                import time
                start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=num_new_tokens, do_sample=False)

            if device == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event) / 1000  # seconds
            else:
                elapsed = time.time() - start_time

            times.append(elapsed)
            total_tokens_generated += num_new_tokens

            # Capture the generated text from the last run
            if run == num_runs - 1:
                generated_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
                response = generated_text.strip()

        avg_time = sum(times) / len(times)
        total_times.append(avg_time)
        generated_responses.append(response)

    overall_avg_time = sum(total_times) / len(total_times)
    overall_throughput = total_tokens_generated / sum(total_times)  # tokens/sec across all runs

    return {
        "avg_time_per_prompt_sec": overall_avg_time,
        "throughput_tokens_per_sec": overall_throughput,
        "total_prompts": len(prompts),
        "num_new_tokens": num_new_tokens,
        "num_runs": num_runs,
        "device": device,
        "prompts": prompts,
        "generated_responses": generated_responses,
    }
