"""Measure inference speed for GPTQ quantized models.

This script measures generation latency and throughput for GPTQ models.

Usage:
    python inference_speed_gptq.py --models gptq_w4

Results are saved as `inference_speed.json` in `quantized_models/<model>/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import utils.eval_utils as eval_utils


OUTPUT_ROOT = Path("./quantized_models")

DEFAULT_MODELS = ["gptq_w4"]

# Default prompts for timing
DEFAULT_PROMPTS = [
    "The capital of France is",
    "In machine learning,",
    "The weather today is",
    "Once upon a time,",
    "The meaning of life is",
]

NUM_NEW_TOKENS = 50
NUM_RUNS = 3


def load_and_prepare_model(config: dict):
    model_path = config["path"]
    model_name = config["name"]

    if "gptq" in model_name.lower():
        from auto_gptq import AutoGPTQForCausalLM
        from transformers import AutoTokenizer

        model = AutoGPTQForCausalLM.from_quantized(
            model_path,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            use_safetensors=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    else:
        raise ValueError("Only GPTQ models are supported by this script.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def measure_model_speed(config: dict, prompts: list, num_new_tokens: int, num_runs: int):
    print("\n" + "=" * 80)
    print(f"Measuring speed for: {config['name']}")
    print("=" * 80)

    model, tokenizer = load_and_prepare_model(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    timing_metrics = eval_utils.measure_inference_time(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        num_new_tokens=num_new_tokens,
        batch_size=1,
        device=device,
        num_runs=num_runs,
    )

    size_metrics = eval_utils.calculate_model_size_and_bandwidth(
        model, bits=config.get("bits", 4)
    )

    results = {
        "model_name": config["name"],
        "model_path": str(config["path"]),
        "bits": config.get("bits", 4),
        **size_metrics,
        **timing_metrics,
    }

    output_dir = Path(config["path"])
    eval_utils.save_eval_result(results, output_dir, file_stem="inference_speed")

    print(f"Saved timing results to: {output_dir / 'inference_speed.json'}")
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure inference speed for GPTQ models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model keys to evaluate (e.g. gptq_w4)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=DEFAULT_PROMPTS,
        help="Prompts to use for generation timing.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=NUM_NEW_TOKENS,
        help="Number of new tokens to generate per prompt.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=NUM_RUNS,
        help="Number of runs to average over.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    model_configs = []
    for model in args.models:
        bits = 4  # GPTQ is typically 4-bit
        model_configs.append(
            {
                "name": model,
                "path": str(OUTPUT_ROOT / model),
                "bits": bits,
            }
        )

    results = []
    for cfg in model_configs:
        try:
            results.append(measure_model_speed(cfg, args.prompts, args.num_tokens, args.num_runs))
        except Exception as e:
            print(f"Failed to measure speed for {cfg['name']}: {e}")

    if results:
        combined_dir = OUTPUT_ROOT / "analysis"
        combined_dir.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        df = pd.DataFrame(results)
        df.to_csv(combined_dir / "inference_speed_gptq_results.csv", index=False)
        df.to_json(
            combined_dir / "inference_speed_gptq_results.json", orient="records", indent=2
        )
        print(f"\nSaved combined results to: {combined_dir}")


if __name__ == "__main__":
    main()