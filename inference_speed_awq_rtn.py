"""Measure inference speed for AWQ/RTN quantized models.

This script measures generation latency and throughput for AWQ and RTN models.

Usage:
    python inference_speed_awq_rtn.py --models baseline_fp16 rtn_w8 rtn_w4 awq_w4

Results are saved as `inference_speed.json` in `quantized_models/<model>/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import utils.eval_utils as eval_utils
import utils.model_loader as model_loader


OUTPUT_ROOT = model_loader.OUTPUT_ROOT

DEFAULT_MODELS = ["baseline_fp16", "rtn_w8", "rtn_w4", "awq_w4"]

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

    if "awq" in model_name.lower():
        from awq import AutoAWQForCausalLM

        model = AutoAWQForCausalLM.from_quantized(model_path, fuse_layers=True)
        tokenizer = model_loader.load_tokenizer(model_path)

    elif "baseline" in model_name.lower():
        model = model_loader.load_base_model()  # uses MODEL_ID
        tokenizer = model_loader.load_tokenizer()

    else:
        # RTN models
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = model_loader.load_tokenizer(model_path)

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
        model, bits=config.get("bits", 16)
    )

    results = {
        "model_name": config["name"],
        "model_path": str(config["path"]),
        "bits": config.get("bits", 16),
        **size_metrics,
        **timing_metrics,
    }

    output_dir = Path(config["path"])
    eval_utils.save_eval_result(results, output_dir, file_stem="inference_speed")

    print(f"Saved timing results to: {output_dir / 'inference_speed.json'}")
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure inference speed for AWQ/RTN models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model keys to evaluate (e.g. baseline_fp16 rtn_w4 awq_w4)",
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
        bits = 16 if model == "baseline_fp16" else 4 if model.endswith("_w4") else 8
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
        df.to_csv(combined_dir / "inference_speed_awq_rtn_results.csv", index=False)
        df.to_json(
            combined_dir / "inference_speed_awq_rtn_results.json", orient="records", indent=2
        )
        print(f"\nSaved combined results to: {combined_dir}")


if __name__ == "__main__":
    main()