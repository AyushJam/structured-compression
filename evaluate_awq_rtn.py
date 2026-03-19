"""Evaluate AWQ/RTN quantized models from the command line.

This script is a CLI wrapper around the notebook logic in
`evaluation_awq_rtn.ipynb`. It evaluates model size, perplexity, and optionally
runs the LM Harness evaluation for a small set of tasks.

Usage:
    python evaluate_awq_rtn.py --models baseline_fp16 rtn_w8 rtn_w4 awq_w4

The evaluation results are saved as `evaluation.json` (and `evaluation.csv`) in
`quantized_models/<model>/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import utils.eval_utils as eval_utils
import utils.model_loader as model_loader


OUTPUT_ROOT = model_loader.OUTPUT_ROOT
PERPLEXITY_SAMPLES = 100
MAX_LENGTH = 512


DEFAULT_MODELS = ["baseline_fp16", "rtn_w8", "rtn_w4", "awq_w4"]


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


def evaluate_model_config(config: dict, run_lm_harness: bool = True):
    print("\n" + "=" * 80)
    print(f"Evaluating: {config['name']}")
    print("=" * 80)

    model, tokenizer = load_and_prepare_model(config)

    size_metrics = eval_utils.calculate_model_size_and_bandwidth(
        model, bits=config.get("bits", 16)
    )

    ppl_metrics = eval_utils.evaluate_perplexity(
        model,
        tokenizer,
        max_samples=PERPLEXITY_SAMPLES,
        max_length=MAX_LENGTH,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    lm_metrics = {}
    if run_lm_harness:
        if "awq" in config["name"].lower():
            model = getattr(model, "model", model)

        lm_metrics = eval_utils.evaluate_lm_harness(
            model_obj=model,
            tokenizer_obj=tokenizer,
            tasks=["hellaswag", "piqa", "mmlu_stem", "mmlu_humanities"],
            num_fewshot=3,
            limit=20,
        )

    results = {
        "model_name": config["name"],
        "model_path": str(config["path"]),
        "bits": config.get("bits", 16),
        **size_metrics,
        **ppl_metrics,
        "lm_eval": lm_metrics,
    }

    output_dir = Path(config["path"])
    eval_utils.save_eval_result(results, output_dir, file_stem="evaluation")

    print(f"Saved evaluation to: {output_dir / 'evaluation.json'}")
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate AWQ/RTN quantized models and save results to JSON/CSV."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model keys to evaluate (e.g. baseline_fp16 rtn_w4 awq_w4)",
    )
    parser.add_argument(
        "--no-lm",
        action="store_true",
        help="Skip running the LM Harness evaluations (faster).",
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
            results.append(evaluate_model_config(cfg, run_lm_harness=not args.no_lm))
        except Exception as e:
            print(f"Failed to evaluate {cfg['name']}: {e}")

    if results:
        combined_dir = OUTPUT_ROOT / "analysis"
        combined_dir.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        df = pd.DataFrame(results)
        df.to_csv(combined_dir / "evaluation_awq_rtn_results.csv", index=False)
        df.to_json(
            combined_dir / "evaluation_awq_rtn_results.json", orient="records", indent=2
        )
        print(f"\nSaved combined results to: {combined_dir}")


if __name__ == "__main__":
    main()
