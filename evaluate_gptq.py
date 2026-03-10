"""Evaluate GPTQ quantized models from the command line.

This script is a CLI wrapper around the notebook logic in `evaluation_gptq.ipynb`.
It evaluates model size, perplexity, and optionally runs the LM Harness evaluation.

Usage:
    python evaluate_gptq.py --models gptq_w4

The evaluation results are saved as `evaluation.json` (and `evaluation.csv`) in
`quantized_models/<model>/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import utils.eval_utils as eval_utils


OUTPUT_ROOT = Path("./quantized_models")
PERPLEXITY_SAMPLES = 100
MAX_LENGTH = 512


DEFAULT_MODELS = ["gptq_w4"]


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


def evaluate_model_config(config: dict, run_lm_harness: bool = True):
    print("\n" + "=" * 80)
    print(f"Evaluating: {config['name']}")
    print("=" * 80)

    model, tokenizer = load_and_prepare_model(config)

    size_metrics = eval_utils.calculate_model_size_and_bandwidth(
        model, bits=config.get("bits", 4)
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
        # auto-gptq wraps the model; ensure we pass the underlying model object
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
        "bits": config.get("bits", 4),
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
        description="Evaluate GPTQ quantized models and save results to JSON/CSV."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model keys to evaluate (e.g. gptq_w4)",
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
        model_configs.append(
            {
                "name": model,
                "path": str(OUTPUT_ROOT / model),
                "bits": 4,
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
        df.to_csv(combined_dir / "evaluation_gptq_results.csv", index=False)
        df.to_json(
            combined_dir / "evaluation_gptq_results.json", orient="records", indent=2
        )
        print(f"\nSaved combined results to: {combined_dir}")


if __name__ == "__main__":
    main()
