# Structured Compression

A project for model quantization and evaluation using various quantization techniques.

## Quantization Methods

- **AWQ (Activation-aware Weight Quantization)**: Quantizes weights while considering activation distributions.
- **GPTQ (GPT Quantization)**: Gradient-based post-training quantization for GPT models.
- **RTN (Round-to-Nearest)**: Simple round-to-nearest quantization method.

## Features

- Quantization notebooks for different methods
- Evaluation scripts for accuracy and inference speed
- Analysis and plotting tools
- Pre-quantized models for testing

## Results

Quantized models and evaluation results are stored in `quantized_models/` and `quantized_models_3b/` directories. Figures are available in `figures/` and `figures_3b/`.

## Project Structure

- `*.ipynb`: Jupyter notebooks for quantization and analysis
- `evaluate_*.py`: Evaluation scripts
- `inference_speed_*.py`: Inference speed measurement scripts
- `utils/`: Utility functions for quantization and evaluation
- `quantized_models/`: Quantized model checkpoints and results
- `figures/`: Generated plots and visualizations
