#!/usr/bin/env bash
set -euo pipefail

# Distributed Multi-GPU Training script using HuggingFace Accelerate
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Launching HuggingFace Accelerate across ${NUM_GPUS} GPUs..."
accelerate launch \
    --multi_gpu \
    --num_processes="${NUM_GPUS}" \
    --mixed_precision=fp16 \
    "${DIR}/train.py" "$@"
