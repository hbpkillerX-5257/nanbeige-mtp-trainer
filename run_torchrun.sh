#!/usr/bin/env bash
set -euo pipefail

# Distributed Multi-GPU Training script using PyTorch torchrun
# Automatically detects number of CUDA GPUs (e.g. 2x T4 on Kaggle)

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected ${NUM_GPUS} CUDA GPU(s)"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Starting PyTorch torchrun DDP training across ${NUM_GPUS} GPUs..."
    torchrun --nproc_per_node="${NUM_GPUS}" "${DIR}/train.py" "$@"
else
    echo "Starting single-GPU training..."
    python3 "${DIR}/train.py" "$@"
fi
