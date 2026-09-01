#!/bin/bash
set -euo pipefail

NUM_GPUS="1"
CONFIG_PATH="qwen3-0.6b.yaml"
ACCELERATE_CONFIG="single_gpu.yaml"
TRAINING_SCRIPT="sft.py"

python3 -m pip install --upgrade uv
uv pip install --system -r requirements.txt

accelerate launch --config_file "$ACCELERATE_CONFIG" --num_processes "$NUM_GPUS" \
    "$TRAINING_SCRIPT" --config "$CONFIG_PATH"