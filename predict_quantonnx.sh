#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/DStereo"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export HAT_SUPPRESS_OPTIONAL_WARNINGS=1
export HAT_USE_WANDB=1
export HAT_LOG_PRETRAINED_BASELINE=0
export WANDB_PROJECT="dstereo_vis"
export WANDB_NAME="DStereoV23_predict_onnx"
export WANDB_LOG_EVERY_STEPS=1
export WANDB_VIS_NUM_SAMPLES=5
export WANDB_VIS_STRATEGY="random"
export WANDB_VIS_SEED=0
export WANDB_VIS_EVERY_EPOCHS=1
export WANDB_VIS_EVERY_STEPS=0
export WANDB_VIS_AT_START=1
export WANDB_API_KEY="ccbc765e15286047df6262193193083e2cc3c48b"

ONNX_PATH="${ONNX_PATH:-ptq_V21/DStereoV23_quantized_model.onnx}"
export ONNX_PATH

mkdir -p logs
ts=$(date +"%Y%m%d_%H%M%S")
log_file="logs/predict_onnx_${ts}.log"

cmd=(
  python3.10 -u tools/predict.py
  -s quantonnx
  -c DStereo/DStereoPlus.py
)

"${cmd[@]}" \
  2>&1 | tee -a "${log_file}"
