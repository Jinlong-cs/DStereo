#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/DStereo_V2.3"
DATA_ROOT="/root/ballcar_datasets"
PRETRAINED_CKPT="/root/DStereo_V2.3/tmp_pretrained_models/mixvargenet_imagenet/float-checkpoint-last.pth.tar"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
export BALLCAR_ROOT="${DATA_ROOT}"
export HAT_SUPPRESS_OPTIONAL_WARNINGS=1
export HAT_USE_WANDB=1
export HAT_LOG_PRETRAINED_BASELINE=0
export WANDB_PROJECT="dstereo_vis"
export WANDB_NAME="DStereoV23_predict_pretrained"
export WANDB_LOG_EVERY_STEPS=1
export WANDB_VIS_NUM_SAMPLES=5
export WANDB_VIS_STRATEGY="random"
export WANDB_VIS_SEED=0
export WANDB_VIS_EVERY_EPOCHS=1
export WANDB_VIS_EVERY_STEPS=0
export WANDB_VIS_AT_START=1
export WANDB_API_KEY="ccbc765e15286047df6262193193083e2cc3c48b"


mkdir -p logs
ts=$(date +"%Y%m%d_%H%M%S")
log_file="logs/predict_float_${ts}.log"

cmd=(
  python3 -u tools/predict.py
  -s float
  -c DStereo/DStereoPlus.py
  -ids 0
  --ckpt "${PRETRAINED_CKPT}"
)

"${cmd[@]}" \
  2>&1 | tee -a "${log_file}"
