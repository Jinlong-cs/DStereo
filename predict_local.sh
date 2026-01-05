#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:/root/DStereo"
export NUMBA_DISABLE_CUDA=1
export WANDB_API_KEY="ccbc765e15286047df6262193193083e2cc3c48b"


mkdir -p logs
ts=$(date +"%Y%m%d_%H%M%S")
log_file="logs/predict_float_${ts}.log"

python3.10 -u tools/predict.py \
  -s float \
  -c DStereo/DStereoPlus.py \
  2>&1 | tee -a "${log_file}"
