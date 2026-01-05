#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=$PYTHONPATH:/root/DStereo
export NUMBA_DISABLE_CUDA=1 # to avoid crash
export HAT_SUPPRESS_OPTIONAL_WARNINGS=1
export HAT_VAL_INTERVAL=100
# 冻结bn层用来调试。
export HAT_FREEZE_BN=1
export HAT_FREEZE_BN_AFFINE=0
export HAT_FREEZE_BN_UNTIL_STEP=2000  # 设置为-1表示一直冻结
export WANDB_API_KEY="ccbc765e15286047df6262193193083e2cc3c48b"

VIS_NUM_SAMPLES=5
VIS_STRATEGY="random"
VIS_SEED=0
export WANDB_VIS_AT_START=1

mkdir -p logs
ts=$(date +"%Y%m%d_%H%M%S")
log_file="logs/train_float_${ts}.log"

cmd=(
  python3.10 -u tools/train.py
  -s float
  -c DStereo/DStereoPlus.py
  -ids 0
  --use-wandb
  --wandb-project dstereo_vis
  --wandb-name DStereoV23_freezeBN
  --wandb-log-every-steps 1
  --fixed-train-index 0
  --fixed-val-index 0
  --vis-num-samples "${VIS_NUM_SAMPLES}"
  --vis-strategy "${VIS_STRATEGY}"
  --vis-seed "${VIS_SEED}"
  --vis-every-steps 100
  --vis-every-epochs 0
)

"${cmd[@]}" \
  2>&1 | tee -a "${log_file}"
