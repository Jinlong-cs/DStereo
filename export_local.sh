#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:/root/DStereo"
export NUMBA_DISABLE_CUDA=1
export WANDB_API_KEY="ccbc765e15286047df6262193193083e2cc3c48b"

python3.10 -u tools/deploy/export_onnx.py \
 -c DStereo/DStereoPlus.py 


