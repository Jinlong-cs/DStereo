#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/DStereo_V2.3"
DATA_ROOT="/root/ballcar_datasets"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
export BALLCAR_ROOT="${DATA_ROOT}"
export HAT_SUPPRESS_OPTIONAL_WARNINGS=1

#python3 -u tools/export_onnx.py \
#  -c DStereo/DStereoPlus.py 

python3 tools/deploy/export_onnx_batch.py \
  --config DStereo/DStereoPlus.py \
  --ckpt_dir work_dirs/tmp_models_szp1/DStereoV23_60000 \
  --out_dir work_dirs/tmp_models_szp1/DStereoV23_60000_onnx_test/ \
  --skip_existing
