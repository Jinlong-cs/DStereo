#!/usr/bin/env bash
set -euo pipefail

stage="${1:-}"
shift || true

export PYTHONPATH="${PYTHONPATH:-}:/root/DStereo"
export WANDB_API_KEY="${WANDB_API_KEY:-ccbc765e15286047df6262193193083e2cc3c48b}"
ids="${DEVICE_IDS:-0}"

case "${stage}" in
  float)
    exec python3.10 tools/train.py -s float -c DStereo/DStereoPlus.py -ids "${ids}" "$@"
    ;;
  calibration|qat|int_infer)
    exec python3.10 tools/train.py -s "${stage}" -c DStereo/DStereoPlus_qat.py -ids "${ids}" "$@"
    ;;
  compile)
    exec python3.10 tools/deploy/compile_perf.py -c DStereo/DStereoPlus_qat.py "$@"
    ;;
  *)
    echo "Usage: bash train_qat.sh {float|calibration|qat|int_infer|compile} [extra args]"
    exit 1
    ;;
esac
