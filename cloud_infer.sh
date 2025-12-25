#!/bin/bash

# 确保脚本在遇到错误时退出
set -e

# Log function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 开始离线推理
log "start inferring..."
python3 ptq_V21/infer_float_onnx.py --onnx_path input/float.onnx --left_img input/infer/left/ --right_img input/infer/right/ --result_path output/infer_result/  || { log "Inferring float ONNX model failed"; exit 1; }
log "inferring done."