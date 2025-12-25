#!/bin/bash
# Ensure the script exits on error
set -e

# Log function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 读取参数
TASK_JSON="input/task.json"
ONNX_PATH=$(python3 -c "import json; print(json.load(open('$TASK_JSON'))['quantize']['onnx_path'])")
CALIB_DATA=$(python3 -c "import json; print(json.load(open('$TASK_JSON'))['quantize']['cal_data_path'])")

log "ONNX_PATH: $ONNX_PATH"
log "CALIB_DATA: $CALIB_DATA"

# Perform operator replacement to improve quantization accuracy
# By default, the float_modify.onnx file is generated in the ptq_V21 path
# log "start export onnx_float_modify..."
# python3 ptq_V21/replace_mul_reducesum.py
# log "export onnx_float_modify done."

# Generate data for quantization
log "Starting to generate data for quantization..."
mkdir -p output/calib_data_npy/infra1 || { log "Failed to create directory: infra1"; exit 1; }
mkdir -p output/calib_data_npy/infra2 || { log "Failed to create directory: infra2"; exit 1; }
python3 ptq_V21/infer_float.py --left_img ${CALIB_DATA}/left/ --right_img ${CALIB_DATA}/right/ --save_npy_path output/calib_data_npy/ --ptq_num 100 || { log "Failed to generate npy data"; exit 1; }
log "Generate data done"

# Start quantization, the process may take about half an hour
log "Starting quantization..."
# cp -rf input/float.onnx ptq_V21/float.onnx || { log "Failed to copy float.onnx"; exit 1; }
hb_mapper makertbin -c ptq_V21/D-StereoPlus-cloud.yaml --model-type onnx || { log "Quantization failed"; exit 1; }
log "Quantization done."

# Calculate metrics
log "Starting to calculate metrics..."
python3 tools/metrics.py -c quant || { log "Failed to calculate metrics"; exit 1; }
log "Metrics calculation done."