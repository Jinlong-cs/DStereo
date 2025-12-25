#!/bin/bash

# 设置环境变量
export PYTHONPATH=/workspace:$PYTHONPATH

# 确保脚本在遇到错误时退出
set -e

# Log function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 训练，传出是checkpoint和loss、epe指标
log "start training..."
python3 tools/train.py -s float -c DStereo/DStereoPlus-Cloud.py  || { log "Training failed"; exit 1; }
log "training done."

# 预测，传出是预测结果
log "start predicting..."
python3 tools/predict.py -s float -c DStereo/DStereoPlus-Cloud.py --ckpt output/DStereoTask/float-checkpoint-last.pth.tar  || { log "Prediction failed"; exit 1; }
log "predicting done."

# 导出onnx模型
log "export onnx..."
python3 tools/deploy/export_onnx.py -c DStereo/DStereoPlus-Cloud.py  || { log "Exporting ONNX model failed"; exit 1; }
log "export onnx done."

# 计算指标
log "start calculating metrics..."
python3 tools/metrics.py -c train  || { log "Metrics calculation failed"; exit 1; }
log "metrics calculation done."