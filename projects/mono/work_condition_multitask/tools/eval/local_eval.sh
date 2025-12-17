export HAT_WK_MODEL_VERSION="v0.0.1"
export HAT_WK_PREDICTION_NAME_SUFFIX="sd_"$HAT_WK_MODEL_VERSION

python tools/predict.py \
    --config $(dirname $0)/../../config/work_condition/eval_multitask.py \
    --stage freeze_bn_2 \
    --device-ids 0,1,2,3 \
    --hat-wk-model-setting sd_work_condition_multitask \
    --hat-num-machines 1
