export HAT_TL_MODEL_VERSION="v9.0.0"
export HAT_TL_PREDICTION_NAME_SUFFIX="sd_"$HAT_TL_MODEL_VERSION

python tools/predict.py \
    --config $(dirname $0)/../../config/traffic_light/eval_multitask.py \
    --stage freeze_bn_2 \
    --device-ids 0,1,2,3 \
    --hat-tl-model-setting tl_cn_2pe_day_multitask \
    --hat-num-machines 1
