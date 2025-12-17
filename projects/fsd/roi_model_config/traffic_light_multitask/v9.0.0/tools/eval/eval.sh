# aidi集群评测脚本
export MODEL_TYPE="mono_tl_mtl"
export STAGE="freeze_bn_2"
export JOB_NAME="traffic_light_multitask"
export MODEL_SETTING="hat_eval"
export MODEL_VERSION="v0.0.13"
export MOUNT_BUCKET="matrix,auto_eval,adas,mono,MultiMode_2"
export PROJECT_ID="PDT20220004"
export CURRENT_CLUSTER="share-3090-small-tcloud"
export MODEL_NAME_POSTFIX="fan.lv"
export HAT_TL_PREDICTION_NAME_SUFFIX="sd_"$MODEL_VERSION


python3 projects/mono/traffic_light_multitask/tools/eval/mono_eval_pipeline.py \
        --model-type ${MODEL_TYPE} \
        --stage ${STAGE} \
        --job-name ${JOB_NAME} \
        --model-setting ${MODEL_SETTING} \
        --model-version ${MODEL_VERSION} \
        --num-machines 1 \
        --num-gpus-per-machine 8 \
        --mount-bucket ${MOUNT_BUCKET} \
        --project-id ${PROJECT_ID} \
        --current-cluster ${CURRENT_CLUSTER} \
        --model-name-postfix ${MODEL_NAME_POSTFIX}
