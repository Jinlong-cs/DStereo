# 交通灯hat多任务训练提交脚本
export MODEL_TYPE="mono_tl_mtl"
export SETTING="mono_cn_2pe_day_multitask_"
export NAME_POSTFIX="fan.lv_mono_day"
export CLUSTER="share-debug-queue-idc"
export NUM_MACHINES=1
# export MODEL_VESION="v0.0.1" # 不指定则默认按照文件夹版本号 projects/mono/traffic_light_multitask/vx.x.x
export NUM_GPUS_PER_MACHINE=1
export START_STAGE="with_bn"
export END_STAGE="freeze_bn_2"
export PROJECT_ID="PDT20220004"
export PRETRAIN_MODEL_VERSION="v0.0.9"
MODEL_SETTING=$SETTING$VERSION


python3 $(dirname $0)/mono_train_pipeline.py \
    --model-type $MODEL_TYPE \
    --model-setting $MODEL_SETTING \
    --model-name-postfix $NAME_POSTFIX  \
    --current-cluster $CLUSTER \
    --num-machines $NUM_MACHINES \
    --num-gpus-per-machine $NUM_GPUS_PER_MACHINE \
    --start-stage $START_STAGE \
    --end-stage $END_STAGE \
    --pretrain-model-version $PRETRAIN_MODEL_VERSION \
    --project-id $PROJECT_ID \
    --pipeline-test \
    # --local \
