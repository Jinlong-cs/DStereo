# 工况hat多任务训练提交脚本
export MODEL_TYPE="sd_wk_mtl"
export SETTING="sd_wk_multitask_"
export NAME_POSTFIX="fan.lv"
export VERSION="v0.0.1"
export CLUSTER="share-3090-small-tcloud"
export NUM_MACHINES=1
export NUM_GPUS_PER_MACHINE=8
export START_STAGE="with_bn"
export END_STAGE="freeze_bn_2"
export PROJECT_ID="PDT2021004-vision"
MODEL_SETTING=$SETTING


python3 $(dirname $0)/mono_train_pipeline.py \
    --model-type $MODEL_TYPE \
    --model-setting $MODEL_SETTING \
    --model-name-postfix $NAME_POSTFIX  \
    --current-cluster $CLUSTER \
    --num-machines $NUM_MACHINES \
    --model-version $VERSION \
    --num-gpus-per-machine $NUM_GPUS_PER_MACHINE \
    --start-stage $START_STAGE \
    --end-stage $END_STAGE \
    --project-id $PROJECT_ID \
    --trace_compile_model \
    # --pipeline-test \
    # --local \
