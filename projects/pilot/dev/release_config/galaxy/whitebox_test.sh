#!/usr/bin/env bash

set -e

AIDI_TOKEN = ""
aidisdk config -t $RELEASE_PACKAGE

RELEASE_PACKAGE="release"
export PYTHONPATH="$(pwd)/$RELEASE_PACKAGE":${PYTHONPATH}

MODEL_ROOT_PATH="/pilot_data_raw/models"

cd $RELEASE_PACKAGE
# ========================GALAXY-SIDE===========================
export HAT_PILOT_MODEL_SETTING="galaxy_x3c_side_lmdb"
export HAT_PILOT_MODEL_CHECKPOINT="$MODEL_ROOT_PATH/side/side_with_bn-checkpoint-last-4a37ce12.pth.tar"
# # train
python3 tools/train.py \
    --config projects/pilot/configs/resize_2_side_bayes/multitask.py \
    --stage with_bn \
    --ids 0,1,2,3 \
    --pipeline-test \

# val
export HAT_TRAINING_STEP="sparse_3d_freeze_bn_2"
export HAT_PILOT_MODEL_CHECKPOINT="$MODEL_ROOT_PATH/side/side_sparse_3d_freeze_bn_2-checkpoint-last-dd121c79.pth.tar"
python3 tools/predict.py \
    --config projects/pilot/configs/resize_2_side_bayes/val_multitask.py \
    --stage sparse_3d_freeze_bn_2 \
    --ids 0 \

# vis
python3 tools/predict.py \
    --config projects/pilot/configs/resize_2_side_bayes/vis_multitask.py \
    --stage sparse_3d_freeze_bn_2 \
    --ids 0 \

# ========================GALAXY-REAR===========================
export HAT_PILOT_MODEL_SETTING="galaxy_0233_rear_lmdb"
export HAT_PILOT_MODEL_CHECKPOINT="$MODEL_ROOT_PATH/rear/rear_with_bn-checkpoint-last-8396222c.pth.tar"
# # train
python3 tools/train.py \
    --config projects/pilot/configs/resize_2_rear_bayes/multitask.py \
    --stage with_bn \
    --ids 2,3 \
    --pipeline-test \

# val
export HAT_TRAINING_STEP="sparse_3d_freeze_bn_2"
export HAT_PILOT_MODEL_CHECKPOINT="$MODEL_ROOT_PATH/rear/rear_sparse_3d_freeze_bn_2-checkpoint-last-bd5a4470.pth.tar"

python3 tools/predict.py \
    --config projects/pilot/configs/resize_2_rear_bayes/val_multitask.py \
    --stage sparse_3d_freeze_bn_2 \
    --ids 0 \

# vis
python3 tools/predict.py \
    --config projects/pilot/configs/resize_2_rear_bayes/vis_multitask.py \
    --stage sparse_3d_freeze_bn_2 \
    --ids 0 \


# ========================GALAXY-CROP===========================
# train
export HAT_PILOT_MODEL_SETTING="galaxy_0233_rear_lmdb"
export HAT_PILOT_MODEL_CHECKPOINT="$MODEL_ROOT_PATH/crop/crop_with_bn-checkpoint-last.pth.tar"

python3 tools/train.py \
    --config projects/pilot/configs/crop_bayes/multitask.py \
    --stage with_bn \
    --ids 2,3 \
    --pipeline-test 

# vis
export HAT_TRAINING_STEP="freeze_bn_3"
export HAT_PILOT_MODEL_CHECKPOINT="$MODEL_ROOT_PATH/crop/crop_freeze_bn_3-checkpoint-last-88d5987a.pth.tar"

python3 tools/predict.py \
    --config projects/pilot/configs/crop_bayes/vis_multitask.py \
    --stage freeze_bn_3 \
    --ids 0 \


# ========================IQA===========================
# train
export HAT_PILOT_MODEL_SETTING="galaxy_0233_rear_lmdb"
export HAT_PILOT_MODEL_CHECKPOINT="$MODEL_ROOT_PATH/iqa/iqa-float-checkpoint-last-dea6860d.pth.tar"

python3 tools/train.py \
    --config projects/pilot/configs/single_task/image_fail_segmentation.py \
    --stage float \
    --ids 0,1,2,3 \
    --pipeline-test \

# vis
export HAT_PILOT_MODEL_NAME="galaxy_whitebox_iqa"
export HAT_TRAINING_STEP="int_infer"
export HAT_PILOT_MODEL_CHECKPOINT="$MODEL_ROOT_PATH/iqa/iqa-qat-checkpoint-last-3e3e6cf3.pth.tar"

python3 tools/predict.py \
    --config projects/pilot/configs/single_task/pred_image_fail_segmentation.py \
    --stage int_infer \
    --ids 0 \


# ========================COMPILE===========================
# compile
python3 projects/pilot/dev/publish/trace_compile_model.py \
    --sub-project galaxy_x3c_pe \
    --publish-version 0.0.1 \
    --compile-mode local \