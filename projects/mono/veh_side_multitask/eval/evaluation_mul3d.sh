#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$(pwd -P):$PYTHONPATH

evalpy=$(dirname $0)/evaluation_mul3d.py
model_name=veh_side_multitask
# model_version=v1.8.1
# pretrained_ckpt=http://fm-xiufeng-zhou.bcloud-2nd.hogpu.cc/plat_gpu/hobot-dag-3429671_veh-side-multitask-v1-8-1-20230717-102004/output/models/vehicle_side/qat-checkpoint-last.pth.tar
# model_version=v1.5.3
# pretrained_ckpt=http://fm-xiufeng-zhou.bcloud-2nd.hogpu.cc/plat_gpu/veh_side_multitask-v1.5.3-20230613_190749/output/models/vehicle_side/qat-checkpoint-last.pth.tar
model_version=v1.5.2
pretrained_ckpt=http://fm-xiufeng-zhou.bcloud-2nd.hogpu.cc/plat_gpu/veh_side_multitask-v1.5.2-20230613_190930/output/models/vehicle_side/qat-checkpoint-last.pth.tar

# pipeline test with ckpt
# python3.8 $evalpy \
#     --model-name $model_name \
#     --model-version $model_version \
#     --stage Qat \
#     --ckpt $pretrained_ckpt \
#     --merge-faceplate \
#     --vis-result \
#     --pipeline-test \
#     --data-paths /home/users/xiufeng.zhou/main/HAT/tmp_data/ADAS_20220613-160510_922_5_aeb_v4_1655107661890.json \
#     --tasks real3d vehicleside face plate

# pipeline test with aidi 
model_version=v5.1.1
python3.8 $evalpy \
    --model-name $model_name \
    --model-version $model_version \
    --stage IntInference \
    --merge-faceplate \
    --vis-result \
    --data-paths /home/users/xiufeng.zhou/main/HAT/tmp_data/vehicleside/ADAS_20230823-122842_204_5.json \
    --tasks real3d vehicleside face plate
    # --pipeline-test \

# main eval progress
python3.8 $evalpy \
    --model-name $model_name \
    --model-version $model_version \
    --stage Qat \
    --merge-faceplate \
    --tasks real3d vehicleside face plate
