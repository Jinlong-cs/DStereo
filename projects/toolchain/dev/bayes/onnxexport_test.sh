#!/usr/bin/env bash

set -e
export PYTHONPATH=`pwd`:${PYTHONPATH}

num_workers=12
save_dir="onnx_export_test"

echo "run align bpu on ${num_workers} wokers"

# rm -rf ./tmp_models/*

# hdfs dfs -get hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bayes/v2.0.3/py38/release_models-2.0.3.tgz 

# tar xvf release_models-2.0.3.tgz -C ./tmp_models
# mv ./tmp_models/release_models/* ./tmp_models

# projects/toolchain/configs/detection/fcos3d/fcos3d_efficientnetb0_nuscenes.py # 有问题
# projects/toolchain/configs/bev/bev_mt_gkt_mixvargenet_nuscenes.py
# projects/toolchain/configs/bev/bev_mt_ipm_efficientnetb0_nuscenes.py
# projects/toolchain/configs/bev/bev_mt_lss_efficientnetb0_nuscenes.py
# projects/toolchain/configs/bev/detr3d_efficientnetb3_nuscenes.py
# projects/toolchain/configs/bev/petr_efficientnetb3_nuscenes.py
# projects/toolchain/configs/bev/bev_mt_ipm_4d_efficientnetb0_nuscenes.py
# "
listTest=(
projects/toolchain/configs/classification/resnet18.py
projects/toolchain/configs/classification/resnet50.py
projects/toolchain/configs/classification/vargconvnet.py
projects/toolchain/configs/classification/efficientnasnetm.py
projects/toolchain/configs/classification/efficientnasnets.py
projects/toolchain/configs/classification/efficientnet.py
projects/toolchain/configs/classification/horizon_swin_transformer.py
projects/toolchain/configs/classification/mixvargenet.py
projects/toolchain/configs/classification/mobilenetv1.py
projects/toolchain/configs/classification/mobilenetv2.py
projects/toolchain/configs/classification/vargdarknet.py
projects/toolchain/configs/classification/vargnetv2.py
projects/toolchain/configs/detection/centerpoint/centerpoint_pointpillar_nuscenes.py
projects/toolchain/configs/detection/detr/detr_efficientnetb3_mscoco.py
projects/toolchain/configs/detection/detr/detr_resnet50_mscoco.py
projects/toolchain/configs/detection/fcos/fcos_efficientnetb0_mscoco.py
projects/toolchain/configs/detection/fcos/fcos_efficientnetb1_mscoco.py
projects/toolchain/configs/detection/fcos/fcos_efficientnetb2_mscoco.py
projects/toolchain/configs/detection/fcos/fcos_efficientnetb3_mscoco.py
projects/toolchain/configs/detection/pointpillars/pointpillars_kitti_car.py
projects/toolchain/configs/detection/retinanet/retinanet_vargnetv2_fpn_mscoco.py
projects/toolchain/configs/detection/yolov3/yolo_mobilenetv1_det.py
projects/toolchain/configs/detection/yolov3/yolo_varg_darknet53_det.py
projects/toolchain/configs/disparity_pred/stereonet/stereonet_stereonetneck_sceneflow.py
projects/toolchain/configs/lane_pred/ganet/ganet_mixvargenet_culane.py
projects/toolchain/configs/opticalflow_pred/pwcnet/pwcnet_pwcnetneck_flyingchairs.py
projects/toolchain/configs/segmentation/deeplabv3plus_efficientnetm0.py
projects/toolchain/configs/segmentation/deeplabv3plus_efficientnetm1.py
projects/toolchain/configs/segmentation/deeplabv3plus_efficientnetm2.py
projects/toolchain/configs/segmentation/fastscnn_unet.py
projects/toolchain/configs/segmentation/unet.py
projects/toolchain/configs/track_pred/motr_efficientnetb3_mot17.py
projects/toolchain/configs/detection/fcos3d/fcos3d_efficientnetb0_nuscenes.py
projects/toolchain/configs/keypoint/efficientnet_heatmap.py
projects/toolchain/configs/lidar_multi_task/lidar_multitask_nuscenes.py
)

len=${#listTest[@]}

div=$((len/num_workers))
mod=$((len%num_workers))

sublists=()
index=0
for ((i=0;i<num_workers;i++))
do
    sublist_len=$div
    if [ $i -lt $mod ]; then
        sublist_len=$((sublist_len+1))
    fi
    for ((j=0;j<sublist_len;j++))
    do
        sublist[$i]+="${listTest[$index]} "
        index=$((index+1))
    done
done

function run_align_bpu(){
    device=$1
    config_list=$2
    save_dir=$3
    resultlogfile="result_${device}.log"
    resultlogpath="$save_dir/$resultlogfile"
    echo $resultlogpath
    for cfg in ${config_list};
    do
        echo ${cfg}
        task_name=`cat ${cfg} | grep "task_name = " | awk '{print $3}' | sed 's/\"//g'`
        echo $task_name
        cfg_log="${save_dir}/${task_name}.log"
        # echo ${cfg_log}
        python3 tools/export_onnx.py --config ${cfg} >> ${cfg_log} 2>&1
        # var=$(cat work_dirs/hat_logss/${task_name}_align_bpu_validation.log | awk 'END {print}'| awk 'BEGIN { FS = "Validation "} ; {print $2}')
        # echo "${cfg} ${var}" >> $resultlogpath
    done
}


for ((i=0;i<num_workers;i++))
do
    tmpresultlog="$save_dir/tmpresult_${i}.log"
    echo $tmpresultlog
    run_align_bpu $i "${sublist[$i][*]}"  $save_dir > $tmpresultlog &
done
