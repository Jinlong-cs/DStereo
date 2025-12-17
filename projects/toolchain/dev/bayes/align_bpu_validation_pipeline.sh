#!/usr/bin/env bash

set -e
export PYTHONPATH=`pwd`:${PYTHONPATH}

touch Result.log

#########################
# Divide by dataset
##########################

imagenet_config_list="
configs/classification/efficientnet.py
configs/classification/mobilenetv1.py
configs/classification/mobilenetv2.py
configs/classification/resnet18.py
configs/classification/resnet50.py
configs/classification/vargnetv2.py
configs/classification/mixvargenet.py
configs/classification/horizon_swin_transformer.py
"
for cfg in ${imagenet_config_list};
do
  echo ${cfg}
  net_name=($(echo ${cfg} | sed "s/\// /g" | sed s'/\./ /g'))
  python3 projects/toolchain/tools/align_bpu_validation.py --config ${cfg} --dataset imagenet
  cat work_dirs/hat_logss/${net_name[-2]}_cls_align_bpu_validation.log | awk 'END {print}'| awk 'BEGIN { FS = "Validation "} ; {print $2}' >> Result.log
done


fcos_coco_config_list="
configs/detection/fcos/fcos_efficientnetb0_mscoco.py
configs/detection/fcos/fcos_efficientnetb2_mscoco.py
configs/detection/fcos/fcos_efficientnetb3_mscoco.py
"
for cfg in ${fcos_coco_config_list};
do
  echo ${cfg}
  net_name=($(echo ${cfg} | sed "s/\// /g" | sed s'/\./ /g'))
  python3 projects/toolchain/tools/align_bpu_validation.py --config ${cfg} --dataset fcos_coco
  cat work_dirs/hat_logss/${net_name[-2]}_align_bpu_validation.log | awk 'END {print}'| awk 'BEGIN { FS = "Validation "} ; {print $2}' >> Result.log
done


coco_config_list="
configs/detection/retinanet/retinanet_vargnetv2_fpn_mscoco.py
"
for cfg in ${coco_config_list};
do
  echo ${cfg}
  net_name=($(echo ${cfg} | sed "s/\// /g" | sed s'/\./ /g'))
  python3 projects/toolchain/tools/align_bpu_validation.py --config ${cfg} --dataset coco
  cat work_dirs/hat_logss/${net_name[-2]}_align_bpu_validation.log | awk 'END {print}'| awk 'BEGIN { FS = "Validation "} ; {print $2}' >> Result.log
done


detr_coco_config_list="
configs/detection/detr/detr_resnet50_mscoco.py
configs/detection/detr/detr_efficientnetb3_mscoco.py
"
for cfg in ${detr_coco_config_list};
do
  echo ${cfg}
  net_name=($(echo ${cfg} | sed "s/\// /g" | sed s'/\./ /g'))
  python3 projects/toolchain/tools/align_bpu_validation.py --config ${cfg} --dataset detr_coco
  cat work_dirs/hat_logss/${net_name[-2]}_align_bpu_validation.log | awk 'END {print}'| awk 'BEGIN { FS = "Validation "} ; {print $2}' >> Result.log
done


culane_config_list="
configs/lane_pred/ganet/ganet_mixvargenet_culane.py
"
for cfg in ${culane_config_list};
do
  echo ${cfg}
  net_name=($(echo ${cfg} | sed "s/\// /g" | sed s'/\./ /g'))
  python3 projects/toolchain/tools/align_bpu_validation.py --config ${cfg} --dataset culane
  cat work_dirs/hat_logss/${net_name[-2]}_align_bpu_validation.log | awk 'END {print}'| awk 'BEGIN { FS = "Validation "} ; {print $2}' >> Result.log
done


flyingchairs_config_list="
configs/opticalflow_pred/pwcnet/pwcnet_pwcnetneck_flyingchairs.py
"
for cfg in ${flyingchairs_config_list};
do
  echo ${cfg}
  net_name=($(echo ${cfg} | sed "s/\// /g" | sed s'/\./ /g'))
  python3 projects/toolchain/tools/align_bpu_validation.py --config ${cfg} --dataset flyingchairs
  cat work_dirs/hat_logss/${net_name[-2]}_align_bpu_validation.log | awk 'END {print}'| awk 'BEGIN { FS = "Validation "} ; {print $2}' >> Result.log
done

cityscapes_config_list="
configs/segmentation/dwunet_seg.py
"
for cfg in ${cityscapes_config_list};
do
  echo ${cfg}
  net_name=($(echo ${cfg} | sed "s/\// /g" | sed s'/\./ /g'))
  python3 projects/toolchain/tools/align_bpu_validation.py --config ${cfg} --dataset cityscapes
  cat work_dirs/hat_logss/${net_name[-2]}_align_bpu_validation.log | awk 'END {print}'| awk 'BEGIN { FS = "Validation "} ; {print $2}' >> Result.log
done

cat Result.log