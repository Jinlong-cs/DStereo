#!/usr/bin/env bash

set -e
export PYTHONPATH=`pwd`:${PYTHONPATH}
rm -rf ./tmp_models/*

hdfs dfs -get hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bernoulli/v2.0.4/py38/release_models-2.0.4.tgz

tar xvf release_models-2.0.4.tgz -C ./tmp_models
mv ./tmp_models/release_models/* ./tmp_models

touch /job_data/Result_xj3.log

#########################
# Divide by dataset
##########################

imagenet_config_list="
projects/toolchain/configs/bernoulli2/classification/efficientnasnetm.py
projects/toolchain/configs/bernoulli2/classification/efficientnasnets.py
projects/toolchain/configs/bernoulli2/classification/efficientnet.py
projects/toolchain/configs/bernoulli2/classification/mobilenetv1.py
projects/toolchain/configs/bernoulli2/classification/resnet18.py
projects/toolchain/configs/bernoulli2/classification/vargconvnet.py
projects/toolchain/configs/bernoulli2/classification/vargnetv2.py
projects/toolchain/configs/bernoulli2/detection/fcos/fcos_efficientnetb0_mscoco.py
projects/toolchain/configs/bernoulli2/detection/fcos/fcos_efficientnetb1_mscoco.py
projects/toolchain/configs/bernoulli2/detection/fcos/fcos_efficientnetb2_mscoco.py
projects/toolchain/configs/bernoulli2/detection/fcos/fcos_efficientnetb3_mscoco.py
projects/toolchain/configs/bernoulli2/detection/pointpillars/pointpillars_kitti_car.py
projects/toolchain/configs/bernoulli2/segmentation/deeplabv3plus_efficientnetm0.py
projects/toolchain/configs/bernoulli2/segmentation/deeplabv3plus_efficientnetm1.py
projects/toolchain/configs/bernoulli2/segmentation/deeplabv3plus_efficientnetm2.py
projects/toolchain/configs/bernoulli2/segmentation/fastscnn_unet.py
projects/toolchain/configs/bernoulli2/segmentation/unet.py
"

for cfg in ${imagenet_config_list};
do
  task_name=`cat ${cfg} | grep "task_name = " | awk '{print $3}' | sed 's/\"//g'`
  python3 projects/toolchain/tools/align_bpu_validationV2.py --config ${cfg}
  cat work_dirs/hat_logss/${task_name}_align_bpu_validation.log | awk 'END {print}'| awk 'BEGIN { FS = "Validation "} ; {print $2}' >> /job_data/Result_xj3.log
done

cat /job_data/Result_xj3.log