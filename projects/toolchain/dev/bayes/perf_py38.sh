#!/usr/bin/env bash

set -e
export PYTHONPATH=`pwd`:${PYTHONPATH}

rm -rf release_models*
rm -rf release_hbms*
rm -rf tmp_models*

hdfs dfs -get hdfs://hobot-bigdata/user/kongtao.hu/horizon_algorithm_toolkit/python3.8/release_models.tgz
tar -xvf release_models.tgz
mv release_models tmp_models

find ./ -name "compile" | xargs rm -rf

config_list="
configs/classification/efficientnet.py
configs/classification/mobilenetv1.py
configs/classification/mobilenetv2.py
configs/classification/resnet18.py
configs/classification/resnet50.py
configs/classification/vargnetv2.py
configs/classification/mixvargenet.py
configs/classification/horizon_swin_transformer.py
configs/classification/torchvision/resnet18.py
configs/classification/torchvision/resnet50.py
configs/classification/torchvision/mobilenetv2.py
configs/detection/retinanet/retinanet_vargnetv2_fpn_mscoco.py
configs/detection/fcos/fcos_efficientnetb0_mscoco.py
configs/detection/fcos/fcos_efficientnetb2_mscoco.py
configs/detection/fcos/fcos_efficientnetb3_mscoco.py
configs/detection/yolov3/pascalvoc_mobilenetv1.py
configs/detection/yolov3/mscoco_varg_darknet53.py
configs/segmentation/unet.py
configs/opticalflow_pred/pwcnet/pwcnet_lg.py
configs/detection/pointpillars/pointpillars_kitti_car.py
configs/detection/detr/detr_resnet50_mscoco.py
configs/detection/detr/detr_efficientnetb3_mscoco.py
configs/lane_pred/ganet/ganet.py
configs/detection/fcos3d/fcos3d_efficientnetb0_nuscenes.py
"

for cfg in ${config_list};
do
  python3 tools/compile_perf.py --config ${cfg}
done

cd tmp_models
rm -rf *.LOG
rm -rf dwunet_seg/tensorboard
find ./ -name "*.html" | xargs rm -rf
find ./ -name "*.json" | xargs rm -rf
find ./ -name "*.hbir" | xargs rm -rf

hbm=$(find ./ -name *.hbm)
for i in $hbm; do
    dir=${i%/*}
    cd $dir
    mv *hbm model.hbm
    cd -
done
cd ..

cp -r tmp_models release_hbms
find ./release_hbms -name "*.pth.tar" | xargs rm -rf
find ./release_hbms -name "*.pth" | xargs rm -rf
find ./release_hbms -name "*.pt" | xargs rm -rf
rm -rf release_hbms/deeplabv3plus_efficientnetm0_seg/
rm -rf release_hbms/deeplabv3plus_efficientnetm1_seg/
rm -rf release_hbms/deeplabv3plus_efficientnetm2_seg/
rm -rf release_hbms/efficientnasnetm_cls/
rm -rf release_hbms/efficientnasnets_cls/
rm -rf release_hbms/fastscnn_efficientnetb0_seg/
rm -rf release_hbms/fcos3d_efficientnetb0_nuscenes_pretrain/
rm -rf release_hbms/vargconvnet_cls/
rm -rf release_hbms/varg_darknet53_cls/

# update
cp projects/toolchain/dev/bayes/Result_py38.LOG tmp_models/Result.LOG
mv tmp_models release_models
find ./release_models -name "*.hbm" | xargs rm -rf

rm -rf release_models.tgz
tar -zcvf release_models.tgz release_models/
tar -zcvf release_hbms.tgz release_hbms/
hdfs dfs -rm -r hdfs://hobot-bigdata/user/kongtao.hu/horizon_algorithm_toolkit/python3.8/release_models.tgz
hdfs dfs -rm -r hdfs://hobot-bigdata/user/kongtao.hu/horizon_algorithm_toolkit/python3.8/release_hbms.tgz
hdfs dfs -copyFromLocal release_models.tgz hdfs://hobot-bigdata/user/kongtao.hu/horizon_algorithm_toolkit/python3.8/
hdfs dfs -copyFromLocal release_hbms.tgz hdfs://hobot-bigdata/user/kongtao.hu/horizon_algorithm_toolkit/python3.8/
# rm -rf release_models*
# rm -rf release_hbm*
