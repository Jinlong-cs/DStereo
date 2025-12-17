#!/usr/bin/env bash

set -e
export PYTHONPATH=`pwd`:${PYTHONPATH}

rm -rf release_models*
rm -rf release_hbms*
rm -rf tmp_models*

hdfs dfs -get hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/release_models/bernoulli2
mv bernoulli2 tmp_models

find ./ -name "compile" | xargs rm -rf

config_list="
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

for cfg in ${config_list};
do
  python3 tools/compile_perf.py --config ${cfg} --jobs 32
done

cd tmp_models
rm -rf *.LOG
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

# update
cp projects/toolchain/dev/bernoulli2/Result.LOG  tmp_models/Result.LOG
mv tmp_models release_models
find ./release_models -name "*.hbm" | xargs rm -rf

#rm -rf release_models.tgz
#tar -zcvf release_models.tgz release_models/
#tar -zcvf release_hbms.tgz release_hbms/
hdfs dfs -rm -r  hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bernoulli/python3.8/release_models
hdfs dfs -rm -r  release_hbms   hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bernoulli/python3.8/release_hbms
hdfs dfs -copyFromLocal release_models hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bernoulli/python3.8/
hdfs dfs -copyFromLocal release_hbms   hdfs://hobot-bigdata/user/rui.xu/horizon_algorithm_toolkit/bernoulli/python3.8/
# rm -rf release_models*
# rm -rf release_hbm*
