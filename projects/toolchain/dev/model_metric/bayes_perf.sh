#!/usr/bin/env bash

set -e

rm -rf tmp_models

# cp release models from bucket
cp -r tmp_orig_models/bayes_release_models/ tmp_models
rm -rf tmp_models/Result.LOG

rm -rf *.LOG
touch validation.LOG
touch config.LOG
touch Performance.LOG

config_list="
configs/classification/efficientnet.py
configs/classification/mobilenetv1.py
configs/classification/mobilenetv2.py
configs/classification/resnet18.py
configs/classification/resnet50.py
configs/classification/vargnetv2.py
configs/detection/retinanet/retinanet_vargnetv2_fpn_mscoco.py
configs/detection/yolov3/pascalvoc_mobilenetv1.py
configs/segmentation/unet.py
"

for cfg in ${config_list};
do
  rm -rf work_dirs/hat_logss/
  python3 tools/train.py --config ${cfg} --step int_infer --val-only
  echo ${cfg} >> config.LOG
  python3 tools/compile_perf.py --config ${cfg}
  grep -r 'Epoch\[0\] Validation' work_dirs/hat_logss/|awk -F 'Validation' '{print $2}' >> Performance.LOG
  rm -rf work_dirs/hat_logss/
done

rm -rf tmp_models/dwunet_seg/tensorboard
find ./tmp_models -name "*.html" | xargs rm -rf
find ./tmp_models -name "*.json" | xargs rm -rf

# gather performance
paste config.LOG Performance.LOG > tmp_models/Result.LOG

mv tmp_models release_models
tar -cvf model_zoo/bayes_release_models.tar release_models

rm -rf release_models
rm -rf *LOG
