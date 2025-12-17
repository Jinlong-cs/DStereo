#!/usr/bin/env bash

set -e

cp tests/data/resnet18_with_deploy_model.py tests/data/resnet18_part_ddp.py
sed -i 's/task_name = "resnet18_cls"/task_name = "resnet18_cls_part_ddp"/g' tests/data/resnet18_part_ddp.py
sed -i 's/quanti_head=True/quanti_head=False/g' tests/data/resnet18_part_ddp.py

# do part ddp test
sed -i 's/# convert_submodule_list/convert_submodule_list/g' tests/data/resnet18_part_ddp.py

cfg=tests/data/resnet18_part_ddp.py

# float train and predict
python3 tools/train.py -s float -c ${cfg} $@
python3 tools/predict.py -s float -c ${cfg} $@

# calibration train
python3 tools/train.py -s calibration -c ${cfg} $@

# model checker
python3 tools/deploy/model_checker.py -c ${cfg} $@

# qat train and predict
python3 tools/train.py -s qat -c ${cfg} $@
python3 tools/predict.py -s qat -c ${cfg} $@

# int_infer predict
python3 tools/predict.py -s int_infer -c ${cfg} $@

rm -rf ${cfg}
