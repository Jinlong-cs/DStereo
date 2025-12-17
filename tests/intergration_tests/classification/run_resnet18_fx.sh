#!/usr/bin/env bash

set -e

cp tests/data/resnet18_with_deploy_model.py tests/data/resnet18_fx.py
sed -i 's/convert_mode = "eager"/convert_mode = "fx"/g' tests/data/resnet18_fx.py

cfg=tests/data/resnet18_fx.py

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

rm -rf tests/data/resnet18_fx.py
