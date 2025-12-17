#!/usr/bin/env bash

version=$1
set -e

cp plugins/aidi_inference/configs/aidi_config.py test_aidi_config.py

sed -i 's/model_name =.*/model_name = "test_tmp_aidi_model_inference"/g' test_aidi_config.py
sed -i "s/param_file =.*/param_file = None/g" test_aidi_config.py

python3 plugins/aidi_inference/aidi_deploy.py --config test_aidi_config.py --publish-version $version --disable-inference --project RDS20220011
python3 plugins/aidi_inference/run_predict.py --config test_aidi_config.py --local --version $version

rm -rf test_aidi_config.py
