#!bin/bash
export PYTHONPATH=/home/users/ziyang02.wang/home_works/HAT/
config_file="./projects/mono/toll_gate/multitask.py"
python3 tools/train.py --config ${config_file} --stage float --hat-num-machines 1 --pipeline-test
python3 tools/train.py --config ${config_file} --stage qat --hat-num-machines 1 --pipeline-test
python3 tools/train.py --config ${config_file} --stage int_infer --hat-num-machines 1 --pipeline-test
