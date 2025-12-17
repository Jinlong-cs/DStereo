#!/bin/bash
export PYTHONPATH=`pwd`
rm -rf ${HOME}/.cache/torch/hub/checkpoints
cd plugins/k8s_submit 
python3 submit.py \
    --job-name hat_mono_toll_gate \
    --num-machines 1 \
    --config ../../projects/mono/toll_gate/multitask.py \
    --cluster share-3090-small-tcloud \

