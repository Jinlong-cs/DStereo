export PYTHONPATH=`pwd`
rm -rf ${HOME}/.cache/torch/hub/checkpoints
cd plugins/k8s_submit 
python3 submit.py \
    --job-name hat_superdrive_data_masking_front \
    --num-machines 1 \
    --num-gpus-per-machine 8 \
    --config ../../projects/fsd/app/configs/data_masking_front/multitask.py \
    --cluster share-3090-small-bcloud
