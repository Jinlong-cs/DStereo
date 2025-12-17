export PYTHONPATH=`pwd`
rm -rf ${HOME}/.cache/torch/hub/checkpoints
cd plugins/k8s_submit 
python3 submit.py --config ../../projects/mono/real3d_multitask/multitask.py --current-cluster share-2080ti-aliyun


