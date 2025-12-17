export PYTHONPATH=`pwd`
rm -rf ${HOME}/.cache/torch/hub/checkpoints

job_dir=`realpath -m --relative-to=$(pwd) $(dirname $0)`
echo $job_dir

cluster=$1
echo $cluster
python3 plugins/k8s_submit/submit.py --config ${job_dir}/multitask.py --cluster ${cluster}
