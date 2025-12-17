export PYTHONPATH=`pwd`
rm -rf ${HOME}/.cache/torch/hub/checkpoints

job_dir=`realpath -m --relative-to=$(pwd) $(dirname $0)`
echo $job_dir

queue=$1
echo $queue
python3 plugins/k8s_submit/submit.py --config ${job_dir}/multitask.py --queue ${queue}
