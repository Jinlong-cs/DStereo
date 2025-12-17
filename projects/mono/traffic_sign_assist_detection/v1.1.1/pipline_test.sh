set -e
cur_dir="$( cd "$( dirname "$0" )" && pwd )"

HAT_ROOT=${cur_dir%\/projects\/*}
echo cur_dir $cur_dir
echo HAT_ROOT $HAT_ROOT

export PYTHONPATH=${HAT_ROOT}:${PYTHONPATH}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.2/lib64:/usr/local/cuda-11.6/lib64:${LD_LIBRARY_PATH}

python3 tools/train.py --config ${cur_dir}/multitask.py --stage float --pipeline-test
python3 tools/train.py --config ${cur_dir}/multitask.py --stage qat --pipeline-test --device 1
python3 tools/train.py --config ${cur_dir}/multitask.py --stage int_infer --pipeline-test --device 1
