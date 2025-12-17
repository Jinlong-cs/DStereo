SHELL_FOLDER=$(dirname $(readlink -f "$0"))
HAT_PROJECT_PATH=${SHELL_FOLDER}/../../../
export PYTHONPATH=${HAT_PROJECT_PATH}:$PYTHONPATH

cfg_file=projects/halo/nlu/configs/tcn_ptq_demo.py

cd utils
if ! ./download_norm_dict.sh; then
    echo "ERROR: ./download_norm_dict.sh failed running" >&2
    exit 1
fi

# 模型训练
cd ../../../../
if ! python tools/train.py --stage float --config ${cfg_file}; then
    echo "ERROR: train.py float stage failed running" >&2
    exit 1
fi

# 模型评估
export NLU_NORM_FILE_PATH=${HAT_PROJECT_PATH}projects/halo/nlu/utils/nlu/nlu/data/standard/dict
if ! python tools/predict.py --stage float --config ${cfg_file}; then
    echo "ERROR: predict.py float stage failed running" >&2
    exit 1
fi
