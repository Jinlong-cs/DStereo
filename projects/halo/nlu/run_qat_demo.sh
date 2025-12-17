SHELL_FOLDER=$(dirname $(readlink -f "$0"))
HAT_PROJECT_PATH=${SHELL_FOLDER}/../../../
export PYTHONPATH=${HAT_PROJECT_PATH}:$PYTHONPATH

cfg_file=projects/halo/nlu/configs/tcn_qat_demo.py
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

if ! python tools/train.py --stage calibration --config ${cfg_file}; then
    echo "ERROR: train.py calibration stage failed running" >&2
    exit 1
fi

if ! python tools/train.py --stage qat --config ${cfg_file}; then
    echo "ERROR: train.py qat stage failed running" >&2
    exit 1
fi

if ! python tools/train.py --stage int_infer --config ${cfg_file}; then
    echo "ERROR: train.py int_infer stage failed running" >&2
    exit 1
fi

# 模型评估
export NLU_NORM_FILE_PATH=${HAT_PROJECT_PATH}projects/halo/nlu/utils/nlu/nlu/data/standard/dict
if ! python tools/predict.py --stage float --config ${cfg_file}; then
    echo "ERROR: predict.py float stage failed running" >&2
    exit 1
fi
