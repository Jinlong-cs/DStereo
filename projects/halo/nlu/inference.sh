export SETUPTOOLS_USE_DISTUTILS=stdlib
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
HAT_PROJECT_PATH=${SHELL_FOLDER}/../../../
export PYTHONPATH=${HAT_PROJECT_PATH}:$PYTHONPATH

python inference_float.py --config configs/tcn_ptq_demo.py --input utils/test_queries.txt
