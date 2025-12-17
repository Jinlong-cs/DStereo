RUNNING_PATH="$(pwd)"
SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
HAT_PROJECT_PATH=${SCRIPT_PATH}/../../../../
export PYTHONPATH=${HAT_PROJECT_PATH}:$PYTHONPATH
export NLU_NORM_FILE_PATH=${SCRIPT_PATH}/../utils/nlu/nlu/data/standard/nn/norm
export SETUPTOOLS_USE_DISTUTILS=stdlib
cd ${SCRIPT_PATH}/../utils
if ! ./download_norm_dict.sh; then
    echo "ERROR: ./download_norm_dict.sh failed running" >&2
    exit 1
fi
cd ${RUNNING_PATH}
