set -e

test_level=${1:-"commit"}

export PATH=~/.local/bin:$PATH

PROJECT_BASE=projects/superparking

# build env
source ${PROJECT_BASE}/dev/build_env.sh

# run doc tests
echo "---------- release_package test (including code stripping and doc building) -------------"

cd ${PROJECT_BASE}
bash release_package.sh ${test_level}

echo "--------------------------- release package test success! -------------------------------"

echo "-------------------------------- integration test ---------------------------------------"

cd release_package
export PYTHONPATH=`pwd`:$PYTHONPATH
HAT_SP_TEST_LEVEL=${test_level} pytest -s -x ${PROJECT_BASE}/tests/test_pipeline.py

echo "---------------------------- integration test success! ----------------------------------"
