#!/usr/bin/env bash

set -e

PROJECT_BASE=projects/superparking

source ${PROJECT_BASE}/dev/build_env.sh
export PATH=~/.local/bin:${PATH}
hbdk-model-verifier -h

# run doc tests
echo "---------- release_package test (including code stripping and doc building) -------------"
current_path=`pwd`
cd ${PROJECT_BASE}
bash release_package.sh ${test_level}
cd ${current_path}

python3 plugins/code_stripping/code_check.py --target-dir ${PROJECT_BASE}/release_package

# run daily level ci tests
echo "-------------------run pipeline test----------------------"
cd ${PROJECT_BASE}/release_package/
HAT_SP_TEST_LEVEL=daily pytest -s -x ${PROJECT_BASE}/tests/test_pipeline.py
echo "----------------- daily ci test success!-------------------"
