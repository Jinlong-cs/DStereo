#!/usr/bin/env bash

set -e


export PYTHONPATH=$(pwd):$PYTHONPATH
# build env
echo "-------------------build env----------------------"
source projects/pilot/dev/build_env.sh daily
echo "---------------do build env success!--------------"

# run doc tests
echo "-------------------run relase_package test (including code stripping and doc building)----------------------"
# cd docs && make html && cd ..
cd projects/pilot/
bash release_package.sh
cd ../../
echo "-----------------release package test success!-------------------"

# run code stripping
echo "-----------stripped code test----------------------"
python3 plugins/code_stripping/code_check.py --target-dir projects/pilot/release_package
echo "-----------do code stripping success!--------------"

# run pipeline tests
echo "-------------------run pipeline tests----------------------"
pytest -s -x projects/pilot/tests/test_packing.py
HAT_PILOT_TEST_LEVEL=daily pytest -s -x projects/pilot/tests/test_pipeline.py
echo "-----------------pipeline tests success-----------------"
