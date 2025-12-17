set -e

# for gallery-cli
mkdir -p ~/.local/bin/
export PATH=~/.local/bin:~/bin/$PATH

# build env
source projects/pilot/dev/build_env.sh

# skip code style since its done in ut

# run doc tests
echo "-------------------run relase_package test (including code stripping and doc building)----------------------"
# cd docs && make html && cd ..
cd projects/pilot/
bash release_package.sh
echo "-----------------release package test success!-------------------"

echo "---------------------run build integration test --------------------------"
cd release_package
export PYTHONPATH=`pwd`:$PYTHONPATH
HAT_PILOT_TEST_LEVEL=commit pytest -s -x projects/pilot/tests/test_pipeline.py
echo "-------------------build integration test success!------------------------"
