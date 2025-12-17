set -e

export PATH=~/.local/bin:$PATH

# prepare hat
export PYTHONPATH=$(pwd):${PYTHONPATH}

# run tests
echo "----------------run stable test with pytest----------------"
make serial-unit-test
make intergration-tests
echo "----------------test success!-----------------------"

# run doc tests
echo "-------------------run doc test----------------------"
# cd docs && make html && cd ..
cd dev/api_generator
python3 api_generator.py --api-module-list api_module_list.yaml
cd ../../

cd docs
sphinx-build -M html source build -W
cd ..
echo "-----------------doc test success!-------------------"

# run optional library tests
# echo "----------------run test with pytest----------------"
# bash optional_lib_test.sh
# echo "----------------test success!-----------------------"
