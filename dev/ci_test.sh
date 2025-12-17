set -e

export PATH=~/.local/bin:$PATH

# prepare hat
export PYTHONPATH=$(pwd):${PYTHONPATH}

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

# run tests
echo "----------------run test with pytest----------------"
make unit-test
echo "----------------test success!-----------------------"
