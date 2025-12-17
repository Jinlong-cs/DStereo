set -e

export PATH=/root/.local/bin:/home/cicd/.local/bin/:$PATH

# prepare hat
export PYTHONPATH=$(pwd):${PYTHONPATH}

# run code stripping
echo "-----------run code stripping test-------------------"
cd plugins/code_stripping/
python3 code_stripping.py --file-list configs/toolchain-file-list.py --target-dir ../../release/HAT --override
cd ../../release/HAT

ln -s ../../tmp_data tmp_data
ln -s ../../tmp_orig_data tmp_orig_data
ln -s ../../tmp_models tmp_models

# prepare stripping hat
export PYTHONPATH=$(pwd):${PYTHONPATH}

# run tests
echo "----------------run test with pytest----------------"
make unit-test
echo "----------------test success!-----------------------"

# do docs test
cd dev/api_generator
python3 api_generator.py --api-module-list api_module_list.yaml
cd ../../docs
sphinx-build -M html source build -W
cd ..
echo "-----------do code stripping success!--------------"
