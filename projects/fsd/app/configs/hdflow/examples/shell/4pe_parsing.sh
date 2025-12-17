cd ../

root_dir=/jfs-public/adas/yue01.shi/code/hdflow_code/hdflow

rm -rf output
# export TEST_MODE=1
# conda activate hdflow
export PYTHONPATH=${root_dir}/deps/py_deps:$PYTHONPATH
export PYTHONPATH=${root_dir}/deps/py_deps_gpu:$PYTHONPATH
export PYTHONPATH=${root_dir}/deps/py_deps_horizon_open_source_algo:$PYTHONPATH
export PATH=${root_dir}/deps/bin:$PATH

python3 -u mono/data_management_and_deploy/batch_runner_rear.py