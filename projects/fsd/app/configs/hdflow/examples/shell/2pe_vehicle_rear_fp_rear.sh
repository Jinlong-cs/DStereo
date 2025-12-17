cd ../

rm -rf output
# export TEST_MODE=1
# conda activate hdflow
python3 -u mono/data_management_and_deploy/batch_runner_rear.py
