
export CUDA_VISIBLE_DEVICES=3

job_dir=`realpath -m --relative-to=$(pwd) $(dirname $0)`
config_file=$job_dir/multitask.py
march=bayes
echo $config_file

python3 tools/train.py --config $config_file --stage float --pipeline-test
python3 tools/train.py --config $config_file --stage freeze_bn --pipeline-test
python3 tools/train.py --config $config_file --stage qat --pipeline-test
python3 tools/train.py --config $config_file --stage int_infer --pipeline-test
python3 tools/compile_standalone.py tmp_output/vehicle_side/int_infer-deploy-checkpoint-last.pt \
    --input-size 1x3x192x960 \
    --output tmp_output/vehicle_side/compile/ \
    --opt O3 --march $march \
    --name real3d_with_desensitization \
    --extra-args '--dev-remove-extra-output-cpu-op --max-time-per-fc 1000 --balance 10'