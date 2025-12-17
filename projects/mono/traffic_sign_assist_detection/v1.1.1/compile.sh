set -e
cur_dir="$( cd "$( dirname "$0" )" && pwd )"

HAT_ROOT=${cur_dir%\/projects\/*}
echo cur_dir $cur_dir
echo HAT_ROOT $HAT_ROOT

export PYTHONPATH=${HAT_ROOT}:${PYTHONPATH}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.2/lib64:/usr/local/cuda-11.6/lib64:${LD_LIBRARY_PATH}

python3 tools/train.py --config ${cur_dir}/multitask.py --stage int_infer --pipeline-test --device 1

# j5
python tools/deploy/compile_standalone.py \
    ./tmp_output/TrafficSignAssistDetection_v1.1.1/int_infer-deploy-checkpoint-last.pt \
    --input-size 6x3x128x128 \
    --output-layout NCHW  \
    --opt O3  \
    --march bayes \
    --name traffic_sign_assist \
    --input-source resizer \
    --extra-args "--dev-remove-extra-output-cpu-op --debug" \
    --output ./tmp_output/TrafficSignAssistDetection_v1.1.1/compile

# j3
python tools/deploy/compile_standalone.py \
    ./tmp_output/TrafficSignAssistDetection_v1.1.1/int_infer-deploy-checkpoint-last.pt \
    --input-size 1x3x128x128 \
    --output-layout BPU_RAW  \
    --opt O3  \
    --march bernoulli2 \
    --name traffic_sign_assist \
    --input-source resizer \
    --extra-args "--dev-remove-extra-output-cpu-op --debug" \
    --output ./tmp_output/TrafficSignAssistDetection_v1.1.1/compile

