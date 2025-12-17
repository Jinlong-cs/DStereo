set -e
cur_dir="$( cd "$( dirname "$0" )" && pwd )"

HAT_ROOT=${cur_dir%\/projects\/*}
echo cur_dir $cur_dir
echo HAT_ROOT $HAT_ROOT


export HPFLOW_INFER=True

hpflow_run evaluate \
    --data ${cur_dir}/datasets.py \
    --config ${cur_dir}/traffic_sign_hat.py \
    --ctx 0,1,2,3 \
    --num-workers 12 \
    --project-id PDT20220004 \
    --overwrite \
    --upload \
    --upload-num-workers 8 \
    --vis-render \
    --cfg_path ${cur_dir}/../v1.1.9/multitask.py \
    --qat_mode 1 \
    --march bernoulli2 \
    --model_ckpt "http://fm-yanlei-zhang.tcloud.hogpu.cc/plat_gpu/hobot-dag-3589811_TrafficSignMultitask2PE-types258-occ4-v1-1-9-20230802-153718/output/models/mono_multitask_2pe_traffic_sign_v1.1.9/qat-checkpoint-last-e5df4751.pth.tar"
