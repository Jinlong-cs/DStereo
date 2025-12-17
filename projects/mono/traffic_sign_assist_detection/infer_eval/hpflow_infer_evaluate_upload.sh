set -e
cur_dir="$( cd "$( dirname "$0" )" && pwd )"

HAT_ROOT=${cur_dir%\/projects\/*}
echo cur_dir $cur_dir
echo HAT_ROOT $HAT_ROOT

hpflow_run evaluate \
    --data ${cur_dir}/datasets.py \
    --config ${cur_dir}/traffic_sign_assist_hat.py \
    --ctx 0,1,2,3 \
    --num-workers 4 \
    --project-id PDT20220004 \
    --overwrite \
    --upload \
    --upload-num-workers 8 \
    --vis-render \
    --cfg_path ${cur_dir}/../v1.1.0/multitask.py \
    --qat_mode 1 \
    --march bernoulli2 \
    --model_ckpt "http://fm-yanlei-zhang.tcloud.hogpu.cc/plat_gpu/hobot-dag-3599297_TrafficSignAssistDetection-v1-0-0-20230803-115607/output/models/mono_traffic_sign_assist_detection_v1.0.0/qat-checkpoint-last-5a4ee8db.pth.tar"
