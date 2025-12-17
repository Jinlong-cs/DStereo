current_cluster=share-3090-small
job_name=image_fail_parsing_vargnet_float3w_fbn5k_qat5k_cls5_320x512_test
project_id=TD20220003
model_setting=as33
python3 projects/halo/cv/configs/imagequality/train/submit_image_fail_segmentation.py \
    --current-cluster ${current_cluster} \
    --job-name ${job_name} \
    --project-id ${project_id} \
    --model-setting ${model_setting} \
    --num-machines 1 \
    --num-gpus-per-machine 8 \