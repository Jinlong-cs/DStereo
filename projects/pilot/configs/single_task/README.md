# Pilot 图像质量模型

## 环境准备

环境安装参考[Pilot多任务](../../README.md)

## 训练

1. 集群训练
   在hat项目根目录，执行Example:
   ```shell
   current_cluster=project-3090-model-experiment
   job_name=pilot_image_fail_parsing
   project_id=xxx
   model_setting=as33
   python3 projects/pilot/tools/train/submit_image_fail_segmentation.py \
       --current-cluster ${current_cluster} \
       --job-name ${job_name} \
       --project-id ${project_id} \
       --model-setting ${model_setting} \
       --num-machines 2 \
   ```
2. 本地训练
   在hat项目根目录，执行Example:
   ```shell
   job_name=pilot_image_fail_parsing
   model_setting=as33
   python3 projects/pilot/tools/train/submit_image_fail_segmentation.py \
       --job-name ${job_name} \
       --model-setting ${model_setting} \
       --pipeline-test \
   ```

## 评测

1. 本地评测
   在hat项目根目录, 使用 `python3 tools/predict.py`, Example:

   ```shell
   export HAT_PILOT_MODEL_SETTING=as33
   export PROJECT_ID=xxx
   config_path=projects/pilot/configs/single_task/eval_image_fail_segmentation.py
   eval_stage=qat
   python3 tools/predict.py \
       --config ${config_path} \
       --stage ${eval_stage} \
       --device-ids 0,1 \
   ```

注意：

- 当前图像质量支持`qat`和`int_infer`阶段模型的infer
- [eval_image_fail_segmentation.py](eval_image_fail_segmentation.py) 指定checkpoint_path，请指定为`qat`阶段的checkpoint
- [image_fail_seg_eval_datasets.py](../datasets/image_fail_seg_eval_datasets.py) 指定评测集

2. 集群评测
   在hat根目录, 使用 `python3 projects/pilot/tools/eval/pilot_cluster_eval.py`, Example:
   ```shell
   config_path=projects/pilot/configs/single_task/eval_image_fail_segmentation.py
   eval_stage=qat
   job_name=pilot_image_fail_parsing
   model_setting=as33
   mount_bucket=auto_eval,matrix
   project_id=xxx
   current_cluster=project-titanx-pilot

   python3 projects/pilot/tools/eval/pilot_cluster_eval.py \
       --config ${config_path} \
       --eval-stage ${eval_stage} \
       --job-name ${job_name} \
       --model-setting ${model_setting} \
       --model-version ${model_version} \
       --num-machines 1 \
       --num-gpus-per-machine 4 \
       --mount-bucket ${mount_bucket} \
       --project-id ${project_id} \
       --current-cluster ${current_cluster} \
   ```
