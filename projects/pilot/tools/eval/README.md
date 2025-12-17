# 模型评测

## 本地评测

在hat项目根目录, 使用`python3 tools/predict.py`, Example:

        # cd to hat root
        export HAT_PILOT_MODEL_SETTING=sedan_x3c_day
        export HAT_PILOT_EVAL_DATA_SETTING=c385_x3c_day
        export HAT_PILOT_MODEL_VERSION=5.0

        config_path=projects/pilot/configs/resize_2/eval_multitask.py
        stage=sparse_3d_freeze_bn_2
        project_id=xxx


        python3 tools/predict.py \
                --config ${config_path} \
                --stage ${stage} \
                --device-ids 0 \
                --project-id ${project_id}$

对应参数有:

- `--config`: 评测config本地路径, 相对于hat根目录或绝对路径
- `--stage`: 需要评测的模型stage, 支持训练过程中的各个阶段模型评测
- `--device-ids`: 本地预测使用gpus

此外, 我们通过环境变量确定模型:
- `HAT_PILOT_MODEL_SETTING`: pilot模型的训练数据setting（一般是火车头、模组和场景的组合，如sedan_x3c_day)
- `HAT_PILOT_EVAL_DATA_SETTING`: pilot模型的评测数据setting（一般是ltc、模组和场景的组合，如c385_x3c_day)
- `HAT_PILOT_MODEL_VERSION`: pilot模型的version

需要注意的是环境变量会用来确定使用的模型.

## 集群评测

在hat根目录, 使用`python3 projects/pilot/tools/eval/pilot_cluster_eval.py`, Example:

        # cd hat root
        model_type=resize_2
        stage=sparse_3d_freeze_bn_2
        job_name=pilot_resize_2
        model_setting=sedan_x3c_night
        eval_data_setting=c385_x3c_night
        model_version=v5.0.0
        mount_bucket=matrix,auto_eval
        project_id=xxx
        current_cluster=project-titanxp-pilot  # [project-titanx-pilot, idc-share-titanxp-8]
        task_scene=model_experiment

        python3 projects/pilot/tools/eval/pilot_cluster_eval.py \
                --model-type ${model_type} \
                --stage ${stage} \
                --job-name ${job_name} \
                --model-setting ${model_setting} \
                --eval-data-setting ${eval_data_setting} \
                --model-name-postfix ${model_name_postfix}
                --eval-postfix ${eval_postfix}
                --model-version ${model_version} \
                --num-machines 1 \
                --num-gpus-per-machine 4 \
                --mount-bucket ${mount_bucket} \
                --project-id ${project_id} \
                --current-cluster ${current_cluster} \
                --task-scene ${task_scene} \

部分参数含义与本地评测一致, 其他参数有:

- `--model-type`: 模型类型，可为resize_2、resize_4、crop、resize_2_rear_bayes、resize_2_side_bayes、crop_bayes，具体可参考[这里](../../model_meta.yaml)。
- `--stage`: 需要评测的模型stage, 支持训练过程中的各个阶段模型评测, 但需求注意模型必须保存在aidi model平台方可使用
- `--model-setting`: pilot模型的setting（一般是火车头、模组和场景的组合，如sedan_x3c_day)
- `--eval-data-setting`: pilot模型的setting（一般是ltc、模组和场景的组合，如c385_x3c_day)
- `--model-name-postfix`: pilot模型的名称后缀，可为空。
- `--eval-postfix`: 评测结果的名称后缀，可为空。
- `--model-version`: pilot模型的version，以v开头，如`v0.0.1`。
- `--job-name`: 集群job name前缀, 和model-setting, model-version组成完整job name
- `--num-gpus-per-machine`: 使用gpu数量
- `--mount-bucket`: 需要挂载的bucket
- `--project-id`: 项目id
- `--current-cluster`: 预测集群, 由于数据读取效率问题, 强烈建议使用idc集群
- `--task-scene`：任务场景，用于模型生产埋点，提供3个选项，data_verify表示发版数据验证任务，model_verify表示与模型发版相关的非数据验证任务，model_experiment表示除上述两个选项和正式发版任务外的其他任务，正式发版任务请使用发版模型生产链路
