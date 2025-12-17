# Pilot多任务集群训练启动脚本

进入到hat**根目录**后，运行

```shell
python3 projects/pilot/tools/train/pilot_train_pipeline.py \
    --model-type $MODEL_TYPE \
    --model-setting $SETTING \
    --model-name-postfix $NAME_POSTFIX  \
    --model-version $VERSION \
    --current-cluster $CLUSTER \
    --num-machines $NUM_MACHINES \
    --num-gpus-per-machine $NUM_GPUS_PER_MACHINE \
    --start-stage $START_STAGE \
    --end-stage $END_STAGE \
    --project-id $PROJECT_ID \
    --task-scene $TASK_SCENE \
    --mount-bucket "matrix2,NIOFY_JFS,SD_Algorithm" \
    --val-last-stage \
    (--pipeline-test) \
    (--local) \
    (--enable-tracking)
```

其中

- `MODEL_TYPE`：模型类型，可为resize_2、resize_4、crop、resize_2_rear_bayes、resize_2_side_bayes、crop_bayes，具体可参考[这里](../../model_meta.yaml)。
- `SETTING`：模型设置，一般是火车头、模组和场景的组合，如`sedan_x3c_day`、`galaxy_x3c_side`等，表现为xxx_dataset.py的前缀xxx部分。
- `NAME_POSTFIX`：模型名的后缀，影响集群任务名，以及enable_tracking的情况下上传模型的名称。
- `VERSION`：当前发版版本号，需要严格遵循`vx.x.x`的格式，默认为`v0.0.1`。在enable_tracking的情况下，决定了上传模型的版本号。
- `NUM_MACHINES`：机器数
- `NUM_GPUS_PER_MACHINE`：每台机器使用的GPU数，默认为`8`。
- `START_STAGE`： 训练启动的stage。
- `END_STAGE`：训练结束的stage。
- `TASK_SCENE`：任务场景，用于模型生产埋点，提供3个选项，data_verify表示发版数据验证任务，model_verify表示与模型发版相关的非数据验证任务，model_experiment表示除上述两个选项和正式发版任务外的其他任务，正式发版任务请使用发版模型生产链路

- `--mount-bucket`: 需要挂载的bucket
- `--val-last-stage`: 训练结束会不是放资源，直接评测最后一个stage的模型
- `--local`: 进入本地调试模式，否则，则需要相关参数来提交集群任务：

- `CLUSTER`：集群名，可通过aidi model集群管理页面查看
- `PROJECT_ID`: 项目号

- `--enable-tracking`: 会将训练阶段的各种信息上报至艾迪平台，方便后续查看。
