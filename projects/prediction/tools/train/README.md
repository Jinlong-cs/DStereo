# 预测多任务集群训练启动脚本

进入到hat**根目录**后使用下列命令自定义提交

```shell
python3 projects/prediction/tools/train/train_pipeline.py \
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
    (--pipeline-test) \
    (--local) \
    (--enable-tracking)
```

其中

- `MODEL_TYPE`：模型类型，具体可参考[这里](../../model_meta.yaml)。
- `SETTING`：模型设置，可以描述模型做了什么修改，会成为job名称中的一部分。
- `NAME_POSTFIX`：模型名的后缀，影响集群任务名，以及enable_tracking的情况下上传模型的名称。
- `VERSION`：当前发版版本号，需要严格遵循`vx.x.x`的格式，默认为`v0.0.1`。在enable_tracking的情况下，决定了上传模型的版本号。
- `NUM_MACHINES`：机器数
- `NUM_GPUS_PER_MACHINE`：每台机器使用的GPU数，默认为`8`。
- `START_STAGE`： 训练启动的stage。
- `END_STAGE`：训练结束的stage。

在使用 `--local`时，进入本地调试模式，否则，则需要相关参数来提交集群任务：

- `CLUSTER`：集群名，可通过aidi model集群管理页面查看
- `PROJECT_ID`: 项目号

在使用`--enable-tracking`时，会将训练阶段的各种信息上报至艾迪平台，方便后续查看。

使用案例：
HAT根目录下
- 本地训练
python3 -W ignore tools/train.py --config projects/prediction/configs/vectornet_traj.py --stage float

- 集群训练
python3 projects/prediction/tools/train/train_pipeline.py \
--model-type vectornet_traj \
--model-version 'v6.1.0' \
--model-setting 'add_SH_dataset.' \
--project-id 'PDT2021004-predic' \
--current-cluster 'share-3090-small-bcloud' \