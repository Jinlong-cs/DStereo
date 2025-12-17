# Copyright (c) Horizon Robotics. All rights reserved.
import os

import torch
from horizon_plugin_pytorch.march import March

from hat.data.collates.collates import collate_nlu_with_pad
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

# 训练相关参数
task_name = "tcn_ptq_official"
batch_size_per_gpu = 512
# 不建议多机 可能是通信带来的成本 实际多机训练时速度很慢
device_ids = [2]
learning_rate = 1e-3
base_jfs_path = "file:///jfs-hdfs/user/nlp/nlu_model/1.0.0/data"
vocab_path = os.path.join(base_jfs_path, "label/dict.txt")
label_path = os.path.join(base_jfs_path, "label/")
train_path = os.path.join(base_jfs_path, "dataset/trainset")
val_path = os.path.join(base_jfs_path, "dataset/valset")
predict_path = os.path.join(base_jfs_path, "dataset/testset")


# 部署相关参数
# 用来验证量化后的模型性能损失，一般为内部测试集的子集。
deploy_evalset_path = "file:///jfs-hdfs/user/nlp/nlu_model/testset/testset_chery_220921_shuf_5k.json"
# 部署平台类型： j3 或 j5
platform = "j5"
hbdk_version = "3.37.1"
# 量化损失阈值，百分比
quantinized_acc_loss = "1"
upload_path = "file:///jfs-hdfs/user/nlp/nlu_model/"
upload_version = "upload_test"

# 下列参数一般不需修改
ckpt_dir = os.path.join(CURRENT_PATH, "../output/%s" % task_name)
ckpt_file = os.path.join(ckpt_dir, "float-checkpoint-best.pth.tar")
log_dir = os.path.join(CURRENT_PATH, "../output/%s/log" % task_name)
train_stat_report_file = os.path.join(log_dir, "train_stat_report.log")
predict_stat_report_file = os.path.join(log_dir, "predict_stat_report.log")
deploy_dir = os.path.join(CURRENT_PATH, "../output/%s/deploy" % task_name)
calibration_num = 500
max_query_length = 30
embed_dim = 512
tcn_levels = 5
cudnn_benchmark = True
seed = None
log_rank_zero_only = False
march = March.BAYES if platform == "j5" else March.BERNOULLI2


tokenizer = dict(
    type="NluBasicTokenizer",
    vocab_path=vocab_path,
)

label_processor = dict(
    type="NluLabelProcessor",
    label_path=label_path,
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NluMultiTaskDataset",
        data_path=train_path,
        tokenizer=tokenizer,
        label_processor=label_processor,
        max_query_length=max_query_length,
        data_report_path=train_stat_report_file,
    ),
    # DistributedSampler在DDP模式中为每个GPU分发数据
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    collate_fn=collate_nlu_with_pad,
    num_workers=1,
    pin_memory=False,  # 迫使Tensor是锁页内存，加快转到GPU的速度，电脑性能不行时设为Fasle，不然会卡死
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NluMultiTaskDataset",
        data_path=val_path,
        tokenizer=tokenizer,
        label_processor=label_processor,
        max_query_length=max_query_length,
    ),
    batch_size=batch_size_per_gpu,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    shuffle=False,
    collate_fn=collate_nlu_with_pad,
    num_workers=1,
    pin_memory=False,
)

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=None,
    loss_collector=collect_loss_by_index(2),
)

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=None,
    loss_collector=None,
)

model = dict(  # 参与 training 过程中的模型结构
    type="NluModel",  # type 表示模型的类型，余下的参数都是用于初始化这个类
    backbone=dict(
        type="NluMultiTaskBackbone",
        tokenizer=tokenizer,
        label_processor=label_processor,
        embed_dim=embed_dim,
        tcn_levels=tcn_levels,
        dropout_rate=0.2,
    ),
    losses=dict(type="CrossEntropyLoss", reduction="none"),
)


# callbacks 表示训练过程中进行的一些列操作，例如模型保存、学习率更新等
def update_metric(metrics, batch, model_outs):
    preds, targets, losses = model_outs
    loss_metric, f1_metric = metrics
    loss_metric.update(losses)
    f1_metric.update(preds, targets)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=500,
    epoch_log_freq=1,
    log_prefix=task_name,
)
val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=500,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=500,
)  # 报告epoch，训练时间等

ckpt_callback = dict(
    type="Checkpoint",  # 保存模型的callback
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,  # 保持train model和test model的结构严格一致
    mode="min",  # 指标值越大越好就用max，指标值越小越好就用min
    save_hash=False,  # 不做hash
)

val_callback = dict(
    type="Validation",  # 拟合训练模型和验证集数据，计算指标并报告
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    val_on_train_end=False,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        # params={"weight": dict(weight_decay=4e-4)},
        params={},
        lr=learning_rate,
    ),
    batch_processor=batch_processor,
    num_epochs=2,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",  # 调整lr
            update_by="epoch",
            lr_decay_id=[1],
            lr_decay_factor=0.6,
            step_log_interval=1,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    # train_metric 表示训练过程的指标验证
    train_metrics=[
        dict(type="LossShow"),
        dict(
            type="NluEvalMetric",
            name=["domain_f1", "intent_f1", "slots_f1", "multi_task_f1"],
            average="macro",
        ),
    ],
    # val_metric 表示每个 epoch 结束模型验证过程的指标验证。
    val_metrics=[
        dict(type="LossShow"),
        dict(
            type="NluEvalMetric",
            name=["domain_f1", "intent_f1", "slots_f1", "multi_task_f1"],
            report=False,
            out_path=os.path.join(
                ckpt_dir, "predict_result/float_trainer_validation.output"
            ),
            average="macro",
        ),
    ],
)


# predict 流程
predict_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=500,
    epoch_log_freq=1,
    log_prefix="prediction" + task_name,
)

predict_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NluMultiTaskDataset",
        data_path=predict_path,
        tokenizer=tokenizer,
        label_processor=label_processor,
        max_query_length=max_query_length,
        data_report_path=predict_stat_report_file,
    ),
    batch_size=batch_size_per_gpu,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    shuffle=False,
    collate_fn=collate_nlu_with_pad,
    num_workers=0,
    pin_memory=False,
)

float_predictor = dict(
    type="Predictor",
    model=model,  # 参与验证的模型结构
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=ckpt_file,
            ),
        ],
    ),
    data_loader=[predict_data_loader],  # 验证的数据集
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="LossShow"),
        dict(
            type="NluEvalMetric",
            name=["domain_f1", "intent_f1", "slots_f1", "multi_task_f1"],
            report=True,
            out_path=os.path.join(
                ckpt_dir, "predict_result/float_predictor.output"
            ),
            average="macro",
            use_protocol=True,
        ),
    ],
    callbacks=[
        predict_metric_updater,
    ],
    log_interval=500,
)

# deploy 流程
# 模型编译过程的模拟输入。不用关心具体的数值，只要保证格式满足输入要求即可
deploy_inputs = dict(
    input_emb=torch.randn((1, embed_dim, 1, 30)),
)

# 参与 deploy 过程的模型结构，主要用于模型编译
deploy_model = dict(
    type="NluModel",
    backbone=dict(
        type="NluMultiTaskBackbone",
        tokenizer=tokenizer,
        label_processor=label_processor,
        embed_dim=embed_dim,
        tcn_levels=tcn_levels,
        dropout_rate=0.2,
    ),
    losses=None,
)

onnx_cfg = dict(
    model=deploy_model,
    stage="float",
    inputs=deploy_inputs,
    model_convert_pipeline=float_predictor["model_convert_pipeline"],
    kwargs=dict(
        verbose=True,
        opset_version=11,
    ),
)

deploy_cfg = dict(
    extracted_ckpt=ckpt_file,
    layers_path=os.path.join(deploy_dir, "special_layers/"),
)
