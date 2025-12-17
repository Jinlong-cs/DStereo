# Copyright (c) Horizon Robotics. All rights reserved.
import copy
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
task_name = "tcn_qat_official"
batch_size_per_gpu = 512
# 不建议多机 可能是通信带来的成本 实际多机训练时速度很慢
device_ids = [1]
learning_rate = 1e-3
base_jfs_path = "file:///jfs-hdfs/user/nlp/nlu_model/1.0.0/data"
vocab_path = os.path.join(base_jfs_path, "label/dict.txt")
label_path = os.path.join(base_jfs_path, "label/")
train_path = os.path.join(base_jfs_path, "dataset/trainset")
val_path = os.path.join(base_jfs_path, "dataset/valset")
predict_path = (
    "file:///jfs-hdfs/user/nlp/nlu_model/testset/testset_chery_220921.json"
)

# 部署相关参数
# 部署平台类型： qat部署目前只能设置j5
platform = "j5"
hbdk_version = "3.37.1"
# 量化损失阈值，百分比
quantinized_acc_loss = "1"
upload_path = "file:///jfs-hdfs/user/nlp/nlu_model/"
upload_version = "upload_test"

# 下列参数一般不需修改
ckpt_dir = os.path.join(CURRENT_PATH, "../output/%s" % task_name)
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


metric_updater = dict(  # 是一个callback，根据model_output计算指标
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

# train
# 各stage训练配置

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
            type="StepDecayLrUpdater",
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


# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader["dataset"]["data_path"] = train_path
calibration_data_loader.pop("sampler")

calibration_batch_processor = copy.deepcopy(val_batch_processor)

calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
            dict(type="Float2Calibration"),
        ],
    ),
    # 设置 data_loader 和 batch_processor
    # 做 Calibration 的数据集(dataset)不能是测试集(可以是训练集或其他数据)
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    # 设置 calibration 迭代的 batch 数目
    num_steps=4,
    device=None,
    callbacks=[
        stat_callback,
        val_callback,
        ckpt_callback,
    ],
    val_metrics=[
        dict(type="LossShow"),
        dict(
            type="NluEvalMetric",
            name=["domain_f1", "intent_f1", "slots_f1", "multi_task_f1"],
            average="macro",
        ),
    ],
    log_interval=1,
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_qat_qkwargs=dict(
                averaging_constant=0,
            ),  # 0.0 ~ 1.0 之间的浮点数
            weight_qat_qkwargs=dict(
                averaging_constant=1,
            ),  # 0.0 ~ 1.0 之间的浮点数
        ),
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={},
        lr=5e-4,
    ),
    resume_optimizer=False,  # 加载ckpt时是否加载optimizer模块
    resume_epoch_or_step=False,  # 加载ckpt时是否恢复epoch和step
    batch_processor=batch_processor,
    num_epochs=1,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            update_by="epoch",
            lr_decay_id=[1],
            lr_decay_factor=0.6,
            step_log_interval=1,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
        dict(
            type="NluEvalMetric",
            name=["domain_f1", "intent_f1", "slots_f1", "multi_task_f1"],
            average="macro",
        ),
    ],
    val_metrics=[
        dict(type="LossShow"),
        dict(
            type="NluEvalMetric",
            name=["domain_f1", "intent_f1", "slots_f1", "multi_task_f1"],
            report=False,
            out_path=os.path.join(
                ckpt_dir, "predict_result/qat_trainer_validation.output"
            ),
            average="macro",
        ),
    ],
)

int_infer_trainer = dict(
    type="Trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[
        val_callback,
        ckpt_callback,
    ],
    val_metrics=[
        dict(type="LossShow"),
    ],
)


# predictor
# 各stage验证流程配置
predict_metric_updater = copy.deepcopy(metric_updater)
predict_metric_updater["log_prefix"] = "prediction " + task_name

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
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
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

qat_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[predict_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="LossShow"),
        dict(
            type="NluEvalMetric",
            name=["domain_f1", "intent_f1", "slots_f1", "multi_task_f1"],
            report=True,
            out_path=os.path.join(
                ckpt_dir, "predict_result/qat_predictor.output"
            ),
            average="macro",
            use_protocol=True,
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

int_infer_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=[predict_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="LossShow"),
        dict(
            type="NluEvalMetric",
            name=["domain_f1", "intent_f1", "slots_f1", "multi_task_f1"],
            report=True,
            out_path=os.path.join(
                ckpt_dir, "predict_result/int_infer_predictor.output"
            ),
            average="macro",
            use_protocol=True,
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

# deploy
# 部署流程相关配置
deploy_cfg = dict(
    extracted_ckpt=os.path.join(ckpt_dir, "qat-checkpoint-best.pth.tar"),
    layers_path=os.path.join(deploy_dir, "special_layers/"),
    float_emb=os.path.join(
        deploy_dir, "special_layers/embedding.float.weight.bin"
    ),
    quantized_emb=os.path.join(
        deploy_dir, "special_layers/embedding.weight.bin"
    ),
    scale=0.0078125,
)

# 模型编译过程的模拟输入。不用关心具体的数值，只要保证格式满足输入要求即可
deploy_inputs = dict(
    input_emb=torch.randn((1, embed_dim, 1, max_query_length)),
)


# models/model_convert.py
# 这个类作用就是把模型转换的过程和加载参数的过程 `compose` 到一起，
# 使得最终参与训练的模型是用户希望的模型。

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT"),
        dict(type="QAT2Quantize"),
    ],
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


compile_cfg = dict(
    march=march,
    name=task_name,  # perf 信息的文件名
    out_dir=deploy_dir,  # perf 信息的输出路径
    hbm=os.path.join(deploy_dir, "base_best.hbm"),  # 编译生成的 hbm 文件（部署模型）的输出路径
    layer_details=True,  # 是否输出每层的性能情况
    input_source=["ddr"],  # 模型上板时的输入来源,非图片选ddr
    input_layout="NHWC",
    output_layout="NHWC",
    opt="O3",  # 优化等级，可以选择 O0, O1, O2, O3，优化程度从低到高
)
