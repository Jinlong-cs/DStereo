import os
from importlib import import_module

import torch

# -------------------------- common --------------------------
from common import (
    batch_size_per_gpu,
    batch_transforms,
    ckpt_dir,
    device_ids,
    expand_type,
    input_hw,
    rec_prefix,
    tasks,
    test_transforms,
    training_step,
)
from horizon_plugin_pytorch.quantization import March

from hat.core.task_sampler import TaskSampler
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.utils.apply_func import _as_list
from hat.utils.config import ConfigVersion

# -------------------------- multitask --------------------------
seed = None
log_rank_zero_only = True
cudnn_benchmark = True
sync_bn = True
enable_amp = False

VERSION = ConfigVersion.v2
# march = March.BERNOULLI  # J2
march = March.BERNOULLI2  # J3
# march = March.BAYES            # J5
# job_name = "face_quality_mtl_hat_speed_numworker3_batchsize512"
# job_name = "facequality_mtl_align_hat_batchsize64_moment0.1"
job_name = "facequality_mtl_hat_batchsize256_longside_square1.2_moment0.1"
device_ids = device_ids

ckpt_dir = ckpt_dir + "/" + job_name

inputs = dict(img=torch.zeros((2, 3, *input_hw)))
test_inputs = dict(img=torch.randn((1, 3, *input_hw)))
deploy_inputs = test_inputs
opt_inputs = dict(
    train=dict(),
    val=dict(),
    test=dict(),
    deploy=dict(),
)

# -------------------------- pretrain model --------------------------
pretrain_model_path = ""

# -------------------------- train data --------------------------
task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(t) for t in task_names]

train_task_sampler_config = {
    T.task_name: dict(sampling_factor=1) for T in TASK_CONFIGS
}

train_task_sampler = TaskSampler(
    task_config=train_task_sampler_config, method="sample_all"
)

train_loaders = {T.task_name: T.train_data_loader for T in TASK_CONFIGS}
train_data_loader = dict(
    type="MultitaskLoader",
    loaders=train_loaders,
    task_sampler=train_task_sampler,
    return_task=True,
    mode="max_size",
    __build_recursive=True,
)


# -------------------------- test data --------------------------
test_rec_list = [
    # blur, brightness, leye, reye, forehead, mouth
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_J2DMS_allattr_ir549197_num5988.rec",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_J2DMS_allattr_rgb547413_num18697.rec",  # noqa
    # mouth
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/mouth_ir_20210810_num10637.rec",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/mouth_rgb_20210810_num11399.rec",  # noqa
    # f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Specific_Testset/{expand_type}/test_mouth_badcase_yawn_20220722_num973.rec",  # noqa
    # f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Specific_Testset/{expand_type}/test_mouth_badcase_phone_occlusion_20220624_num16139.rec",  # noqa
    # glass
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_glass_ir_20220329_num25695.rec",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_glass_rgb_20220329_num18906.rec",  # noqa
    # mask
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_mask_ir_20220329_num17956.rec",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_mask_rgb_20220329_num7243.rec",  # noqa
    # hat
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_hat_ir_20220329_num18963.rec",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_hat_rgb_20220329_num13271.rec",  # noqa
]
test_idx_list = [
    # blur, brightness, leye, reye, forehead, mouth
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_J2DMS_allattr_ir549197_num5988.idx",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_J2DMS_allattr_rgb547413_num18697.idx",  # noqa
    # mouth
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/mouth_ir_20210810_num10637.idx",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/mouth_rgb_20210810_num11399.idx",  # noqa
    # f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Specific_Testset/{expand_type}/test_mouth_badcase_yawn_20220722_num973.idx",  # noqa
    # f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Specific_Testset/{expand_type}/test_mouth_badcase_phone_occlusion_20220624_num16139.idx",  # noqa
    # glass
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_glass_ir_20220329_num25695.idx",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_glass_rgb_20220329_num18906.idx",  # noqa
    # mask
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_mask_ir_20220329_num17956.idx",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_mask_rgb_20220329_num7243.idx",  # noqa
    # hat
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_hat_ir_20220329_num18963.idx",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_hat_rgb_20220329_num13271.idx",  # noqa
]
test_label_list = [
    # blur, brightness, leye, reye, forehead, mouth
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_J2DMS_allattr_ir549197_num5988.label.npy",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_J2DMS_allattr_rgb547413_num18697.label.npy",  # noqa
    # mouth
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/mouth_ir_20210810_num10637.label.npy",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/mouth_rgb_20210810_num11399.label.npy",  # noqa
    # f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Specific_Testset/{expand_type}/test_mouth_badcase_yawn_20220722_num973.label.npy",  # noqa
    # f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Specific_Testset/{expand_type}/test_mouth_badcase_phone_occlusion_20220624_num16139.label.npy",  # noqa
    # glass
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_glass_ir_20220329_num25695.label.npy",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_glass_rgb_20220329_num18906.label.npy",  # noqa
    # mask
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_mask_ir_20220329_num17956.label.npy",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_mask_rgb_20220329_num7243.label.npy",  # noqa
    # hat
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_hat_ir_20220329_num18963.label.npy",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Testset/{expand_type}/test_hat_rgb_20220329_num13271.label.npy",  # noqa
]
info_path = f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/label_idx_map_v1.0.yaml"  # noqa


test_name_list = [
    ["blur", "brightness", "leye", "reye", "forehead"],
    ["blur", "brightness", "leye", "reye", "forehead"],
    ["mouth"],
    ["mouth"],
    # ["mouth"],
    # ["mouth"],
    ["glass"],
    ["glass"],
    ["mask"],
    ["mask"],
    ["hat"],
    ["hat"],
]

test_metric_list = []
test_metric_updaters = []
test_data_loaders = []
# collect test_metric
for test_names in test_name_list:
    test_metric = []
    for name in test_names:
        test_metric.append(dict(type="Accuracy", name=f"{name}_accuracy"))
        if name == "brightness":
            test_metric.append(dict(type="Recall", cls_num=4, task_name=name))
        else:
            test_metric.append(dict(type="Recall", cls_num=2, task_name=name))
    test_metric_list.append(test_metric)


# collect test_data_loader
for i, test_names in enumerate(test_name_list):
    loader_key = tuple(test_names)
    loader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="FaceQualityDataset",
            rec_path=test_rec_list[i],
            idx_path=test_idx_list[i],
            label_path=test_label_list[i],
            info_path=info_path,
            task_name=test_names,
            need_flag=False,
            transforms=test_transforms,
        ),
        sampler=dict(type=torch.utils.data.DistributedSampler),
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # example loaders: {('blur', 'mouth'): loader}
    test_loader = dict(
        type="MultitaskLoader",
        task_sampler=None,
        return_task=True,
        mode="validation",
        loaders={loader_key: loader},
        __build_recursive=True,
    )
    test_data_loaders.append(test_loader)


def test_update_metric(metrics, batch, model_outs):
    for metric in metrics:
        # get task name
        cur_task_name = metric.name
        if isinstance(cur_task_name, list):
            cur_task_name = cur_task_name[0]
        cur_task_name = cur_task_name.split("_")[0]
        gt = batch[0][cur_task_name][f"gt_{cur_task_name}"]
        preds = model_outs[cur_task_name]["pred"]
        if cur_task_name == "brightness":
            preds = torch.argmax(preds, axis=1)
        else:
            preds = preds > 0

        # ignore data labeled 0.5 or -1 (necessary when calculate acc)
        preds = preds[gt != -1]
        gt = gt[gt != -1]
        preds = preds[gt != 0.5]
        gt = gt[gt != 0.5]
        metric.update(gt, preds)


# collect test_metric_updater
for i in range(len(test_name_list)):
    test_metric_updater = dict(
        type="MetricUpdater",
        metrics=test_metric_list[i],
        metric_update_func=test_update_metric,
        step_log_freq=0,
        epoch_log_freq=1,
        log_prefix=f"test_rec{i}",
    )
    test_metric_updaters.append(test_metric_updater)


# -------------------------- model --------------------------
def get_model(mode):
    return dict(
        type="MultitaskGraphModel",
        inputs=inputs,
        opt_inputs=opt_inputs[mode],
        task_inputs={T.task_name: T.inputs[mode] for T in TASK_CONFIGS},
        task_modules={T.task_name: T.get_model(mode) for T in TASK_CONFIGS},
        funnel_modules=None,
        flatten_outputs=True if mode == "deploy" else False,
        lazy_forward=False,
    )


train_model = get_model("train")
# test_model = get_model("test")
deploy_model = get_model("deploy")


# ----------------- callbacks ---------------------------
# train callbacks
stat_callback = dict(
    type="StatsMonitor",
    log_freq=500,
)

lr_callback = dict(
    type="StepDecayLrUpdater",
    lr_decay_id=[10, 20],
    lr_decay_factor=0.1,
    warmup_by="step",
    warmup_len=1000,
    step_log_interval=500,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    save_hash=False,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=test_inputs,
)

train_metric_updaters = [T.metric_updater for T in TASK_CONFIGS]
callbacks = [
    stat_callback,
    lr_callback,
    ckpt_callback,
]
callbacks += train_metric_updaters
train_metrics = []
for T in TASK_CONFIGS:
    train_metrics += _as_list(T.train_metrics)


# test callbacks
test_metrics = []
for metrics in test_metric_list:
    test_metrics += _as_list(metrics)


# -------------------------- training ----------------------------
# -------------------------- step float --------------------------
train_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    batch_transforms=batch_transforms,
    loss_collector=collect_loss_by_regex("^.*loss.*"),
    enable_amp=enable_amp,
)


float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=train_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=pretrain_model_path,
                allow_miss=False,
                ignore_extra=False,
            ),
        ],
    )
    if pretrain_model_path
    else None,
    data_loader=train_data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=5e-4)},
        lr=0.001,
    ),
    # find_unused_parameters=False,
    batch_processor=train_batch_processor,
    stop_by="epoch",
    num_epochs=30,
    device=None,  # set when building
    sync_bn=sync_bn,
    callbacks=callbacks,
    train_metrics=train_metrics,
    # profiler=profiler,
)


# ------------------------ step qat ------------------------
qat_lr_callback = dict(
    type="StepDecayLrUpdater",
    lr_decay_id=[10, 20],
    lr_decay_factor=0.1,
    step_log_interval=500,
)
# qat_callbacks = [
#     stat_callback,
#     lr_callback,
#     ckpt_callback,
# ]


qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=train_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
            ),
            dict(type="Float2QAT"),
        ],
    ),
    # find_unused_parameters=False,
    data_loader=train_data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=5e-4)},
        lr=0.00001,
    ),
    batch_processor=train_batch_processor,
    num_epochs=6,
    device=None,
    callbacks=callbacks,
    train_metrics=train_metrics,
)


# -------------------------- step int_infer --------------------------
int_infer_trainer = dict(
    type="Trainer",
    model=deploy_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
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
    callbacks=[ckpt_callback, trace_callback],
)


# ----------------------- compile -----------------------------
compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=job_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    output_layout="NHWC",
    input_source=["pyramid"],
)


# -------------------------- prediction ----------------------------
# -------------------------- step float --------------------------
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    batch_transforms=batch_transforms,
    loss_collector=None,
)

float_predictor = dict(
    type="Predictor",
    model=train_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    # ckpt_dir, "float-checkpoint-last.pth.tar"
                    ckpt_dir,
                    "float-checkpoint-epoch-0019.pth.tar",
                ),
            ),
        ],
    ),
    data_loader=test_data_loaders,
    batch_processor=val_batch_processor,
    device=None,
    metrics=test_metrics,
    callbacks=test_metric_updaters,
    log_interval=0,
    share_callbacks=False,
)

qat_predictor = dict(
    type="Predictor",
    model=train_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    # ckpt_dir, "qat-checkpoint-last.pth.tar"
                    ckpt_dir,
                    "qat-checkpoint-epoch-0004.pth.tar",
                ),
            ),
        ],
    ),
    data_loader=test_data_loaders,
    batch_processor=val_batch_processor,
    device=None,
    metrics=test_metrics,
    callbacks=test_metric_updaters,
    log_interval=0,
    share_callbacks=False,
)

int_infer_predictor = dict(
    type="Predictor",
    model=train_model,
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
    data_loader=test_data_loaders,
    batch_processor=val_batch_processor,
    device=None,
    metrics=test_metrics,
    callbacks=test_metric_updaters,
    log_interval=0,
    share_callbacks=False,
)
