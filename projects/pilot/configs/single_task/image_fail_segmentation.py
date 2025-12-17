import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.proj_spec.descs import get_image_fail_parsing_desc
from hat.core.task_sampler import TaskSampler
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.utils import Config

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from project_common import (  # noqa
    enable_model_tracking,
    eval_data_setting,
    model_checkpoint,
    model_name_postfix,
    model_setting,
    model_version,
    num_machines,
    pipeline_test,
    resume_training,
    training_step,
)

sys.modules.pop("project_common")

model_type = "pilot_image_fail_segmentation"
model_name = "_".join([model_type, model_setting, model_version])
if model_name_postfix:
    model_name = "_".join([model_name, model_name_postfix])

with_rle = False
local_train = not os.path.exists("/running_package")

device_ids = [0]

if local_train:
    batch_size = 4
    if pipeline_test:
        batch_size = 2
    log_freq = 5
    save_prefix = "tmp_output"
else:
    batch_size = 10
    log_freq = 25
    save_prefix = "/job_data/models/"


task_name = "image_fail_parsing"
tasks = (dict(name="image_fail_parsing", important=True),)

seed = None
log_rank_zero_only = True
cudnn_benchmark = True
sync_bn = True
enable_amp = False
ckpt_dir = Path(save_prefix) / task_name
log_dir = ckpt_dir / "logs"
redirect_config_logging_path = (log_dir / "config_log.log").as_posix()

# input configs
if "galaxy_0233_rear" in model_setting:
    input_hw = resize_hw = [256, 480]
    input_padding = [16, 16, 0, 0]
    vanishing_point = (480 // 2, 270 // 2)
    roi_region = (0, 0, 480, 256)
    actual_input_hw = (256, 512)
elif "x3c" in model_setting:
    input_hw = resize_hw = [320, 480]
    input_padding = [16, 16, 0, 0]
    vanishing_point = (input_hw[1] // 2, input_hw[0] // 2)
    roi_region = (0, 0, *input_hw[::-1])
    actual_input_hw = (320, 512)
else:
    input_hw = resize_hw = [320, 512]
    input_padding = [0, 0, 0, 0]
    vanishing_point = (input_hw[1] // 2, input_hw[0] // 2)
    roi_region = (0, 0, *input_hw[::-1])
    actual_input_hw = (320, 512)

lmdb_data = "lmdb" in model_setting.lower()

if lmdb_data:
    ds_path = os.path.join(
        os.path.dirname(__file__),
        "../datasets/image_fail_seg_lmdb_datasets.py",
    )
    if model_setting == "test_lmdb":
        ds_path = os.path.join(
            os.path.dirname(__file__),
            "../datasets/test_lmdb_datasets.py",
        )
else:
    ds_path = os.path.join(
        os.path.dirname(__file__),
        "../datasets/image_fail_seg_datasets.py",
    )

datapaths = Config.fromfile(ds_path).datapaths

J5_ltc = ["galaxy", "niofy"]
# 编译
if any(ltc in model_setting for ltc in J5_ltc):
    march = "bayes"
else:
    march = "bernoulli2"

# ============================= model params ================================
num_classes = 7
desc_id = "gl_%d" % num_classes
# bn_kwargs = dict(eps=1e-5, momentum=0.1)
bn_kwargs = dict(eps=1e-4, momentum=0.9)

head_out_strides = [2, 8, 16, 32, 64]

inputs = dict(img=torch.zeros((1, 3, *resize_hw)))
task_inputs = dict(
    train=dict(
        labels=[
            torch.zeros(
                (1, 1, actual_input_hw[0] // s, actual_input_hw[1] // s)
            )
            for s in head_out_strides
        ],
    ),
    val=dict(),
    test=dict(),
)

backbone = dict(
    type="ZeroPad2DPatcher",
    backbone=dict(
        type="VargNetV2Stage2631",
        num_classes=1000,
        multiplier=0.5,
        group_base=4,
        last_channels=1024,
        stages=(1, 2, 3, 4, 5),
        include_top=False,
        extend_features=True,
        bn_kwargs=bn_kwargs,
        node_name="backbone",
    ),
    input_padding=input_padding,
)

fpn_neck = dict(
    type="FPN",
    in_strides=[2, 4, 8, 16, 32, 64],
    in_channels=[16, 16, 32, 64, 128, 128],
    out_strides=[4, 8, 16, 32, 64],
    out_channels=[16, 32, 64, 128, 128],
    bn_kwargs=bn_kwargs,
    node_name="fpn_neck",
)

ufpn_seg_neck = dict(
    type="UFPN",
    in_strides=[4, 8, 16, 32, 64],
    in_channels=[16, 32, 64, 128, 128],
    out_channels=[16, 32, 64, 128, 128],
    bn_kwargs=bn_kwargs,
    group_base=4,
    node_name="ufpn_seg_neck",
)


def get_train_model(mode):
    out_strides = (
        head_out_strides if mode == "train" else [min(head_out_strides)]
    )

    return dict(
        type="SegmentorV2",
        backbone=backbone,
        neck=dict(type="ExtSequential", modules=[fpn_neck, ufpn_seg_neck]),
        head=dict(
            type="FRCNNSegHead",
            group_base=4,
            in_strides=[4, 8, 16, 32, 64],
            in_channels=[16, 32, 64, 128, 128],
            out_strides=out_strides,
            out_channels=[num_classes] * len(out_strides),
            bn_kwargs=bn_kwargs,
            argmax_output="train" not in mode,
            with_score="test" in mode and with_rle,
            rle_label="test" in mode and with_rle,
            dequant_output="train" in mode,
            with_extra_conv=False,
            node_name=f"{task_name}_head",
        ),
        loss=dict(
            type="MultiStrideLosses",
            num_classes=num_classes,
            out_strides=head_out_strides,
            loss=dict(
                type="WeightedSquaredHingeLoss",
                reduction="mean",
                weight_low_thr=0.6,
                weight_high_thr=1.0,
                hard_neg_mining_cfg=dict(
                    keep_pos=True,
                    neg_ratio=0.999,
                    hard_ratio=1.0,
                    min_keep_num=255,
                ),
            ),
            loss_weights=[4, 2, 2, 2, 2],
            node_name=f"{task_name}_loss",
        )
        if mode == "train"
        else None,
        desc=dict(
            type="AddDesc",
            per_tensor_desc=get_image_fail_parsing_desc(
                desc_id=desc_id,
                roi_regions=roi_region,
                vanishing_point=vanishing_point,
                input_padding=input_padding,
                with_rle=with_rle,
            ),
            node_name=f"{task_name}_desc",
        )
        if "test" in mode
        else None,
        postprocess=dict(
            type="VargNetSegDecoder",
            out_strides=out_strides,
            input_padding=input_padding,
            node_name=f"{task_name}_decoder",
        )
        if "val" in mode
        else None,
    )


def get_model(mode):
    return dict(
        type="MultitaskGraphModel",
        inputs=inputs,
        task_inputs={task_name: task_inputs[mode]},
        task_modules={task_name: get_train_model(mode)},
        lazy_forward=False,
        flatten_outputs="val" not in mode,
    )


model = get_model("train")
val_model = get_model("val")
test_model = get_model("test")

# ========================== dataset and dataloader ===========================
if not lmdb_data:
    ds = datapaths.image_fail_parsing
    rec_paths = [d["rec_path"] for d in ds["train_data_paths"]]
    anno_paths = [d["anno_path"] for d in ds["train_data_paths"]]
    sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

    data_loader = dict(
        type="MultiCachedDataLoader",
        __build_recursive=False,
        last_batch="pad",
        batch_size=batch_size // 2,
        num_workers=1,
        shuffle=True,
        chunk_size=32,
        min_prefetch=1,
        max_prefetch=2,
        min_chunk_num=4,
        max_chunk_num=8,
        batched_transform=True,
        skip_batchify=False,
        prefetcher_using_thread=True,
        dataset=dict(
            type="MultiFusedIterableDataset",
            dataset=[
                dict(
                    type="SplitDataset",
                    dataset=dict(
                        type="LegacyDenseBoxImageRecordDataset",
                        rec_path=rec_path_i,
                        anno_path=anno_path_i,
                        read_only=True,
                        with_seg_label=False,
                        to_rgb=True,
                    ),
                    even_split=False,
                )
                for rec_path_i, anno_path_i in zip(rec_paths, anno_paths)
            ],
            prob=sample_weights,
            balance=True,
        ),
        transform=[
            dict(
                type="LegacyDenseBoxImageRecordDatasetDecoder",
                to_rgb=True,
                with_seg_label=True,
                seg_label_dtype=np.int8,
            ),
            dict(
                type="DecodeDenseBoxDatasetToSemanticSegFormat",
            ),
            dict(
                type="MapSemanticSegLabels",
                src_values=[255],
                dst_values=[-1],
            ),
            dict(
                type="SemanticSegAffineAugTransformerEx",
                target_wh=resize_hw[::-1],
                norm_wh=None,
                inter_method=10,
                label_scales=[1.0 / stride_i for stride_i in head_out_strides],
                use_pyramid=True,
                pyramid_min_step=0.7,
                pyramid_max_step=0.8,
                flip_prob=0.5,
                label_padding_value=-1,
                rand_translation_ratio=0.0,
                center_aligned=False,
                rand_scale_range=(0.8, 1.3),
                resize_wh=resize_hw[::-1],
                pixel_center_aligned=False,
                adapt_diff_resolution=True,
            ),
            dict(
                type="ReshapeAndCastSemanticSegLabels",
                label_dtype=np.float32,
            ),
            dict(
                type="PackImgAndLabels",
            ),
        ],
    )
else:
    ds = datapaths.image_fail_parsing
    data_paths = [d["data_path"] for d in ds["train_data_paths"]]
    sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

    data_loader = dict(
        type=torch.utils.data.DataLoader,
        drop_last=True,
        num_workers=0,
        batch_size=batch_size // 2,
        dataset=dict(
            type="DistributedComposeRandomDataset",
            sample_weights=sample_weights,
            datasets=[
                dict(
                    type="DetSeg2DAnnoDataset",
                    idx_path=os.path.join(path, "idx"),
                    img_path=os.path.join(path, "img"),
                    anno_path=os.path.join(path, "anno"),
                    transforms=[
                        dict(
                            type="SemanticSegAffineAugTransformerEx",
                            target_wh=resize_hw[::-1],
                            norm_wh=None,
                            inter_method=10,
                            label_scales=[
                                1.0 / stride_i for stride_i in head_out_strides
                            ],
                            use_pyramid=True,
                            pyramid_min_step=0.7,
                            pyramid_max_step=0.8,
                            flip_prob=0.5,
                            label_padding_value=-1,
                            rand_translation_ratio=0.0,
                            center_aligned=False,
                            rand_scale_range=(0.8, 1.3),
                            resize_wh=resize_hw[::-1],
                            pixel_center_aligned=False,
                            adapt_diff_resolution=True,
                            padding_in_network_lrub=input_padding,
                        ),
                    ],
                )
                for path in data_paths
            ],
        ),
    )


loaders = {task_name: data_loader}
task_sampler_configs = {task_name: dict(sampling_factor=1)}

task_sampler = TaskSampler(
    task_config=task_sampler_configs,
    method="sample_all",
)
train_data_loader = dict(
    type="MultitaskInfLoader",
    loaders=loaders,
    task_sampler=task_sampler,
    return_task=True,
)
# -------------------------- trainer configs --------------------------
if pipeline_test:
    num_steps = dict(
        float=5,
        freeze_bn_1=5,
        freeze_bn_2=5,
        qat=5,
        int_infer=0,
    )
    warmup_steps = 0
    save_interval = 5
else:
    num_steps = dict(
        float=50000,
        freeze_bn_1=10000,
        freeze_bn_2=10000,
        qat=10000,
        int_infer=0,
    )
    warmup_steps = 1000
    save_interval = 2000

base_lr = dict(
    float=0.0015,
    freeze_bn_1=0.00005,
    freeze_bn_2=0.00001,
    qat=0.00001,
    int_infer=0.0,
)

interval_by = "step"
num_steps = num_steps[training_step]
base_lr = base_lr[training_step]

# freeze bn config
f1 = [
    "backbone",  # backbone
    "fpn_neck",  # fpn
]
f2 = ["ufpn_seg_neck"]

freeze_bn_modules = dict(
    float=None,
    freeze_bn_1=f1,
    freeze_bn_2=f1 + f2,
    qat=None,
    int_infer=None,
)

train_stages = [
    "float",
    "freeze_bn_1",
    "freeze_bn_2",
    "qat",
]

# -------------------------- callbacks --------------------------
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_regex("^.*loss.*"),
    enable_amp=enable_amp,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)

lr_callback = dict(
    type="PolyLrUpdater",
    max_update=num_steps // num_machines,
    power=1.0,
    warmup_len=warmup_steps,
    step_log_interval=25,
)

checkpoint_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir.as_posix(),
    name_prefix=training_step + "-",
    strict_match=False,
    best_refer_metric=None,
    save_interval=save_interval,
    interval_by=interval_by,
    save_on_train_end=True,
)

tensorboard_callback = dict(
    type="TensorBoard",
    save_dir=log_dir.as_posix() if local_train else "/job_tboard/",  # noqa
    update_freq=log_freq,
    tb_update_funcs=[],
)

freeze_bn_callback = dict(
    type="FreezeModule",
    modules=[freeze_bn_modules[training_step]],
    step_or_epoch=[0],
    update_by=interval_by,
    only_batchnorm=True,
)


metric_updaters = [
    dict(
        type="MetricUpdater",
        metrics=[
            dict(type="LossShow", name=f"stride_{s}_loss")
            for s in head_out_strides
        ],
        metric_update_func=update_metric_using_regex(
            per_metric_patterns=[  # corresponding to metrics
                dict(
                    label_pattern=None,
                    pred_pattern=f"^.*{task_name}_stride_{s}_loss$",
                )
                for s in head_out_strides
            ]
        ),
        step_log_freq=log_freq,
        epoch_log_freq=1,
        log_prefix=task_name,
        reset_metrics_by="log",
    )
]

aidi_expmodel_callback = dict(
    type="AIDIExperimentManager",
    model_name=model_name,
    save_model="last",
    model_version=model_version,
    upload_progressive_checkpoint=True,
)


# the order of callbacks affects the logging order
callbacks = [
    stat_callback,
    lr_callback,
    checkpoint_callback,
    tensorboard_callback,
]

callbacks += metric_updaters

if training_step not in ("float", "qat", "int_infer"):
    callbacks.append(freeze_bn_callback)


if enable_model_tracking:
    callbacks.append(aidi_expmodel_callback)

profiler = dict(
    type="SimpleProfiler",
    # dirpath=log_dir.as_posix(),
    # filename='profile.log',
)

# ============================= solver ==============================
base_trainer = dict(
    type="DistributedDataParallelTrainer",
    model=model,
    data_loader=train_data_loader,
    optimizer=dict(
        type="LegacyNadamEx",
        params={"weight": dict(weight_decay=5e-5)},
        lr=base_lr * num_machines,
        rescale_grad=1,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=(num_steps + warmup_steps) // num_machines,
    device=None,  # set when building
    sync_bn=sync_bn,
    callbacks=callbacks,
    assign_module_buffers=False,
    # profiler=profiler,
)

# -------------------------- trainer --------------------------
pretrain_checkpoint = "http://fm-tian-li.alitrain.hogpu.cc/plat_gpu/multitask-pretrain-2631-20211109-130832/output/models/vargnetv2_cls/float-checkpoint-best.pth.tar"  # noqa
stage_extend_params = dict(
    float=dict(
        checkpoint_path=pretrain_checkpoint,
        allow_miss=True,
        ignore_extra=True,
        # verbose=0, # 1
        # checkpoint_mode="pre_stage", # "resume"
        # resume_epoch_or_step=False,
        # resume_optimizer=False,
    ),
    freeze_bn_1=dict(
        checkpoint_path=os.path.join(
            ckpt_dir, "float-checkpoint-last.pth.tar"
        ),
    ),
    freeze_bn_2=dict(
        checkpoint_path=os.path.join(
            ckpt_dir, "freeze_bn_1-checkpoint-last.pth.tar"
        ),
    ),
    qat=dict(
        checkpoint_path=os.path.join(
            ckpt_dir, "freeze_bn_2-checkpoint-last.pth.tar"
        ),
    ),
)

for stage in train_stages:
    params = stage_extend_params.get(stage, {})
    checkpoint_mode = (
        "resume"
        if resume_training
        else params.get("checkpoint_mode", "pre_stage")
    )

    if model_checkpoint:
        checkpoint_path = model_checkpoint
        track_input_model = True
    else:
        checkpoint_path = params["checkpoint_path"]
        track_input_model = False

    track_input_model = track_input_model and enable_model_tracking

    trainer = deepcopy(base_trainer)
    trainer.update(
        dict(
            model_convert_pipeline=dict(
                type="ModelConvertPipeline",
                converters=[
                    dict(
                        type="LoadCheckpoint",
                        checkpoint_path=checkpoint_path,
                        state_dict_update_func=params.get(
                            "state_dict_update_func", None
                        ),
                        enable_tracking=track_input_model,
                        allow_miss=params.get("allow_miss", False),
                        ignore_extra=params.get("ignore_extra", False),
                        verbose=params.get("verbose", 1),
                    )
                ],
            )
        )
    )
    if checkpoint_mode == "resume":
        trainer.update(
            resume_optimizer=params.get("resume_optimizer", True),
            resume_epoch_or_step=params.get("resume_epoch_or_step", True),
        )

    if stage == "qat":
        trainer["model_convert_pipeline"]["converters"].append(
            dict(type="Float2QAT")
        )

    globals()[f"{stage}_trainer"] = trainer

# -------------------------- stage int_infer --------------------------

int_checkpoint = dict(
    type="SaveTraced",
    save_dir=ckpt_dir.as_posix(),
    trace_inputs=inputs,
    name_prefix=training_step + "-",
)

int_infer_trainer = dict(
    type="Trainer",
    model=test_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=model_checkpoint
                if model_checkpoint
                else os.path.join(
                    ckpt_dir,
                    "qat-checkpoint-last.pth.tar",
                ),
                allow_miss=False,
                ignore_extra=True,
                verbose=1,
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    num_epochs=0,
    device=None,
    optimizer=None,
    data_loader=None,
    batch_processor=None,
    callbacks=[int_checkpoint],
)
