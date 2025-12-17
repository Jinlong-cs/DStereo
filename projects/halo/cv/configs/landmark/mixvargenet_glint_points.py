import copy
import os
import warnings

import torch
from horizon_plugin_pytorch.quantization import March

from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.utils.config import ConfigVersion

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
root = os.path.join(bucket_root, "interaction")

if not os.path.isdir(root):
    raise FileNotFoundError("interaction bucket")

VERSION = ConfigVersion.v2
enable_model_tracking = True
enable_amp = False
warnings.filterwarnings("ignore")
NUM_LDMK = 2
LDMK_PAIRS = [[0, 1]]
bn_kwargs = dict(
    eps=2e-5, momentum=0.1
)  # because the different implement from pytorch and mxnet, momentum in gluon face is 0.9. # noqa


training_step = os.environ.get("HAT_TRAINING_STEP", "float")
task_name = "exp_01"

log_freq = 50

if is_local_train:
    batch_size_per_gpu = 8
    device_ids = [0, 2]
    # ckpt_dir = "./tmp_models/%s" % task_name
    ckpt_dir = (
        "/home/users/jiaqi.quan/project/gaze/glint_landmark/log/exp_sigma2/"
    )
    # if not os.path.isdir("tmp_models"):
    #     os.makedirs("./tmp_models")
else:
    batch_size_per_gpu = 8
    device_ids = [0, 1]
    ckpt_dir = "/job_data/models/%s" % task_name


num_epoch = 60
qat_epoch = 50
LR = 1e-3
QAT_LR = 1e-4
STRIDE = 1
INPUT_H = 64
INPUT_W = 64
TARGET_H = INPUT_H // STRIDE
TARGET_W = INPUT_W // STRIDE
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES
qat_mode = "fuse_bn"

test_inputs = dict(img=torch.randn((1, 3, INPUT_H, INPUT_W)))


# src mixvargnet 0.75
src_mixvargenet_075_config = [
    [
        MixVarGENetConfig(
            in_channels=24,
            out_channels=24,
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=24,
            out_channels=24,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=24,
            out_channels=48,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=48,
            out_channels=64,
            head_op="mixvarge_f2_gb16",
            stack_ops=[
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=128,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 32
]

backbone = dict(
    type="MixVarGENet",
    net_config=src_mixvargenet_075_config,
    input_channels=3,
    num_classes=1000,
    bn_kwargs=bn_kwargs,
    include_top=False,
    bias=True,
)

# Train Model
model = dict(
    type="LdmkModel",
    # backbone=dict(
    #     type="ResNet18",
    #     num_classes=1,
    #     quant_input=False,
    #     bn_kwargs=bn_kwargs,
    #     include_top=False,
    # ),
    backbone=backbone,
    mode="train",
    decoder=dict(
        type="LdmkDecoder",
        in_stride=32,
        out_stride=STRIDE,
        in_channels=128,
        out_channels=256,
        bn_kwargs=bn_kwargs,
    ),
    vector_head=dict(
        type="LdmkVectorHead",
        in_channels=256,
        num_ldmk=NUM_LDMK,
        band_width=1,
        vector_size=(TARGET_H, TARGET_W),
        band_module_type="pool",
        loss_func=dict(type="LdmkLoss", loss_type="l2"),
        bn_kwargs=bn_kwargs,
    ),
    coords_head=None,
    feat_stride=STRIDE,
    heatmap_head=None,
    cls_head=None,
    loss_weights={"vector": 1.0},
)

val_model = copy.deepcopy(model)
val_model["mode"] = "val"
# Val Model
deploy_model = copy.deepcopy(val_model)
deploy_model["mode"] = "deploy"

deploy_inputs = dict(img=torch.randn((1, 3, INPUT_H, INPUT_W)))

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT"),
        dict(type="QAT2Quantize"),
    ],
)


# transform
train_transforms = [
    dict(
        type="RandomRotateCrop",
        net_input_size=(INPUT_W, INPUT_H),
        rot_prob=0.0,
        rot_angle_range=20,
        center_shift_prob=0.0,
        center_shift_range=0.01,
        norm_ratio=1.0,
        norm_jitter_range=0.0,
        return_normalized_ldmk=False,
    ),
    # dict(
    #     type="RandomFlip",
    #     px=0.0,
    #     flip_ldmk_wo_1=True,
    # ),
    # dict(
    #     type="GaussianNoise",
    #     prob=0.2,
    #     mean=0,
    #     sigma=2,
    # ),
    # dict(
    #     type="RandomNoise",
    #     prob=0.2,
    #     min=-5,
    #     max=5,
    # ),
    # dict(
    #     type="SaltPepperNoise",
    #     prob=0.2,
    #     s_ratio=0.05,
    #     p_ratio=0.05,
    # ),
    # dict(
    #     type="GaussianBlur",
    #     p=0.2,
    #     kernel_size_min=3,
    #     kernel_size_max=9,
    #     sigma_min=1.0,
    #     sigma_max=5.0,
    # ),
    # dict(
    #     type="MotionBlur",
    #     p=0.2,
    #     length_min=2,
    #     length_max=10,
    #     angle_min=1,
    #     angle_max=359,
    # ),
    # dict(
    #     type="RandomGray",
    #     p=0.3,
    # ),
    # dict(
    #     type="RandomOcclusion",
    #     prob=0.5,
    #     occ_type="whole",
    #     occ_num=1,
    #     size_ratio=0.7,
    # ),
    dict(
        type="GenerateGaussianVector",
        num_ldmk=NUM_LDMK,
        feat_stride=STRIDE,
        vector_size=(TARGET_H, TARGET_W),
        sigma=2,
        encoding_method="unbiased",
    ),
    dict(type="ToTensor"),
    # dict(type="Normalize", mean=128.0, std=128.0),
]
val_transforms = [
    dict(
        type="RandomRotateCrop",
        net_input_size=(INPUT_W, INPUT_H),
        rot_prob=0.0,
        rot_angle_range=20,
        center_shift_prob=0.0,
        center_shift_range=0.01,
        norm_ratio=1.0,
        norm_jitter_range=0.0,
        return_normalized_ldmk=False,
    ),
    dict(type="ToTensor"),
    # dict(type="Normalize", mean=128.0, std=128.0),
]


# datasets
# train_data = ["glint_points_baseline_b"]
# val_data = ["glint_points_baseline_a"]
train_data = [
    "/home/users/jiaqi.quan/project/gaze/glint_landmark/data/lmdb/20231106/train/"
]
val_data = [
    "/home/users/jiaqi.quan/project/gaze/glint_landmark/data/lmdb/20231106/val/"
]
test_data = [
    "/home/users/jiaqi.quan/project/gaze/glint_landmark/data/lmdb/20231106/test/"
]


# dataloader
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="LdmkDataset",
        image_lmdb_list=[os.path.join(t, "image_lmdb") for t in train_data],
        anno_lmdb_list=[os.path.join(t, "anno_lmdb") for t in train_data],
        num_ldmk=NUM_LDMK,
        ldmk_pairs=LDMK_PAIRS,
        transforms=train_transforms,
        task_type="glint",
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
)


val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="LdmkDataset",
        image_lmdb_list=[os.path.join(t, "image_lmdb") for t in val_data],
        anno_lmdb_list=[os.path.join(t, "anno_lmdb") for t in val_data],
        num_ldmk=NUM_LDMK,
        ldmk_pairs=LDMK_PAIRS,
        transforms=val_transforms,
        task_type="glint",
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
)

test_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="LdmkDataset",
        image_lmdb_list=[os.path.join(t, "image_lmdb") for t in test_data],
        anno_lmdb_list=[os.path.join(t, "anno_lmdb") for t in test_data],
        num_ldmk=NUM_LDMK,
        ldmk_pairs=LDMK_PAIRS,
        transforms=val_transforms,
        task_type="glint",
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
)

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        # dict(
        #     type="Lighting",
        #     prob=1.0,
        #     alphastd=0.2,
        # ),
        # dict(
        #     type="TorchVisionAdapter",
        #     interface="ColorJitter",
        #     brightness=0.2,
        #     contrast=0.0,
        #     saturation=0.0,
        #     hue=0,
        # ),
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_regex("total_loss"),
    enable_amp=enable_amp,
)
val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
)


def update_metric(metrics, batch, model_outs):
    for metric, key in zip(metrics, model_outs):
        metric.update(model_outs[key])


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
)


def update_val_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


val_metrics = [
    dict(
        type="NormalizedMeanError",
        num_ldmk=NUM_LDMK,
        name="EPE",
        norm_type=None,
        mode="vector",
        decoding_method="null",
        feat_stride=STRIDE,
    )
]


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_val_metric,
    step_log_freq=-1,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)


stat_callback = dict(
    type="StatsMonitor",
    log_freq=50,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    interval_by="epoch",
    strict_match=False,
)

# val_callback = dict(
#     type="Validation",
#     data_loader=val_data_loader,
#     batch_processor=val_batch_processor,
#     callbacks=val_metric_updater_list,
#     # callbacks=[val_metric_updater],
#     interval_by="epoch",
#     val_interval=1,
#     val_model=val_model,
#     val_on_train_end=True,
#     log_interval=10,
# )

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        converters=[
            dict(
                type="LoadCheckpoint",
                # checkpoint_path=os.path.join(
                #     "/horizon-bucket/HDLTAlgorithm/models/bayes_release_models/resnet18_cls/float-checkpoint-best.pth.tar"
                # ),
                # checkpoint_path="/horizon-bucket/HDLTAlgorithm/models/bayes_release_models/mobilenetv2_cls/float-checkpoint-best.pth.tar",
                checkpoint_path="/home/users/jiaqi.quan/project/gaze/glint_landmark/log/exp_pool/float-checkpoint-last.pth.tar",
                # checkpoint_path=None,
                allow_miss=False,
                ignore_extra=False,
                verbose=True,  # Show unexpect_key and miss_key info.
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=1e-5)},
        lr=LR,
    ),
    batch_processor=batch_processor,
    num_epochs=num_epoch,
    stop_by="epoch",
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            warmup_by="epoch",
            warmup_len=0,
            step_log_interval=log_freq,
            # lr_decay_id=[120, 180],
            lr_decay_id=[20, 40],
            lr_decay_factor=0.1,
        ),
        # dict(
        #     type="PolyLrUpdater",
        #     max_update=100000,
        #     step_log_interval=log_freq,
        #     power=2,
        #     final_lr=1e-5,
        #     warmup_by="step",
        #     warmup_len=1000,
        #     warmup_begin_lr=0,
        #     warmup_mode="linear",
        # ),
        metric_updater,
        # val_callback,
        ckpt_callback,
        # trace_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="LdmkLoss"),
        dict(type="LossShow", name="Loss"),
    ],
)


qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,  # Show unexpect_key and miss_key info.
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=1e-5)},
        lr=QAT_LR,
    ),
    batch_processor=batch_processor,
    num_epochs=qat_epoch,
    stop_by="epoch",
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            warmup_by="epoch",
            warmup_len=0,
            step_log_interval=log_freq,
            lr_decay_id=[30],
            lr_decay_factor=0.1,
        ),
        # dict(
        #     type="PolyLrUpdater",
        #     max_update=100000,
        #     step_log_interval=log_freq,
        #     power=2,
        #     final_lr=1e-5,
        #     warmup_by="step",
        #     warmup_len=1000,
        #     warmup_begin_lr=0,
        #     warmup_mode="linear",
        # ),
        metric_updater,
        # val_callback,
        ckpt_callback,
        # trace_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="LdmkLoss"),
        dict(type="LossShow", name="Loss"),
    ],
)


# predictor
float_predictor = dict(
    type="Predictor",
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                allow_miss=False,
                ignore_extra=True,
                verbose=True,  # Show unexpect_key and miss_key info.
            ),
        ],
    ),
    data_loader=[test_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=[val_metric_updater],
    metrics=val_metrics,
    log_interval=0,
    share_callbacks=False,
)

qat_predictor = dict(
    type="Predictor",
    model=val_model,
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
                allow_miss=False,
                ignore_extra=True,
                verbose=True,  # Show unexpect_key and miss_key info.
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=[val_metric_updater],
    metrics=val_metrics,
    log_interval=0,
    share_callbacks=False,
)


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
                ignore_extra=True,
                verbose=True,
                # check_hash=False,
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


onnx_cfg = dict(
    model=deploy_model,
    stage="float",
    inputs=deploy_inputs,
    model_convert_pipeline=float_predictor["model_convert_pipeline"],
)
