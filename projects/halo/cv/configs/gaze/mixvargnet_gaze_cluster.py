import copy
import os

import torch
import torchvision
from horizon_plugin_pytorch.march import March

from hat.data.datasets.gaze import GazeDataset
from hat.data.transforms.detection import (
    Normalize,
    RandomFlip,
    Resize,
    ToTensor,
)
from hat.data.transforms.gaze import Clip, GazeYUVTransform, RandomColorJitter
from hat.data.transforms.gaze.gaze import GazeRotate3DWithCrop
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

DEBUG = False
CHIP = "J5"

task_name = "gaze_estimation_one_head_online_augm_20230331_01_continue"
num_classes = 1000
batch_size_per_gpu = 128 if not DEBUG else 32
if DEBUG:
    device_ids = [0]
else:
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES if CHIP == "J5" else March.BERNOULLI2

input_h = 192
input_w = 320
# two_head
use_glass = False  # use_glass=True Not Implemented

bucket_root = (
    "/bucket/input" if os.path.exists("/bucket") else "/horizon-bucket"
)
rec_root = (
    f"{bucket_root}/MultiMode_2/mm_algorithms_data/gaze/train_wt_online_rot"
)
sub_rec_roots = (
    [
        "recs_20220620",
        "recs_20220707_wldmk_wogaze",
        "recs_20220707_wldmk_wogaze",
        "recs_20220707_wldmk_wogaze",
        "recs_20220811_add_missing_data",
    ]
    if not DEBUG
    else ["recs_20220811_add_missing_data"]
)
sub_rec_roots = [os.path.join(rec_root, _, "training") for _ in sub_rec_roots]
rec_list = []
for sub_rec_root in sub_rec_roots:
    rec_list += [
        os.path.join(sub_rec_root, _)
        for _ in os.listdir(sub_rec_root)
        if _.endswith(".rec")
    ]
repeat = [
    _
    for _ in rec_list
    if "_da05" in _
    and "_left_" not in _
    and "_mid_" not in _
    and "_right_" not in _
]
if not DEBUG:
    rec_list += repeat * 2


dataset = dict(
    type=GazeDataset,
    rec_list=rec_list,
    input_size=(input_w, input_h),
    rotate_3d_augm=True,
    transforms=torchvision.transforms.Compose(
        [
            RandomFlip(0.5),
            RandomColorJitter(
                brightness=0.05,
                contrast=0.05,
                saturation=0.05,
                hue=0.0,
                prob=0.5,
            ),
            Clip(),
            GazeRotate3DWithCrop(
                is_train=True,
                rotate_augm_prob=0.6,
            ),
            GazeYUVTransform(rgb_data=False, nc=3),
            Resize(img_scale=(192, 320), keep_ratio=False),
            ToTensor(),
            Normalize(mean=128.0, std=128.0),
        ]
    ),
)
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=4 if DEBUG else 16,
    pin_memory=False,
)

rec_list_val = [
    f"{bucket_root}/MultiMode_2/mm_algorithms_data/gaze/test_wt_online_rot/testing/da04_clean_20221220/da04_clean_20221220.rec",
    # f"{bucket_root}/MultiMode_2/mm_algorithms_data/gaze/test_wt_online_rot/testing/cd569_clean_20221221/cd569_clean_20221221.rec",
    # f"{bucket_root}/MultiMode_2/mm_algorithms_data/gaze/test_wt_online_rot/testing/t18_clean_20221221/t18_clean_20221221.rec",
]
dataset_val = dict(
    type=GazeDataset,
    rec_list=rec_list_val,
    input_size=(input_w, input_h),
    rotate_3d_augm=True,
    transforms=torchvision.transforms.Compose(
        [
            GazeRotate3DWithCrop(
                is_train=False,
            ),
            GazeYUVTransform(rgb_data=False, nc=3),
            Resize(img_scale=(192, 320), keep_ratio=False),
            ToTensor(),
            Normalize(mean=128.0, std=128.0),
        ]
    ),
)
data_loader_val = dict(
    type=torch.utils.data.DataLoader,
    dataset=dataset_val,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4 if DEBUG else 16,
    pin_memory=False,
)

# network args
in_channels = [32, 32, 32, 64, 128]
out_channels = [32, 32, 64, 128, 256]
alpha = 0.75
in_channels = [int(x * alpha) for x in in_channels]
out_channels = [int(x * alpha) for x in out_channels]
shared_feat_size = max(int(alpha * 1024), 1024)
dropout_ratio = 0.5
bn_kwargs = {
    "eps": 2e-05,
    "momentum": 0.1,
}

# mixvargnet config
net_config = [
    [
        MixVarGENetConfig(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=in_channels[3],
            out_channels=out_channels[3],
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
        ),  # noqa
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=in_channels[4],
            out_channels=out_channels[4],
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 32
]

model = dict(
    type="GazeModel",
    backbone=dict(
        type="MixVarGENet",
        net_config=net_config,
        num_classes=1000,
        include_top=False,
        bn_kwargs=bn_kwargs,
        bias=True,
    ),
    head=dict(
        type="GazeEyeldmkHead",
        input_size=[input_w, input_h],
        last_channel_out=out_channels[4],
        shared_feat_size=shared_feat_size,
        gaze_head_params=dict(
            use_pool=True,
            dropout_ratio=dropout_ratio,
            channels=(shared_feat_size, 128, 128, 4),
            output_add_bias=False,
            bn_kwargs=bn_kwargs,
        ),
        eyeldmk_head_params=dict(
            alpha=0.1875,
            bn_kwargs=bn_kwargs,
            channels=[128, 128, 16],
            J5_efficient=True,
            use_groupstyle=True,
        ),
    ),
    losses=dict(
        type="GazeEyeldmkLoss",
        eye_ldmk_num=21,
        gaze_loss_name="wing",
        gaze_loss_weight=1,
        gaze_loss_params=dict(
            pitch_w=15,
            yaw_w=15,
            pitch_e=0.5,
            yaw_e=0.5,
            averaged_output=False,
            batch_axis=0,
        ),
        eye_ldmk_loss_name="wing",
        eye_ldmk_loss_weight=5,
        eye_ldmk_loss_params=dict(
            w=15,
            epsilon=0.1,
            averaged_output=False,
            batch_axis=0,
        ),
    ),
)

val_model = dict(
    type="GazeModel",
    backbone=dict(
        type="MixVarGENet",
        net_config=net_config,
        num_classes=1000,
        include_top=False,
        bn_kwargs=bn_kwargs,
        bias=True,
    ),
    head=dict(
        type="GazeEyeldmkHead",
        input_size=[input_w, input_h],
        last_channel_out=out_channels[4],
        shared_feat_size=shared_feat_size,
        gaze_head_params=dict(
            use_pool=True,
            dropout_ratio=dropout_ratio,
            channels=(shared_feat_size, 128, 128, 4),
            output_add_bias=False,
            bn_kwargs=bn_kwargs,
        ),
        eyeldmk_head_params=dict(
            alpha=0.1875,
            bn_kwargs=bn_kwargs,
            channels=[128, 128, 16],
            J5_efficient=True,
            use_groupstyle=True,
        ),
    ),
    losses=None,
)

# Model for deploy, save checkpoint.
deploy_model = copy.deepcopy(val_model)
deploy_model["deploy"] = True
test_inputs = dict(img=torch.randn((1, 3, 192, 320)))
deploy_inputs = test_inputs

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=None,
    loss_collector=collect_loss_by_index(1),
)

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=None,
    loss_collector=collect_loss_by_index(1),  # not sure
)


def update_metric(metrics, batch, model_outs):
    target_gaze = batch["gaze_label"]
    target_eye_ldmk = batch["gaze_label"]["gt_normed_eye_ldmk"]
    preds, losses = model_outs
    for metric in metrics:
        if metric._get_name() == "LossShow":
            metric.update(losses)
        elif metric._get_name() == "AngleDifferenceMetric":
            metric.update(target_gaze, preds)
        elif metric._get_name() == "EyeLdmksDist":
            metric.update(target_eye_ldmk, preds)
        elif metric._get_name() == "GazeModelTestMetric":
            metric.update(target_gaze, preds)
        else:
            raise ValueError


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=100 if DEBUG else 1000,
    epoch_log_freq=1,
    log_prefix=task_name,
)

val_metric_updater = copy.deepcopy(metric_updater)
val_metric_updater["log_prefix"] = "Validation " + task_name

stat_callback = dict(
    type="StatsMonitor",
    log_freq=20,  # not sure
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode="max",
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=test_inputs,
)


val_callback = dict(
    type="Validation",
    data_loader=data_loader_val,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=0.0)},
        lr=5e-3,
    ),
    batch_processor=batch_processor,
    num_epochs=250 if not DEBUG else 5,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=1000,
            lr_decay_id=[150, 170, 190, 210, 230],
            lr_decay_factor=0.5,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
        dict(
            type="AngleDifferenceMetric", name="left_eye", use_glass=use_glass
        ),
        dict(
            type="AngleDifferenceMetric", name="right_eye", use_glass=use_glass
        ),
        dict(type="EyeLdmksDist", name="eye_ldmk", size=(input_w, input_h)),
    ],
    val_metrics=[
        dict(
            type="AngleDifferenceMetric", name="left_eye", use_glass=use_glass
        ),
        dict(
            type="AngleDifferenceMetric", name="right_eye", use_glass=use_glass
        ),
        dict(type="EyeLdmksDist", name="eye_ldmk", size=(input_w, input_h)),
    ],
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
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
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=0)},
        lr=1e-4,
    ),
    batch_processor=batch_processor,
    num_epochs=30 if not DEBUG else 3,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=1000,
            lr_decay_id=[15, 25],
            lr_decay_factor=0.1,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
        dict(type="AngleDifferenceMetric", name="left_eye"),
        dict(type="AngleDifferenceMetric", name="right_eye"),
        dict(type="EyeLdmksDist", name="eye_ldmk", size=(input_w, input_h)),
    ],
    val_metrics=[
        dict(type="AngleDifferenceMetric", name="left_eye"),
        dict(type="AngleDifferenceMetric", name="right_eye"),
        dict(type="EyeLdmksDist", name="eye_ldmk", size=(input_w, input_h)),
    ],
)

# just for saving int_infer pth and pt
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
    callbacks=[
        ckpt_callback,
        trace_callback,
    ],
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
)

# predictor
float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[data_loader_val],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="AngleDifferenceMetric", name="left_eye"),
        dict(type="AngleDifferenceMetric", name="right_eye"),
        dict(
            type="GazeModelTestMetric",
            name="gaze diff",
            save_txt=f"tmp_models/{task_name}/float_test.txt",
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
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
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[data_loader_val],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="AngleDifferenceMetric", name="left_eye"),
        dict(type="AngleDifferenceMetric", name="right_eye"),
        dict(
            type="GazeModelTestMetric",
            name="gaze diff",
            save_txt=f"tmp_models/{task_name}/float_test.txt",
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
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
                ),
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=[data_loader_val],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="AngleDifferenceMetric", name="left_eye"),
        dict(type="AngleDifferenceMetric", name="right_eye"),
        dict(
            type="GazeModelTestMetric",
            name="gaze diff",
            save_txt=f"tmp_models/{task_name}/float_test.txt",
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

onnx_cfg = dict(
    model=copy.deepcopy(deploy_model),
    stage="float",
    inputs=test_inputs,
    model_convert_pipeline=dict(
        type="LoadCheckpoint",
        checkpoint_path=os.path.join(
            ckpt_dir, "float-checkpoint-last.pth.tar"
        ),
    ),
)

k8s_config = dict(
    job_name=task_name,
    job_password="TD20220003",
    num_machines=1,
    num_gpus_per_machine=8,
    framework="pytorch",
    task_label="halo_multitask",
    project_id="TD20220003",
    input_bucket="MultiMode_2",
    priority=5,
    docker_image="docker.hobot.cc/imagesys/hat:halo-runtime-py38-cu111-torch1-102",  # with ablu and trackeval  # noqa
    # default 7200 = 5days
    max_jobtime=16000,
    # launcher only for multi-machines
    launcher="mpi",
    # upload folder
    upload_folder_name="k8s_job",
    folder_list=[
        "../../hat",
        "../../plugins",
        "../../tools",
        "../../projects",
        "url2IP.py",
        "ssh_launcher.py",
    ],
    job_list=[
        "pip3 install albumentations pytorch-crf==0.7.2 -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc",
        f"hdfs dfs -mkdir hdfs://hobot-bigdata/user/shiyao.tang/output/hat/gaze_eyeldmk/{task_name}",
        "python3 tools/train.py --config projects/halo/cv/configs/gaze/mixvargnet_gaze_cluster.py -s float",
        "python3 tools/train.py --config projects/halo/cv/configs/gaze/mixvargnet_gaze_cluster.py -s qat",
        f"hdfs dfs -put -f tmp_models/* hdfs://hobot-bigdata/user/shiyao.tang/output/hat/gaze_eyeldmk/{task_name}",
    ],
)
