import os
from pathlib import Path

import torch
from hatbc.filestream.bucket.client import get_bucket_mount_root
from horizon_plugin_pytorch.quantization import March

from hat.models.backbones.mixvargenet import (
    MixVarGENetConfig,
    get_mixvargenet_stride2channels,
)
from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels

# ---------------------------------------------------------

torch.set_printoptions(precision=6)
# mount bucket to local manually: dmp mount exec mono ./mono
buckets = [
    "mono",
]
bucket2mount_root = get_bucket_mount_root()
tasks = [
    dict(name="toll_lmks_det"),
]
# use mount_root to construct rec, json path
mount_root = bucket2mount_root.get(buckets[0], None)
if mount_root is None:
    raise FileNotFoundError("mono bucket")

num_machines = int(os.getenv("HAT_NUM_MACHINES", "1"))
pipeline_test = os.environ.get("HAT_PIPELINE_TEST", "0") == "1"
enable_model_tracking = os.getenv("HAT_ENABLE_MODEL_TRACKING") == "1"
model_name_postfix = os.getenv("HAT_TL_MODEL_NAME_POSTFIX", "ziyang02.wang")
model_version = os.getenv("HAT_TL_MODEL_VERSION", "v0.0.6")
training_step = os.getenv("HAT_TRAINING_STEP", "float")
assert training_step in ["float", "float_freeze_bn", "qat", "int_infer"]

task_type = "sd"
assert task_type in ["mono", "sd", "others"]

if task_type == "mono":
    input_size = os.getenv("INPUT_SIZE", (256, 480))
    crop_h, img_w, img_h = (*input_size, int(input_size[0] * 270 / 256))
    sigma = 2
elif task_type == "sd":  # SD model
    input_size = os.getenv("INPUT_SIZE", (512, 960))
    crop_h, img_w, img_h = (*input_size, int(input_size[0] * 540 / 512))
    sigma = 4

local_or_remote_debug = False
compile_model = True
multitask_task_name = f"{task_type}_toll_gate"
march = March.BERNOULLI2 if task_type == "mono" else March.BAYES
use_calibration_step = march
is_local_train = not os.path.exists("/running_package")
save_prefix = "tmp_output" if is_local_train else "/job_data/models/"
ckpt_dir = Path(save_prefix) / multitask_task_name
# ------------------------------------------------------
# train
# ------------------------------------------------------
num_gpus_per_machine = 4
device_ids = [0] if is_local_train else list(range(num_gpus_per_machine))
log_freq = 15 if is_local_train else 25
float_steps = 100000 if not pipeline_test else 20
qat_steps = 10000 if not pipeline_test else 20
warmup_steps = 500 if not pipeline_test else 10
save_interval = 2000 if not pipeline_test else 100
float_lr = 0.001
qat_lr = 0.0005

interval_by = "step"
pretrain_checkpoint = "http://fm-hao-chen.ucloudtrain.hogpu.cc/plat_gpu/TinyVarGNetV2_CLS-HAT-pretrain-20220223-113926/output/models/TinyVarGNetV2_CLS/float-checkpoint-best-9b796482.pth.tar"  # noqa
# pretrain_checkpoint = "http://fm-jing-li.tcloud.hogpu.cc/plat_gpu/hobot-dag-4093655_hat-toll-gate-mixvargnet-20230908-171527/output/models/mono_vw_toll_gate/float-checkpoint-last.pth.tar"  # noqa
pretrain_checkpoint = "http://fm-jing-li.bcloud-2nd.hogpu.cc/plat_gpu/hobot-dag-4189908_hat-toll-gate-tinyvargnet-weight-noflip-pretrain-iterfalf-20230916-103404/output/models/mono_vw_toll_gate/float-checkpoint-last.pth.tar"  # noqa
float_checkpoint = None
float_resume_checkpoint = None
qat_resume_checkpoint = None
# common structures
# -------------------------------- Backbone configuration ----------------------------------  # noqa
bn_kwargs = {"eps": 1e-05, "momentum": 0.1}
# backbone
alpha = 0.5
input_channels = 3
input_sequence_length = 1
factor = 2
head_factor = 2
group_base = 8
include_top = False
extend_features = True
use_mixvargnet = False
# -------------------------------- Neck configuration ----------------------------------  # noqa
unet_in_strides = [2, 4, 8, 16, 32, 64]
unet_out_strides = [4]
stride2channels = get_vargnetv2_stride2channels(alpha)
unet_factor = 2
# -------------------------------- Backbone ----------------------------------   # noqa
backbone = dict(
    type="TinyVargNetV2",
    num_classes=1000,
    bn_kwargs=bn_kwargs,
    alpha=0.5,
    group_base=8,
    include_top=include_top,
    extend_features=extend_features,
    input_resize_scale=None,
    channel_list=[32, 32, 64, 128, 256, 512],
    node_name="backbone",
)


# LargeMixVarGENet
input_resize_scale = None


def get_mixvargenet_config(
    net_config, output_list, input_channels=3, disable_quanti_input=False
):
    mixvargenet_backbone = dict(
        type="MixVarGENet",
        net_config=net_config,
        output_list=output_list,
        input_channels=input_channels,
        input_sequence_length=input_sequence_length,
        input_resize_scale=input_resize_scale,
        num_classes=1000,
        include_top=False,
        bn_kwargs=bn_kwargs,
        bias=True,
        disable_quanti_input=disable_quanti_input,
        node_name="backbone",
    )

    return mixvargenet_backbone


mixvargenet_stride64_example = [
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=32,
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=2,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=2,
        ),  # noqa
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=32,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f2_r", "mixvarge_k3k3_f2"],
            stack_factor=2,
            stride=2,
            fusion_strides=[2],
            extra_downsample_num=2,
        ),  # noqa
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=64,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f2_r"],
            stack_factor=2,
            stride=2,
            fusion_strides=[2, 4],
            extra_downsample_num=2,
        ),  # noqa
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=96,
            head_op="mixvarge_f2_gb16",
            stack_ops=[
                "mixvarge_f2_gb16",
                "mixvarge_f2_r_gb16",
                "mixvarge_k3k3_f2_gb16",
                "mixvarge_f2_r_gb16",
                "mixvarge_k3k3_f2_gb16",
                "mixvarge_f2_r_gb16",
            ],
            stack_factor=2,
            stride=2,
            fusion_strides=[4, 8],
            extra_downsample_num=2,
        ),  # noqa
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=96,
            out_channels=160,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_r_gb16"],
            stack_factor=2,
            stride=2,
            fusion_strides=[8, 16],
            extra_downsample_num=1,
        ),  # noqa
    ],  # stride 32
    [
        MixVarGENetConfig(
            in_channels=160,
            out_channels=320,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_r_gb16"],
            stack_factor=2,
            stride=2,
            fusion_strides=[16, 32],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 64
]

if use_mixvargnet:
    stride2channels = get_mixvargenet_stride2channels(
        net_config=mixvargenet_stride64_example
    )
    backbone = get_mixvargenet_config(
        net_config=mixvargenet_stride64_example, output_list=[0, 1, 2, 3, 4, 5]
    )


# -------------------------------- neck ----------------------------------   # noqa
neck = dict(
    type="Unet",
    in_strides=unet_in_strides,
    out_strides=unet_out_strides,
    stride2channels={2: 16, 4: 16, 8: 32, 16: 64, 32: 128, 64: 256},
    factor=unet_factor,
    bn_kwargs=bn_kwargs,
    group_base=group_base,
    node_name="neck",
)

if use_mixvargnet:
    neck = dict(
        type="Unet",
        in_strides=unet_in_strides,
        out_strides=unet_out_strides,
        stride2channels=stride2channels,
        factor=unet_factor,
        use_bias=False,
        bn_kwargs=bn_kwargs,
        group_base=32,
        node_name="neck",
    )
# -------------------------------- Submit configuration ----------------------------------  # noqa
# job_name = "mono_4pe_toll_gate"
for task in tasks:
    job_name = multitask_task_name + "_" + task["name"]
job_password = "newk8s666"
framework = "pytorch"
task_label = "HAT"
project_id = "LTCS20230017"
input_bucket = "mono,matrix"
priority = 5
docker_image = "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-1.2.1"
max_jobtime = 60 if local_or_remote_debug else 10000
# max_jobtime = 60
launcher = "mpi"
upload_folder_name = "k8s_job"
folder_list = [
    "../../hat",
    "../../tools",
    "../../projects",
    "url2IP.py",
]
config_file = "./projects/mono/toll_gate/multitask.py"
job_list = [
    "pip install hatbc==0.9.0b202306151509+99e57ce",
    "pip3 install albumentations -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc",  # noqa
    "pip install 'gevent<=23.7.0'",
    f"python3 tools/train.py --config {config_file} --stage float",  # noqa
    f"python3 tools/train.py --config {config_file} --stage qat",  # noqa
    f"python3 tools/train.py --config {config_file} --stage int_infer",  # noqa
]

# -------------------------------- val_transforms ----------------------------------  # noqa
# val_transforms = [
#     dict(
#         type="Resize",
#         img_scale=(img_h, img_w),
#         ratio_range=(1.0, 1.0),
#         keep_ratio=False,
#     ),
#     dict(
#         type="FixedCrop",
#         size=(0, 0, img_w, crop_h),
#     ),
#     dict(type="ToTensor", to_yuv=True),
#     dict(type="Normalize", mean=128.0, std=128.0),
#     # dict(
#     #     type="RandomFlip",
#     #     px=0.5,
#     #     py=0,
#     # ),
#     dict(
#         type="OvalHmTargetGenerator",
#         img_scale=(crop_h, img_w),
#         encode_lmks=2,
#         stride=4,
#         sigma=sigma,
#         ng_weights=1,
#     ),
# ]
