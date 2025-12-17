import datetime
import os
from importlib import __import__, import_module  # noqa

from datasets.auto_2d.datasets import load_mono_data_v12

from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels

# ----------------------- common env -----------------------
job_name = "mono_resize_multitask"

training_step = os.getenv("HAT_TRAINING_STEP", "float")
pipeline_test = os.environ.get("HAT_PIPELINE_TEST", "0") == "1"
pipeline_test_dump = os.environ.get("HAT_PIPELINE_TEST_DUMP_DATA", "0") == "1"
val_only = os.getenv("HAT_VAL_ONLY", "0") == "1"
aidi_eval = os.getenv("HAT_AIDI_EVAL", "0") == "1"
val_ckpt = os.getenv("HAT_VAL_CKPT", None)
predict_only = os.getenv("HAT_PRED_ONLY", "0") == "1"
local_train = not os.path.exists("/running_package")

config_file = "projects/" + os.path.abspath(__file__).split("projects/")[-1]
config_file_root = os.path.dirname(config_file)
abs_config_file_root = os.path.abspath(config_file_root)
ctimestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_version = f"{job_name}-stage[{training_step}]-V{ctimestr}"
is_int_infer = training_step == "int_infer"

num_machines = 2
num_gpus_per_machine = 8
log_freq = 25

float_steps = 100000 // num_machines
qat_num_steps = 40000 // num_machines
warmup_steps = 500 // num_machines
do_freeze_bn = True
do_grad_scale = True

freezebn_step_ids = [
    75000 // num_machines,
]

lr = 5e-3
freeze_bn_lr = 2e-3
wd = 1e-5
qat_lr = 1e-4
use_step_lr = False
save_interval = 5000 // num_machines
interval_by = "step"

val_while_train = False  # TODO: not ready, ModuleShareError exists.
val_interval = 5000 // num_machines
val_interval_by = "step"

tasks_batch_size = {
    "vehicle_detection": {"train": 16, "val": 16, "test": 1},
    "vehicle_detection_3d": {"train": 16, "val": 16, "test": 1},
    "rear_detection": {"train": 16, "val": 16, "test": 1},
    "person_detection": {"train": 16, "val": 16, "test": 1},
    "person_detection_3d": {"train": 16, "val": 16, "test": 1},
    "cyclist_detection": {"train": 16, "val": 16, "test": 1},
    "cyclist_detection_3d": {"train": 16, "val": 16, "test": 1},
    "parsing_panoptic": {"train": 16, "val": 8, "test": 1},
    "parsing_lane": {"train": 16, "val": 8, "test": 1},
}

use_mosaic_aug = True
use_mixup_aug = "yolox"  # ["yolox", "yolov5", False / None / ""]

fcos_stacked_convs = 2
fcos_max_per_img = 30

custom_loader_length = None

if pipeline_test:
    # quick debug
    float_steps = 1
    freeze_bn_steps = 1
    qat_num_steps = 1
    warmup_steps = 1
    save_interval = 1
    custom_loader_length = 1

    for task_name in tasks_batch_size:
        tasks_batch_size[task_name]["train"] = 4
        tasks_batch_size[task_name]["val"] = 4


if local_train:
    device_ids = [1]
    log_freq = 25

    bucket_root = "/horizon-bucket"
    save_prefix = "tmp_output"
    train_num_workers, val_num_workers, test_num_workers = (
        (2, 4, 4) if not pipeline_test_dump else (0, 0, 0)
    )

    real3d_num_workers = (
        {"train": 2, "val": 2, "test": 1}
        if not pipeline_test_dump
        else {"train": 0, "val": 0, "test": 0}
    )
else:
    device_ids = list(range(num_gpus_per_machine))
    log_freq = 25
    bucket_root = "/bucket/input"
    save_prefix = "/job_data/models/"
    redirect_config_logging_path = "/job_log/hat_config_output.log"
    train_num_workers, val_num_workers, test_num_workers = 2, 0, 0
    real3d_num_workers = {"train": 2, "val": 2, "test": 1}

datasets = load_mono_data_v12(bucket_root)

loss_weights = {
    "vehicle_detection": 5,
    "rear_detection": 5,
    "cyclist_detection": 5,
    "person_detection": 5,
    "parsing_panoptic": 20,
    "parsing_lane": 20,
}

# out_strides of necks
bifpn_out_strides = [4, 8, 16, 32, 64]
rpn_out_strides = [8, 16, 32, 64]
rpn_out_strides_index = [
    bifpn_out_strides.index(stride) for stride in rpn_out_strides
]
# input: H * W
model_input_size = (512, 960)
resize_hw_for_val = (540, 960)

# network batchnorm kwargs
bn_kwargs = {"eps": 1e-5, "momentum": 0.1}

val_decoders = {}

float_resume_checkpoint = None
qat_resume_checkpoint = None  # noqa
pretrain_name = "imagenet"
pretrain_dict = {
    "imagenet": "hdfs://hobot-bigdata/user/mengao.zhao/hdlt_model/pretrained/vargnetv2_05_headfactor2.pth.tar",  # noqa
    "parsing_panoptic": os.path.join(
        bucket_root,
        "SuperParking/fangquan.hu/MonoData/float-checkpoint-step-39999-38f60de2.pth.tar",  # noqa
    ),  # noqa
    "vehicle2d3d": "http://fm-fangquan-hu.alitrain.hogpu.cc/plat_gpu/mono-resize-multitask-vehicle2d3d-add_flip-20230206_232149/output/models/mono_multitask/float-checkpoint-step-39999-47483ea9.pth.tar",  # noqa
    "no_pretrain": None,
    "debug": "http://fm-fangquan-hu.alitrain.hogpu.cc/plat_gpu/mono_v12_sp_solution_all_vdvru_detection_bs16_float-20221215-120103_resume-20221216-212904-COPY/output/models/mono_v12_sp/float-checkpoint-step-39999-a119e270.pth.tar",
}
pretrain_checkpoint = pretrain_dict[pretrain_name]

compare_version = "Resize_Day_v12.0.0"

test_image_dir = "your image dir (type:*/test_image_dir/*.jpg)"

# ------------------- backboone/neck -------------------
# backbone setting
alpha = 0.5
head_factor = 2
feat_channels = 32
neck_stack = 3

# stride2channels of bifpn
bifpn_stride2channels = {stride: feat_channels for stride in bifpn_out_strides}
stride2channels = get_vargnetv2_stride2channels(alpha)

backbone = dict(
    type="VargNetV2",
    input_channels=3,
    input_sequence_length=1,
    num_classes=1000,
    factor=2,
    alpha=alpha,
    bias=True,
    bn_kwargs=bn_kwargs,
    group_base=8,
    include_top=False,
    head_factor=head_factor,
    node_name="backbone",
)

neck = dict(
    type="BiFPN",
    fpn_name="bifpn_sum",
    in_strides=[2, 4, 8, 16, 32],
    out_strides=bifpn_out_strides,
    stride2channels=stride2channels,
    out_channels=feat_channels,
    stack=neck_stack,
    start_level=1,
    end_level=-1,
    num_outs=5,
    node_name="neck",
)

train_gpu_transforms = [
    dict(
        type="ColorJitter",
        brightness=0.2,
        contrast=(0.5, 1.5),
        saturation=(0.5, 1.5),
        hue=0.1,
    ),
    dict(type="BgrToYuv444", rgb_input=True),
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=128.0,
        std=128.0,
    ),
]

val_only_transforms = [
    dict(type="Resize", img_scale=resize_hw_for_val, keep_ratio=True),
    dict(type="FixedCrop", size=model_input_size[:2][::-1]),
    dict(type="ToTensor", to_yuv=False),
    dict(type="RenameKeys", keys=["imgs|img"]),
    dict(type="DeleteKeys", keys=["img_buf"]),
]

bpu_transforms = [
    {"type": "YUVTurboJPEGDecoder", "to_string": True},
    {
        "type": "BPUPyramidResizer",
        "scale_wh": (0.25, 0.25),
        "pyramid_type": "ips",
        "resize_gt": True,
    },
    {"type": "ImgBufToYUV444"},
    dict(type="FixedCrop", size=model_input_size[:2][::-1]),
    {"type": "ToTensor", "to_yuv": False},
    {"type": "Normalize", "mean": 128.0, "std": 128.0},
    {"type": "RenameKeys", "keys": ["imgs|img"]},
    {"type": "DeleteKeys", "keys": ["img_buf", "transform_meta"]},
]

val_gpu_transforms = train_gpu_transforms[-2:]

val_only_transforms += val_gpu_transforms

project_id = "PDT20220001"
