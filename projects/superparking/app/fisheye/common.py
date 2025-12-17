import copy
import datetime
import os
from importlib import __import__, import_module  # noqa

from lib.aidieval_helper import TASK2AIDI_EVAL_IDS
from lib.utils import get_ci_bucket_map, regain_dataset_auto2d_paths

from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels

# ----------------------- common env -----------------------
job_name = "Parking_FE_MT"

training_step = os.getenv("HAT_TRAINING_STEP", "float")
pipeline_test = os.environ.get("HAT_PIPELINE_TEST", "0") == "1"
pipeline_test_dump = os.environ.get("HAT_PIPELINE_TEST_DUMP_DATA", "0") == "1"
use_mini_dataset = os.environ.get("HAT_USE_MINI_DATASET", "0") == "1"
commit_level_ci = os.environ.get("HAT_SP_TEST_LEVEL", None) == "commit"
val_only = os.getenv("HAT_VAL_ONLY", "0") == "1"
aidi_eval = os.getenv("HAT_AIDI_EVAL", "0") == "1"
val_ckpt = os.getenv("HAT_VAL_CKPT", None)
infer_model_ckpt = os.getenv("HAT_INFER_MODEL_CKPT")
int_infer_stage = os.getenv("HAT_INFER_STAGE", None)
assert int_infer_stage in [None, "stage_one", "stage_two"]
local_train = not os.path.exists("/running_package")
if commit_level_ci:
    use_mini_dataset = True
if infer_model_ckpt is not None and infer_model_ckpt.startswith("http:"):
    job_name = infer_model_ckpt.split("/plat_gpu/")[1].split("/")[0]

# 2D tasks use pilot dataset or sp dataset
config_file_root = os.path.dirname(os.path.abspath(__file__))
multitask_path = "projects.superparking.app.fisheye.datasets.auto_2d.dataset"
data_string = "sp"
data_pack_name = multitask_path + "." + data_string
datasets = import_module(multitask_path)
datapaths = datasets.datapaths
ctimestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

num_machines = 4
num_gpus_per_machine = 8
log_freq = 25

float_steps = 80000 // num_machines
freeze_bn_steps = 7200 // num_machines
qat_num_steps = 40000 // num_machines
warmup_steps = 500 // num_machines
do_freeze_bn = True
add_freeze_bn_step = False
do_grad_scale = True

freezebn_step_ids = [
    60000 // num_machines,
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

model_thresh = dict(
    det_thresh=dict(
        person=0.42,
        cyclist=0.50,
        vehicle=0.55,
        rear=0.60,
    ),
    roi_det_thresh=dict(
        vehicle=0.3,
        rear=0.52,
        person=0.67,
    ),
)

tasks_batch_size = {
    "vehicle_detection": {"train": 8, "val": 16, "test": 1},
    "vehicle_detection_3d": {"train": 16, "val": 16, "test": 8},
    "vehicle_category_classification": {"train": 8, "val": 1, "test": 1},
    "vehicle_occlusion_classification": {"train": 8, "val": 1, "test": 1},
    "vehicle_truncation_classification": {"train": 8, "val": 1, "test": 1},
    "vehicle_wheel_kps": {"train": 8, "val": 1, "test": 1},
    "rear_detection": {"train": 8, "val": 16, "test": 1},
    "rear_plate_detection": {"train": 8, "val": 1, "test": 1},
    "rear_occlusion_classification": {"train": 8, "val": 1, "test": 1},
    "rear_part_classification": {"train": 8, "val": 1, "test": 1},
    "person_detection": {"train": 8, "val": 16, "test": 1},
    "person_detection_3d": {"train": 16, "val": 16, "test": 1},
    "person_face_detection": {"train": 8, "val": 1, "test": 1},
    "person_occlusion_classification": {"train": 8, "val": 1, "test": 1},
    "person_orientation_classification": {"train": 8, "val": 1, "test": 1},
    "person_pose_classification": {"train": 8, "val": 1, "test": 1},
    "cyclist_detection": {"train": 8, "val": 16, "test": 1},
    "cyclist_detection_3d": {"train": 16, "val": 16, "test": 1},
    "sod": {"train": 8, "val": 16, "test": 1},
    "parsing": {"train": 8, "val": 16, "test": 1},
}

if local_train:
    device_ids = [0]
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

sp_bucket = os.path.join(bucket_root, "SuperParking")
pt_bucket = os.path.join(bucket_root, "HDLTAlgorithm")

custom_loader_length = None

if pipeline_test:
    # quick debug
    float_steps = 1
    freeze_bn_steps = 1
    qat_num_steps = 1
    warmup_steps = 1
    save_interval = 1
    custom_loader_length = 1
    job_name = "sp_pipeline_test"

algo_version = os.environ.get("HAT_ALGO_VERSION", ctimestr)
model_version = f"{job_name}-stage[{training_step}]-V{algo_version}"
# for AIDI Report comparison
compare_version = "sp_fisheye_multitask_align_2dvdvru_fix_tasksampler_add_freeze_bn_2x8-20230126_225303-stage[qat]-V0"  # noqa

datapaths = regain_dataset_auto2d_paths(
    bucket_root, datapaths, use_mini_dataset, commit_level_ci
)

parsing_loss_weight = 40.0
det2d_loss_weight = 25
sod_loss_weight = 25

# backbone setting
alpha = 0.5
head_factor = 2
feat_channels = 32
neck_stack = 3

fcos_stacked_convs = 2

# out_strides of necks
bifpn_out_strides = [4, 8, 16, 32, 64]
rpn_out_strides = [8, 16, 32, 64]
rpn_out_strides_index = [
    bifpn_out_strides.index(stride) for stride in rpn_out_strides
]
# input: H * W
fisheye_input_size = (1152, 1408)
input_scale = 0.5
input_size = tuple([int(input_scale * d) for d in fisheye_input_size])

# centernet_head
bifpn_stride2levels = {
    stride: level for level, stride in enumerate(bifpn_out_strides)
}
# stride2channels of bifpn
bifpn_stride2channels = {stride: feat_channels for stride in bifpn_out_strides}
stride2channels = get_vargnetv2_stride2channels(alpha)

# network batchnorm kwargs
bn_kwargs = {"eps": 1e-5, "momentum": 0.1}

val_decoders = {}
is_int_infer = training_step == "int_infer"

float_resume_checkpoint = None
qat_resume_checkpoint = None  # noqa
checkpoint_dict = {
    "last_version": os.path.join(
        sp_bucket
        if not commit_level_ci
        else os.path.join(
            get_ci_bucket_map(bucket_root)[pt_bucket], "users", "fangquan.hu"
        ),
        "VWVersion/Models/Multitask/Fisheye/pretrain/float-checkpoint-step-19999-cf6b3258.pth.tar",  # noqa
    ),
}
pretrain_checkpoint = checkpoint_dict["last_version"]
test_image_dir = "your image dir (type:*/test_image_dir/*.jpg)"
# --------------------- tasks ---------------------

roi_task_train_post_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.0,
    nms=dict(name="nms", iou_threshold=0.7),
    max_per_img=30,
)

roi_task_test_post_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.3,
    nms=dict(name="nms", iou_threshold=0.5),
    max_per_img=15,
)

# image transforms for just val-only
bpu_transforms = [
    {"type": "YUVTurboJPEGDecoder", "to_string": True},
    {
        "type": "BPUPyramidResizer",
        "scale_wh": (input_scale, input_scale),
        "pyramid_type": "ips",
        "resize_gt": True,
    },
    {"type": "ToTensor", "to_yuv": False},
    {"type": "Normalize", "mean": 128.0, "std": 128.0},
    {"type": "RenameKeys", "keys": ["imgs|img"]},
    {"type": "DeleteKeys", "keys": ["img_buf", "transform_meta"]},
]

val_gpu_transforms = [
    dict(
        type="DiffCamImageProjector",
        src_cam_key="source_cam",
        dst_cam_key="warp_cam",
        from_meta=False,
        image_key="img",
    ),
]

train_gpu_transforms = copy.deepcopy(val_gpu_transforms)
train_gpu_transforms.append(
    dict(
        type="ColorJitter",
        brightness=0.2,
        contrast=(0.5, 1.5),
        saturation=(0.5, 1.5),
        hue=0.1,
    ),
)

task_common_transforms = [
    dict(type="BgrToYuv444", rgb_input=True),
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=128.0,
        std=128.0,
    ),
]

val_decoders = {}

hbdk_input_path = os.path.join(
    os.path.join(sp_bucket, "ci_data")
    if not commit_level_ci
    else os.path.join(get_ci_bucket_map(bucket_root)[pt_bucket], "data"),
    "yuv420_test_data/fisheye_multitask_1st_stage.yuv",
)

# ------------------- backboone/neck -------------------
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

project_id = "LTCS20230017"
# project_id = "PDT20220001"

AIDI_PREDICT_TAG = ["sp", "fisheye", "multitask"]

save_all_data_paths = False  # to save data paths of all tasks to a file
all_task_data_files = os.path.join(save_prefix, "all_task_data_files.txt")
os.makedirs(os.path.dirname(all_task_data_files), exist_ok=True)


def get_aidi_eval_info_common(
    task_name,
    task_type,
    reformat_output_fn,
    reformat_out_fn_kwargs,
    extra_tags=None,
):
    aidi_eval_page_label = task_name
    aidi_eval_dataset_id = []
    aidi_eval_callbacks = []
    aidi_eval_type = task_type
    old_prediction = compare_version
    new_prediction = model_version
    dataset_ids = TASK2AIDI_EVAL_IDS[task_name]
    predition_tags = AIDI_PREDICT_TAG
    if extra_tags is not None:
        predition_tags = predition_tags + extra_tags
    for dataset_id in dataset_ids:
        aidi_eval_callback = dict(
            type="AIDIEval",
            output_root=os.path.join(save_prefix, job_name, "prediction"),
            prediction_tags=predition_tags,
            project_id=project_id,
            prediction_name=new_prediction,
            aidi_eval_dataset_id=[dataset_id],
            reformat_output_fn=reformat_output_fn,
            reformat_out_fn_kwargs=reformat_out_fn_kwargs,
        )
        aidi_eval_callbacks.append(
            [
                aidi_eval_callback,
                dict(
                    type="StatsMonitor",
                    log_freq=log_freq,
                ),
            ]
        )
        aidi_eval_dataset_id.append(dataset_id)
    return (
        aidi_eval_page_label,
        aidi_eval_dataset_id,
        aidi_eval_callbacks,
        aidi_eval_type,
        old_prediction,
        new_prediction,
    )
