import datetime
import os

from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels
from projects.superparking.app.ipm.lib.parser import DataYamlParser

# ----------------------- HAT env -----------------------
pipeline_test = os.environ.get("HAT_PIPELINE_TEST", "0") == "1"
pipeline_test_dump = os.environ.get("HAT_PIPELINE_TEST_DUMP_DATA", "0") == "1"
pipeline_test_ckpt = os.environ.get("HAT_PIPELINE_TEST_CKPT", None)

qat_pretrain_checkpoint = os.environ.get("HAT_SP_IPM_QAT_PRETRAIN_CKPT", None)

training_step = os.getenv("HAT_TRAINING_STEP", "float")
assert training_step in ["float", "float_freeze_bn", "qat", "int_infer"]

val_only = os.getenv("HAT_VAL_ONLY", "0") == "1"
aidi_eval = os.getenv("HAT_AIDI_EVAL", "0") == "1"
infer_model_ckpt = os.getenv("HAT_INFER_MODEL_CKPT")

# -------------------------- common --------------------------
ctimestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
job_name = "ipm-multitask"
if infer_model_ckpt is not None and infer_model_ckpt.startswith("http:"):
    job_name = infer_model_ckpt.split("/")[4] + f"-{ctimestr}"
local_train = not os.path.exists("/running_package")
input_sequence_length = 1

num_machines = 2
num_gpus_per_machine = 8
batch_size_factor = 1.0

cfg_dir = os.path.dirname(__file__)
CONFIGS = [
    os.path.join(cfg_dir, "parsing/ipm_parsing.py"),
    os.path.join(cfg_dir, "psd/ipm_psd.py"),
]

local_save_prefix = "tmp_output"
remote_save_prefix = "/job_data/models/"
if local_train:
    device_ids = [1]
    log_freq = 25
    bucket_root = "/horizon-bucket/SuperParking"
    save_prefix = local_save_prefix
    redirect_config_logging_path = None
    batch_size_per_gpu = dict(
        parsing=16,
        psd=16,
    )
    train_num_workers = dict(parsing=4, psd=4)
    val_num_workers = dict(parsing=2, psd=2)
    test_num_workers = dict(parsing=1, psd=1)
else:
    device_ids = list(range(num_gpus_per_machine))
    log_freq = 50
    bucket_root = "/bucket/input/SuperParking"
    save_prefix = remote_save_prefix
    redirect_config_logging_path = "/job_log/hat_config_output.log"
    # batch_size_per_gpu should be set in task config,
    # we put here for convenience.
    batch_size_per_gpu = dict(
        parsing=int(4 * batch_size_factor),
        psd=int(4 * batch_size_factor),
    )
    train_num_workers = dict(parsing=2, psd=2)
    val_num_workers = dict(parsing=1, psd=1)
    test_num_workers = dict(parsing=2, psd=2)
pretrain_checkpoint = os.path.join(
    bucket_root,
    "VWVersion/Models/Multitask/IPM/pretrain/ipm-multitask-use-step-20230321-150904/float-checkpoint-last-90d29e9b.pth.tar",  # noqa
)
float_resume_checkpoint = None

predition_tags = ["test"]

# loss weights
parsing_loss_weight = 5.0
psd_loss_weight = 1.0

# train config
start_lr = 0.009  # * 0.1
weight_decay = 4.0e-5
momentum = 0.9
freeze_bn_lr = 1e-3
qat_lr = 2e-5
use_step_lr = False
# if no IPM pretrained model, float epochs need  set 32
num_epoch_float = 12
num_epoch_freezebn = 12
num_epoch_qat = 12

# 36w steps = 1.5w steps/epoch * 2machine * 12epoch (batch size = 4)
num_steps_float = 360000 // num_machines
num_steps_qat = 120000 // num_machines

warmup_by = "epoch"
if warmup_by == "step":
    warmup_steps = 1000
else:
    warmup_steps = 1
do_freeze_bn = False
add_freeze_bn_step = False
freezebn_step_ids = [
    40,
]
do_freeze_backbone_neck = False
sync_bn = True
enable_amp = False
save_interval = 20000 // num_machines
interval_by = "step"
do_validation = False
qat_train_only = False


# model input size
height = 896
width = 896
input_size = (width, height)

# backbone & neck config
alpha = 0.5  # 1.0
head_factor = 2  # 1
feat_channels = 32
bifpn_out_strides = [4, 8, 16, 32, 64]
unet_out_strides = [4, 8, 16, 32]
use_bifpn_only = True
neck_stack = 3

# stride2channels of bifpn
bifpn_stride2channels = {stride: feat_channels for stride in bifpn_out_strides}

# stride2channels
stride2channels = get_vargnetv2_stride2channels(alpha)

# network batchnorm kwargs
bn_kwargs = dict(eps=2e-5, momentum=0.1)


def get_data_paths(task, collection, split):
    parser = DataYamlParser(
        task=task,
        collection_name=collection,
        data_yaml=os.path.join(os.path.dirname(__file__), "dataset/data.yaml"),
        version_yaml=os.path.join(
            os.path.dirname(__file__), "dataset/data_version.yaml"
        ),
    )

    paths = parser.get_lmdb_paths(split)
    weights = parser.get_lmdb_paths("weight")
    return paths, weights


task = "parsing"
collection = "last_version"
parsing_train_paths, parsing_train_weights = get_data_paths(
    task, collection, "train"
)
parsing_val_paths, _ = get_data_paths(task, collection, "val")

parsing_num_classes = 20
parsing_use_auxi_loss = False  # True
parsing_head_num_auxi_layer = 2
parsing_head_out_strides = [2]  # [2, 8, 16]
parsing_head_share_conv = False
parsing_head_stacked_convs = 3  # 6
parsing_head_start_level = 0
parsing_head_end_level = 2
if parsing_use_auxi_loss:
    assert parsing_head_num_auxi_layer == (
        parsing_head_end_level - parsing_head_start_level
    )
    assert parsing_head_num_auxi_layer + 1 == len(parsing_head_out_strides)
parsing_head_group_base = 8
parsing_head_conv_method = "varg_conv"
parsing_head_aggregation_method = "concat"

parsing_head_loss_weight = 1.0
parsing_ignore_index = 255
# parsing_class_weight = [2, 2, 15, 10, 10, 4, 4, 20]
parsing_class_weight = None
parsing_losses_weight = [1.0, 0.5, 0.5]
parsing_atuo_class_weight = False
parsing_weight_min = 0.0
parsing_weight_noobj = 0.75

# psd config
# -------------------- data ----------------------------------

task = "psd"
collection = "last_version"
psd_train_paths, psd_train_weights = get_data_paths(task, collection, "train")
psd_val_paths, _ = get_data_paths(task, collection, "val")

psd_num_slot_type = 3
psd_local_stride_idx = 0
psd_global_stride_idx = 3

# local loss config
psd_gaussian_loss_alpha = 2.0
psd_gaussian_loss_gamma = 4.0
psd_local_loss_radius = 6
psd_local_loss_weights = [1, 1, 1, 1]
psd_local_loss_feat_size = 224
psd_local_loss_far_weight = 1.0

# global loss config
psd_global_loss_weights = [1, 1, 1, 1, 1]
psd_global_loss_feat_size = 28

# slot loss config
psd_slot_weight_dict = {0: 1.0, 1: 5.0, 2: 1.0, -1: 0.0}

# metric config
metric_global_threshold = 0.18
metric_global_topk = 50
metric_global_downsample_factor = 32
metric_global_max_distance = 20
metric_global_kernel_size = 3
metric_local_threshold = 0.25
metric_local_topk = 30
metric_local_downsample_factor = 4
metric_local_max_distance = 5
metric_local_kernel_size = 7
metric_threshold_fuse_distance = 55
metric_iou_threshold = 0.5
metric_nms_distance_threshold = 20
# -------------------------- task_description --------------------------
task_desc = {
    "parsing": [dict(size=(448, 448))],
    "psd": [
        dict(
            name="global_cls",
            metric=dict(
                metric_global_threshold=metric_global_threshold,
                metric_global_topk=metric_global_topk,
                metric_global_downsample_factor=metric_global_downsample_factor,  # noqa
                metric_global_max_distance=metric_global_max_distance,
                metric_global_kernel_size=metric_global_kernel_size,
                metric_local_threshold=metric_local_threshold,
                metric_local_topk=metric_local_topk,
                metric_local_downsample_factor=metric_local_downsample_factor,
                metric_local_max_distance=metric_local_max_distance,
                metric_local_kernel_size=metric_local_kernel_size,
                metric_threshold_fuse_distance=metric_threshold_fuse_distance,
                metric_iou_threshold=metric_iou_threshold,
            ),
        ),
        dict(name="global_offset"),
        dict(name="global_occupancy"),
        dict(name="global_slot_type"),
        dict(name="global_direction"),
        dict(name="local_cls"),
        dict(name="local_offset"),
        dict(name="local_sline_angle"),
        dict(name="local_point_type"),
    ],
}

project_id = "PDT20220004"  # mono: "PDT20220001"
