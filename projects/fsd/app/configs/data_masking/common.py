import os

import hat

multitask = [
    dict(name="plate_detection"),
    dict(name="face_detection"),
]

# -------------------------------- Training configuration ----------------------------------  # noqa

pipeline_test = os.getenv("HAT_PIPELINE_TEST", "0") == "1"
training_step = os.getenv("HAT_TRAINING_STEP", "with_bn")

is_int_infer = training_step == "int_infer"
task_name = "superdrive_data_masking"
local_train = not os.path.exists("/running_package")
march = "bayes"

is_debug = True
num_machines = 1
num_gpus_per_machine = 2 if is_debug else 8
log_rank_zero_only = True
os.system(f"rm -rf {os.getenv('HOME')}/.cache/torch/hub/checkpoints")

img_height = 640
img_width = 960
roi_region = (
    0,
    0,
    img_height,
    img_width,
)

vanishing_point = (int(img_height / 2), int(img_width / 2))
pretrain_checkpoint = "http://aidi.hobot.cc/user/homespace/kai.liu/plat_gpu/data_anon_v8_as33_day_sw5-20230611-104331/output/models/sd_datamasking_4_12_pretrain_no_ts_with_mono_sw_5/with_bn-checkpoint-last.pth.tar"  # noqa
# -------------------------------- Step and lr configuration ----------------------------------  # noqa
log_freq = 10
if is_debug:
    with_bn_steps = 100
    fuse_bn_steps = 100
    warmup_steps = 10
    save_interval = 20
else:
    with_bn_steps = 40000
    fuse_bn_steps = 60000
    warmup_steps = 1000
    save_interval = 20000
interval_by = "step"
with_bn_lr = 1e-3
fuse_bn_lr = 1e-4
warmup_begin_lr = 1e-5
weight_decay = 0

# -------------------------------- Directories configuration ----------------------------------   # noqa
local_save_prefix = "tmp_output"
remote_save_prefix = "/job_data/models/"
save_prefix = local_save_prefix if local_train else remote_save_prefix
redirect_config_logging_path = (
    None if local_train else f"/job_log/{task_name}_{training_step}_step.log"
)
ckpt_dir = os.path.join(save_prefix, task_name)
log_dir = os.path.join(ckpt_dir, "logs")
compile_dir = os.path.join(ckpt_dir, "compile")
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(compile_dir, exist_ok=True)
bucket_root = "/horizon-bucket" if local_train else "/bucket/input"

# -------------------------------- Dataloader configuration ---------------------------------- """  # noqa
batch_size_per_gpu = 8 if local_train else 32
num_workers = 4 if local_train else 4
pin_memory = False if local_train else True
drop_last = True

# -------------------------------- Backbone configuration ----------------------------------  # noqa
num_classes = 1000
multiplier = 0.125
group_base = 4
last_channels = 1024
stages = (1, 2, 3)
use_bias = True
include_top = False
extend_features = False
bn_kwargs = dict(eps=1e-5, momentum=0.1)
backbone = dict(
    type="VargNetV2Stage2631",
    num_classes=num_classes,
    multiplier=multiplier,
    group_base=group_base,
    last_channels=last_channels,
    stages=stages,
    use_bias=use_bias,
    include_top=include_top,
    extend_features=extend_features,
    bn_kwargs=bn_kwargs,
    node_name="backbone",
)

# -------------------------------- neck ----------------------------------   # noqa
in_strides = [2, 4, 8]
in_channels = [16, 16, 32]
in_channels = list(map(lambda x: max(8, int(2 * x * multiplier)), in_channels))
out_strides = [8]
out_channel = 16
neck_out_strides = [8]  # , 16, 32, 64
feat_channels = 24
neck_stride2channels = {stride: out_channel for stride in neck_out_strides}

neck = dict(
    type="FixChannelNeck",
    in_strides=in_strides,
    in_channels=in_channels,
    out_strides=out_strides,
    out_channel=out_channel,
    bn_kwargs=bn_kwargs,
    node_name="fix_channel_neck",
)

# -------------------------------- Submit configuration ----------------------------------  # noqa
job_name = task_name
for task in multitask:
    job_name = job_name + "_" + task["name"]
job_password = "newk8s666"
framework = "pytorch"
task_label = "HAT"
project_id = "PDT20220001"
input_bucket = "SD_Algorithm,matrix,matrix2"
priority = 5
version = hat.__version__.split(".dev")[0]
docker_image = (
    "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.13.0-cu116-%s" % version
)
max_jobtime = 60 if is_debug else 10000
launcher = "mpi"
upload_folder_name = "k8s_job"
folder_list = [
    "../../hat",
    "../../tools",
    "../../projects",
    "url2IP.py",
]
config_file = "./projects/fsd/app/configs/data_masking/multitask.py"
job_list = [
    f"python3 tools/train.py --config {config_file} --stage with_bn",
    f"python3 tools/train.py --config {config_file} --stage fuse_bn",
    f"python3 tools/train.py --config {config_file} --stage int_infer",
]
