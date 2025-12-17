import os

from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels

multitask = [
    dict(name="vehicle_detection"),
    dict(name="face_detection"),
    dict(name="plate_detection"),
]
# -------------------------------- Training configuration ----------------------------------  # noqa

pipeline_test = os.getenv("HAT_PIPELINE_TEST", "0") == "1"
training_step = os.getenv("HAT_TRAINING_STEP", "float")
assert training_step in ["float", "float_freeze_bn", "qat", "int_infer"]
is_int_infer = training_step == "int_infer"
task_name = "mono_real3d_multitask"
local_train = not os.path.exists("/running_package")
march = "bernoulli2"

local_or_remote_debug = True
num_machines = 1
num_gpus_per_machine = 2 if local_or_remote_debug else 8
os.system(f"rm -rf {os.getenv('HOME')}/.cache/torch/hub/checkpoints")

# -------------------------------- Step and lr configuration ----------------------------------  # noqa
log_freq = 10
float_steps = 100000
qat_steps = 50000
warmup_steps = 10
save_interval = 20000
interval_by = "step"
float_lr = 1e-3
qat_lr = 1e-4
warmup_begin_lr = 5e-5
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
pretrain_checkpoint = "http://fm-xiang-yan.ucloudtrain.hogpu.cc/plat_gpu/hat_mono_real3d_vehicle_detection_face_detection_plate_detection-20221012-200954/output/models/mono_real3d_multitask/float-checkpoint-last.pth.tar"  # noqa

# -------------------------------- Dataloader configuration ---------------------------------- """  # noqa
batch_size_per_gpu = 8 if local_train else 16
num_workers = 4 if local_train else 4
pin_memory = False if local_train else True
drop_last = True

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
# -------------------------------- Neck configuration ----------------------------------  # noqa
unet_in_strides = [2, 4, 8, 16, 32, 64]
unet_out_strides = [4]
stride2channels = get_vargnetv2_stride2channels(alpha)
unet_factor = 2
# -------------------------------- Backbone ----------------------------------   # noqa
backbone = dict(
    type="VargNetV2",
    input_channels=input_channels,
    input_sequence_length=input_sequence_length,
    num_classes=1000,
    factor=factor,
    alpha=alpha,
    bias=True,
    bn_kwargs=bn_kwargs,
    group_base=group_base,
    include_top=include_top,
    head_factor=head_factor,
    extend_features=extend_features,
    node_name="backbone",
)
# -------------------------------- neck ----------------------------------   # noqa
neck = dict(
    type="Unet",
    in_strides=unet_in_strides,
    out_strides=unet_out_strides,
    stride2channels=stride2channels,
    factor=unet_factor,
    bn_kwargs=bn_kwargs,
    group_base=group_base,
    node_name="neck",
)
# -------------------------------- Submit configuration ----------------------------------  # noqa
job_name = "hat_mono_real3d"
for task in multitask:
    job_name = job_name + "_" + task["name"]
job_password = "newk8s666"
framework = "pytorch"
task_label = "HAT"
project_id = "PDT2020005"
input_bucket = "depth_data,mono,matrix"
priority = 5
docker_image = "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-1.2.1"
max_jobtime = 60 if local_or_remote_debug else 10000
launcher = "mpi"
upload_folder_name = "k8s_job"
folder_list = [
    "../../hat",
    "../../tools",
    "../../projects",
    "url2IP.py",
]
config_file = "./projects/mono/real3d_multitask/multitask.py"
job_list = [
    f"python3 tools/train.py --config {config_file} --stage float",
    f"python3 tools/train.py --config {config_file} --stage qat",
    f"python3 tools/train.py --config {config_file} --stage int_infer",
]
