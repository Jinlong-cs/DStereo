import os

import hat

# ---------------------------------------------------------
# some history or settings
# ---------------------------------------------------------
clusters = [
    "share-3090-small-bcloud",
    "share-3090-small",
    "project-v100-mono-debug",
    "share-3090-idc",
    "share-debug-queue-idc",
    "share-3090-small-tcloud",
]

version = hat.__version__.split(".dev")[0]
dockers = [
    # "docker.hobot.cc/imagesys/base:py38cu111torch1102mmdet",
    # "docker.hobot.cc/imagesys/base:py38cu111torch1102hat121mmdet2230mmcv142",
    f"docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.13.0-cu116-{version}",
    f"docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-{version}",
    f"docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu102-{version}",
    f"docker.hobot.cc/dlp/hat:runtime-py3.6-torch1.10.2-cu102-{version}",
    f"docker.hobot.cc/dlp/hat:runtime-py3.6-torch1.10.2-cu111-{version}",
    f"docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.9.1-cu102-{version}",
    f"docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.9.1-cu111-{version}",
    f"docker.hobot.cc/dlp/hat:runtime-py3.6-torch1.9.1-cu102-{version}",
    f"docker.hobot.cc/dlp/hat:runtime-py3.6-torch1.9.1-cu111-{version}",
]

project_ids = [
    "PDT2020005",
    "PDT20220002",
    "PDT20220004",  # mono
    "PDT2021004-bevdata",
]

# ---------------------------------------------------------
# external
# ---------------------------------------------------------
# basic envs
debug = True
do_eval = False
do_infer = False

job_name = "test"
job_password = "newk8s666"
job_name = job_name if not debug else "debug_" + job_name
priority = 5

num_machines = 1 if not debug else 1
num_gpus_per_machine = 8 if not debug else 2
launcher = "mpi"  # launcher only for multi-machines
max_jobtime = 20160 if not debug else 60  # default 7200 = 5days

cluster = (
    "share-3090-small-tcloud" if not debug else "share-debug-queue-tcloud"
)
gpu_ids = ",".join([str(_) for _ in range(num_gpus_per_machine)])

framework = "pytorch"
task_label = "HAT"
project_id = "PD20230003"
input_bucket = "mono,matrix,adas,auto_eval,SuperParking"
docker_image = f"docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-{version}"  # noqa E501

# upload folder
dir_name = os.path.dirname(__file__)
upload_folder_name = "k8s_job"
folder_list = [
    os.path.join(dir_name, "../../../../../hat"),
    os.path.join(dir_name, "../../../../../tools"),
    os.path.join(dir_name, "../../../.."),
    os.path.join(dir_name, "../../../../../plugins/k8s_submit/url2IP.py"),
    os.path.join(
        dir_name, "../../../../../plugins/k8s_submit/ssh_launcher.py"
    ),
]

job_list = [
    f"python3 tools/train.py -c projects/cloudmodel/configs/perception_2d/cloudmodel_entry_trainval.py -s float  -ids {gpu_ids}",  # noqa E501
]
if do_eval:
    job_list.append(
        f"python3 tools/predict.py --config projects/cloudmodel/configs/perception_2d/cloudmodel_entry_aidieval.py --stage float -ids {gpu_ids}",  # noqa E501
    )
if do_infer:
    job_list = [
        f"python3 tools/predict.py --config projects/cloudmodel/configs/perception_2d/inference.py --stage float -ids {gpu_ids}",  # noqa E501
    ]

custom_cmds_before_job_list = [
    "pip3 install --upgrade --force-reinstall numpy==1.20.0 -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc",  # noqa E501
]
