# Copyright (c) Horizon Robotics, All rights reserved.
# flake8: noqa
import os

from hat.utils.root_helper import RootHelper
from projects.halo.avspeech.utils.global_config import get_config

# ---------------------------------------------------------------------------
#
# 提交任务配置
#
# ---------------------------------------------------------------------------
# REQUIRED
k8s_config = dict(
    job_name=get_config("task_name"),
    job_password="yaoyaoqiekenao",
    num_machines=2,
    num_gpus_per_machine=8,
    # OPTIONAL
    framework="pytorch",
    task_label="CARP",
    project_id="TD20220003",
    input_bucket="J2MM,jfs-hdfs,speech",  # default use HDLTAlgorithm Bucket
    priority=5,
    docker_image=(
        "docker.hobot.cc/imagesys/hat:runtime-py3.8-torch1.10.2-cu111-1.1.3-multidevice-pyhisf"
    ),
    max_jobtime=7200,  # default 7200 = 5days
    # launcher only for multi-machines
    launcher="mpi",
    # upload folder
    upload_folder_name="k8s_job",
    folder_list=[
        os.path.join(RootHelper.HAT_ROOT, "hat"),
        os.path.join(RootHelper.HAT_ROOT, "tools"),
        os.path.join(RootHelper.HAT_ROOT, "projects"),
        os.path.join(RootHelper.HAT_ROOT, "plugins/k8s_submit/url2IP.py"),
    ],
    job_list=[
        f"python3 tools/train.py \
        --config {get_config('current_config')} \
            --stage float",
    ],
    custom_cmds_before_job_list=[
        "export PYTHONUNBUFFERED=0",
        "export LANG=en_US.UTF-8",
        "export NCCL_P2P_LEVEL=NVL",
        "export NCCL_DEBUG=INFO",
    ],
)
