import os

from common import (
    config_file_root,
    job_name,
    num_gpus_per_machine,
    num_machines,
    project_id,
)

dockers = dict(
    master_cu116=f"docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.13.0-cu116-1.3.1",  # noqa
    master_cu111=f"docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-1.2.1",  # noqa
)

config_file = os.path.join(config_file_root, "multitask.py")
job_list = [
    f"python3 -W ignore tools/train.py --config {config_file} --stage float",
    f"python3 -W ignore tools/train.py --config {config_file} --stage qat",
]

k8s_config = dict(
    job_name=job_name,
    job_password="newk8s666",
    num_machines=num_machines,
    num_gpus_per_machine=num_gpus_per_machine,
    framework="pytorch",
    task_label="fisheye_multitask",
    project_id=project_id,
    input_bucket="SuperParking,mono,depth_data",
    # priority description:
    # 3: common experiment job, default
    # 4: model publish job
    # 5: data-related job, used by model publish
    priority=5,
    docker_image=dockers["master_cu116"],
    # default 7200 = 5days
    max_jobtime=10000,
    # launcher only for multi-machines
    launcher="mpi",
    # upload folder
    upload_folder_name="k8s_job",
    folder_list=[
        "../../hat",
        "../../tools",
        "../../projects",
        "url2IP.py",
        # "../../Makefile",
    ],
    job_list=job_list,
)
