import os

DOCKER_IMAGE = "docker.hobot.cc/imagesys/hat:pilot-runtime-cu111-hdflow-20230611-torch1102-aidisdk0111-py38-mmcv142-nuplanv2"  # noqa
INPUT_BUCKET = [
    "matrix",
    "matrix2",
    "auto_eval",
    "adas",
    "SD_Algorithm",
    "mono",
]


def get_k8s_config(cfg):
    """get k8s config for cloudsparse4d."""
    task_name = cfg.task_name
    os.system(f"rm -rf {os.getenv('HOME')}/.cache/torch/hub/checkpoints")
    local_or_remote_debug = cfg.local_or_remote_debug
    aidi_eval = True if os.getenv("aidi_eval") else False
    job_name = "hat_job"
    job_name = job_name + "_" + task_name
    job_password = "6150"
    framework = "pytorch"
    task_label = "HAT_cloudsparse4d"
    project_id = cfg.get("project_id", "PD20230003")
    input_bucket = ",".join(INPUT_BUCKET)
    docker_image = DOCKER_IMAGE
    priority = 5
    max_jobtime = 60 if local_or_remote_debug else 20080  # 60 for debug
    launcher = "mpi"
    upload_folder_name = "k8s_job"
    if local_or_remote_debug or aidi_eval:
        num_machines = 1
    else:
        num_machines = cfg.get("num_machines", 2)

    if local_or_remote_debug:
        num_gpus_per_machine = cfg.get("num_gpus_debug", 1)
    elif aidi_eval:
        num_gpus_per_machine = cfg.get("num_gpus_eval", 4)
    else:
        num_gpus_per_machine = cfg.get("num_gpus", 8)

    folder_list = [
        "../../hat",
        "../../tools",
        "../../projects",
        "url2IP.py",
    ]
    cur_file_path = (
        "projects" + os.path.dirname(cfg.filename).split("projects")[-1]
    )
    config_file = f"{cur_file_path}/entry.py"

    base_job_list = [
        "pip3 install hatbc==0.10.0b202309070700+b39a57d",
    ]
    job_list = [
        f"python3 tools/train.py --config {config_file} --stage float --device-ids "
        + ",".join(list(map(str, range(num_gpus_per_machine)))),
        f"python3 tools/predict.py --config {config_file} --stage float --device-ids "
        + ",".join(list(map(str, range(num_gpus_per_machine)))),
    ]
    if aidi_eval:
        # only eval
        job_list = base_job_list + job_list[1:]
    else:
        job_list = base_job_list + job_list[:1]

    k8s_config = dict(
        job_name=job_name,
        job_password=job_password,
        num_machines=num_machines,
        num_gpus_per_machine=num_gpus_per_machine,
        framework=framework,
        task_label=task_label,
        project_id=project_id,
        input_bucket=input_bucket,
        priority=priority,
        docker_image=docker_image,
        max_jobtime=max_jobtime,
        launcher=launcher,
        upload_folder_name=upload_folder_name,
        folder_list=folder_list,
        job_list=job_list,
    )

    return k8s_config


def get_ckpt_dir(cfg):
    is_local_train = not os.path.exists("/running_package")
    if is_local_train:
        os.makedirs("tmp_models", exist_ok=True)
        ckpt_dir = "./tmp_models/%s" % cfg.task_name
    else:
        ckpt_dir = "/job_data/models/%s" % cfg.task_name
        os.environ["HAT_USE_CHECKPOINT"] = "1"
    return ckpt_dir
