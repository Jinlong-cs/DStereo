# Copyright (c) Horizon Robotics. All rights reserved.
# submit jobs

import argparse
import logging
import os
import re
import subprocess
from datetime import datetime
from typing import Optional

from aidisdk.compute.job_abstract import (
    JobMountType,
    MountItem,
    MountMode,
    RunningResourceConfig,
    StartUpConfig,
)
from aidisdk.compute.package_abstract import (
    CodePackageConfig,
    GitCodeItem,
    LocalPackageItem,
)
from create_job import create_job
from generate_submit_files import (
    generate_bash_file,
    generate_elastic_bash_file,
    generate_upload_folder,
    generate_watchdog_bash_file,
)

from hat.utils.aidi import get_aidi_client
from hat.utils.config import Config
from hat.utils.setup_env import setup_args_env
from hat.version import get_git_repo_info

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--cluster",
        type=str,
        required=False,
        help="queue name, please set up queue name of current cluster",
    )
    parser.add_argument(
        "--queue",
        type=str,
        required=False,
        help="queue name, please set up queue name",
    )
    parser.add_argument(
        "--dag-name",
        default=None,
        type=str,
        required=False,
        help="if set, will override dag_name in config",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        type=str,
        required=False,
        help="if set, will override job_name in config",
    )
    parser.add_argument(
        "--dag-desc",
        default=None,
        type=str,
        required=False,
        help="Description of the DAG, will be set to None in single job mode",
    )
    parser.add_argument(
        "--job-desc",
        default=None,
        type=str,
        required=False,
        help="Description of the job",
    )
    parser.add_argument(
        "--num-machines",
        default=None,
        type=int,
        required=False,
        help="if set, will override num_machines in config",
    )
    parser.add_argument(
        "--num-gpus-per-machine",
        default=None,
        type=int,
        required=False,
        help="if set, will override num_gpus_per_machine in config",
    )
    parser.add_argument(
        "--job-type",
        default="train",
        type=str,
        required=False,
        help="job type, use `aidisdk.compute.job_abstract.JobType` enum",
    )
    parser.add_argument(
        "--schedule-type",
        default=None,
        type=str,
        required=False,
        help="schedule type of queue.",
    )
    parser.add_argument("--upload-folder", type=str, default="./")
    parser.add_argument(
        "--save-upload-folder",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--sleep",
        dest="sleep",
        action="store_true",
        help="sleep in cluster for debug",
    )

    parser.add_argument(
        "--experiment-name",
        default=None,
        type=str,
        required=False,
        help="AIDI MLOPs Exmeriment name",
    )
    parser.add_argument(
        "--experiment-path",
        default="",
        type=str,
        required=False,
        help="bucket for saving files in AIDI tracking",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        type=str,
        required=False,
        help="Run name in AIDI MLOPs Exmeriment",
    )
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        default=False,
        help="use AIDI MLOPs Exmeriment and enable tracking",
    )
    parser.add_argument(
        "--single-job",
        action="store_true",
        default=False,
        help="submit job by aidisdk.single_job interface.",
    )
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


def submit(
    config: str,
    queue: str = None,
    dag_name: str = None,
    dag_desc: str = None,
    job_name: str = None,
    job_desc: str = None,
    num_machines: int = None,
    num_gpus_per_machine: int = None,
    job_type: str = "train",
    schedule_type: str = None,
    upload_folder: str = "./",
    save_upload_folder: bool = False,
    sleep: bool = False,
    experiment_name: Optional[str] = None,
    experiment_path: str = "",
    run_name: str = None,
    enable_tracking: bool = False,
    args_env: list = None,
    use_dag: bool = True,
):
    """Submit cluster function.

    Args:
        config: Config file path.
        queue: queue name of the cluster being used.
        dag_name: dag name, If None, use default in config.
        job_name: job name, if None, use default in config.
        num_machines: number of machines, if None, use default in config.
        num_gpus_per_machine: number of gpus per machine.
            If None, use default in config.
        job_type: job type, default train.
        schedule_type: schedule type of queue, default None.
        upload_folder: Upload folder path.
        save_upload_folder: Wheather save the upload folder.
        sleep: Sleep in cluster for debug.
        experiment_name: AIDI MLOPs Experiment name.
        experiment_path: Bucket path for saving in AIDI MLOPs.
        enable_tracking: Use AIDI MLOPs Exmeriment and enable tracking.
        args_env: The args will be set in env.
    """
    if args_env:
        setup_args_env(args_env)

    cfg = Config.fromfile(config)
    if "k8s_config" in cfg:
        cfg = Config(cfg["k8s_config"])
    opts = {}
    if num_machines is not None:
        opts["num_machines"] = str(num_machines)
    if num_gpus_per_machine is not None:
        opts["num_gpus_per_machine"] = str(num_gpus_per_machine)
    cfg.merge_from_list_or_dict(opts, overwrite=True)

    launcher = cfg.get("launcher", "torch")
    if launcher == "mpi":
        if os.getenv("HAT_DDP_LAUNCHER_FORCE_MPI", "0") == "1":
            logger.warning("Note: Forced use of the `mpi`(mpirun) launcher.")
        else:
            logger.warning(
                "Note: The `mpi`(mpirun) launcher will be deprecated and will "
                " be replaced by `torch`(torchrun) launcher by default."
                "Please update the `launcher='torch'` in your config."
            )
            cfg.launcher = "torch"
    use_elastic = cfg.get("launcher", "torch") == "torchrun"

    upload_folder = generate_upload_folder(cfg, upload_folder, cfg.upload_folder_name)

    if use_elastic:
        generate_watchdog_bash_file(cfg, upload_folder, sleep)
        generate_elastic_bash_file(cfg, upload_folder, sleep)
    else:
        generate_bash_file(cfg, upload_folder, sleep)

    try:
        client = get_aidi_client()
        job_name = job_name if job_name is not None else cfg.job_name
        num_machines = cfg.num_machines
        num_gpu = cfg.num_gpus_per_machine

        # http://user-manual.aidi.hobot.cc/docs/aidi-model/aidi-model-1d2kctujvq092#bdwm44
        if num_machines > 1 and job_type == "prediction":
            assert schedule_type == "topo", (
                f"`schedule_type` should be `topo` when `job_type=prediction` "
                f"and `num_machines>1`, but get `schedule_type={schedule_type}`."  # noqa E501
                f"See http://user-manual.aidi.hobot.cc/docs/aidi-model/aidi-model-1d2kctujvq092#bdwm44"  # noqa E501
            )

        if queue is None:
            if "cluster" in cfg:
                logger.warning(
                    "`cluster` will be deprecated, please use `queue` instead."  # noqa E501
                )
                queue = cfg.cluster
            elif "queue" in cfg:
                queue = cfg.queue
            else:
                raise ValueError("`queue` can not be None.")
        mount_list = []
        # input bucket
        if cfg.get("input_bucket") is not None:
            bucket_list = cfg.input_bucket.split(",")
            for bucket in bucket_list:
                mount_item = MountItem(
                    mount_type=JobMountType.BUCKET,
                    name=bucket,
                    mode=MountMode.READ_ONLY,
                )
                mount_list.append(mount_item)
        # output bucket
        if cfg.get("output_bucket") is not None:
            bucket_list = cfg.output_bucket.split(",")
            for bucket in bucket_list:
                mount_item = MountItem(
                    mount_type=JobMountType.BUCKET,
                    name=bucket,
                    mode=MountMode.READ_AND_WRITE,
                )
                mount_list.append(mount_item)

        repo_url, commit_id = get_git_repo_info()
        time_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

        if job_desc is None:
            job_desc = cfg.get("job_desc", None)

        if use_dag:
            hat_git = GitCodeItem(
                repo=None if repo_url == "unknow" else repo_url,
                commit_id=None if commit_id == "unknow" else commit_id,
            )

            def _check_name(value, re_exp="\W"):  # noqa W605
                res = re.findall(re_exp, value)
                if res:
                    logger.warning(
                        f"Task name must match ^[a-zA-Z]+[a-zA-Z0-9_]*$, "
                        f"but get {res} in your task name, "
                        f"which will be replaced by `_`."
                    )
                    for c in res:
                        value = value.replace(c, "_")
                return value

            dag_name = dag_name if dag_name is not None else cfg.get("dag_name", None)
            dag_name = (
                f"{dag_name}_{time_suffix}"
                if dag_name is not None
                else f"DAG_{job_name}_{time_suffix}"
            )

            if dag_desc is None:
                dag_desc = cfg.get("dag_desc", None)

            # check git commit only when using AIDI Experiment.
            git_packages = [hat_git] if experiment_name is not None else []

            dag = client.dag.new_dag(
                name=_check_name(dag_name),
                desc=dag_desc,
                project_id=cfg.project_id,
                queue_name=queue,
                code_package=CodePackageConfig(
                    raw_package=LocalPackageItem(
                        lpath=upload_folder,
                        encrypt_passwd=cfg.job_password,
                        follow_softlink=True,
                    ).set_as_startup_dir(),
                    git_packages=git_packages,
                ),
            )

            _ = dag.new_job(
                name=_check_name(job_name + f"_{time_suffix}"),
                desc=job_desc,
                job_type=job_type,
                schedule_type=schedule_type,
                queue_name=queue,
                running_resource=RunningResourceConfig(
                    docker_image=cfg.docker_image,
                    instance=num_machines,
                    cpu=cfg.get("num_cpu", 1),
                    gpu=num_gpu,
                    walltime=cfg.max_jobtime,
                ),
                mount=mount_list,
                startup=StartUpConfig(
                    command="${WORKING_PATH}/job.sh",
                ),
                code_package=CodePackageConfig(
                    raw_package=LocalPackageItem(
                        lpath=upload_folder,
                        encrypt_passwd=cfg.job_password,
                        follow_softlink=True,
                    ).set_as_startup_dir(),
                    git_packages=git_packages,
                ),
                priority=cfg.priority,
            )

            if experiment_name is not None:
                if client.experiment.get_experiment(experiment_name) is None:
                    client.experiment.create_experiment(
                        name=experiment_name,
                        project_id=cfg.get("project_id", None),
                        experiment_path=experiment_path,
                    )
                with client.experiment.init(
                    experiment_name=experiment_name,
                    run_name=f"{job_type if run_name is None else run_name}",  # noqa E501
                    enabled=enable_tracking,
                ) as run:
                    # TODO: fix config_file
                    run.log_runtime(runtime=dag, config_file=config)

                    if use_elastic:
                        # TODO(mengyang.duan): 使用新提交接口
                        pass
                    else:
                        client.dag.submit_dag(dag)
            else:
                if use_elastic:
                    # TODO(mengyang.duan): 使用新提交接口
                    pass
                else:
                    client.dag.submit_dag(dag)

            print(dag)
            print("Submit dag successfully.")
        else:
            logger.warning(
                "`SingleJob` will be deprecated and not support AIDI MLOPs v3,"
                " please use `DAG` instead."
            )
            if not use_elastic:
                job = client.single_job.create(
                    job_name=job_name + "-" + "@datetime",
                    desc=job_desc,
                    job_type=job_type,
                    schedule_type=schedule_type,
                    ipd_number=cfg.project_id,
                    queue_name=queue,
                    running_resource=RunningResourceConfig(
                        docker_image=cfg.docker_image,
                        instance=num_machines,
                        cpu=cfg.get("num_cpu", 1),
                        gpu=num_gpu,
                        walltime=cfg.max_jobtime,
                    ),
                    mount=mount_list,
                    startup=StartUpConfig(
                        command="${WORKING_PATH}/job.sh",
                    ),
                    code_package=CodePackageConfig(
                        raw_package=LocalPackageItem(
                            lpath=upload_folder,
                            encrypt_passwd=cfg.job_password,
                            follow_softlink=True,
                        ).set_as_startup_dir(),
                    ),
                    priority=cfg.priority,
                )
                print(job)
                print("Submit job successfully.")
            else:
                create_job(
                    job_name=job_name + "-" + "@datetime",
                    job_type=job_type,
                    job_desc=job_desc,
                    project_id=cfg.project_id,
                    queue_name=queue,
                    running_resource=RunningResourceConfig(
                        docker_image=cfg.docker_image,
                        instance=num_machines,
                        cpu=cfg.get("num_cpu", 1),
                        gpu=num_gpu,
                        walltime=cfg.max_jobtime,
                    ),
                    mount=mount_list,
                    startup=StartUpConfig(
                        command="${WORKING_PATH}/job.sh",
                    ),
                    code_package=CodePackageConfig(
                        raw_package=LocalPackageItem(
                            lpath=upload_folder,
                            encrypt_passwd=cfg.job_password,
                            follow_softlink=True,
                        ).set_as_startup_dir(),
                    ),
                    priority=cfg.priority,
                    max_reschedulings=cfg.get("job_max_restarts", 0),
                    reschedule_policy=cfg.get("job_restart_mode", ""),
                )

    except Exception as e:
        logger.error("submit failed! " + str(e))
        raise e
    finally:
        if not save_upload_folder:
            if os.path.exists(upload_folder):
                subprocess.check_call(["rm", "-rf", upload_folder])


if __name__ == "__main__":
    args, args_env = parse_args()
    if args.cluster:
        logger.warning(
            "`--cluster` args will be deprecated, please use `--queue` instead."  # noqa E501
        )
        queue = args.cluster
    else:
        queue = args.queue
    submit(
        config=args.config,
        queue=queue,
        dag_name=args.dag_name,
        dag_desc=args.dag_desc,
        job_name=args.job_name,
        job_desc=args.job_desc,
        num_machines=args.num_machines,
        num_gpus_per_machine=args.num_gpus_per_machine,
        job_type=args.job_type,
        schedule_type=args.schedule_type,
        upload_folder=args.upload_folder,
        save_upload_folder=args.save_upload_folder,
        sleep=args.sleep,
        experiment_name=args.experiment_name,
        experiment_path=args.experiment_path,
        run_name=args.run_name,
        enable_tracking=args.enable_tracking,
        args_env=args_env,
        use_dag=not args.single_job,
    )
