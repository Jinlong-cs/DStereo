# Copyright (c) Horizon Robotics. All rights reserved.
# submit jobs

import logging
import os
import random
import subprocess

from hat.version import get_git_repo_info

logger = logging.getLogger(__name__)


def generate_upload_folder(cfg, upload_folder, upload_folder_name):
    upload_folder = os.path.join(upload_folder, upload_folder_name)
    subprocess.check_call(["mkdir", "-p", upload_folder])

    assert "folder_list" in cfg
    folder_list = cfg.folder_list
    # TODO(mengyang.duan): for compatibility, delete later.

    if cfg.launcher == "torch":
        for f in ["ssh_launcher.py", "torchrun", "nccl_check.py"]:
            if f not in folder_list:
                folder_list.append(f)

    for path in folder_list:
        if not os.path.exists(path):
            print(f"{path} not exists, skip")
            continue
        # support converting soft links to files.
        subprocess.check_call(["rsync", "-aL", path, upload_folder])
        print("copy %s to %s" % (path, upload_folder))

    return upload_folder


def write_mpi_multi_machines_cmd(fn, cfg, job):
    """Write job to fn."""
    command = "mpirun -n %d -ppn %d --hostfile %s %s %s" % (
        cfg.num_machines * cfg.num_gpus_per_machine,
        cfg.num_gpus_per_machine,
        "/job_data/mpi_hosts",
        job,
        "--dist-url tcp://$dis_url:8000 --launcher mpi ",
    )
    fn.write("%s\n" % command)


def write_torch_multi_machines_cmd(fn, cfg, job):
    """Write job to fn."""

    if job.startswith("python3"):
        cmd = (
            "./torchrun --nnodes=%d --nproc_per_node=%d --rdzv_id=%d --rdzv_backend=c10d --rdzv_endpoint=$HOST_NODE_ADDR %s %s"
            % (  # noqa
                cfg.num_machines,
                cfg.num_gpus_per_machine,
                random.randint(0, 100000),
                job.replace("python3 ", "").replace("-W ignore ", ""),
                "--launcher torch ",
            )
        )
        command_args = " --check --monitor"
    else:
        cmd = job
        command_args = ""
    command = "python3 ssh_launcher.py %s -n %d -g %d -H %s '%s'" % (
        command_args,
        cfg.num_machines,
        cfg.num_gpus_per_machine,
        "/job_data/mpi_hosts",
        cmd,
    )

    fn.write("%s\n" % command)


def write_multi_machines_cmd(fn, cfg, job, launcher):
    if launcher == "mpi":
        write_mpi_multi_machines_cmd(fn, cfg, job)
    elif launcher == "torch":
        write_torch_multi_machines_cmd(fn, cfg, job)
    else:
        raise ValueError("launcher only supports %s, %s" % ("mpi", "torch"))


def generate_bash_file(cfg, upload_folder, run_in_sleep):
    bash_file = os.path.join(upload_folder, "job.sh")
    with open(bash_file, "w") as fn:
        fn.write("set -e\n")
        fn.write("export PYTHONPATH=${WORKING_PATH}:$PYTHONPATH\n")
        fn.write("date\n")
        fn.write("env\n")
        fn.write("pip3 list\n")
        fn.write("export PYTHONUNBUFFERED=0\n")

        repo_url, commit_id = get_git_repo_info()
        fn.write(f"export HAT_GIT_REPO={repo_url}\n")
        fn.write(f"export HAT_GIT_COMMIT_ID={commit_id}\n")

        fn.write("cd ${WORKING_PATH}\n")
        if run_in_sleep:
            fn.write("sleep 1000000m\n")

        for cmd in getattr(cfg, "prefix_cmds_on_master", []):
            fn.write(f"{cmd}\n")

        if cfg.num_machines > 1:  # multi-machines
            fn.write("python3 url2IP.py\n")
            fn.write("cat /job_data/mpi_hosts\n")
            fn.write("dis_url=$(head -n +1 /job_data/mpi_hosts)\n")

            custom_cmds_before_job_list = cfg.get("custom_cmds_before_job_list", [])
            if cfg.launcher == "torch":
                chmod_torchrun = "chmod +x ./torchrun"
                custom_cmds_before_job_list.append(chmod_torchrun)

            if len(custom_cmds_before_job_list) > 0:
                cmds_file = os.path.join(
                    upload_folder, "custom_cmds_before_job_list.sh"
                )
                with open(cmds_file, "w") as cus:
                    for cmd in custom_cmds_before_job_list:
                        cus.write("%s\n" % cmd)
                job = "bash ${WORKING_PATH}/custom_cmds_before_job_list.sh"
                write_multi_machines_cmd(fn, cfg, job, cfg.launcher)

            for job in cfg.job_list:
                write_multi_machines_cmd(fn, cfg, job, cfg.launcher)

            if hasattr(cfg, "custom_cmds_after_job_list"):
                cmds_file = os.path.join(upload_folder, "custom_cmds_after_job_list.sh")
                with open(cmds_file, "w") as cus:
                    for cmd in cfg.custom_cmds_after_job_list:
                        cus.write("%s\n" % cmd)
                job = "bash ${WORKING_PATH}/custom_cmds_after_job_list.sh"
                write_multi_machines_cmd(fn, cfg, job, cfg.launcher)
        else:
            if hasattr(cfg, "custom_cmds_before_job_list"):
                for cmd in cfg.custom_cmds_before_job_list:
                    fn.write("%s\n" % cmd)
            for job in cfg.job_list:
                fn.write("%s\n" % job)
            if hasattr(cfg, "custom_cmds_after_job_list"):
                for cmd in cfg.custom_cmds_after_job_list:
                    fn.write("%s\n" % cmd)

        for cmd in getattr(cfg, "suffix_cmds_on_master", []):
            fn.write(f"{cmd}\n")

    subprocess.check_call(["chmod", "777", bash_file])
    generate_nccl_test_bash_file(cfg, upload_folder)


def generate_nccl_test_bash_file(cfg, upload_folder):
    bash_file = os.path.join(upload_folder, "nccl_check.sh")
    with open(bash_file, "w") as fn:
        fn.write("set -e\n")
        fn.write("export PYTHONPATH=${WORKING_PATH}:$PYTHONPATH\n")
        fn.write("export PYTHONUNBUFFERED=0\n")

        fn.write("cd ${WORKING_PATH}\n")

        if cfg.num_machines > 1:  # multi-machines
            fn.write("python3 url2IP.py\n")
            fn.write("cat /job_data/mpi_hosts\n")
            fn.write("dis_url=$(head -n +1 /job_data/mpi_hosts)\n")
            nccl_test_cmd = f"python3 nccl_check.py --driver --ngpus {cfg.num_gpus_per_machine} --nccl"  # noqa E501
            command = "python3 ssh_launcher.py -n %d -g %d -H %s '%s'" % (
                cfg.num_machines,
                cfg.num_gpus_per_machine,
                "/job_data/mpi_hosts",
                nccl_test_cmd,
            )
            fn.write("%s\n" % command)
        else:
            nccl_test_cmd = f"python3 nccl_check.py --driver --ngpus {cfg.num_gpus_per_machine} --nccl"  # noqa E501
            fn.write(f"{nccl_test_cmd}\n")

    subprocess.check_call(["chmod", "777", bash_file])


def generate_watchdog_bash_file(cfg, upload_folder, run_in_sleep):
    bash_file = os.path.join(upload_folder, "watch_dog.sh")
    with open(bash_file, "w") as fn:
        fn.write("set -e\n")
        # TODO(mengyang.duan):
        fn.write("export WORKING_PATH=/running_package/code_package\n")
        fn.write("export PYTHONPATH=${WORKING_PATH}:$PYTHONPATH\n")
        fn.write("export PYTHONUNBUFFERED=0\n")
        fn.write("cd ${WORKING_PATH}\n")
        pattern_files = cfg.elastic_pattern_files
        pattern_files = " ".join(pattern_files)
        if run_in_sleep:
            fn.write("sleep 1d\n")
        fn.write(
            "python3 watch_dog.py --log-path ${JOB_LOG_PATH} --pattern-files %s \n"  # noqa E501
            % pattern_files
        )

    subprocess.check_call(["chmod", "777", bash_file])


def generate_elastic_bash_file(cfg, upload_folder, run_in_sleep):
    bash_file = os.path.join(upload_folder, "job.sh")
    with open(bash_file, "w") as fn:
        fn.write("set -e\n")
        fn.write("export PYTHONPATH=${WORKING_PATH}:$PYTHONPATH\n")
        fn.write("date\n")
        fn.write("env\n")
        fn.write("pip3 list\n")
        fn.write("export PYTHONUNBUFFERED=0\n")

        repo_url, commit_id = get_git_repo_info()
        fn.write(f"export HAT_GIT_REPO={repo_url}\n")
        fn.write(f"export HAT_GIT_COMMIT_ID={commit_id}\n")

        if cfg.get("job_max_restarts", 0) > 0:
            fn.write("export USE_ELASTIC=1\n")

        fn.write("cd ${WORKING_PATH}\n")
        if run_in_sleep:
            fn.write("sleep 1d\n")

        for cmd in getattr(cfg, "prefix_cmds_on_master", []):
            fn.write(f"{cmd}\n")

        if hasattr(cfg, "custom_cmds_before_job_list"):
            for cmd in cfg.custom_cmds_before_job_list:
                fn.write("%s\n" % cmd)
        for idx, job in enumerate(cfg.job_list):
            if cfg.launcher == "torchrun":
                job = (
                    "python3 elastic_launcher.py --job-idx %d --nnodes %s --nproc_per_node %d '%s'"
                    % (  # noqa E501
                        idx + 1,
                        cfg.num_machines,
                        cfg.num_gpus_per_machine,
                        job + " --launcher torch",
                    )
                )
            else:
                raise NotImplementedError

            fn.write("%s\n" % job)
        if hasattr(cfg, "custom_cmds_after_job_list"):
            for cmd in cfg.custom_cmds_after_job_list:
                fn.write("%s\n" % cmd)

        for cmd in getattr(cfg, "suffix_cmds_on_master", []):
            fn.write(f"{cmd}\n")

    subprocess.check_call(["chmod", "777", bash_file])
