import argparse
import json
import os
import subprocess
import sys

import yaml

from projects.pilot.configs.project_utils.enum import BEVModelType

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, "../"))
from production.config.project_info import (  # noqa: E402
    get_project_name,
    standardized_model_version,
)


def parse_full_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument(
        "--current-cluster",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--num-machines",
        type=int,
        required=False,
        default=1,
        help="Number of machines used to run the training",
    )
    parser.add_argument(
        "--num-gpus-per-machine",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--start-stage",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--end-stage",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--project-id",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--model-setting",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model-name-postfix",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--model-version",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--val-last-stage",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--task-scene",
        type=str,
        required=True,
        choices=[
            "model_release",
            "model_experiment",
            "data_verify",
            "model_verify",
            "test",
        ],
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
    )
    parser.add_argument("--mount-bucket", type=str, default=None)
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument(
        "--pipeline-test",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--dag-name",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--sleep",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--max-split-size",
        type=str,
        required=False,
        default=None,
    )

    args = parser.parse_args()

    return args


def run_pipeline(
    model_type,
    model_name,
    config,
    model_setting,
    model_name_postfix,
    model_version,
    task_scene,
    num_machines,
    num_gpus_per_machine,
    start_stage,
    end_stage,
    project_id,
    train_stages,
    current_cluster,
    mount_bucket,
    dry_run,
    local,
    pipeline_test,
    enable_tracking,
    experiment_name,
    dag_name,
    sleep,
    max_split_size,
    val_last_stage: bool = False,
    val_config: str = None,
):
    if task_scene in ["model_verify", "data_verify"]:
        model_version = standardized_model_version(model_version)

    k8s_config = f"{DIR_PATH}/../pilot_cluster_cfg.yaml"
    if model_type in BEVModelType.values():
        k8s_config = f"{DIR_PATH}/../pilot_bev_cluster_cfg.yaml"

    with open(k8s_config, "r") as rf:
        k8s_config = yaml.safe_load(rf)
        train_config = k8s_config["base"]

    # generate job name
    names = [model_name, model_setting, model_version]
    if model_name_postfix is not None:
        names.append(model_name_postfix)
    job_name = "_".join(names)

    train_cmd_pattern = (
        "python3 tools/train.py --config {} --stage {} --device-ids "  # noqa
        + ",".join(list(map(str, range(num_gpus_per_machine))))
        + f" --hat-pilot-model-setting {model_setting}"
        + f" --hat-num-machines {num_machines}"
    )

    if model_name_postfix is not None:
        train_cmd_pattern += (
            f" --hat-pilot-model-name-postfix {model_name_postfix}"
        )
    if model_version is not None:
        train_cmd_pattern += f" --hat-pilot-model-version {model_version}"
    if max_split_size is not None:
        train_cmd_pattern += (
            f" --PYTORCH_CUDA_ALLOC_CONF max_split_size_mb:{max_split_size}"
        )
    if pipeline_test:
        train_cmd_pattern += " --pipeline-test"

    tracking_cmd = ""
    if enable_tracking:
        tracking_cmd = " --enable-tracking"
        if experiment_name is None:
            experiment_name = job_name
        if dag_name is None:
            dag_name = "dag_" + job_name

        tracking_cmd += "".join(
            [
                f" --experiment-name {experiment_name}",
                f" --dag-name {dag_name}",
                f" --job-name {job_name}",
            ]
        )
        train_cmd_pattern += tracking_cmd

    if start_stage is not None:
        idx = train_stages.index(start_stage)
        train_stages = train_stages[idx:]

    if end_stage is not None:
        idx = train_stages.index(end_stage)
        train_stages = train_stages[: idx + 1]

    # DAG desc
    dag_project = get_project_name([[model_setting]])
    exp_label = []
    if pipeline_test:
        exp_label.append("pipeline_test")
    exp_info = dict(label=exp_label)
    dag_desc_dict = dict(
        task_scene=f"{task_scene.upper()}",
        domain="perception",
        model_version=model_version,
        subprojects=dag_project,
        exp_info=exp_info,
    )
    # JOB desc
    job_desc_dict = dict(
        model_type=model_type,
        extra_name=model_name_postfix,
        extra_info=model_setting,
    )

    dag_desc = json.dumps(dag_desc_dict)
    job_desc = json.dumps(job_desc_dict)

    assert (
        len(dag_desc) <= 255
    ), "len(dag_desc) should <= 255, please delete some dag info"
    assert (
        len(job_desc) <= 255
    ), "len(job_desc) should <= 255, please delete some job info"

    desc_cmd = f" --dag-desc '{dag_desc}'" + f" --job-desc '{job_desc}'"

    job_command = ""
    if sleep:
        job_command += " --sleep"

    if local:
        for stage in train_stages:
            cmd = train_cmd_pattern.format(config, stage)
            print(cmd)
            if not dry_run:
                subprocess.check_call(cmd, shell=True)
        if val_last_stage and val_config:
            val_cmd = build_val_cmd(
                project_id=project_id,
                num_machines=num_machines,
                num_gpus_per_machine=num_gpus_per_machine,
                config=val_config,
                model_setting=model_setting,
                stage=train_stages[-1],
                model_checkpoint=f"tmp_output/{model_name}/{train_stages[-1]}-checkpoint-last.pth.tar",  # noqa
                model_version=model_version,
                model_name_postfix=model_name_postfix,
                pipeline_test=pipeline_test,
                enable_tracking=enable_tracking,
            )
            print(val_cmd)
            if not dry_run:
                subprocess.check_call(val_cmd, shell=True)
    else:
        assert job_name is not None, "job-name should be assigned"
        assert project_id is not None, "project-id should be assigned"

        rel_config = os.path.relpath(config, "projects")

        job_list = [
            train_cmd_pattern.format(rel_config, stage)
            for stage in train_stages
        ]

        if val_last_stage and val_config:
            val_cmd = build_val_cmd(
                project_id=project_id,
                num_machines=num_machines,
                num_gpus_per_machine=num_gpus_per_machine,
                config=os.path.relpath(val_config, "projects"),
                model_setting=model_setting,
                stage=train_stages[-1],
                model_checkpoint=f"/job_data/models/{model_name}/{train_stages[-1]}-checkpoint-last.pth.tar",  # noqa
                model_version=model_version,
                model_name_postfix=model_name_postfix,
                pipeline_test=pipeline_test,
                enable_tracking=enable_tracking,
            )
            job_list.append(val_cmd)

        train_config.update(
            dict(
                job_name=job_name,
                num_machines=num_machines,
                num_gpus_per_machine=num_gpus_per_machine,
                project_id=project_id,
                job_list=job_list,
                prefix_cmds_on_master=[
                    "export HAT_PROCESS_GROUP_TIMEOUT=6000",
                ],
            )
        )
        if mount_bucket:
            train_config["input_bucket"] = mount_bucket

        with open("__tmp_train_cfg.yaml", "w") as wf:
            yaml.dump(train_config, wf)

        cmd = [
            "cd",
            "plugins/k8s_submit",
            "&&",
            "python3",
            "submit.py",
            "--config",
            "../../__tmp_train_cfg.yaml",
            "--cluster",
            current_cluster,
            tracking_cmd,
            desc_cmd,
            job_command,
        ]
        if not dry_run:
            subprocess.run(" ".join(cmd), shell=True, check=True)
            os.remove("__tmp_train_cfg.yaml")
        else:
            print(" ".join(cmd))


def build_val_cmd(
    project_id: str,
    num_machines: int,
    num_gpus_per_machine: int,
    config: str,
    model_setting: str,
    stage: str,
    model_checkpoint: str,
    model_version: str = None,
    model_name_postfix: str = None,
    pipeline_test: bool = False,
    enable_tracking: bool = False,
):
    cmd = (
        "python3 tools/predict.py"
        + f" --config {config}"
        + f" --stage {stage}"
        + f" --project-id {project_id}"
        + f" --device-ids {','.join(map(str, range(num_gpus_per_machine)))}"
        + f" --project-id {project_id}"
        + f" --hat-pilot-model_setting {model_setting}"
        + f" --hat-num-machines {num_machines}"
        + f" --auto-cv-conn SbtKTQf/Awk2lgHlVhezZvEjwnYEbQAh4KmvDkxVhGtmDPnXaC4dF0zw+XlSV+5IuieJrfReSkIiYRK4njmRfTh1rvlU2xqyNoRHgSCdyc9QA9SYXjh/vb2ZeYF4CPh57vfSeXKnJXl8hUPFe5HFxQ=="  # noqa
    )

    if model_name_postfix:
        cmd += f" --hat-pilot-model-name-postfix {model_name_postfix}"

    if model_version:
        cmd += f" --hat-pilot-model-version {model_version}"

    if model_checkpoint:
        cmd += f" --hat-pilot-model-checkpoint {model_checkpoint}"

    if pipeline_test:
        cmd += " --pipeline-test"

    if enable_tracking:
        cmd += " --enable-tracking"

    return cmd


if __name__ == "__main__":

    args = parse_full_args()
    with open(f"{DIR_PATH}/../../model_meta.yaml", "r") as rf:
        model_meta = yaml.safe_load(rf)
    model_type_meta = model_meta[args.model_type]

    args.model_name = model_type_meta["model_name"]
    args.config = model_type_meta["train"]["entry"]
    args.train_stages = model_type_meta["train"]["stages"]
    args.val_config = model_type_meta["eval"]["entry"]

    run_pipeline(**vars(args))
