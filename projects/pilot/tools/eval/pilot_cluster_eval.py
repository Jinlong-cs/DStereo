import argparse
import json
import os
import subprocess
import sys
import time

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
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--current-cluster",
        default=None,
        required=True,
        type=str,
        help="running cluster",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        required=False,
        default=None,
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
        default=4,
    )
    parser.add_argument(
        "--project-id",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model-setting",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--eval-data-setting", type=str, required=False, default=None
    )
    parser.add_argument(
        "--model-name-postfix", type=str, required=False, default=None
    )
    parser.add_argument(
        "--model-version",
        type=str,
        required=True,
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
        "--eval-postfix",
        type=str,
        required=False,
        default="",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
    )
    parser.add_argument("--mount-bucket", type=str, default=None)
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
        "--pipeline-test",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--job-type",
        type=str,
        default="train",
        choices=["prediction", "train"],
    )

    args = parser.parse_args()

    return args


def run_submit(
    job_name,
    config,
    model_name,
    model_type,
    model_setting,
    eval_data_setting,
    model_name_postfix,
    model_version,
    stage,
    eval_postfix,
    num_machines,
    num_gpus_per_machine,
    project_id,
    current_cluster,
    mount_bucket,
    dry_run,
    enable_tracking,
    experiment_name,
    dag_name,
    task_scene,
    model_checkpoint,
    pipeline_test,
    job_type,
):
    if task_scene in ["model_verify", "data_verify"]:
        model_version = standardized_model_version(model_version)

    k8s_config = f"{DIR_PATH}/../pilot_cluster_cfg.yaml"
    if model_type in BEVModelType.values():
        k8s_config = f"{DIR_PATH}/../pilot_bev_cluster_cfg.yaml"

    with open(k8s_config, "r") as rf:
        k8s_config = yaml.safe_load(rf)
        eval_config = k8s_config["base"]

    rel_config = os.path.relpath(config, "projects")

    cmd = (
        f"python3 tools/predict.py --config {rel_config} --stage {stage} --project-id {project_id} --device-ids "
        + ",".join(list(map(str, range(num_gpus_per_machine))))
        + f" --project-id {project_id}"
        + f" --hat-pilot-model_setting {model_setting}"
        + f" --hat-num-machines {num_machines}"
        + f" --AUTO_CV_CONN SbtKTQf/Awk2lgHlVhezZvEjwnYEbQAh4KmvDkxVhGtmDPnXaC4dF0zw+XlSV+5IuieJrfReSkIiYRK4njmRfTh1rvlU2xqyNoRHgSCdyc9QA9SYXjh/vb2ZeYF4CPh57vfSeXKnJXl8hUPFe5HFxQ=="  # noqa
    )

    if eval_data_setting:
        cmd += f" --hat-pilot-eval_data_setting {eval_data_setting}"

    if model_name_postfix:
        cmd += f" --hat-pilot-model-name-postfix {model_name_postfix}"

    if model_version:
        cmd += f" --hat-pilot-model-version {model_version}"

    if eval_postfix:
        cmd += f" --hat-pilot-prediction-name-suffix {eval_postfix}"

    if model_checkpoint:
        cmd += f" --hat-pilot-model-checkpoint {model_checkpoint}"

    if pipeline_test:
        cmd += " --pipeline-test"

    eval_name = "_".join(
        [model_name, model_setting, model_name_postfix]
        if model_name_postfix
        else [model_name, model_setting]
    )

    tracking_cmd = ""

    if enable_tracking:

        tracking_cmd = " --enable-tracking"

        cmd += tracking_cmd

        if eval_name == "_":
            eval_name = "eval_test"

        if experiment_name is None:
            experiment_name = eval_name

        if dag_name is None:
            dag_name = "dag_" + eval_name

        tracking_cmd += f" --experiment-name {experiment_name} --dag-name {dag_name} --job-name {eval_name}"

    eval_config.update(
        dict(
            job_name=job_name,
            num_machines=num_machines,
            num_gpus_per_machine=num_gpus_per_machine,
            project_id=project_id,
            job_list=[cmd],
            prefix_cmds_on_master=[
                "export HAT_PROCESS_GROUP_TIMEOUT=6000",
            ],
        )
    )
    if mount_bucket:
        eval_config["input_bucket"] = mount_bucket

    # DAG desc
    dag_project = get_project_name([[model_setting]])
    dag_desc_dict = dict(
        task_scene=f"{task_scene.upper()}",
        domain="perception",
        model_version=model_version,
        subprojects=dag_project,
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

    with open("__tmp_eval_cfg.yaml", "w") as wf:
        yaml.dump(eval_config, wf)

    assert (
        current_cluster is not None
    ), "current-cluster should be assigned"  # noqa
    cmd = [
        "cd",
        "plugins/k8s_submit",
        "&&",
        "python3",
        "submit.py",
        "--config",
        "../../__tmp_eval_cfg.yaml",
        "--cluster",
        current_cluster,
        tracking_cmd,
        desc_cmd,
        "--job-type",
        job_type,
    ]
    if not dry_run:
        subprocess.run(" ".join(cmd), shell=True, check=True)
        os.remove("__tmp_eval_cfg.yaml")
    else:
        print(" ".join(cmd))


if __name__ == "__main__":
    args = parse_full_args()

    with open(f"{DIR_PATH}/../../model_meta.yaml", "r") as rf:
        model_meta = yaml.safe_load(rf)

    model_type_meta = model_meta[args.model_type]
    args.config = model_type_meta["eval"]["entry"]

    args.model_name = model_type_meta["model_name"]

    job_name = []
    if args.job_name is not None:
        job_name.append(args.job_name)
    if args.model_setting is not None:
        job_name.append(args.model_setting)
    if args.eval_data_setting is not None:
        job_name.append(args.eval_data_setting)
    if args.model_name_postfix is not None:
        job_name.append(args.model_name_postfix)
    if args.model_version is not None:
        job_name.append(args.model_version)
    if len(job_name) > 0:
        args.job_name = "_".join(job_name)
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        args.job_name = "hat_eval_test_" + timestamp
    run_submit(**vars(args))
