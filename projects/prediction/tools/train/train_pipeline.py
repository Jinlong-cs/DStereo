import argparse
import json
import os
import subprocess

import yaml

DIR_PATH = os.path.dirname(__file__)


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
        "--task-scene",
        type=str,
        required=False,
        default="model_experiment",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
    )
    parser.add_argument("--mount-bucket", type=str, default="matrix")
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
        dest="sleep",
        action="store_true",
        help="sleep in cluster for debug",
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
    max_jobtime,
    sleep,
):

    k8s_config = f"{DIR_PATH}/../cluster_cfg.yaml"

    with open(k8s_config, "r") as rf:
        k8s_config = yaml.safe_load(rf)
        train_config = k8s_config["base"]

    # generate job name
    names = [model_name, model_setting]
    if model_name_postfix is not None:
        names.append(model_name_postfix)
    job_name = "_".join(names)

    train_cmd_pattern = (
        "python3 -W ignore tools/train.py --config {} --stage {} --device-ids "  # noqa
        + ",".join(list(map(str, range(num_gpus_per_machine))))
        # + f" --hat-pred-model-setting {model_setting}"
        + f" --hat-num-machines {num_machines}"
    )

    if model_name_postfix is not None:
        train_cmd_pattern += (
            f" --hat-pred-model-name-postfix {model_name_postfix}"
        )
    if model_version is not None:
        train_cmd_pattern += f" --hat-pred-model-version {model_version}"

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
    dag_desc_dict = {
        "task_scene": f"{task_scene.upper()}",
        "domain": "perception",
        "model_version": model_version,
        "model_setting": model_setting,
    }

    # JOB desc
    job_desc_dict = {
        "model_type": model_type,
        "extra_name": model_name_postfix,
    }

    dag_desc = json.dumps(dag_desc_dict)
    job_desc = json.dumps(job_desc_dict)

    desc_cmd = ""

    max_length = 255

    if len(dag_desc) <= max_length:
        desc_cmd += f" --dag-desc '{dag_desc}'"

    if len(job_desc) <= max_length:
        desc_cmd += f" --job-desc '{job_desc}'"

    if local:
        for stage in train_stages:
            cmd = train_cmd_pattern.format(config, stage)
            if not dry_run:
                subprocess.check_call(cmd, shell=True)
    else:
        assert job_name is not None, "job-name should be assigned"
        assert project_id is not None, "project-id should be assigned"
        rel_config = os.path.relpath(config, "projects")
        job_list = [
            train_cmd_pattern.format(rel_config, stage)
            for stage in train_stages
        ]
        print(job_list)
        train_config.update(
            dict(
                job_name=job_name,
                num_machines=num_machines,
                num_gpus_per_machine=num_gpus_per_machine,
                project_id=project_id,
                # input_bucket=mount_bucket,
                job_list=job_list,
                max_jobtime=max_jobtime,
            )
        )

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
        ]

        if sleep:
            cmd.append("--sleep")

        if not dry_run:
            subprocess.run(" ".join(cmd), shell=True, check=True)
            os.remove("__tmp_train_cfg.yaml")
        else:
            print(" ".join(cmd))


if __name__ == "__main__":

    args = parse_full_args()
    with open(f"{DIR_PATH}/../../model_meta.yaml", "r") as rf:
        model_meta = yaml.safe_load(rf)
    model_type_meta = model_meta[args.model_type]

    args.model_name = model_type_meta["model_name"]
    args.config = model_type_meta["train"]["entry"]
    args.train_stages = model_type_meta["train"]["stages"]
    args.num_machines = model_type_meta["train"]["resource"]["num_machines"]
    args.num_gpus_per_machine = model_type_meta["train"]["resource"][
        "num_gpus_per_machine"
    ]
    args.max_jobtime = model_type_meta["train"]["resource"]["max_jobtime"]

    run_pipeline(**vars(args))
