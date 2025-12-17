import argparse
import os
import subprocess

import yaml

try:
    from hdflow.data.anno_data_manage.utils import EnumAction
    from hdflow.tagging.tags.environment import Time
    from hdflow.tagging.tags.project import Name as ProjectName

    hdflow_err_msg = None
except ImportError as e:
    hdflow_err_msg = e

from local_train import check_args, run_local

k8s_config = f"{os.path.dirname(__file__)}/cluster_cfg.yaml"  # noqa
with open(k8s_config) as f:
    k8s_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    train_config = k8s_config["base"]
    if "train" in k8s_config:
        train_config.update(k8s_config["train"])


def parse_full_args(config=None, train_stages=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True if config is None else False,
    )
    parser.add_argument(
        "--current-cluster",
        default=None,
        type=str,
        help="[traincli option] If None, use default in yaml",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        required=False,
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
        "--project-id",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--train-stages",
        type=str,
        required=True if train_stages is None else False,
    )
    parser.add_argument(
        "--model-setting",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--model-version",
        type=str,
        required=False,
        default="test",
    )
    if hdflow_err_msg is None:
        parser.add_argument(
            "--project-name",
            type=ProjectName,
            required=False,
            default=None,
            action=EnumAction,
            help="the project name for training",
        )
        parser.add_argument(
            "--work-condition-time",
            type=Time,
            required=False,
            default=None,
            action=EnumAction,
            help="the work condition time for training",
        )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
    )
    parser.add_argument("--mount-bucket", type=str, default="MultiMode_2")
    parser.add_argument(
        "--pipeline-test",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    if config is not None:
        args.config = config
    if train_stages is not None:
        args.train_stages = train_stages
    return args


def get_model_setting(args):
    if args.model_setting is None:
        model_settings = []
        if args.project_name is not None:
            model_settings.append(args.project_name.value)
        if args.work_condition_time is not None:
            model_settings.append(args.work_condition_time.value)
        if len(model_settings) > 0:
            return "_".join(model_settings)
        else:
            return "all"
    else:
        return args.model_setting


def run_submit(
    job_name,
    config,
    model_setting,
    model_version,
    num_machines,
    num_gpus_per_machine,
    project_id,
    train_stages,
    current_cluster,
    mount_bucket,
    dry_run,
    project_name=None,
    work_condition_time=None,
    pipeline_test=False,
):
    if pipeline_test:
        return run_local(
            config,
            model_setting,
            model_version,
            train_stages,
            project_name,
            work_condition_time,
        )
    check_args(
        model_setting,
        project_name,
        work_condition_time,
    )
    envs = dict(  # noqa
        HAT_PILOT_MODEL_VERSION=model_version,
        HAT_NUM_MACHINES=num_machines,
        HAT_PILOT_MODEL_SETTING=model_setting,
        HDFLOW_ENABLE_AUTO_DP_DEBUG_LOGGING="0",
        # for auto cv database connection
        AUTO_CV_CONN="SbtKTQf/Awk2lgHlVhezZvEjwnYEbQAh4KmvDkxVhGtmDPnXaC4dF0zw+XlSV+5IuieJrfReSkIiYRK4njmRfTh1rvlU2xqyNoRHgSCdyc9QA9SYXjh/vb2ZeYF4CPh57vfSeXKnJXl8hUPFe5HFxQ==",  # noqa
    )

    if hdflow_err_msg is None:
        if project_name is not None:
            envs.update(
                dict(  # noqa
                    HAT_TRAIN_PROJECT_NAME=project_name.value,
                )
            )
        if work_condition_time is not None:
            envs.update(
                dict(  # noqa
                    HAT_TRAIN_WORK_CONDITION_TIME=work_condition_time.value,
                )
            )

    train_cmd_pattern = (
        "python3 -W ignore tools/train.py --config {} --stage {} --device-ids "  # noqa
        + ",".join(list(map(str, range(num_gpus_per_machine))))
    )
    train_stages = [stage.strip() for stage in train_stages.split(",")]
    rel_config = os.path.relpath(config, "projects")

    assert job_name is not None, "job-name should be assigned"
    assert project_id is not None, "project-id should be assigned"
    train_config.update(
        dict(  # noqa
            job_name=job_name,
            num_machines=num_machines,
            num_gpus_per_machine=num_gpus_per_machine,
            project_id=project_id,
            envs=envs,
            input_bucket=mount_bucket,
            job_list=[
                train_cmd_pattern.format(rel_config, stage)
                for stage in train_stages
            ],
        )
    )

    with open("__tmp_train_cfg.yaml", "w") as wf:
        yaml.dump(train_config, wf)

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
        "../../__tmp_train_cfg.yaml",
        "--cluster",
        current_cluster,
    ]
    if not dry_run:
        subprocess.run(" ".join(cmd), shell=True, check=True)
        os.remove("__tmp_train_cfg.yaml")
    else:
        print(" ".join(cmd))


if __name__ == "__main__":
    args = parse_full_args()
    run_submit(**vars(args))
