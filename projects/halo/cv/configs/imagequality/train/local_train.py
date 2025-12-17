import argparse
import os
import subprocess

from hatbc.utils import Enum

try:
    from hdflow.data.anno_data_manage.utils import EnumAction
    from hdflow.tagging.tags.environment import Time
    from hdflow.tagging.tags.project import Name as ProjectName

    hdflow_err_msg = None
except ImportError as e:
    hdflow_err_msg = e


def parse_full_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train-stages",
        type=str,
        required=True,
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

    return parser.parse_args()


def set_local_env(
    model_version,
    model_setting,
    num_machines,
    project_name,
    work_condition_time,
):
    def _set_env(env_name, env_val):
        if env_val is not None:
            if isinstance(env_val, Enum):
                os.environ[env_name] = str(env_val.value)
            else:
                os.environ[env_name] = str(env_val)
        else:
            os.environ.pop(env_name, None)

    _set_env("HAT_PILOT_MODEL_VERSION", model_version)
    _set_env("HAT_PILOT_MODEL_SETTING", model_setting)
    _set_env("HAT_NUM_MACHINES", num_machines)
    _set_env("HAT_TRAIN_PROJECT_NAME", project_name)
    _set_env("HAT_TRAIN_WORK_CONDITION_TIME", work_condition_time)


def run_local_pipeline_test(config, train_stages):
    train_cmd_pattern = "python36 -W ignore tools/train.py --config {} --stage {} --pipeline-test"  # noqa
    cmds = [train_cmd_pattern.format(config, stage) for stage in train_stages]
    for cmd in cmds:
        print(cmd)
        subprocess.check_call(cmd, shell=True)


def check_args(
    model_setting,
    project_name,
    work_condition_time,
):
    if model_setting is None:
        if project_name is None and work_condition_time is None:
            model_setting = "all"


def run_local(
    config,
    model_setting,
    model_version,
    train_stages,
    project_name=None,
    work_condition_time=None,
):
    check_args(
        model_setting,
        project_name,
        work_condition_time,
    )
    train_stages = [stage.strip() for stage in train_stages.split(",")]
    num_machines = 1
    set_local_env(
        model_version,
        model_setting,
        num_machines,
        project_name,
        work_condition_time,
    )
    run_local_pipeline_test(config, train_stages)


if __name__ == "__main__":
    args = parse_full_args()
    run_local(**vars(args))
