import argparse
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
        "--pretrain-model-version",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--mount-bucket",
        type=str,
        default="adas,mono,auto_eval",
    )
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
    # 编译模型
    parser.add_argument(
        "--trace_compile_model",
        action="store_true",
        default=True,
    )
    parser.add_argument("--sub-project", type=str, default="tl")
    parser.add_argument("--overwrite", action="store_true", default=True)
    # 模型评测
    parser.add_argument("--eval_online", action="store_true", default=False)

    args = parser.parse_args()

    return args


def run_pipeline(
    model_name,
    config,
    model_setting,
    model_name_postfix,
    model_version,
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
    trace_compile_model,
    sub_project,
    overwrite,
    eval_online,
    pretrain_model_version,
):

    k8s_config = f"{DIR_PATH}/../mono_tl_cluster_cfg.yaml"

    with open(k8s_config, "r") as rf:
        k8s_config = yaml.safe_load(rf)
        train_config = k8s_config["base"]
        if "train" in k8s_config:
            train_config.update(k8s_config["train"])
    if pipeline_test:
        train_config.update(
            max_jobtime=60,
        )

    train_cmd_pattern = (
        "python3 -W ignore tools/train.py --config {} --stage {} --device-ids "  # noqa
        + ",".join(list(map(str, range(num_gpus_per_machine))))
        + f" --hat-tl-model-setting {model_setting}"
        + f" --hat-num-machines {num_machines}"
    )

    if model_name_postfix is not None:
        train_cmd_pattern += (
            f" --hat-tl-model-name-postfix {model_name_postfix}"
        )
    if model_version is None:
        model_version = f"{DIR_PATH}".split("/")[-3]
    train_cmd_pattern += f" --hat-tl-model-version {model_version}"

    # generate job name
    names = [model_name, model_setting, model_version]
    if model_name_postfix is not None:
        names.append(model_name_postfix)
    job_name = "_".join(names)

    if pipeline_test:
        train_cmd_pattern += " --pipeline-test"

    if enable_tracking:
        train_cmd_pattern += " --enable-tracking"

    if start_stage is not None:
        idx = train_stages.index(start_stage)
        train_stages = train_stages[idx:]

    if end_stage is not None:
        idx = train_stages.index(end_stage)
        train_stages = train_stages[: idx + 1]

    if local:
        for stage in train_stages:
            cmd = train_cmd_pattern.format(config, stage)
            print(cmd)
            if not dry_run:
                subprocess.check_call(cmd, shell=True)
    else:
        assert job_name is not None, "job-name should be assigned"
        assert project_id is not None, "project-id should be assigned"

        # rel_config = os.path.relpath(config, "projects")
        rel_config = config

        # 添加环境变量
        job_list = [
            f"export HAT_TL_MODEL_VERSION={model_version}",
        ]
        if pretrain_model_version is not None:
            job_list = [
                f"export HAT_TL_PRETRAIN_MODEL_VERSION={pretrain_model_version}"  # noqa
            ]

        #  添加训练cmd
        job_list += [
            train_cmd_pattern.format(rel_config, stage)
            for stage in train_stages
        ]
        if trace_compile_model:
            compile_cmd_pattern = (
                f"python3 {DIR_PATH}/../../dev/publish/trace_compile_model.py"
                + f" --sub-project {sub_project} --publish-version {model_version}"  # noqa
                + " --compile-mode aidi"
                + f" --name-postfix {model_name_postfix}"
            )
            if overwrite:
                compile_cmd_pattern += " --overwrite"
            job_list += [compile_cmd_pattern]

        # 添加评测cmd
        if eval_online:
            eval_cmd_pattern = [
                f"export HAT_TL_PREDICTION_NAME_SUFFIX=sd_{model_version}"
            ]
            eval_cmd_pattern += [
                (
                    "python3 -W ignore tools/predict.py"
                    + " --config {DIR_PATH}/../../config/traffic_light/eval_multitask.py"  # noqa
                    + " --stage freeze_bn_2 --device-ids "
                    + ",".join(list(map(str, range(num_gpus_per_machine))))
                    + " --hat-tl-model-setting tl_cn_2pe_day_multitask"
                    + " --hat-num-machines 1"
                )
            ]
            job_list += eval_cmd_pattern
            # job_list = job_list[:1] + eval_cmd_pattern

        train_config.update(
            {
                "job_name": job_name,
                "num_machines": num_machines,
                "num_gpus_per_machine": num_gpus_per_machine,
                "project_id": project_id,
                "input_bucket": mount_bucket,
                "job_list": job_list,
            }
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
        ]
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
    args.config = os.path.join(
        os.path.dirname(__file__), model_type_meta["train"]["entry"]
    )

    args.train_stages = model_type_meta["train"]["stages"]

    del args.model_type

    run_pipeline(**vars(args))
