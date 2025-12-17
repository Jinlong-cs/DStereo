import argparse
import os
import subprocess
import time

import yaml

DIR_PATH = os.path.dirname(__file__)


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
        "--model-name-postfix", type=str, required=False, default=None
    )
    parser.add_argument(
        "--model-version",
        type=str,
        required=True,
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
    parser.add_argument("--mount-bucket", type=str, default="matrix,mono,adas")

    args = parser.parse_args()

    return args


def run_submit(
    job_name,
    config,
    model_setting,
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
):

    k8s_config = f"{os.path.dirname(__file__)}/../mono_wk_cluster_cfg.yaml"

    with open(k8s_config, "r") as rf:
        k8s_config = yaml.safe_load(rf)
        eval_config = k8s_config["base"]

    # rel_config = os.path.relpath(config, "projects")
    rel_config = config

    cmd = (
        f"python3 -W ignore tools/predict.py --config {rel_config} "
        + f"--stage {stage} --project-id {project_id} --device-ids "
        + ",".join(list(map(str, range(num_gpus_per_machine))))
        + f" --hat-wk-model_setting {model_setting}"
        + f" --hat-num-machines {num_machines}"
        + f" --AUTO_CV_CONN SbtKTQf/Awk2lgHlVhezZvEjwnYEbQAh4KmvDkxVhGtmDPnXaC4dF0zw+XlSV+5IuieJrfReSkIiYRK4njmRfTh1rvlU2xqyNoRHgSCdyc9QA9SYXjh/vb2ZeYF4CPh57vfSeXKnJXl8hUPFe5HFxQ=="  # noqa
    )

    if model_name_postfix:
        cmd += f" --hat-wk-model-name-postfix {model_name_postfix}"

    if model_version:
        cmd += f" --hat-wk-model-version {model_version}"

    if eval_postfix:
        cmd += f" --hat-wk-prediction-name-suffix {eval_postfix}"

    eval_model_checkpoint = os.getenv("HAT_WK_MODEL_CHECKPOINT", None)
    prediction_name = os.getenv("HAT_WK_PREDICTION_NAME_SUFFIX", None)
    # 添加环境变量
    job_list = [
        f"export HAT_WK_MODEL_VERSION={model_version}",
        f"export HAT_TRAINING_STEP={stage}",
        f"export HAT_WK_MODEL_NAME_POSTFIX={model_name_postfix}",
        f"export HAT_WK_PREDICTION_NAME_SUFFIX={prediction_name}",
    ]
    if eval_model_checkpoint is not None:
        job_list += [f"export HAT_WK_MODEL_CHECKPOINT={eval_model_checkpoint}"]
    # 添加评测脚本
    job_list += [cmd]
    eval_config.update(
        dict(
            job_name=job_name,
            num_machines=num_machines,
            num_gpus_per_machine=num_gpus_per_machine,
            project_id=project_id,
            input_bucket=mount_bucket,
            job_list=job_list,
        )
    )
    with open("__tmp_train_cfg.yaml", "w") as wf:
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
    args.config = model_type_meta["eval"]["entry"]

    del args.model_type

    job_name = []
    if args.job_name is not None:
        job_name.append(args.job_name)
    if args.model_setting is not None:
        job_name.append(args.model_setting)
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
