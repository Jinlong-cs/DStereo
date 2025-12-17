import argparse
import os
import subprocess

import yaml

DIR_PATH = os.path.dirname(__file__)


def parse_full_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", type=str, required=True)
    parser.add_argument(
        "--current-cluster",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--num-gpus-per-machine",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--aidi-eval",
        type=str,
        default=False,
    )
    parser.add_argument(
        "--temporal-eval",
        type=str,
        default=False,
    )
    parser.add_argument(
        "--eval-dataset-id",
        type=int,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--sleep",
        dest="sleep",
        action="store_true",
        help="sleep in cluster for debug",
    )
    parser.add_argument(
        "--local",
        dest="local_run",
        action="store_true",
        help="run in local",
    )

    args = parser.parse_args()

    return args


def run_pipeline(
    model_version,
    num_gpus_per_machine,
    current_cluster,
    val_batch_size,
    aidi_eval,
    temporal_eval,
    eval_dataset_id,
    sleep,
    local_run,
):
    k8s_config = f"{DIR_PATH}/../cluster_cfg.yaml"
    with open(k8s_config, "r") as rf:
        k8s_config = yaml.safe_load(rf)
        cluster_config = k8s_config["base"]
    job_list = []
    # virtual_envs
    perf_job_list = [
        "pip3 install hatbc==0.10.0b202309070700+b39a57d",
        "virtualenv --python=python3 py3.8",
        "source py3.8/bin/activate",
        "make run-env-cu111-torch1102",
        "cd hat/models/task_modules/sparse4d/ops",
        "python3 setup.py develop",
        "cd ../../../../../",
        "deactivate",
    ]
    # envs
    if aidi_eval is not None:
        job_list += [f"export aidi_eval={aidi_eval}"]
    if val_batch_size is not None:
        job_list += [f"export val_batch_size={val_batch_size}"]
    if temporal_eval is not None:
        job_list += [f"export temporal_eval={temporal_eval}"]
    if eval_dataset_id is not None:
        job_list += [f"export eval_dataset_id={eval_dataset_id}"]
    eval_config = os.path.join(
        DIR_PATH, f"../model_configs/{model_version}/entry.py"
    )
    eval_cmd_pattern = (
        f"python3 -W ignore tools/predict.py --config {eval_config} "
        + "--stage float --device-ids "
        + ",".join(list(map(str, range(num_gpus_per_machine))))
    )
    job_list += [eval_cmd_pattern]

    if local_run:
        os.system(eval_cmd_pattern)
    else:
        job_list = perf_job_list + job_list
        cluster_config.update(
            {
                "job_name": f"hat_job_{model_version}",
                "num_machines": 1,
                "num_gpus_per_machine": num_gpus_per_machine,
                "job_list": job_list,
            }
        )

        with open("__tmp_train_cfg.yaml", "w") as wf:
            yaml.dump(cluster_config, wf)

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
        if sleep:
            cmd += ["--sleep"]
        print(cmd)
        subprocess.run(" ".join(cmd), shell=True, check=True)
        os.remove("__tmp_train_cfg.yaml")


if __name__ == "__main__":

    args = parse_full_args()

    run_pipeline(**vars(args))
