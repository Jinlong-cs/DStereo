import argparse
import os
import subprocess
import time

import yaml

from projects.pilot.configs.project_utils.enum import BEVSubPorject


def run_submit(
    sub_project,
    version,
    fillback_type,
    app,
    update_hbm,
    disable_eval,
    enable_report_diff,
    project_id,
    current_cluster,
    fillback_cluster,
):
    __dir = os.path.dirname(__file__)
    k8s_config = f"{__dir}/../pilot_cluster_cfg.yaml"
    if sub_project in BEVSubPorject.values():
        k8s_config = f"{__dir}/../pilot_bev_cluster_cfg.yaml"

    with open(k8s_config) as f:
        k8s_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        cluster_config = k8s_config["base"]
        cluster_config["docker_image"] = cluster_config["docker_image_cpu"]
        cluster_config.update(k8s_config["fillback_eval"])

    _cmd = "python3 pilot/tools/fillback_eval/pipeline.py "
    _cmd += " ".join(
        [
            "--sub-project",
            sub_project,
            "--version",
            version,
            "--fillback-type",
            fillback_type,
        ]
    )
    if app is not None:
        _cmd += f" --app {app}"
    if update_hbm is not None:
        _cmd += f" --update-hbm {update_hbm}"
    if disable_eval:
        _cmd += " --disable-eval"
    if fillback_cluster is not None:
        _cmd += f" --fillback-cluster {fillback_cluster}"
    if enable_report_diff:
        _cmd += " --enable-report-diff"
    print(f"Run script on cluster: \n {_cmd}")
    cluster_config.update(
        dict(  # noqa: C408
            job_name="_".join(
                [
                    "fillback_eval_" + sub_project,
                    version,
                    time.strftime("%Y%m%d%H%M", time.localtime()),
                ]
            ),
            project_id=project_id,
            job_list=[_cmd],
        )
    )
    with open("__tmp_train_cfg.yaml", "w") as wf:
        yaml.dump(cluster_config, wf)

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
        "--job-type",
        "app_eval",
        "--cluster",
        current_cluster,
    ]

    subprocess.run(" ".join(cmd), shell=True, check=True)
    os.remove("__tmp_train_cfg.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sub-project", type=str, required=True)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--fillback-type", type=str, required=True)
    parser.add_argument("--app", type=str, default=None)
    parser.add_argument("--update-hbm", type=str, default=None)
    parser.add_argument("--disable-eval", action="store_true", default=False)
    parser.add_argument(
        "--enable-report-diff", action="store_true", default=False
    )
    parser.add_argument(
        "--fillback-cluster", type=str, required=False, default=None
    )
    parser.add_argument("--project-id", type=str, required=True)
    parser.add_argument("--current-cluster", type=str, required=True)
    args = parser.parse_args()

    run_submit(
        args.sub_project,
        args.version,
        args.fillback_type,
        args.app,
        args.update_hbm,
        args.disable_eval,
        args.enable_report_diff,
        args.project_id,
        args.current_cluster,
        args.fillback_cluster,
    )
