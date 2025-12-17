import argparse
import os
import subprocess

import yaml

from projects.pilot.configs.project_utils.enum import BEVSubPorject


def run_submit(
    sub_project,
    publish_version,
    project_id,
    current_cluster,
):
    __dir = os.path.dirname(__file__)
    k8s_config = f"{__dir}/../pilot_cluster_cfg.yaml"
    if sub_project in BEVSubPorject.values():
        k8s_config = f"{__dir}/../pilot_bev_cluster_cfg.yaml"

    with open(k8s_config) as f:
        k8s_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        compile_config = k8s_config["base"]
        if "compile" in k8s_config:
            compile_config.update(k8s_config["compile"])

    tag = "-".join(["refs/tags/pilot", sub_project, publish_version])
    setup_git_pattern = f"export gitlabTargetBranch={tag}"
    setup_path_pattern = "export PILOTPATH=."
    cmd_pattern = "bash pilot/dev/publish/publish.sh"

    compile_config.update(
        dict(  # noqa: C408
            job_name="_".join(
                [
                    "pub_" + sub_project,
                    publish_version,
                    "model_trace_and_compile",
                ]
            ),
            project_id=project_id,
            job_list=[setup_git_pattern, setup_path_pattern, cmd_pattern],
        )
    )
    with open("__tmp_train_cfg.yaml", "w") as wf:
        yaml.dump(compile_config, wf)

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

    subprocess.run(" ".join(cmd), shell=True, check=True)
    os.remove("__tmp_train_cfg.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sub-project", type=str, required=True)
    parser.add_argument("--publish-version", type=str, required=True)
    parser.add_argument("--project-id", type=str, required=True)
    parser.add_argument("--current-cluster", type=str, required=True)
    args = parser.parse_args()

    run_submit(
        args.sub_project,
        args.publish_version,
        args.project_id,
        args.current_cluster,
    )
