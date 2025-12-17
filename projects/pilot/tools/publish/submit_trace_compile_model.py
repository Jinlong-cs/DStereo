import argparse
import os
import subprocess

import yaml

from projects.pilot.configs.project_utils.enum import BEVSubPorject


def run_submit(
    sub_project,
    publish_version,
    overwrite,
    project_id,
    current_cluster,
    compile_mode,
    release: bool = False,
):

    k8s_config = f"{os.path.dirname(__file__)}/../pilot_cluster_cfg.yaml"
    if sub_project in BEVSubPorject.values():
        k8s_config = (
            f"{os.path.dirname(__file__)}/../pilot_bev_cluster_cfg.yaml"
        )
    with open(k8s_config) as f:
        k8s_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        compile_config = k8s_config["base"]
        if "compile" in k8s_config:
            compile_config.update(k8s_config["compile"])

    cmd_pattern = (
        "python3 pilot/dev/publish/trace_compile_model.py"
        + f" --sub-project {sub_project} --publish-version {publish_version}"
        + f" --compile-mode {compile_mode}"
    )

    if overwrite:
        cmd_pattern += " --overwrite"
    if release:
        cmd_pattern += " --release"
    job_name = [
        "pub_" + sub_project,
        publish_version,
        "model_trace_and_compile",
    ]
    if release:
        job_name.append("release")
    compile_config.update(
        dict(  # noqa: C408
            job_name="_".join(job_name),
            project_id=project_id,
            job_list=[cmd_pattern],
        )
    )
    print(compile_config)
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
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--project-id", type=str, required=True)
    parser.add_argument("--current-cluster", type=str, required=True)
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="aidi",
        help="choose compile in aidi or local, default is local.",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="whether is release model",
    )
    args = parser.parse_args()

    run_submit(
        args.sub_project,
        args.publish_version,
        args.overwrite,
        args.project_id,
        args.current_cluster,
        args.compile_mode,
        args.release,
    )
