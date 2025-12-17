import argparse
import os
import subprocess

import yaml

k8s_config = f"{os.path.dirname(__file__)}/../mono_tl_cluster_cfg.yaml"  # noqa
with open(k8s_config) as f:
    k8s_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    compile_config = k8s_config["base"]
    if "compile" in k8s_config:
        compile_config.update(k8s_config["compile"])


def run_submit(
    sub_project,
    publish_version,
    overwrite,
    project_id,
    current_cluster,
):
    DIR_PATH = os.path.dirname(__file__)
    cmd_pattern = (
        f"python3 {DIR_PATH}/../../dev/publish/trace_compile_model.py"
        + f" --sub-project {sub_project} --publish-version {publish_version}"
        + " --compile-mode aidi"
    )

    if overwrite:
        cmd_pattern += " --overwrite"
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

    parser.add_argument("--sub-project", type=str, default="tl")
    parser.add_argument("--publish-version", type=str, default="v0.1.0")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--project-id", type=str, default="PDT2021004-vision")
    parser.add_argument(
        "--current-cluster", type=str, default="idc-share-titanxp-8"
    )
    args = parser.parse_args()

    run_submit(
        args.sub_project,
        args.publish_version,
        args.overwrite,
        args.project_id,
        args.current_cluster,
    )
