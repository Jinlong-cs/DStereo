import argparse
import json
import os
import subprocess

import yaml

from hat.utils.config import Config
from projects.pilot.configs.project_utils.enum import (
    BEVModelName,
    BEVModelSetting,
    BEVModelType,
    BEVSubPorject,
    version_matcher,
)
from projects.pilot.tools.production.config.project_info import (
    get_project_name,
)

DIR_PATH = os.path.dirname(os.path.abspath(__file__))


def parse_full_args():
    parser = argparse.ArgumentParser()

    # model configs
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--model-setting", type=str, required=True)
    parser.add_argument("--model-version", type=str, required=True)
    parser.add_argument(
        "--model-name-postfix", type=str, required=False, default=None
    )

    # checkpoint config
    parser.add_argument("--model-checkpoint", type=str, default=None)
    parser.add_argument("--model-thresh", type=str, default=None)
    parser.add_argument("--load-publish", action="store_true", default=False)

    # pack configs
    parser.add_argument("--pack-consist", action="store_true", default=False)
    parser.add_argument("--pack-viz", action="store_true", default=False)
    parser.add_argument("--multiview-pack", type=str, default=None)
    parser.add_argument("--bev-pack", type=str, default=None)
    parser.add_argument("--homo-offset", type=str, default=None)
    parser.add_argument("--temporal-hommo-offset", type=str, default=None)

    # visualize configs
    parser.add_argument("--to-video", action="store_true", default=False)

    # cluster configs
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--project-id", type=str, required=True)
    parser.add_argument(
        "--current-cluster",
        default=None,
        required=True,
        type=str,
        help="running cluster",
    )
    parser.add_argument("--max-jobtime", type=int, default=None)
    parser.add_argument(
        "--num-machines",
        type=int,
        required=False,
        default=1,
        help="Number of machines used to run the training",
    )
    parser.add_argument("--num-gpus-per-machine", type=int, default=1)
    parser.add_argument(
        "--task-scene",
        type=str,
        default="model_experiment",
        choices=[
            "model_release",
            "model_experiment",
            "data_verify",
            "model_verify",
            "test",
        ],
    )
    parser.add_argument("--mount-bucket", type=str, default=None)
    parser.add_argument("--pipeline-test", action="store_true", default=False)
    parser.add_argument(
        "--job-type",
        type=str,
        default="prediction",
        choices=["prediction", "train"],
    )

    args = parser.parse_args()

    return args


def load_publish_config(model_type: str, model_setting: str):
    _type2thesh = {
        BEVModelType.bev_7v: BEVModelName.bev_multitask_stage2,
        BEVModelType.bev_5v: BEVModelName.bev_multitask_small_stage2,
        BEVModelType.bev_7v_temporal: BEVModelName.bev_multitask_stage2_temporal,  # noqa
    }
    _cfg_map = {
        (
            BEVModelType.bev_7v,
            BEVModelSetting.pilot51_master,
        ): BEVSubPorject.master_bev,
        (
            BEVModelType.bev_5v,
            BEVModelSetting.pilot51_master,
        ): BEVSubPorject.master_bev,
        (
            BEVModelType.bev_7v_temporal,
            BEVModelSetting.pilot51_master,
        ): BEVSubPorject.master_bev_temporal,
        (BEVModelType.bev_7v, BEVModelSetting.ek_bev): BEVSubPorject.ek_bev,
        (BEVModelType.bev_5v, BEVModelSetting.ek_bev): BEVSubPorject.ek_bev,
        (
            BEVModelType.bev_7v_temporal,
            BEVModelSetting.ek_bev,
        ): BEVSubPorject.ek_bev_temporal,
    }

    _cfg_dir = os.path.join(DIR_PATH, "../../dev/publish/cfg")
    subproject_type = _cfg_map[(model_type, model_setting)]
    cfg_file = os.path.join(_cfg_dir, f"pub_cfg_{subproject_type}.py")
    model_config = Config.fromfile(cfg_file)["models"][_type2thesh[model_type]]

    return model_config["model_address"], model_config["update_cfg"]


def run_submit(
    model_type,
    model_setting,
    model_name_postfix,
    model_version,
    model_checkpoint,
    model_thresh,
    load_publish,
    pack_consist,
    pack_viz,
    multiview_pack,
    bev_pack,
    homo_offset,
    temporal_hommo_offset,
    project_id,
    current_cluster,
    num_machines,
    num_gpus_per_machine,
    mount_bucket,
    task_scene,
    pipeline_test,
    job_type,
    max_jobtime=None,
    local=False,
    to_video=False,
):
    assert version_matcher.search(model_version) is not None

    with open(f"{DIR_PATH}/../../model_meta.yaml", "r") as rf:
        model_meta = yaml.safe_load(rf)
    config = model_meta[model_type]["pack_infer"]["entry"]

    command = [
        "python3",
        "tools/predict.py",
        "--config",
        config,
        "--stage",
        "pack_infer",
    ]

    # model configs
    command.extend(
        [
            "--hat-pilot-model-setting",
            model_setting,
            "--hat-pilot-model-version",
            model_version,
        ]
    )
    if model_name_postfix:
        command.extend(["--hat-pilot-model-name-postfix", model_name_postfix])

    # checkpoint config
    if load_publish:
        model_checkpoint, model_thresh = load_publish_config(
            model_type=model_type, model_setting=model_setting
        )
    assert model_thresh is not None and model_checkpoint is not None
    command.extend(
        [
            "--hat-pilot-model-checkpoint",
            model_checkpoint,
            f"--hat-pilot-model-thresh '{json.dumps(model_thresh)}'",
        ]
    )

    # pack configs
    if pack_consist:
        command.extend(["--hat-pilot-pack-infer-consist", "1"])
    if pack_viz:
        command.extend(["--hat-pilot-pack-infer-vis", "1"])
    if multiview_pack:
        command.append(f"--hat-pilot-multiview-pack '{multiview_pack}'")
    if bev_pack:
        command.extend(["--hat-pilot-bev-pack", bev_pack])
    if homo_offset:
        command.extend(["--hat-pilot-homo-offset", homo_offset])
    if temporal_hommo_offset:
        command.extend(
            f"--hat-pilot-temporal-hommo-offset '{temporal_hommo_offset}'"
        )

    # others
    if pipeline_test:
        command.append("--pipeline-test")

    command = " ".join(map(str, command))

    job_cmds = [command]

    if to_video:
        _tmp_output = "tmp_output" if local else "/job_data/models"
        video_cmd = [
            "python3",
            "projects/pilot/tools/pack_infer/imgs2video.py",
            "--imgs-root",
            f"{_tmp_output}/visualize",
            "--out-root",
            f"{_tmp_output}/packinfer_video",
            "--multi-process",
            "--del-imgs",
        ]
        video_cmd = " ".join(map(str, video_cmd))
        job_cmds.append(video_cmd)

    if local:
        for job_cmd in job_cmds:
            subprocess.check_call(job_cmd, shell=True)
        return

    job_name = ["PackInfer", model_type, model_setting]
    if model_name_postfix:
        job_name.append(model_name_postfix)
    job_name.append(model_version)
    k8s_config = f"{DIR_PATH}/../pilot_bev_cluster_cfg.yaml"
    if model_type in BEVModelType.values():
        k8s_config = f"{DIR_PATH}/../pilot_bev_cluster_cfg.yaml"
    with open(k8s_config, "r") as rf:
        k8s_config = yaml.safe_load(rf)["base"]
    k8s_config.update(
        job_name="_".join(job_name),
        num_machines=num_machines,
        num_gpus_per_machine=num_gpus_per_machine,
        project_id=project_id,
        job_list=job_cmds,
        prefix_cmds_on_master=[
            "export HAT_PROCESS_GROUP_TIMEOUT=6000",
        ],
    )
    if mount_bucket:
        k8s_config["input_bucket"] = mount_bucket
    if max_jobtime:
        k8s_config["max_jobtime"] = max_jobtime

    # DAG desc
    dag_project = get_project_name([[model_setting]])
    dag_desc_dict = {
        "task_scene": f"{task_scene.upper()}",
        "domain": "perception",
        "model_version": model_version,
        "subprojects": dag_project,
    }
    # JOB desc
    job_desc_dict = {
        "model_type": model_type,
        "extra_name": model_name_postfix,
        "extra_info": model_setting,
    }

    dag_desc = json.dumps(dag_desc_dict)
    job_desc = json.dumps(job_desc_dict)
    assert (
        len(dag_desc) <= 255
    ), "len(dag_desc) should <= 255, please delete some dag info"
    assert (
        len(job_desc) <= 255
    ), "len(job_desc) should <= 255, please delete some job info"

    desc_cmd = f" --dag-desc '{dag_desc}'" + f" --job-desc '{job_desc}'"

    with open("__tmp_packinfr_cfg.yaml", "w") as wf:
        yaml.dump(k8s_config, wf)

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
        "../../__tmp_packinfr_cfg.yaml",
        "--cluster",
        current_cluster,
        desc_cmd,
        "--job-type",
        job_type,
    ]
    subprocess.run(" ".join(cmd), shell=True, check=True)
    os.remove("__tmp_packinfr_cfg.yaml")


if __name__ == "__main__":
    args = parse_full_args()
    run_submit(**vars(args))
