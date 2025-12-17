import argparse
import json
import os
import shutil
import subprocess

from hat.utils import Config

test_models = [
    "pilot_legorcnn_multitask_resize_2",
    "pilot_legorcnn_multitask_resize_2_parking",
    "pilot_legorcnn_multitask_resize_2_night",
    "pilot_legorcnn_multitask_resize_4",
    "pilot_legorcnn_multitask_resize_4_parking",
    "pilot_legorcnn_multitask_resize_4_night",
    "pilot_legorcnn_multitask_crop",
    "pilot_legorcnn_multitask_crop_night",
    "pilot_legorcnn_multitask_rear_resize_2_day",
    "pilot_legorcnn_multitask_rear_resize_2_night",
    "pilot_legorcnn_multitask_side_resize_2_day",
    "pilot_legorcnn_multitask_side_resize_2_night",
    "pilot_legorcnn_iqa_parsing_resize_4",
    "pilot_legorcnn_iqa_parsing_resize_4_parking",
    "pilot_legorcnn_iqa_parsing_rear_resize_4",
    "pilot_legorcnn_iqa_parsing_rear_resize_4_night",
    "pilot_legorcnn_iqa_parsing_side_resize_4",
    "pilot_legorcnn_iqa_parsing_side_resize_4_night",
    "pilot_legorcnn_multitask_rear_crop_day",
    "pilot_legorcnn_multitask_rear_crop_night",
    "pilot_legorcnn_multitask_side_crop_day",
    "pilot_legorcnn_multitask_side_crop_night",
    "pilot_bev_5v_multitask_day",
]


niofy_test_models = [
    "pilot_legorcnn_multitask_side_resize_2_day",
    "pilot_legorcnn_multitask_rear_resize_2_day",
    "pilot_legorcnn_multitask_rear_crop_day",
    "pilot_legorcnn_iqa_parsing_rear_resize_4",
    "pilot_legorcnn_iqa_parsing_side_resize_4",
]


def get_infer_res(sub_project, publish_version):

    config = Config.fromfile(
        os.path.join(
            os.path.dirname(__file__),
            f"cfg/pub_cfg_{sub_project}.py",
        ),
    )

    shutil.rmtree("dump_res", ignore_errors=True)

    if "niofy" in sub_project:
        test_models = niofy_test_models

    for model_name in test_models:
        if model_name not in config.models:
            print("Skip " + model_name)
            continue

        os.environ["HAT_PILOT_MODEL_NAME"] = model_name

        model_cfg = config.models[model_name]
        os.environ["HAT_PILOT_MODEL_SETTING"] = model_cfg["model_setting"]
        os.environ["HAT_PILOT_MODEL_CHECKPOINT"] = model_cfg["model_address"]
        if "iqa_parsing" in model_name:
            cmds = [
                "python3",
                "tools/predict.py",
                "--config",
                model_cfg["cfg_path"].replace(
                    "image_fail_segmentation", "pred_image_fail_segmentation"
                ),
                "--stage",
                "int_infer",
            ]
            subprocess.run(" ".join(cmds), shell=True)
        else:
            os.environ["HAT_PILOT_MODEL_THRESH"] = json.dumps(
                model_cfg["update_cfg"]
            )
            cmds = [
                "python3",
                "tools/predict.py",
                "--config",
                model_cfg["cfg_path"].replace("multitask", "pred_multitask"),
                "--stage",
                "int_infer",
            ]
            subprocess.run(" ".join(cmds), shell=True)

    names = [config.infer_result_prefix]
    if publish_version is not None:
        names.append(publish_version)
    names.append("result.tar.gz")
    file_name = "_".join(names)

    subprocess.run(
        f"cd dump_res && tar -zcvf ../{file_name} ./*",
        shell=True,
    )

    return os.path.abspath(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sub-project",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--publish-version",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    get_infer_res(**vars(args))
