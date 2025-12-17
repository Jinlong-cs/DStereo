# Copyright (c) Horizon Robotics. All rights reserved.

import json
import os

import yaml
from torch.utils.data._utils.collate import default_convert

from hat.metrics.metric_3dv import PoseRTE
from hat.utils.filesystem import file_load


def evaluate(
    dataset_fpath: str,
    prediction_fpath: str,
    setting_fpath: str,
    output_dir: str,
):
    """2p5d_pose evaluation.

    Args:
        dataset_fpath: Image path to evaluation.
        prediction_fpath: Prediction results saved by aidi_eval callback in
            fsd inference.It should be a pickle file.
        setting_fpath: Profile the path of the configuration setting.It should
            be aYAML file.
        output_dir: The path to the result output.
    """

    cfg = yaml.load(open(setting_fpath))
    poserte_eval = PoseRTE(gt_pose_path=cfg["gt_pose_path"])
    for eval_data in file_load(prediction_fpath):
        eval_data = default_convert(eval_data)
        batch = eval_data["batch"]
        outputs = eval_data["outputs"]
        poserte_timestamp = batch["timestamp"]
        for key in outputs.keys():
            if "axisangle" in key:
                poserte_axisangle = outputs[key]
            if "translation" in key:
                poserte_translation = outputs[key]
        poserte_eval.update(
            poserte_timestamp, poserte_axisangle, poserte_translation
        )
    _, result_poserte = poserte_eval.get()
    result = {}
    result["result_poserte"] = result_poserte
    json.dump(result, open(os.path.join(output_dir, "result.json"), "w"))
    return result
