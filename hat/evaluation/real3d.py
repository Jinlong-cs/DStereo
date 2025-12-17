# Copyright (c) Horizon Robotics. All rights reserved.

import json
import os
import shutil
import tarfile

import yaml
from torch.utils.data._utils.collate import default_convert

from hat.registry import build_from_registry
from hat.utils.apply_func import convert_numpy
from hat.utils.filesystem import file_load


def evaluate(
    dataset_fpath: str,
    prediction_fpath: str,
    setting_fpath: str,
    output_dir: str,
):
    """real3d evaluation.

    Args:
        dataset_fpath: Image path to evaluation.
        prediction_fpath: Prediction results saved by aidi_eval callback in
            fsd inference.It should be a tar file.
        setting_fpath: Profile the path of the configuration setting.It should
            be aYAML file.
        output_dir: The path to the result output.
    """

    cfg = yaml.load(open(setting_fpath), Loader=yaml.FullLoader)
    # disable save result in real3d metric
    if "save_path" in cfg:
        cfg["save_path"] = None
    real3d_eval = build_from_registry(cfg)
    tar_output = "tmp_real3d"
    if not os.path.exists(tar_output):
        os.makedirs(tar_output)
    tar = tarfile.open(prediction_fpath, "r")
    tar.extractall(path=tar_output)
    pkl_file = [f for f in os.listdir(tar_output) if f.endswith(".pkl")]
    assert (
        len(pkl_file) == 1
    ), f"{prediction_fpath} should only contain one pickle file"

    pkl_file_path = os.path.join(tar_output, pkl_file[0])
    for eval_data in file_load(pkl_file_path):
        eval_data = default_convert(eval_data)
        batch = eval_data["batch"]
        outputs = eval_data["outputs"]
        new_outputs = {}
        for key in outputs.keys():
            if "predict" in key:
                index = key.index("predict") + len("predict")
                new_outputs[key[index + 1 :]] = outputs[key]

        new_outputs["image_id"] = batch["image_id"]
        real3d_eval.update(batch, new_outputs)
    names, values = real3d_eval.get()
    result = {}
    result["name"] = "real3d eval"
    result["header"] = names
    result["data"] = []
    for i in range(len(values[0])):
        data_dict = {}
        for j, name in enumerate(names):
            data_dict[name] = values[j][i]
        result["data"].append(data_dict)
    result = convert_numpy(result, True)
    json.dump(result, open(os.path.join(output_dir, "result.json"), "w"))
    return_dict = {
        "name": "real3d eval",
        "state": "success",
    }
    shutil.rmtree(tar_output)
    return return_dict


# # Comment below, since aidi don't support do visualize in python3.6
# # and `Painter` is not in hat now.
# def visualize(image: np.ndarray, sample: dict):
#     """real3d visualize.
#
#     Args:
#         image: The original image.
#         sample: Data needed for visualization.It should be use result by
#             inference.
#     """
#
#     painter = Painter()
#     img = painter.draw_real3d(sample, image, 0)
#     return img
