import os
import pickle

import torch
from torch.utils.data._utils.collate import default_collate

from hat.models.base_modules.target.heatmap_roi_3d_target import (
    HeatMap3DTargetGenerator,
)
from tests import HAT_BUCKET_PATH

root = f"{HAT_BUCKET_PATH}/users/zihan.qiu/jenkins_test_transform_data"


def test_heatmap_3d_target_generator():
    with open(os.path.join(root, "transformed_3d_output.pkl"), "rb") as f:
        input_data = pickle.load(f)
    if "calib_all" in input_data:
        input_data.pop("calib_all")
    input_data = default_collate([input_data])
    classnames = ["vehicle"]
    classid_map = {1: -1, 2: 0, 3: -1, 4: 0, 5: 0, 6: 0, 7: 0, 8: -1}
    num_classes = len(classnames)
    down_stride = 4
    normalize_depth = True
    focal_length_default = 1114.3466796875

    max_depth = 60
    max_objs = 100
    undistort_depth_uv = False
    depth_min_option = False

    dense_module = HeatMap3DTargetGenerator(
        num_classes=num_classes,
        normalize_depth=normalize_depth,
        classid_map=classid_map,
        focal_length_default=focal_length_default,
        down_stride=down_stride,
        min_box_edge=8,  # original image size
        max_depth=max_depth,
        max_objs=max_objs,
        undistort_2dcenter=True,
        undistort_depth_uv=undistort_depth_uv,
        depth_min_option=depth_min_option,
    )
    output = dense_module(input_data)
    assert isinstance(output, dict)
    assert "heatmap" in output.keys()
    assert "depth" in output.keys()
    assert "rotation_y" in output.keys()
    assert "dimensions" in output.keys()
    assert "img_wh" in output.keys()
    assert torch.all(output["img_wh"] == torch.tensor([[960, 512]]))
    assert output["heatmap"].shape == torch.Size([1, 1, 128, 240])
