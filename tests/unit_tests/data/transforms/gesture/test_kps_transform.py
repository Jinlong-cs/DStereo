# Copyright (c) Horizon Robotics. All rights reserved.
import copy

from init_path import assert_func, get_data
from torch import Tensor

from hat.data.transforms.gesture.kps_transform import (
    ActionKpsJitter,
    ActionKpsNormalize,
    ActionKpsRandScale,
    ActionKpsReshape,
    ActionKpsScoreNormalize,
    ActionKpsSmooth,
    ActionKpsToTensor,
)


def test_kps_reshape():
    # data from gluonperson
    (input_data,) = get_data("flip_all_output3.pkl")
    akr = ActionKpsReshape(
        num_kps=21,
        feat_ch=3,
    )
    pred_data = akr(input_data)

    assert pred_data["clip_keypoints"].shape == (16, 21, 3)


def test_kps_score_norm():
    (data,) = get_data("kps_score_norm_input_output.pkl")
    aksn = ActionKpsScoreNormalize(
        use_kps_score=data["use_kps_score"],
        disable_kps_conf=data["disable_kps_conf"],
        use_3d_kps=False,
    )
    pred_data = aksn(data)
    assert_func(pred_data, data["output"])


def test_kps_smooth():
    (data,) = get_data("kps_smooth_input_output.pkl")
    aks = ActionKpsSmooth(p_smooth_kps=1, smooth_method="GaussianBlur")
    pred_data = aks(data["input"])
    assert_func(pred_data, data["output"])


def test_kps_norm():
    (data,) = get_data("kps_norm_input_output.pkl")
    akn = ActionKpsNormalize(**data["params"])
    pred_data = akn(data["input"])
    assert_func(pred_data, data["output"])


def test_kps_jitter_randscale():
    # jitter - simple check
    (data,) = get_data("kps_norm_input_output.pkl")
    assert "clip_keypoints" in data["output"]
    akj = ActionKpsJitter(p_point_jitter=1)
    pred_data = akj(copy.deepcopy(data["output"]))
    assert akj._point_jitter and "clip_keypoints" in pred_data
    # rand scale - simple check
    akrs = ActionKpsRandScale()
    pred_data = akrs(copy.deepcopy(data["output"]))
    assert akrs._rand_scale


def test_kps_totensor():
    input_data, output_data = get_data(
        "kps_totensor_input_output.pkl",
        "kps_totensor_output.pkl",
    )
    aktt = ActionKpsToTensor(
        tensor_layout="chw",
        use_3d_kps=False,
        hand_orient_concat=False,
        xy_diff_concat=False,
    )
    pred_data = aktt(input_data)

    assert type(pred_data["box_center"]) == Tensor
    assert type(pred_data["clip_keypoints"]) == Tensor
    assert pred_data["kps_layout"] == "chw"
    assert (
        pred_data["box_center"].numpy() == output_data["box_center"]
    ).all()  # noqa
    assert (
        pred_data["clip_keypoints"].numpy() == output_data["clip_keypoints"]
    ).all()  # noqa
