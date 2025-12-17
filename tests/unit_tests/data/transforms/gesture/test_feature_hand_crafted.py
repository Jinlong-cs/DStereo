# Copyright (c) Horizon Robotics. All rights reserved.

from init_path import assert_func, get_data

from hat.data.transforms.gesture.feature_hand_crafted import (
    ActionAddMotionInRgbBranch,
    ActionKpsAddFingerEncoding,
    ActionKpsAddTemporalEncoding,
)


def test_add_motion():
    (data,) = get_data("addmotion_input_output.pkl")
    aamirb = ActionAddMotionInRgbBranch()
    pred_data = aamirb(data["input"])
    assert_func(pred_data, data["output"])


def test_add_finger_encoding():
    (data,) = get_data("finger_encoding_input_output.pkl")
    akafe = ActionKpsAddFingerEncoding(**data["params"])
    pred_data = akafe(data["input"])
    assert_func(pred_data, data["output"])


def test_add_temporal_encoding():
    (data,) = get_data("time_encoding_input_output.pkl")
    akate = ActionKpsAddTemporalEncoding(**data["params"])
    pred_data = akate(data["input"])
    assert_func(pred_data, data["output"])
