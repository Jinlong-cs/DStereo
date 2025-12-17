# Copyright (c) Horizon Robotics. All rights reserved.
from init_path import assert_func, get_data

from hat.data.transforms.gesture import ActionClipDataPreProcess


def test_preprocess():
    # Intermediate results from gluonperson
    input_data, ref_output_data = get_data(
        "proprocess_input2.pkl", "proprocess_output2.pkl"
    )

    params = {
        # net params
        "act_kps_seq_len": 16,
        "act_img_seq_len": 8,
        "num_kps": 21,
        "use_3d_kps": False,
        "use_kps_score": True,
        "use_rgb_branch": True,
        # preprocess params
        "use_replenish": True,
    }
    # ActionClipDataPreProcess for rgb and kps
    acdpp = ActionClipDataPreProcess(**params)
    clip_kps_meta_info = input_data["clip_kps_meta_info"]
    clip_kps_meta_info["clip_keypoints"] = clip_kps_meta_info.pop("keypoints")
    clip_kps_meta_info["clip_boxes"] = clip_kps_meta_info.pop("boxes")
    # input_data could be changed
    data = acdpp(input_data)
    assert_func(data, ref_output_data)
