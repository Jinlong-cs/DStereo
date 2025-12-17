# Copyright (c) Horizon Robotics. All rights reserved.
import pytest
from init_path import PATH_REC_READER_AVAIABLE
from init_path import get_data as _get_data

from hat.data.transforms.gesture import ActionGetImgClip, ActionGetMetaData


def get_data():
    is_read_rec = True
    (data,) = _get_data("roidb_recreader.pkl")
    data["frames"] = data.pop("clip_img")
    data["frames"] = []
    data["clip_rec_label"] = []
    return data, is_read_rec


@pytest.mark.skipif(not PATH_REC_READER_AVAIABLE, reason="need rec reader")
def test_action_reader():
    data, is_read_rec = get_data()
    params = {
        "seq_len": 32,
        "feat_len": 21 * 4,
        "box_len": 4,
        "time_stride": 0.015625,
    }
    av2c = ActionGetMetaData(**params)
    data = av2c(data)
    assert "clip_kps_meta_info" in data
    assert data["clip_kps_meta_info"]["clip_idx"].shape[0] == params["seq_len"]
    assert data["clip_kps_meta_info"]["clip_keypoints"].shape == (
        params["seq_len"],
        params["feat_len"] + 1,
    )
    assert data["clip_kps_meta_info"]["clip_boxes"].shape == (
        params["seq_len"],
        params["box_len"],
    )

    if is_read_rec:
        agic = ActionGetImgClip()
        data = agic(data)
        assert "frames" in data
        assert len(data["frames"]) == params["seq_len"]
        assert data["frames"][0][0].shape[-1] == 3

        assert "clip_rec_label" in data
        assert len(data["clip_rec_label"]) == params["seq_len"]
        assert (
            len(data["clip_rec_label"][0]["crop_boxes"][0])
            == params["box_len"]
        )
