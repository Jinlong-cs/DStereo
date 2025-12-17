# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.task_modules.traj_pred.necks import AnchorEncodeNeck


@pytest.mark.skipif(False, reason="do not need skip")
def test_anchor_encode_neck():
    num_anchors = 14
    traj_len = 12
    anchor_mode = "fuse"
    num_obj = 17

    neck = AnchorEncodeNeck(
        anc_conv_channels=[1024, 128, 16],
        num_anchors=num_anchors,
        traj_len=traj_len,
        anchor_mode=anchor_mode,
        is_int_infer_model=False,
    )

    anc_x_key = f"{anchor_mode}_anchors_diff_x"
    anc_y_key = f"{anchor_mode}_anchors_diff_y"
    anc_key = f"{anchor_mode}_anchors"
    backbone_feat = torch.zeros([num_obj, 128, 1, 1])
    data = {
        "feats": [backbone_feat],
        anc_x_key: torch.zeros([num_obj, num_anchors, traj_len, 1]),
        anc_y_key: torch.zeros([num_obj, num_anchors, traj_len, 1]),
        anc_key: torch.zeros([num_obj, num_anchors, traj_len, 2]),
    }
    neck_feats = neck(data)
    assert "anchor_encode_feat" in neck_feats
    anc_feat = neck_feats["anchor_encode_feat"]
    assert list(anc_feat.shape) == [num_obj, 16, 1, 1]
