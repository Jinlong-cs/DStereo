# Copyright (c) Horizon Robotics. All rights reserved.
# The unit test for trajectory prediction transforms about anchors.

import os

import pytest

from hat.core.traj_pred_typing import ANCHOR_TYPE
from hat.data.transforms.traj_pred.traj_pred_anchor import GetBestAnchors
from projects.prediction.configs.anchor_type_split import (
    ANCHORSET_124_TYPE_DICT,
)
from projects.prediction.configs.processed_dataset import HAT_UNITTEST_PREFIX
from tests import SD_AlGORITHM_BUCKET_PATH
from tests.unit_tests.data.transforms.traj_pred.test_traj_pred_obstacle import (  # noqa: E501
    test_get_traj_pred_info,
)


@pytest.mark.skipif(True, reason="requiring New Dataset")
def test_get_best_anchors():
    sample = test_get_traj_pred_info()
    anchor_file = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        HAT_UNITTEST_PREFIX,
        "anchors/veh_anchor_add_ped_anchors.pkl",
    )
    anchor_num = 124
    anchor_type_version = "classical"
    get_best_anchor = GetBestAnchors(
        anchor_cfg=dict(
            anchor_file=anchor_file,
            anchor_method="kmeans",
            anchor_num=anchor_num,
        ),
        anchor_type=ANCHOR_TYPE[anchor_type_version],
        anchor_type_dict=ANCHORSET_124_TYPE_DICT[anchor_type_version],
    )
    sample = get_best_anchor(sample)
    assert "selected_anchors" in sample
    assert "selected_anchors_mask" in sample
    num_anchor_types = len(ANCHOR_TYPE[anchor_type_version])
    assert sample["selected_anchors"].shape[1] == num_anchor_types
    assert sample["selected_anchors_mask"].shape[1] == num_anchor_types
