# Copyright (c) Horizon Robotics. All rights reserved.
import os

import pytest

from hat.data.datasets.densebox_dataset import LegacyDenseBoxImageRecordDataset
from hat.registry import build_from_registry
from tests import HAT_BUCKET_PATH

CLOUDMODEL_ROOT = os.path.join(HAT_BUCKET_PATH, "unit_test_data/cloudmodel")
CLOUDMODEL_AVAILABLE = os.path.exists(CLOUDMODEL_ROOT)


@pytest.mark.skipif(
    LegacyDenseBoxImageRecordDataset is None,
    reason="auto_matrix or horizon_plugin_pytorch >= 1.0.0 is required.",
)  # noqa
@pytest.mark.skipif(not CLOUDMODEL_AVAILABLE, reason="cloudmodel is required")
@pytest.mark.parametrize(
    ["with_polygon"],
    [
        pytest.param(False),
        pytest.param(True),
    ],
)
def test_auto2d_dataset_det(with_polygon):
    rec_file = "instanceseg/lane/data.rec"
    anno_file = "instanceseg/lane/data.anno.pb_rec"
    config = dict(
        type="InstSegDenseboxDataset",
        data_path=os.path.join(CLOUDMODEL_ROOT, rec_file),
        anno_path=os.path.join(CLOUDMODEL_ROOT, anno_file),
        task_type="instanceseg",
        with_polygon=with_polygon,
    )
    dataset = build_from_registry(config)

    data = dataset[0]
    assert isinstance(data, dict)
    assert "gt_labels" in data
    assert "gt_seg" in data
    if with_polygon:
        assert "gt_polygons" in data
    else:
        assert "gt_polygons" not in data
