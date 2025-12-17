import os

import pytest

from hat.registry import build_from_registry
from tests import (
    HAT_BUCKET_EXISTS,
    HAT_BUCKET_PATH,
    HAT_BUCKET_URL_EXISTS,
    HAT_BUCKET_URL_PATH,
)


@pytest.mark.skipif(
    not HAT_BUCKET_EXISTS,
    reason="requiring HAT_BUCKET bucket",
)
def test_det_seg_2d_anno_dataset():
    root = f"{HAT_BUCKET_PATH}/users/zihan.qiu/jenkins_test_packed_data/cyclist_detection"  # noqa
    config = dict(
        type="DetSeg2DAnnoDataset",
        idx_path=os.path.join(root, "idx"),
        img_path=os.path.join(root, "img"),
        anno_path=os.path.join(root, "anno"),
    )
    dataset = build_from_registry(config)
    # datset_len = len(dataset)
    data = dataset[0]
    assert isinstance(data, dict)
    assert "anno" in data
    assert "img" in data


@pytest.mark.skipif(
    not HAT_BUCKET_URL_EXISTS,
    reason="requiring HAT_BUCKET bucket",
)
def test_det_seg_2d_anno_dataset_from_url_data_path():
    root = f"{HAT_BUCKET_URL_PATH}/users/zihan.qiu/jenkins_test_packed_data/cyclist_detection"  # noqa
    config = dict(
        type="DetSeg2DAnnoDataset",
        idx_path=os.path.join(root, "idx"),
        img_path=os.path.join(root, "img"),
        anno_path=os.path.join(root, "anno"),
    )
    dataset = build_from_registry(config)
    # datset_len = len(dataset)
    data = dataset[0]
    assert isinstance(data, dict)
    assert "anno" in data
    assert "img" in data
