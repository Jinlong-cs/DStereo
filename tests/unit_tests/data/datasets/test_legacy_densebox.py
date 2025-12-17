import os

import pytest

from hat.data.datasets.legacy_densebox import (
    HAT_LEGACYDENSEBOX_AVAILABLE,
    LegacyDenseBoxImageRecordDataset,
)
from tests import HAT_BUCKET_PATH

ADASMINI_ROOT = os.path.join(HAT_BUCKET_PATH, "unit_test_data/J5FSD/adasmini")
ADASMINI_AVAILABLE = os.path.exists(ADASMINI_ROOT)


@pytest.mark.skipif(
    not HAT_LEGACYDENSEBOX_AVAILABLE,
    reason="horizon_plugin_pytorch >= 1.0.0 is required.",
)
@pytest.mark.skipif(not ADASMINI_AVAILABLE, reason="adas-mini is required")
def test_auto2d_dataset_densebox_record(tmpdir):
    rec_file = "vehicle_rear/detection/720_v1/rec/val.rec"
    json_file = "vehicle_rear/detection/720_v1/rec/val.json"
    dataset = LegacyDenseBoxImageRecordDataset(
        rec_path=os.path.join(ADASMINI_ROOT, rec_file),
        anno_path=os.path.join(ADASMINI_ROOT, json_file),
        rec_idx_file_path=os.path.join(tmpdir, "tmp.idx"),
    )
    for data in dataset:
        assert isinstance(data, tuple)
        assert len(data) == 2
        image, anno = data
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        anno = anno.to_dict()
        assert "img_h" in anno
        assert "instances" in anno
        break


def test_auto2d_dataset_densebox_record_with_img_buf(tmpdir):
    rec_file = "vehicle_rear/detection/720_v1/rec/val.rec"
    json_file = "vehicle_rear/detection/720_v1/rec/val.json"
    dataset = LegacyDenseBoxImageRecordDataset(
        rec_path=os.path.join(ADASMINI_ROOT, rec_file),
        anno_path=os.path.join(ADASMINI_ROOT, json_file),
        rec_idx_file_path=os.path.join(tmpdir, "tmp.idx"),
        with_img_buf=True,
    )
    for data in dataset:
        assert isinstance(data, tuple)
        assert len(data) == 3
        image, img_buf, anno = data
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        anno = anno.to_dict()
        assert "img_h" in anno
        assert "instances" in anno
        break
