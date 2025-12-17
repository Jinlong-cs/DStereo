import os

import pytest

from hat.core.anno_ts_utils import get_img_idx_to_img_record_map
from tests import HAT_BUCKET_PATH

ADASMINI_ROOT = os.path.join(HAT_BUCKET_PATH, "unit_test_data/J5FSD/adasmini")
ADASMINI_AVAILABLE = os.path.exists(ADASMINI_ROOT)


@pytest.mark.skipif(not ADASMINI_AVAILABLE, reason="adas-mini is required")
def test_get_img_idx_to_img_record_map():
    json_file = "vehicle_rear/detection/720_v1/rec/val.json"
    anno_path = os.path.join(ADASMINI_ROOT, json_file)
    img_record_map = get_img_idx_to_img_record_map(anno_path)
    assert len(img_record_map.keys()) == 10000


if __name__ == "__main__":
    pytest.main(["-s", __file__])
