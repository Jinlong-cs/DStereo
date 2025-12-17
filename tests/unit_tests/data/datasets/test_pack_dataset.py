import os
from distutils.version import LooseVersion

import pytest

from hat.data.datasets.pack_dataset import PackDataset
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH

try:
    import tat
except ImportError:
    tat = None


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
@pytest.mark.skipif(not tat, reason="tat is required")
def test_pack_dataset():

    pack_path = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/J5FSD/users/ben.hu/test/ADAS_20220623-161529_761_$Index.pack",  # noqa
    )

    dataset = PackDataset(
        pack_path=pack_path,
        camera_view_names=["camera_front"],
        start_idx=10,
        expect_length=100,
        camera_calib=True,
        transforms=None,
        return_odometry=True,
    )

    for ind, data in enumerate(dataset):
        image = data["img"]
        assert "camera_calib" in data
        assert "timestamp" in data
        assert "odo_info" in data
        if LooseVersion(tat.__version__) < LooseVersion("0.2.1"):
            assert image.shape == (2160, 3840, 3)
        if ind > 10:
            break
