import pytest

from hat.data.datasets.nuscenes_perception_dataset import (
    NuScenesPerceptionDataset,
)
from tests import AIDI_PUBLIC_DATA_BUCKET_EXISTS, AIDI_PUBLIC_DATA_BUCKET_PATH

try:
    import nuscenes
    import pyquaternion

    _NUSCENES_IMPORTED = True
except ImportError:
    _NUSCENES_IMPORTED = False


@pytest.mark.skipif(
    not AIDI_PUBLIC_DATA_BUCKET_EXISTS or not _NUSCENES_IMPORTED,
    reason="requiring aidi_public_data bucket",
)
def test_nuscenes_perception_dataset():
    assert hasattr(nuscenes, "NuScenes")
    assert hasattr(pyquaternion, "Quaternion")

    video_frame = 4
    dataset = NuScenesPerceptionDataset(
        version="v1.0-mini",
        dataroot=f"{AIDI_PUBLIC_DATA_BUCKET_PATH}/nuScenes/origin",
        video_frame=video_frame,
    )

    assert len(dataset) == 404

    num_cams = len(NuScenesPerceptionDataset.CAMERAS)
    for i in range(10):
        data = dataset[i]
        assert data["img"].shape == (num_cams, 900, 1600, 3)
        assert data["lidar2global"].shape == (4, 4)
        assert data["cam2global"].shape == (num_cams, 4, 4)
        assert data["cam_intrinsic"].shape == (num_cams, 3, 3)
        assert data["lidar2img"].shape == (num_cams, 3, 4)
        assert data["lidar2ego"].shape == data["lidar2global"].shape
        assert data["cam2ego"].shape == data["cam2global"].shape
        assert data["lidar2cam"].shape == data["cam2global"].shape
        assert "timestamp" in data
        assert "scene_token" in data
        assert len(data["seq_data"]) < video_frame

        if "boxes" in data:
            num_obj = len(data["boxes"])
            assert data["boxes"].shape == (num_obj, 10)
            assert data["names"].shape[0] == num_obj
            assert data["instance_inds"].shape[0] == num_obj
            assert data["category_ids"].shape[0] == num_obj
