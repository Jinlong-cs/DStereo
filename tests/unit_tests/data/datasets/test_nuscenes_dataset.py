import pytest
import torchvision

from hat.data.datasets.nuscenes_dataset import (
    NuscenesBevDataset,
    NuscenesLidarDataset,
    NuscenesLidarWithSegDataset,
    NuscenesMonoDataset,
)
from hat.data.transforms.lidar_utils import LidarReformat

try:
    import nuscenes
except ImportError:
    nuscenes = None

transforms = torchvision.transforms.Compose(
    [
        LidarReformat(),
    ]
)

map_size = (15, 30, 0.15)
bev_size = (51.2, 51.2, 0.4)
bev_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)


@pytest.mark.skipif(nuscenes is None, reason="need nuscenes")
@pytest.mark.parametrize(
    "bev_size, bev_range, with_bev_bboxes, with_ego_bboxes, with_bev_mask",
    [
        (bev_size, None, True, False, True),
        (None, bev_range, False, True, False),
    ],
)
def test_nuscenes(
    bev_size, bev_range, with_bev_bboxes, with_ego_bboxes, with_bev_mask
):
    dataset = NuscenesBevDataset(
        data_path="./tmp_data/nuscenes/v1.0-trainval/val_lmdb",
        bev_size=bev_size,
        bev_range=bev_range,
        map_size=map_size,
        map_path="./tmp_data/nuscenes/meta",
        with_bev_bboxes=with_bev_bboxes,
        with_ego_bboxes=with_ego_bboxes,
        with_bev_mask=with_bev_mask,
    )
    state = dataset.__getstate__()
    dataset.__setstate__(state)

    for ind, data in enumerate(dataset):
        image = data["img"]
        assert len(image) == 6
        if ind > 10:
            break


@pytest.mark.skipif(nuscenes is None, reason="need nuscenes")
def test_nuscenes_mono_dataset():
    dataset = NuscenesMonoDataset(
        data_path="./tmp_data/nuscenes/v1.0-trainval/val_lmdb",
    )
    state = dataset.__getstate__()
    dataset.__setstate__(state)
    for ind, data in enumerate(dataset):
        image = data["img"]
        print(image.shape)
        if ind > 10:
            break


@pytest.mark.skipif(nuscenes is None, reason="need nuscenes")
@pytest.mark.parametrize(
    "num_sweeps, with_velocity, test_mode, filter_empty_gt, transforms",
    [
        (5, True, False, True, None),
        (2, False, True, False, transforms),
    ],
)
def test_nuscenes_lidar(
    num_sweeps,
    with_velocity,
    test_mode,
    filter_empty_gt,
    transforms,
):
    dataset = NuscenesLidarDataset(
        num_sweeps=num_sweeps,
        data_path="./tmp_data/nuscenes/lidar_seg/v1.0-trainval/val_lmdb",
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        use_valid_flag=True,
        with_velocity=with_velocity,
        test_mode=test_mode,
        filter_empty_gt=filter_empty_gt,
        transforms=transforms,
    )
    state = dataset.__getstate__()
    dataset.__setstate__(state)

    for ind, data in enumerate(dataset):
        if transforms is None:
            points = data["lidar"]["points"]
            anns = data["lidar"]["annotations"]
        else:
            points = data["points"]
            anns = data["annotations"]
        assert points.shape[-1] == 5
        assert "boxes" in anns
        if ind > 10:
            break

    dataset_withseg = NuscenesLidarWithSegDataset(
        num_sweeps=num_sweeps,
        data_path="./tmp_data/nuscenes/lidar_seg/v1.0-trainval/val_lmdb",
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        use_valid_flag=True,
        with_velocity=with_velocity,
        test_mode=test_mode,
        filter_empty_gt=filter_empty_gt,
        transforms=transforms,
    )

    for ind, data in enumerate(dataset_withseg):
        if transforms is None:
            points = data["lidar"]["points"]
            anns = data["lidar"]["annotations"]
        else:
            points = data["points"]
            anns = data["annotations"]
        assert points.shape[-1] == 5
        assert "boxes" in anns
        assert "gt_seg_labels" in anns
        if ind > 10:
            break


if __name__ == "__main__":
    pytest.main(["-s", __file__])
