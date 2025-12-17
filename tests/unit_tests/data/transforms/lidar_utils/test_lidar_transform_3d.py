import copy

import numpy as np

import hat.data.datasets.nuscenes_dataset as NuscenesDataset
from hat.data.transforms.lidar_utils.lidar_transform_3d import (
    AssignSegLabel,
    LidarMultiPreprocess,
    LidarReformat,
    ObjectNoise,
    ObjectRangeFilter,
    PointCloudSegPreprocess,
    PointGlobalRotation,
    PointGlobalScaling,
)

points = np.array(
    [[0.5, 0.5, 0.5, 1.0], [0.8, 0.3, 0.4, 0.6]], dtype=np.float32
)
gt_boxes = np.array(
    [
        [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, np.pi / 2],
        [10.0, 10.0, 10.0, 1.0, 1.0, 1.0, np.pi / 2],
    ],
    dtype=np.float32,
)
names = np.array(["car", "ignore"])
gt_seg_labels = np.array([0, 1], dtype=np.uint8)
gt_boxes_mask = np.array([True, False], dtype=np.bool_)
data = {
    "lidar": {
        "points": points,
        "annotations": {
            "gt_boxes": gt_boxes,
            "gt_boxes_mask": gt_boxes_mask,
            "boxes": gt_boxes,
            "names": names,
            "gt_seg_labels": gt_seg_labels,
        },
    },
    "metadata": {
        "image_idx": 1,
    },
    "mode": "train",
}


def test_ObjectNoise():
    obj_noise = ObjectNoise(
        num_try=100,
        gt_loc_noise_std=[0.25, 0.25, 0.25],
        global_random_rot_range=[0.0, 0.0],
        gt_rotation_noise=[-0.15707963267, 0.15707963267],
    )

    out_data = obj_noise(copy.deepcopy(data))
    print(out_data)
    out_gt_box = out_data["lidar"]["annotations"]["gt_boxes"]
    assert out_gt_box.shape == (1, 7)


def test_PointGlobalScaling():
    scale = PointGlobalScaling(
        min_scale=1.0,
        max_scale=1.0,
    )

    out_data = scale(copy.deepcopy(data))
    scale_box = out_data["lidar"]["annotations"]["gt_boxes"]
    scale_points = data["lidar"]["points"]
    assert np.all(gt_boxes == scale_box)
    assert np.all(points == scale_points)


def test_PointGlobalRotation():
    rotation = PointGlobalRotation(rotation=0)

    out_data = rotation(copy.deepcopy(data))

    rot_box = out_data["lidar"]["annotations"]["gt_boxes"]
    rot_points = data["lidar"]["points"]
    assert np.all(gt_boxes == rot_box)
    assert np.all(points == rot_points)


def test_ObjectRangeFilter():
    range_filter = ObjectRangeFilter(point_cloud_range=[-5, -5, 0, 5, 5, 1])
    out_data = range_filter(copy.deepcopy(data))
    filtered_box = out_data["lidar"]["annotations"]["gt_boxes"]
    assert filtered_box.shape == (1, 7)
    assert np.all(filtered_box == gt_boxes[0])


def test_LidarReformat():
    lidar_reformat = LidarReformat()
    new_data = lidar_reformat(data)

    expected_keys = ["metadata", "points", "annotations"]
    for key in expected_keys:
        assert key in new_data


def test_PointCloudSegPreprocess():
    seg_preprocess = PointCloudSegPreprocess(
        global_rot_noise=[-0.3925, 0.3925],
        global_scale_noise=[0.95, 1.05],
    )
    out_data = seg_preprocess(copy.deepcopy(data))
    process_box = out_data["lidar"]["annotations"]["gt_boxes"]
    process_points = data["lidar"]["points"]
    assert np.all(gt_boxes == process_box)
    assert np.all(points.shape == process_points.shape)


def test_LidarMultiPreprocess():
    multi_preprocess = LidarMultiPreprocess(
        class_names=NuscenesDataset.CLASSES,
        global_rot_noise=[-0.3925, 0.3925],
        global_scale_noise=[0.95, 1.05],
    )
    out_data = multi_preprocess(copy.deepcopy(data))
    process_box = out_data["lidar"]["annotations"]["gt_boxes"]
    process_seg_labels = out_data["lidar"]["annotations"]["gt_seg_labels"]
    process_points = data["lidar"]["points"]
    assert np.all(gt_seg_labels == process_seg_labels)
    assert process_box.shape == (1, 7)
    assert np.all(points.shape == process_points.shape)


def test_AssignSegLabel():
    assign_label = AssignSegLabel(
        bev_size=[50, 50],
        num_classes=2,
        class_names=[0, 1],
        point_cloud_range=[-5, -5, 0, 5, 5, 1],
        voxel_size=[0.2, 0.2],
    )
    out_data = assign_label(copy.deepcopy(data))
    assign_seg_labels = out_data["lidar"]["annotations"]["gt_seg_labels"]
    process_points = data["lidar"]["points"]
    assert np.all(process_points == points)
    assert assign_seg_labels.shape == (50, 50)
