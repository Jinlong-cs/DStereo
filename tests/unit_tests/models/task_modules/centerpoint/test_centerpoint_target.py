import torch

from hat.models.task_modules.centerpoint.target import (
    CenterPointLidarTarget,
    CenterPointTarget,
)

CLASSES = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)

tasks = [
    dict(name="car", num_class=1, class_names=["car"]),
    dict(
        name="truck",
        num_class=2,
        class_names=["truck", "construction_vehicle"],
    ),
    dict(name="bus", num_class=2, class_names=["bus", "trailer"]),
    dict(name="barrier", num_class=1, class_names=["barrier"]),
    dict(name="bicycle", num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(
        name="pedestrian",
        num_class=2,
        class_names=["pedestrian", "traffic_cone"],
    ),
]


def test_centerpoint_target():
    target = CenterPointTarget(
        class_names=CLASSES,
        tasks=tasks,
        gaussian_overlap=0.1,
        min_radius=3,
        out_size_factor=1,
        norm_bbox=True,
        max_num=500,
        bbox_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
    )
    preds = list()
    for task in tasks:
        data = {}
        data["reg"] = torch.rand(1, 2, 32, 60)
        data["height"] = torch.rand(1, 1, 32, 60)
        data["dim"] = torch.rand(1, 3, 32, 60)
        data["vel"] = torch.rand(1, 2, 32, 60)
        data["rot"] = torch.rand(1, 2, 32, 60)
        data["heatmap"] = torch.rand(1, task["num_class"], 32, 60)
        preds.append(data)
    label = [torch.randint(10, (10, 10))]
    results = target(label, preds)
    for result in results:
        assert "cls_target" in result
        assert "reg_target" in result
        assert "task_name" in result
        assert "pos_indices" in result


def test_centerpoint_lidar_target():
    target = CenterPointLidarTarget(
        grid_size=[512, 512, 1],
        voxel_size=[0.2, 0.2, 8],
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        tasks=[dict(num_class=1, class_names=["car"])],
        dense_reg=1,
        max_objs=20,
        gaussian_overlap=0.1,
        min_radius=2,
        out_size_factor=4,
        norm_bbox=True,
        with_velocity=True,
    )
    gt_boxes = [torch.randn(20, 9)]
    gt_classess = [torch.randint(1, 11, (20,))]

    heatmaps, anno_boxes, inds, masks = target(gt_boxes, gt_classess)
    assert heatmaps is not None
    assert anno_boxes is not None
    assert inds is not None
    assert masks is not None
