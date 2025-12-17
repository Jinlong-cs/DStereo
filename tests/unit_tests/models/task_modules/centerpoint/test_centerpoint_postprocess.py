import torch

from hat.models.task_modules.centerpoint.bbox_coders import (
    CenterPointBBoxCoder,
)
from hat.models.task_modules.centerpoint.post_process import (
    CenterPointPostProcess,
)

tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]


def test_centerpoint_postprocess():
    bbox_coder = CenterPointBBoxCoder(
        pc_range=[-51.2, -51.2],
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_num=100,
        score_threshold=0.1,
        out_size_factor=4,
        voxel_size=[0.2, 0.2],
    )
    post_process = CenterPointPostProcess(
        tasks=tasks,
        norm_bbox=True,
        bbox_coder=bbox_coder,
        # test_cfg
        max_pool_nms=False,
        score_threshold=0.1,
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        out_size_factor=4,
        nms_type="rotate",
        pre_max_size=1000,
        post_max_size=83,
        nms_thr=0.2,
        box_size=9,
    )

    preds = list()
    for task in tasks:
        data = {}
        data["reg"] = torch.rand(1, 2, 32, 32)
        data["height"] = torch.rand(1, 1, 32, 32)
        data["dim"] = torch.rand(1, 3, 32, 32)
        data["vel"] = torch.rand(1, 2, 32, 32)
        data["rot"] = torch.rand(1, 2, 32, 32)
        data["heatmap"] = torch.rand(1, task["num_class"], 32, 32)
        preds.append(data)

    ret_list = post_process(preds)
    assert "bboxes" in ret_list[0]
    assert "scores" in ret_list[0]
    assert "labels" in ret_list[0]
