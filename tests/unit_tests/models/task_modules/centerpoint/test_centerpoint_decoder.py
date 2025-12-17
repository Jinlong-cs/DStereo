import torch

from hat.models.task_modules.centerpoint.decoder import CenterPointDecoder

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

bev_size = (51.2, 51.2, 0.8)


class test_centerpoint_decoder:
    kpts_hm = torch.randn((1, 1, 40, 100))
    pts_offset = torch.randn((1, 2, 40, 100))
    int_offset = torch.randn((1, 2, 40, 100))
    decoder = CenterPointDecoder(
        class_names=CLASSES,
        tasks=tasks,
        bev_size=bev_size,
        out_size_factor=1,
        score_threshold=0.1,
        use_max_pool=True,
        nms_type=[
            "rotate",
            "rotate",
            "rotate",
            "circle",
            "rotate",
            "rotate",
        ],
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        nms_threshold=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
        decode_to_ego=True,
    )
    meta_data = {}
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

    decoder(preds, meta_data)
