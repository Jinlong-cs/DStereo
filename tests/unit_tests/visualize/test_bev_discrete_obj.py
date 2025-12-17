import os

import numpy as np
import torch

from hat.callbacks.task_visualize.bev_discrete_obj import (
    ANCBevObjVisualize,
    vis_disc_obj,
)


def test_bev_discrete_obj(tmpdir):
    color_map = {
        "static_obstacle": {
            0: (100, 0, 80),  # OpenParkingLock
            1: (150, 50, 150),  # CloseParkingLock
            2: (0, 128, 128),  # CementColumn
        },
    }
    id2label = {
        "static_obstacle": {
            0: "Open_PL",
            1: "Close_PL",
            2: "CColumn",
        },
    }
    vis = ANCBevObjVisualize(
        bev_size=(512, 512),
        vcs_range=(-30.0, -51.2, 72.4, 51.2),
        res_key2anno_name={
            "bev_static_obstacle": "annos_bev_discrete_obj",
        },
        color_map=color_map,
        id2label=id2label,
        camera_view_names=[
            "camera_front_left",
            "camera_front",
            "camera_front_right",
            "camera_rear_left",
            "camera_rear",
            "camera_rear_right",
            "fisheye_front",
            "fisheye_rear",
            "fisheye_left",
            "fisheye_right",
            "camera_front_30fov",
        ],
        out_ori_img_size=(2048 // 2, 1280 // 2),
    )
    batch = (
        {
            "ipm": torch.randint(0, 255, (1, 512, 512, 3)),
            "timestamp": torch.tensor([1]),
            "img": [torch.randint(0, 255, (1, 3, 512, 960))],
            "side_img": [torch.randint(0, 255, (5, 3, 512, 960))],
            "round_img": [torch.randint(0, 255, (4, 3, 512, 960))],
            "narrow_img": [torch.randint(0, 255, (1, 3, 512, 960))],
        },
        "bev_static_obstacle",
    )

    prefix = "bev_static_obstacle_bev_stage2_static_obstacle_head_predict_pred"
    results = {
        f"{prefix}_bev_discobj_ct": [np.random.randn(20, 2)],
        f"{prefix}_bev_discobj_wh": [np.random.randn(20, 2)],
        f"{prefix}_bev_discobj_rot": [np.random.randn(20)],
        f"{prefix}_bev_discobj_score": [np.random.randn(20)],
        f"{prefix}_bev_discobj_cls_id": [np.random.randint(0, 3, (20))],
    }
    vis(batch, results, batch[1], tmpdir)

    assert os.path.exists(os.path.join(tmpdir, "static_obstacle_1000.jpg"))


def test_vis_disc_obj():
    img = np.zeros((512, 512, 3), np.uint8)
    vcs_loc = [0, 0]
    vcs_dim = [10, 5]
    yaw = 0
    vcs_range = (-30.0, -51.2, 72.4, 51.2)
    m_perpixel = [0.2, 0.2]
    vis_disc_obj(
        img,
        vcs_loc,
        vcs_dim,
        yaw,
        vcs_range,
        m_perpixel,
    )

    assert img.shape == (512, 512, 3)
