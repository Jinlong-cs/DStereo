# -*- coding: utf-8 -*-
# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np

from hat.registry import build_from_registry


def test_image_transform():
    image_width, image_height, pitch = 960, 640, 1
    cfg = dict(
        type="CameraStandardization",
        image_width=image_width,
        image_height=image_height,
        pitch=pitch / 180 * np.pi,
        default_calib={
            "camera_x": 0.0,
            "camera_y": 0.0,
            "camera_z": 0.0,
            "center_u": 960,
            "center_v": 640.0,
            "distort": [
                0.31648436188697815,
                -0.03045056015253067,
                2.9076751161483116e-05,
                -2.241096444777213e-05,
                0.0030021648854017258,
                0.6827608942985535,
                0.0,
                0.0,
            ],
            "focal_u": 1149.9775390625,
            "focal_v": 1149.9775390625,
            "fov": 99.30000305175781,
            "image_height": 1280,
            "image_width": 1920,
            "pitch": 0.0,
            "roll": 0.0,
            "valid_height": [1280, 1280],
            "vcs": {
                "rotation": [0.0, 0.0, 0.8257636427879333],
                "translation": [2.2606000900268555, 0.9092429876327515, 0.0],
            },
            "yaw": -0.0,
        },
        meta_key=("anno", "meta"),
    )
    cam_stand = build_from_registry(cfg)
    data = {
        "img": np.concatenate(
            [
                np.indices([1280, 1920]).transpose([1, 2, 0])[..., ::-1],
                np.zeros([1280, 1920, 1]),
            ],
            axis=-1,
        ),
        "anno": {
            "meta": {},
            "objects": [
                {"bbox": [0, 0, 100, 100], "bbox_2d": [0, 0, 100, 100]},
                {
                    "bbox": [0, 0, 200, 200],
                    "bbox_2d": None,
                    "in_camera": {"location": [1, 1, 1], "rotation_y": 0},
                },
            ],
        },
    }
    data_res = cam_stand(data)
    print(data["anno"])
    assert (
        data_res["img"].shape[0] == image_height
        and data_res["img"].shape[1] == image_width
    )
    assert (
        data["anno"]["objects"][0]["bbox"]
        == data["anno"]["objects"][0]["bbox_2d"]
    )
    assert np.allclose(
        data["anno"]["objects"][1]["bbox"],
        [7.659883975982666, 0.0, 192.45655584335327, 186.34500122070312],
        rtol=1e-03,
        atol=1e-03,
    )
    assert (
        abs(data["anno"]["objects"][1]["in_camera"]["rotation_y"] - 0) < 1e-5
    )
