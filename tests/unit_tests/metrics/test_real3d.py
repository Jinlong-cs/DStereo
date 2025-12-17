import torch

from hat.registry import build_from_registry


def get_config():
    cfg = dict(
        type="Real3dEval",
        need_eval_categories={
            #  category_names_to_val: [categories_id_in_gt,
            #                          category_id_in_prediction]
            "Pedestrian": [[1], 0],
            "MergedCar": [[2, 4, 5, 6], 1],
            "Cyclist": [[3], 2],
            "Bus": [[4], 1],
            "Truck": [[5], 1],
            "SpecialCar": [[6], 1],
            # 'Tricycle': [[7], 0],
        },
        eval_camera_names=("__front__", "__front_right__", "__right__"),
        depth_intervals=(20, 50, 80, 130),
        iou_threshold=0.2,
        score_threshold=0.1,
        gt_max_depth=300,
        metrics=("dxyp", "drot"),
        save_path=None,
        num_dist=4,
    )
    return cfg


def get_gt_and_pred():
    gt = {
        "image_name": ["pandar64_20200701__front_left__1588996959200.jpg"],
        "image_height": [1080],
        "image_width": [1920],
        "color": ["bgr"],
        "annotations": [
            [
                {
                    "timestamp": "1588996959200",
                    "image_id": "1588996959200070",
                    "id": "1588996959200070001",
                    "category_id": 2,
                    "dim": [1.8275, 1.7910, 4.6636],
                    "bbox": [1, 612, 390, 340],
                    "depth": 9.2800,
                    "alpha": -3.3953,
                    "location": [-6.3448, 2.2290, 9.2800],
                    "location_offset": [0.143961, -0.14451, 0.0],
                    "rotation_y": 2.2882,
                    "distCoeffs": [-0.34998, 0.13789, 0.00013, -0.000331, 0.0],
                    "ignore": False,
                    "occlusion": "full_visible",
                    "bbox_2d": [1, 612, 390, 340],
                },
                {
                    "timestamp": "1588996959200",
                    "image_id": "1588996959200070",
                    "id": "1588996959200070002",
                    "category_id": 4,
                    "dim": [1.6889, 1.8096, 4.7735],
                    "bbox": [675.7658, 618.8544, 425, 200],
                    "depth": 14.1608,
                    "alpha": -3.7623,
                    "location": [-0.8548, 2.2934, 14.1608],
                    "location_offset": [0.143969, -0.144511, 0.0],
                    "rotation_y": 2.4606,
                    "distCoeffs": [-0.35, 0.13781, 0.00013, -0.00033, 0.0],
                    "ignore": False,
                    "occlusion": "full_visible",
                    "bbox_2d": [675.7658, 618.8544, 425, 200],
                },
            ]
        ],
        "calibration": torch.tensor(
            [
                [
                    [1.5439e03, 0.00, 9.6403e02, 0.000],
                    [0.0000e00, 1.5439e03, 5.6257e02, 0.0000e00],
                    [0.000, 0.00, 1.0000e00, 0.0],
                ]
            ]
        ),
        "dist_coeffs": torch.tensor([[-0.3500, 0.1378, 0.0001, -0.0003]]),
        "image_id": ["1588996959200070"],
        "ignore_mask": None,
        "image_transform": None,
        "target": None,
        "img": None,
    }
    det = {
        "dim": torch.tensor(
            [[[1.8275, 1.7910, 4.6636], [1.6889, 1.8096, 4.7735]]]
        ),
        "category_id": torch.tensor([[1, 1]]),
        "score": torch.tensor([[0.5995, 0.5978]]),
        "bbox": torch.tensor(
            [
                [
                    [19.2579, 620.9153, 348.7421, 883.0847],
                    [675.7658, 618.8544, 1100.2341, 821.1456],
                ]
            ],
            dtype=torch.float64,
        ),
        "center": torch.tensor(
            [[[184.0, 752.0], [888.0, 720.0]]], dtype=torch.float64
        ),
        "dep": torch.tensor([[9.2800, 14.1608]]),
        "alpha": torch.tensor([[-3.3953, -3.7623]]),
        "location": torch.tensor(
            [[[-6.3448, 2.2290, 9.2800], [-0.8548, 2.2934, 14.1608]]],
            dtype=torch.float64,
        ),
        "rotation_y": torch.tensor([[2.2882, 2.4606]], dtype=torch.float64),
        "image_id": ["1588996959200070"],
    }
    return gt, det


def test_real3d_eval_init():
    cfg = get_config()
    instance = build_from_registry(cfg)
    assert instance.depth_intervals == (20, 50, 80, 130)
    assert instance.metrics == ("dxyp", "drot")


def test_real3d_eval_reset():
    cfg = get_config()
    real3d_eval = build_from_registry(cfg)
    real3d_eval.reset()


def test_real3d_update_get():
    cfg = get_config()
    cfg["eval_camera_names"] = ("__front_left__",)
    cfg["need_eval_categories"] = {
        "MergedCar": [[2, 4, 5, 6], 1],
    }
    real3d_eval = build_from_registry(cfg)

    gt, det = get_gt_and_pred()
    real3d_eval.update(gt, det)
    names, _ = real3d_eval.get()
    assert names[0] == "Camera+Category"


if __name__ == "__main__":
    import pytest

    pytest.main(["-s", __file__])
