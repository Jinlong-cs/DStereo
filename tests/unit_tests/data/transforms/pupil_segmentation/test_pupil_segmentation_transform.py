import cv2
import numpy as np

from hat.registry import build_from_registry


def test_generate_ellipse_mask():
    generate_mask_func = build_from_registry(dict(type="GenerateEllipseMask"))
    img = np.random.rand(5, 5, 3)
    gt_mask = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    ellipse_param = np.array([1, 1, 1, 1, 0])
    data = {"img": img, "gt_pupil_ellipse_param": ellipse_param}
    data = generate_mask_func(data)
    assert (gt_mask - data["gt_pupil_mask"]).sum() == 0


def test_generate_edge_weight_map():
    generate_map_func = build_from_registry(
        dict(
            type="GenerateEdgeWeightMap",
        )
    )
    mask = np.zeros((10, 10))
    mask = cv2.ellipse(mask, (5, 5), (3, 3), 0, 0, 360, 1, 1)
    spatial_weights = cv2.Canny(mask.astype(np.uint8), 0, 1) / 255
    gt_spat_weights = (
        1 + cv2.dilate(spatial_weights, (3, 3), iterations=1) * 20
    )
    data = {"gt_pupil_mask": mask}
    data = generate_map_func(data)
    assert (gt_spat_weights - data["spat_weights"]).sum() == 0


def test_generate_dist_map():
    generate_dist_func = build_from_registry(
        dict(
            type="GenerateDistMap",
        )
    )
    mask = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
    a = np.sqrt(2) / np.sqrt(18)
    b = 1 / np.sqrt(18)
    gt_dist_map = np.array(
        [
            [
                [a, b, b, a],
                [b, 0.0, 0.0, b],
                [b, 0.0, 0.0, b],
                [a, b, b, a],
            ]
        ]
    )
    data = {"gt_pupil_mask": mask}
    data = generate_dist_func(data)
    assert (gt_dist_map - data["dist_map"]).sum() < 1e-4


def test_norm_ellipse_param():
    norm_param_func = build_from_registry(dict(type="NormEllipseParam"))
    param = np.array(
        [
            43.043212890625,
            23.693267822265625,
            13.157154083251953,
            9.960553169250488,
            56.03654861450195,
        ]
    )
    gt_norm_param = np.array(
        [0.3451004, -0.25958538, 0.31126729, 0.41116107, 2.5488186]
    )
    data = {"img": np.random.rand(64, 64, 3), "gt_pupil_ellipse_param": param}
    data = norm_param_func(data)
    assert (
        abs(gt_norm_param - data["gt_norm_pupil_ellipse_param"]).sum() < 1e-4
    )
