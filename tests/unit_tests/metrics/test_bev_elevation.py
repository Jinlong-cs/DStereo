import cv2
import numpy as np
import pytest
import torch

from hat.metrics.bev_elevation import (
    ANCBevElevationMetric,
    ANCBEVFreespaceMetric,
    FreespaceBoundaryExtractor,
)
from hat.metrics.bev_vismask import ANCBevVismaskMetric

try:
    import aidisdk
except ImportError:
    aidisdk = None


def get_fake_data():
    voxel_gts = torch.ones(2, 512, 512)
    timestamps = torch.ones(2, 1)
    vis_mask_gts = {
        "vismask": torch.ones(2, 512, 512),
        "agent": torch.ones(2, 512, 512),
    }
    bev_freespace_gts = torch.ones(2, 512, 512)
    voxel_preds = torch.ones(2, 512, 512)
    vis_mask_preds = torch.ones(2, 512, 512)
    confidence_preds = torch.ones(2, 512, 512)

    return (
        voxel_gts,
        timestamps,
        vis_mask_gts,
        bev_freespace_gts,
        voxel_preds,
        vis_mask_preds,
        confidence_preds,
    )


@pytest.mark.parametrize(
    ["bev_elevation_type"],
    [
        pytest.param("HDE_45_7"),
        pytest.param("HDE_REG"),
    ],
)
@pytest.mark.skipif(aidisdk is None, reason="need aidisdk")
def test_bevelevation_metric(bev_elevation_type):
    torch.manual_seed(0)
    height_names = [
        "0|<10cm",
        "1|10~20cm",
        "2|20~30cm",
        "3|30~50cm",
        "4|50~100cm",
        "5|100~200cm",
        "6|>200cmm",
    ]
    eval_cfg = {
        "eval_vcs_range": (-12.8, -12.8, 25.6, 12.8),
        "depth_intervals": (0, 5, 10, 25, 50, 100),  # measured in meter
        "height_intervals": (0, 10, 25, 50),  # measured in meter
    }
    bev_elevation_metric = ANCBevElevationMetric(
        bev_elevation_type=bev_elevation_type,
        bev_size=(512, 512),
        vcs_range=(-30.0, -51.2, 72.4, 51.2),
        eval_cfg=eval_cfg,
        height_names=height_names,
        metrics_type=["elevation"],
        use_mask_type=["vismask"],
        ignore_index=255,
    )

    (
        voxel_gts,
        timestamps,
        vis_mask_gts,
        bev_freespace_gts,
        voxel_preds,
        vis_mask_preds,
        confidence_preds,
    ) = get_fake_data()

    bev_elevation_metric.update(
        voxel_gts,
        timestamps,
        vis_mask_gts,
        bev_freespace_gts,
        voxel_preds,
        vis_mask_preds,
        confidence_preds,
    )
    val = bev_elevation_metric.compute()
    if bev_elevation_type == "HDE_REG":
        assert val.summary is None
    else:
        assert val.summary


def gen_fake_freespace(task_name, gt_name, pred_name):
    """Generate fake gt and pred for freespace.

    Returns:
        ts: batch timestamps
        gt: batch ground truth labels.
        pred: batch pred labels.
    """
    ts = torch.tensor([[162490000000], [162490000001]], dtype=torch.float64)
    gt = {
        task_name: {gt_name: {gt_name: torch.randint(0, 2, (2, 384, 256))}},
        "timestamp": ts,
    }
    pred = {pred_name: [torch.randint(0, 2, (2, 384, 256))]}
    return gt, pred


def gen_fake_freespace_boundary(task_name, gt_name, pred_name):
    """Generate freespace boundary with special pattern."""

    def gen_specific_pattern(radius):
        center = (256, 128)
        front, rear = (41, 11)  # ego size
        front_center = (center[0] - front, center[1])
        rear_center = (center[0] + rear, center[1])
        circle_thickness = 7
        line_thickness = 5
        front_semi_circle = np.zeros((384, 256), np.uint8)
        cv2.circle(
            front_semi_circle,
            front_center[::-1],
            radius,
            1,
            thickness=circle_thickness,
        )
        front_semi_circle[front_center[0] :, :] = 0
        rear_semi_circle = np.zeros((384, 256), np.uint8)
        cv2.circle(
            rear_semi_circle,
            rear_center[::-1],
            radius,
            1,
            thickness=circle_thickness,
        )
        rear_semi_circle[: rear_center[0], :] = 0
        data = np.max(
            np.stack([front_semi_circle, rear_semi_circle], axis=-1), axis=-1
        )
        data[
            front_center[0] : rear_center[0] + 1,
            center[1]
            - radius
            - line_thickness : center[1]
            - radius
            + line_thickness,
        ] = 1
        data[
            front_center[0] : rear_center[0] + 1,
            center[1]
            + radius
            - line_thickness : center[1]
            + radius
            + line_thickness,
        ] = 1
        return data

    gt = gen_specific_pattern(80)
    pred = gen_specific_pattern(40)
    ts = torch.tensor([[162490000000]], dtype=torch.float64)
    gt = torch.tensor(np.expand_dims(gt, 0), dtype=torch.int64)
    pred = torch.tensor(np.expand_dims(pred, 0), dtype=torch.int64)
    gt = {task_name: {gt_name: {gt_name: gt}}, "timestamp": ts}
    pred = {pred_name: [pred]}
    return gt, pred


@pytest.mark.skipif(aidisdk is None, reason="need aidisdk")
def test_bevfreespace_metric():
    torch.manual_seed(0)
    eval_cfg = {
        "eval_vcs_range": (-12.8, -12.8, 25.6, 12.8),
        "smallobj_area_thresh_ub": 12.0,  # measured in m^2
        "smallobj_area_thresh_lb": 0.04,  # measured in m^2
        "smallobj_background_dilate_k": 9,  # in pixel
        "depth_intervals": (-32.4, 0, 20, 50, 80, 100),
        "dist_intervals": (0, 20),  # measured in meters
        "pr_cfg": {
            "area_intervals": (0.5, 1.0, 4.0, 12.0),  # measured in m^2
            "dist_interval_dxy_threshs": {
                (0, 1000): 2.0,
            },  # object center dxy thresh for pr matching, measured in meter
        },
        "boundary_cfg": {
            "dist_interval_dxy_threshs": {
                (0, 20): 10.0,
            },  # dxy thresh for boundary point matching along ray
        },
        "metric_modes": ["iou", "boundary_error"],
    }
    task_name = "bev_freespace"
    gt_name = "gt_bev_freespace"
    pred_name = "pred_bev_freespace"
    bev_freespace_metric = ANCBEVFreespaceMetric(
        name="BEVFreespaceMIOU",
        seg_class=["0", "1"],
        bev_size=(384, 256),
        vcs_range=(-12.8, -12.8, 25.6, 12.8),
        eval_cfg=eval_cfg,
        task_name=task_name,
        gt_name=gt_name,
        pred_name=pred_name,
        ignore_index=255,
        global_ignore_index=255,
    )
    # iou eval
    label, preds = gen_fake_freespace(task_name, gt_name, pred_name)
    bev_freespace_metric.update(label, preds)
    val = bev_freespace_metric.compute()
    if bev_freespace_metric.obj_area_thresh_ub > 0:
        assert len(val.tables) == 2
    else:
        assert len(val.tables) == 1

    # boundary eval
    bev_freespace_metric.reset()
    gt, pred = gen_fake_freespace_boundary(task_name, gt_name, pred_name)
    bev_freespace_metric.update(gt, pred)
    bev_freespace_metric.compute()
    assert abs(bev_freespace_metric.boundary_error_mean[0] - 4.0) < 0.03


def gen_fake_vismask(task_name, gt_name, pred_name):
    """Generate fake gt and pred for vismask.

    Returns:
        ts: batch timestamps
        gt: batch ground truth labels.
        pred: batch pred labels.
    """
    ts = torch.tensor([[162490000.000], [162490000.001]], dtype=torch.float64)
    color_imgs_f = torch.ones(2, 3, 512, 960)  # [2, 3, 512, 960]
    color_imgs_fl = torch.ones(2, 3, 512, 960)  # [2, 3, 512, 960]
    color_imgs_fr = torch.ones(2, 3, 512, 960)  # [2, 3, 512, 960]
    color_imgs_r = torch.ones(2, 3, 512, 960)  # [2, 3, 512, 960]
    color_imgs_rl = torch.ones(2, 3, 512, 960)  # [2, 3, 512, 960]
    color_imgs_rr = torch.ones(2, 3, 512, 960)  # [2, 3, 512, 960]
    color_imgs = [
        [
            color_imgs_f,
            color_imgs_fl,
            color_imgs_fr,
            color_imgs_r,
            color_imgs_rl,
            color_imgs_rr,
        ]
    ]
    gt = {
        task_name: {
            gt_name: {
                "vismask": torch.randint(0, 2, (2, 512, 512)),
                "agent": torch.randint(0, 2, (2, 512, 512)),
            },
        },
        "timestamp": ts,
        "color_imgs": color_imgs,
    }
    pred = {
        pred_name: [torch.randint(0, 2, (2, 512, 512))],
    }
    return gt, pred


def test_vismask_metric():
    torch.manual_seed(0)
    bev_vismask_metric = ANCBevVismaskMetric(
        bev_size=(512, 512),
        vcs_range=(-30, -51.2, 72.4, 51.2),
        eval_vcs_range=(-12.8, -12.8, 25.6, 12.8),
    )
    task_name = "bev_vismask"
    gt_name = "gt_bev_elevation_vismask"
    pred_name = "bev_vismask"
    (
        vis_mask_gts,
        vis_mask_preds,
    ) = gen_fake_vismask(task_name, gt_name, pred_name)
    bev_vismask_metric.update(
        vis_mask_gts,
        vis_mask_preds,
    )
    val = bev_vismask_metric.compute()

    assert val.summary


def test_boundary_extrator():
    data = np.ones((384, 256), np.uint8)
    data[:, 64:192] = 0
    data[128:168, 80:100] = 1
    data[200:210, 130:140] = 1
    data[:, 192] = 255

    boundary_extractor = FreespaceBoundaryExtractor(
        center=(128, 256),
        spatial_resolution=(0.1, 0.1),
        save_dir="./tmp_output/vis_boundary",
        polar_ray_num=720,
        side_ray_num=40,
    )
    bdry_pts, bdry_pts_radius, _ = boundary_extractor.get_boundary_points(
        data, vis_boundary=True
    )
    assert abs(bdry_pts.max() - 25.6) < 1e-1
    assert abs(bdry_pts_radius.max() - 22.5) < 1e-1
