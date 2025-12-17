import os
import pickle

import numpy as np
import numpy.testing as npt
import pytest
import torch

from hat.data.transforms.auto_3dv import ANCOnlineMappingTargetGenerator
from hat.data.transforms.online_mapping_utils import (
    filter_short_instance,
    process_roadedges_occ,
)
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH
from tests.utils import check, check_shape

om_head_groups = {
    "lane": {
        "multi": True,
        "group": "lane",
        "key": "solid_lanes#lane_type",
        "loss_weight": 3,
        "max_patch_segment": 2,
        "loss_type": "ce_loss",
        "cls_remap": {
            "solid": 1,
            "dotted": 1,
            "dashed": 1,
            "solid_fishbone": 1,
            "ldotted_fishbone": 1,
            "boots_dots": 1,
            "other": 1,
            "unknown": 1,
        },
        "cls_list": ["lane"],
        "cls_colors": ["blue"],
        "cls_thr": 0.5,
    },
    "roadedge": {
        "multi": True,
        "group": "roadedge",
        "key": "roadedges#curb_category",
        "max_patch_segment": 2,
        "loss_type": "ce_loss",
        "cls_remap": {
            "groundside": 1,
            "roadside": 1,
            "cone": 1,
            "water_horse": 1,
            "guardrail": 1,
            "other": 1,
            "unknown": 1,
        },
        "cls_list": ["roadside"],
        "cls_colors": ["red"],
        "cls_thr": 0.5,
    },
    "lane_direction": {
        "group": "lane",
        "loss_weight": 0.5,
        "add_cost": False,
        "auto_class_weight": False,
        "global_auto_class_weight": [0.75, 2.0],
        "key": "solid_lanes#lane_direction",
        "cls_remap": {"unidirectional": 0, "bidirectional": 4},
        "cls_list": ["up", "down", "left", "right", "bidirection"],
        "cls_colors": ["white", "red", "green", "blue", "yellow"],
    },
    "roadedge_subtype": {
        "group": "roadedge",
        "key": "roadedges#curb_category",
        "loss_weight": 1.0,
        "auto_class_weight": False,
        "global_auto_class_weight": [1.0, 2.5],
        "cls_remap": {
            "groundside": 0,
            "roadside": 1,
            "cone": 2,
            "water_horse": 3,
            "guardrail": 4,
            "other": 5,
            "unknown": -1,
        },
        "cls_list": [
            "groundside",
            "roadside",
            "cone",
            "water_horse",
            "guardrail",
            "other",
        ],
        "cls_colors": [
            "white",
            "red",
            "blue",
            "orange",
            "green",
            "yellow",
        ],
    },
    "lane_color": {
        "group": "lane",
        "key": "solid_lanes#lane_color",
        "loss_weight": 0.5,
        "add_cost": False,
        "auto_class_weight": False,
        "global_auto_class_weight": [0.75, 2.0],
        "cls_remap": {
            "white": 0,
            "yellow": 1,
            "blue": 2,
            "orange": 2,
            "red": 2,
            "other": 2,
        },
        "cls_list": ["white", "yellow", "other"],
        "cls_colors": ["white", "yellow", "blue"],
        "only_focus_main_roi": True,
    },
    "lane_wide": {
        "group": "lane",
        "key": "solid_lanes#lane_width",
        "loss_weight": 0.5,
        "add_cost": False,
        "auto_class_weight": False,
        "global_auto_class_weight": [1.0, 3.0],
        "cls_remap": {"normal": 0, "wide": 1},
        "cls_colors": ["red", "blue"],
        "only_focus_main_roi": True,
    },
    "lane_double": {
        "group": "lane",
        "key": "solid_lanes#lane_flag",
        "loss_weight": 1.0,
        "add_cost": False,
        "auto_class_weight": False,
        "global_auto_class_weight": [1.0, 2.0],
        "cls_remap": {"single": 0, "double": 1, "triple": 1},
        "cls_list": ["single", "double"],
        "cls_colors": ["red", "blue"],
        "only_focus_main_roi": True,
    },
    "lane_property": {
        "group": "lane",
        "key": "solid_lanes#lane_properties",
        "loss_weight": 0.5,
        "add_cost": False,
        "auto_class_weight": False,
        "global_auto_class_weight": [1.0, 2.0],
        "cls_remap": {
            "general": 0,
            "stay": 1,
            "tide": 2,
            "three_color": 3,
            "bus": 4,
            "other": 5,
        },
        "cls_colors": [
            "white",
            "red",
            "blue",
            "orange",
            "green",
            "yellow",
        ],
        "only_focus_main_roi": True,
    },
    "lane_dashed": {
        "group": "lane",
        "key": "solid_lanes#lane_type",
        "loss_weight": 3,
        "add_cost": True,
        "auto_class_weight": False,
        "global_auto_class_weight": [1.0, 1.5],
        "cls_remap": {
            "solid": 0,
            "dotted": 1,
            "dashed": 1,
            "solid_fishbone": 0,
            "ldotted_fishbone": 1,
            "boots_dots": 1,
            "other": -1,
            "unknown": -1,
        },
        "cls_list": ["solid", "dashed"],
        "cls_colors": ["red", "blue"],
        "only_focus_main_roi": True,
    },
}
om_roadedge_occ_cfg = {
    "anchor_points": [-2, 1, 4],
    "check_close": False,
    "black_list": [],
    "sample_res": 0.5,
    "keep_horizon": True,
    "horizon_thresh": 28,
    "keep_horizon_ratio": 0.5,
}

om_short_filter_cfg = {
    "short_filter_thresh": 3.0,
    "short_keep_ratio": 0.8,
}


@pytest.mark.skip("file cannot load.")
@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HDLTAlgorithm bucket")
def test_om_target_generator(tmpdir):
    test_file = os.path.join(
        HAT_BUCKET_PATH, "unit_test_data/parse_om_test_data.pkl"
    )  # noqa
    assert os.path.exists(test_file)
    with open(test_file, "rb") as fid:
        test_data = pickle.load(fid)
    input_data = test_data["input"]
    image_files = input_data["image_files"]
    input_data["pack_dir"] = "/".join(image_files[0].split("/")[:-2])
    input_data["img_paths"] = [
        "/".join(i.split("/")[-2:]) for i in image_files
    ]
    target_data = test_data["output"]

    om_target_generator = ANCOnlineMappingTargetGenerator(
        out_size=(64, 64),
        roadedge_occ_cfg=om_roadedge_occ_cfg,
        head_groups=om_head_groups,
        visualize_output_dir=tmpdir,
    )

    output = om_target_generator(input_data)["om_target"]
    assert "cls" in output
    assert "instance" in output
    assert "r" in output
    assert "sin" in output
    assert "cos" in output

    npt.assert_equal(output["cls"], target_data["gt_online_mapping_cls"])
    npt.assert_equal(
        output["instance"],
        target_data["gt_online_mapping_instance"],
    )

    # TODO(yueyu.wang): fix ut error #
    with pytest.raises(AssertionError):
        npt.assert_equal(output["r"], target_data["gt_online_mapping_r"])
        npt.assert_equal(output["sin"], target_data["gt_online_mapping_sin"])
        npt.assert_equal(output["cos"], target_data["gt_online_mapping_cos"])


def test_om_generator():
    attr_list = ["cls", "instance", "r", "sin", "cos"]
    data = {
        "pack_dir": "pack",
        "img_paths": "om.jpg",
        "gt_online_mapping": {
            "data_version": "v1.0",
            "solid_lanes": [
                {
                    "lane_type": "solid",
                    "lane_direction": "unidirectional",
                    "lane_flag": "single",
                    "lane_width": "normal",
                    "lane_color": "unknown",
                    "lane_properties": "unknown",
                    "pts": np.array(
                        [[197, 15, 0, 145, 15, 0], [145, 15, 0, 117, 15, 0]],
                        np.float64,
                    ),
                }
            ],
            "roadedges": [
                {
                    "curb_category": "unknown",
                    "pts": np.array(
                        [[36, -71, 0, 35, -26, 0], [35, -26, 0, 34, -21, 0]],
                        np.float64,
                    ),
                }
            ],
        },
        "homography": torch.randn((1, 6, 3, 3)),
    }

    assign_near_instance_cfg = {
        "assign_blur_window": (3, 3),
        "assign_blur_thresh": 0.6,
        "multi_ins_grid_thresh": 6,
        "multi_lane_ignore_thresh": 8,
        "near_instance_weight": 1.4,
        "lane_ch2_weight": {
            "background": 1.0,
            "dilated": 1.0,
            "foreground": 0.95,
        },
        "ebd_channel_weight": [1.0, 0.3],
    }

    om_horizontal_cfg = {
        # (bottom, right, top, left)
        "horizontal_ignore_roi": [
            [52.5, -51.2, 100.8, -10.5],
            [52.5, 10.5, 100.8, 51.2],
        ],
        "horizontal_degree_thresh": 28,  # unit: degree
        "ratio_thresh": 0.70,
    }

    om_roi_weight_cfg = {
        "main_vcs_roi": {
            "region": [(-32.0, -13.2, 100.8, 13.2)],
            "weight": 2.0,
            "save_roi_mask": True,
        },
    }

    om_target_generator = ANCOnlineMappingTargetGenerator(
        out_size=(64, 64),
        head_groups=om_head_groups,
        vcs_range=(-32.0, -51.2, 100.8, 51.2),
        view_bev_size=(512, 512),
        roadedge_occ_cfg=om_roadedge_occ_cfg,
        visualize_output_dir=None,
        assign_near_instance_cfg=assign_near_instance_cfg,
        om_short_filter_cfg=om_short_filter_cfg,
        om_horizontal_cfg=om_horizontal_cfg,
        roi_weight_cfg=om_roi_weight_cfg,
    )

    gt = om_target_generator(data)
    assert "om_target" in gt
    for _, head_infos in om_head_groups.items():
        group = head_infos["group"]
        group_head = om_head_groups[group]
        max_patch_segment = group_head.get("max_patch_segment", 1)
        assert group in gt["om_target"]
        for attr in attr_list:
            assert attr in gt["om_target"][group]
            size = (max_patch_segment, 64, 64)
            check(gt["om_target"][group][attr], check_shape, shape=size)


def test_filter_short_instance():
    online_mapping_gt = {
        "roadedges": [{"pts": np.array([[0, 0, 1], [1, 1, 1], [2, 2, 0]])}]
    }
    online_mapping_gt = filter_short_instance(
        online_mapping_gt, om_short_filter_cfg
    )
    pts = online_mapping_gt["roadedges"][0]["pts"]
    assert np.sum(pts[:, -1] == 1) == 2


def test_process_roadedges_occ():
    online_mapping_gt = {
        "roadedges": [
            {
                "pts": np.array(
                    [
                        [134.17213315, -2.76899968, 0.0],
                        [100.8, -2.98349899, 1.0],
                        [34.24557523, -3.41127738, 1.0],
                        [-32.0, -3.99864533, 0.0],
                        [-72.71547978, -4.35965009, 0.0],
                        [-149.30749233, -4.96178168, 0.0],
                        [-155.97357969, -7.58132119, 0.0],
                        [-165.77034868, -7.63555613, 0.0],
                    ]
                )
            },
            {"pts": np.array([[80, -11, 1], [81, -41, 1]])},
        ]
    }
    online_mapping_gt = process_roadedges_occ(
        online_mapping_gt, om_roadedge_occ_cfg
    )[0]
    instance_filter = online_mapping_gt["roadedges"][1]["pts"]
    assert np.sum(instance_filter[:, -1] == 1) == 2
