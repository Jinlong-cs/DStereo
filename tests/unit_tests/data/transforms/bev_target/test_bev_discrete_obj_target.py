# -*- coding: utf-8 -*-
# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np
import pytest

from hat.data.transforms.auto_3dv import (
    ANCBevDiscreteTargetGenerator,
    ANCBevDiscreteWithClsTargetGenerator,
)
from tests import HAT_BUCKET_EXISTS


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HDLTAlgorithm bucket")
def test_bev_discrete_target_generator():
    name2group = {
        "arrows": "det",
        "crosswalks": "det",
        "speedbumps": "det",
        "noparking_lines": "det",
        "diamond_markings": "det",
        "inverted_triangle_markings": "det",
        "exclusive_lane_signs": "det",
        "parking_locks": "det",
        "comment_columns": "det",
        "intersections": "freespace",
        "stoplines": "om",
        "roadmarking": "det",
    }
    datasets = {
        "stopline": {
            "gt_bev_discrete_raw": {
                "om": {
                    "stoplines": [
                        [0, 0, 1, 0.5],
                        [0, 0, 1, -0.5],
                        [0, 0, -1, 0.5],
                        [0, 0, -1, -0.5],
                    ],
                },
            },
            "timestamp": np.array([1640942465.613]),
        },
        "roadmarking": {
            "gt_bev_discrete_raw": {
                "det": {
                    "crosswalks": [
                        [
                            [-1.5, -14.0, -4.0, 14.0],
                            [-4.0, 14.0, 1.3, 14.1],
                            [1.3, 14.1, 4.1, -14.1],
                        ],
                    ],
                    "speedbumps": [
                        [
                            [-2.9, -32.3, 2.1, -32.4],
                            [2.1, -32.4, 3.1, 32.5],
                            [3.1, 32.5, -2.3, 32.1],
                        ]
                    ],
                },
            },
            "timestamp": np.array([1640942465.613]),
        },
        "arrow": {
            "gt_bev_discrete_raw": {
                "det": {
                    "arrows": [
                        {
                            "type": "Other",
                            "pts": [
                                [3.0, 0.7, -1.3, -0.8],
                                [-1.3, -0.8, -1.6, -0.2],
                                [-1.6, -0.2, 2.8, 1.3],
                            ],
                        },
                        {
                            "type": "Straight",
                            "pts": [
                                [3.0, -0.6, -1.5, -0.4],
                                [-1.5, -0.4, -1.5, 0.5],
                                [-1.5, 0.5, 3.1, 0.3],
                            ],
                        },
                        {
                            "type": "Turnleft",
                            "pts": [
                                [-2.9, 0.7, 1.5, 0.6],
                                [1.5, 0.6, 1.5, -0.6],
                                [1.5, -0.6, -2.9, -0.6],
                            ],
                        },
                        {
                            "type": "Turnright",
                            "pts": [
                                [-2.8, 0.5, 1.4, 0.6],
                                [1.4, 0.6, 1.4, -0.6],
                                [1.4, -0.6, -2.8, -0.6],
                            ],
                        },
                    ]
                },
            },
            "timestamp": np.array([1640942465.613]),
        },
        "crosswalk": {
            "gt_bev_discrete_raw": {
                "det": {
                    "crosswalks": [
                        [
                            [-1.5, -14.0, -4.0, 14.0],
                            [-4.0, 14.0, 1.3, 14.1],
                            [1.3, 14.1, 4.1, -14.1],
                        ],
                        [
                            [-2.9, -32.3, 2.1, -32.4],
                            [2.1, -32.4, 3.1, 32.5],
                            [3.1, 32.5, -2.3, 32.1],
                        ],
                    ],
                },
            },
            "timestamp": np.array([1640942465.613]),
        },
        "intersection": {
            "gt_bev_discrete_raw": {
                "freespace": {
                    "intersections": [
                        [
                            [-3.5, -7.6, 1.8, -7.6],
                            [1.8, -7.6, 3.1, -3.4],
                            [3.1, -3.4, 3.3, 8.3],
                            [3.3, 8.3, -4.3, 8.3],
                            [-4.3, 8.3, -4.4, -3.6],
                        ],
                        [
                            [6.3, -10.2, -4.9, -10.3],
                            [-4.9, -10.3, -12.6, -3.3],
                            [-12.6, -3.3, -11.8, 9.9],
                            [-11.8, 9.9, 16.8, 9.8],
                            [16.8, 9.8, 18.5, -2.1],
                        ],
                    ],
                },
            },
            "timestamp": np.array([1640942465.613]),
        },
    }

    # bev discrete object data (e.g., crosswalk, arrow, etc)
    category2id_maps = {
        "stopline": {
            "Stopline": 0,
        },
        "arrow": {
            "Other": 0,
            "Straight": 1,
            "Turnleft": 2,
            "Turnright": 3,
        },
        "crosswalk": {
            "Crosswalk": 0,
        },
        "intersection": {
            "Junction": 0,
        },
        "roadmarking": {
            "Crosswalk": 0,
            "Stopline": 1,
            "DiamondMarking": 2,
            "InvertedTriangleMarking": 3,
            "SpeedBump": 4,
        },
    }
    name2labels = {
        "stopline": {"stoplines": "Stopline"},
        "crosswalk": {"crosswalks": "Crosswalk"},
        "intersection": {"intersections": "Junction"},
        "arrow": {"arrows": "ARROW"},
        "roadmarking": {
            "crosswalks": "Crosswalk",
            "stoplines": "Stopline",
            "diamond_markings": "DiamondMarking",
            "inverted_triangle_markings": "InvertedTriangleMarking",
            "speedbumps": "SpeedBump",
        },
    }
    valid_vcs_range_percls = {
        "roadmarking": {
            "Crosswalk": (-12.8, -12.8, 25.6, 12.8),
            "Stopline": (-6.4, -6.4, 12.8, 6.4),
            "DiamondMarking": (-6.4, -6.4, 12.8, 6.4),
            "InvertedTriangleMarking": (-6.4, -6.4, 12.8, 6.4),
            "SpeedBump": (-6.4, -6.4, 12.8, 6.4),
        },
    }
    for category, dataset in datasets.items():
        discrete_target_generator = ANCBevDiscreteTargetGenerator(
            num_classes=len(category2id_maps[category].keys()),
            bev_size=(192, 128),
            vcs_range=(-12.8, -12.8, 25.6, 12.8),
            category2id_map=category2id_maps[category],
            max_objs=100,
            vcs_bbox_area_thresh=0,
            name2label=name2labels[category],
            name2group=name2group,
            ego_ignore_range=(-0.6, -0.5, 2.0, 0.5),
            vis_mask_vcs_range_cfg={
                (1024, 768): (-51.2, -76.8, 153.6, 76.8),
                (1024, 800): (-51.2, -80.0, 153.6, 80.0),
            },
            valid_vcs_range_percls=valid_vcs_range_percls.get(category, None),
            psc_phase_factor=1,
            N_steps_PSC_rot=3,
        )
        data = discrete_target_generator(dataset)
        yaw = (
            data["annos_bev_discrete_obj"]["vcs_discobj_yaw"][:5] * 180 / np.pi
        )
        if category == "stopline":
            assert np.abs(yaw[0] - 26) < 1
            assert np.abs(yaw[1] + 26) < 1
            assert np.abs(yaw[2] + 26) < 1
            assert np.abs(yaw[3] - 26) < 1
        if category == "arrow":
            assert np.abs(yaw[0] + 160) < 1
            assert np.abs(yaw[1] - 177) < 1
            assert np.abs(yaw[2] + 1) < 1
            assert np.abs(yaw[3] - 1) < 1
        if category == "crosswalk":
            assert np.abs(yaw[0] + 84) < 1
            assert np.abs(yaw[1] - 89) < 1
        if category == "intersection":
            assert np.abs(yaw[0] - 89) < 1
            assert np.abs(yaw[1] + 0) < 1
        data_gt = data["gt_bev_discrete_obj"]
        assert data_gt["bev_discobj_rot"].shape == (192, 128, 3)
        if valid_vcs_range_percls.get(category, None) is not None:
            assert "bev_discobj_cls_valid_mask" in data_gt
            bev_discobj_cls_valid_mask = data_gt["bev_discobj_cls_valid_mask"]
            assert bev_discobj_cls_valid_mask.shape == (192, 128, 5)
            assert np.sum(bev_discobj_cls_valid_mask) == 49152


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HDLTAlgorithm bucket")
def test_bev_discrete_with_cls_target_generator():
    name2group = {
        "arrows": "det",
        "crosswalks": "det",
        "speedbumps": "det",
        "noparking_lines": "det",
        "diamond_markings": "det",
        "inverted_triangle_markings": "det",
        "exclusive_lane_signs": "det",
        "parking_locks": "det",
        "comment_columns": "det",
        "intersections": "freespace",
        "stoplines": "om",
        "roadmarking": "det",
    }
    datasets = {
        "stopline": {
            "gt_bev_discrete_raw": {
                "om": {
                    "stoplines": [
                        [0, 0, 1, 0.5],
                        [0, 0, 1, -0.5],
                        [0, 0, -1, 0.5],
                        [0, 0, -1, -0.5],
                    ],
                },
            },
            "timestamp": np.array([1640942465.613]),
        },
        "roadmarking": {
            "gt_bev_discrete_raw": {
                "det": {
                    "crosswalks": [
                        [
                            [-1.5, -14.0, -4.0, 14.0],
                            [-4.0, 14.0, 1.3, 14.1],
                            [1.3, 14.1, 4.1, -14.1],
                        ],
                    ],
                    "speedbumps": [
                        [
                            [-2.9, -32.3, 2.1, -32.4],
                            [2.1, -32.4, 3.1, 32.5],
                            [3.1, 32.5, -2.3, 32.1],
                        ]
                    ],
                },
            },
            "timestamp": np.array([1640942465.613]),
        },
        "arrow": {
            "gt_bev_discrete_raw": {
                "det": {
                    "arrows": [
                        {
                            "type": "Other",
                            "pts": [
                                [3.0, 0.7, -1.3, -0.8],
                                [-1.3, -0.8, -1.6, -0.2],
                                [-1.6, -0.2, 2.8, 1.3],
                            ],
                        },
                        {
                            "type": "Straight",
                            "pts": [
                                [3.0, -0.6, -1.5, -0.4],
                                [-1.5, -0.4, -1.5, 0.5],
                                [-1.5, 0.5, 3.1, 0.3],
                            ],
                        },
                        {
                            "type": "Turnleft",
                            "pts": [
                                [-2.9, 0.7, 1.5, 0.6],
                                [1.5, 0.6, 1.5, -0.6],
                                [1.5, -0.6, -2.9, -0.6],
                            ],
                        },
                        {
                            "type": "Turnright",
                            "pts": [
                                [-2.8, 0.5, 1.4, 0.6],
                                [1.4, 0.6, 1.4, -0.6],
                                [1.4, -0.6, -2.8, -0.6],
                            ],
                        },
                    ]
                },
            },
            "timestamp": np.array([1640942465.613]),
        },
        "crosswalk": {
            "gt_bev_discrete_raw": {
                "det": {
                    "crosswalks": [
                        [
                            [-1.5, -14.0, -4.0, 14.0],
                            [-4.0, 14.0, 1.3, 14.1],
                            [1.3, 14.1, 4.1, -14.1],
                        ],
                        [
                            [-2.9, -32.3, 2.1, -32.4],
                            [2.1, -32.4, 3.1, 32.5],
                            [3.1, 32.5, -2.3, 32.1],
                        ],
                    ],
                },
            },
            "timestamp": np.array([1640942465.613]),
        },
        "intersection": {
            "gt_bev_discrete_raw": {
                "freespace": {
                    "intersections": [
                        [
                            [-3.5, -7.6, 1.8, -7.6],
                            [1.8, -7.6, 3.1, -3.4],
                            [3.1, -3.4, 3.3, 8.3],
                            [3.3, 8.3, -4.3, 8.3],
                            [-4.3, 8.3, -4.4, -3.6],
                        ],
                        [
                            [6.3, -10.2, -4.9, -10.3],
                            [-4.9, -10.3, -12.6, -3.3],
                            [-12.6, -3.3, -11.8, 9.9],
                            [-11.8, 9.9, 16.8, 9.8],
                            [16.8, 9.8, 18.5, -2.1],
                        ],
                    ],
                },
            },
            "timestamp": np.array([1640942465.613]),
        },
    }

    # bev discrete object data (e.g., crosswalk, arrow, etc)
    category2id_maps = {
        "stopline": {
            "Stopline": 0,
        },
        "arrow": {
            "Other": 0,
            "Straight": 1,
            "Turnleft": 2,
            "Turnright": 3,
        },
        "crosswalk": {
            "Crosswalk": 0,
        },
        "intersection": {
            "Junction": 0,
        },
        "roadmarking": {
            "Crosswalk": 0,
            "Stopline": 1,
            "DiamondMarking": 2,
            "InvertedTriangleMarking": 3,
            "SpeedBump": 4,
        },
    }
    name2labels = {
        "stopline": {"stoplines": "Stopline"},
        "crosswalk": {"crosswalks": "Crosswalk"},
        "intersection": {"intersections": "Junction"},
        "arrow": {"arrows": "ARROW"},
        "roadmarking": {
            "crosswalks": "Crosswalk",
            "stoplines": "Stopline",
            "diamond_markings": "DiamondMarking",
            "inverted_triangle_markings": "InvertedTriangleMarking",
            "speedbumps": "SpeedBump",
        },
    }
    valid_vcs_range_percls = {
        "roadmarking": {
            "Crosswalk": (-12.8, -12.8, 25.6, 12.8),
            "Stopline": (-6.4, -6.4, 12.8, 6.4),
            "DiamondMarking": (-6.4, -6.4, 12.8, 6.4),
            "InvertedTriangleMarking": (-6.4, -6.4, 12.8, 6.4),
            "SpeedBump": (-6.4, -6.4, 12.8, 6.4),
        },
    }
    for category, dataset in datasets.items():
        discrete_target_generator = ANCBevDiscreteWithClsTargetGenerator(
            num_classes=len(category2id_maps[category].keys()),
            bev_size=(192, 128),
            vcs_range=(-12.8, -12.8, 25.6, 12.8),
            category2id_map=category2id_maps[category],
            max_objs=100,
            vcs_bbox_area_thresh=0,
            name2label=name2labels[category],
            name2group=name2group,
            ego_ignore_range=(-0.6, -0.5, 2.0, 0.5),
            vis_mask_vcs_range_cfg={
                (1024, 768): (-51.2, -76.8, 153.6, 76.8),
                (1024, 800): (-51.2, -80.0, 153.6, 80.0),
            },
            valid_vcs_range_percls=valid_vcs_range_percls.get(category, None),
            psc_phase_factor=1,
            N_steps_PSC_rot=3,
        )
        data = discrete_target_generator(dataset)
        yaw = (
            data["annos_bev_discrete_obj"]["vcs_discobj_yaw"][:5] * 180 / np.pi
        )
        if category == "stopline":
            assert np.abs(yaw[0] - 26) < 1
            assert np.abs(yaw[1] + 26) < 1
            assert np.abs(yaw[2] + 26) < 1
            assert np.abs(yaw[3] - 26) < 1
        if category == "arrow":
            assert np.abs(yaw[0] + 160) < 1
            assert np.abs(yaw[1] - 177) < 1
            assert np.abs(yaw[2] + 1) < 1
            assert np.abs(yaw[3] - 1) < 1
        if category == "crosswalk":
            assert np.abs(yaw[0] + 84) < 1
            assert np.abs(yaw[1] - 89) < 1
        if category == "intersection":
            assert np.abs(yaw[0] - 89) < 1
            assert np.abs(yaw[1] + 0) < 1
        data_gt = data["gt_bev_discrete_obj"]
        assert data_gt["bev_discobj_rot"].shape == (192, 128, 3)
        if valid_vcs_range_percls.get(category, None) is not None:
            assert "bev_discobj_cls_valid_mask" in data_gt
            bev_discobj_cls_valid_mask = data_gt["bev_discobj_cls_valid_mask"]
            assert bev_discobj_cls_valid_mask.shape == (192, 128, 5)
            assert np.sum(bev_discobj_cls_valid_mask) == 49152
