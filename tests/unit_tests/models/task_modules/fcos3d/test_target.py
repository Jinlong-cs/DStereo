import torch

from hat.models.task_modules.fcos3d import FCOS3DTarget

INF = 1e8


def test_fcos3d_target():
    module = FCOS3DTarget(
        num_classes=10,
        background_label=None,
        bbox_code_size=9,
        regress_ranges=((-1, 48), (48, 96), (96, 192), (192, 384), (384, INF)),
        strides=[8, 16, 32, 64, 128],
        pred_attrs=True,
        num_attrs=9,
        center_sampling=True,
        center_sample_radius=1.5,
        centerness_alpha=2.5,
        norm_on_bbox=True,
    )
    cls_scores = [
        torch.rand(2, 10, 116, 200),
        torch.rand(2, 10, 58, 100),
        torch.rand(2, 10, 29, 50),
        torch.rand(2, 10, 15, 25),
        torch.rand(2, 10, 8, 13),
    ]
    bbox_preds = [
        torch.rand(2, 9, 116, 200),
        torch.rand(2, 9, 58, 100),
        torch.rand(2, 9, 29, 50),
        torch.rand(2, 9, 15, 25),
        torch.rand(2, 9, 8, 13),
    ]
    gt_bboxes = [
        torch.tensor(
            [
                [357.9058, 455.4693, 478.1024, 713.2845],
                [359.6490, 470.6709, 432.2777, 616.9516],
                [866.7497, 473.8965, 983.6488, 565.5880],
                [536.9000, 457.5315, 603.5837, 604.4177],
                [267.6915, 464.1540, 358.5342, 680.3337],
                [937.0857, 350.8254, 1085.0278, 546.8214],
                [998.3503, 475.7121, 1146.8167, 563.1693],
                [641.0569, 463.6450, 679.6312, 550.3938],
                [426.6460, 445.1522, 464.2877, 503.0703],
                [0.0000, 430.3711, 2.1077, 556.9604],
                [772.9155, 450.3050, 835.6003, 604.1185],
                [835.1089, 469.0152, 912.4780, 533.9516],
            ]
        ),
        torch.tensor(
            [
                [1338.1815, 516.8099, 1600.0000, 610.8064],
                [1152.6719, 492.9074, 1600.0000, 650.9208],
                [0.0000, 515.1469, 306.9224, 664.4988],
                [0.0000, 529.0372, 226.0330, 660.1250],
                [1489.3954, 493.9149, 1600.0000, 581.0579],
            ]
        ),
    ]
    gt_labels = [
        torch.tensor([7, 7, 0, 7, 7, 3, 0, 7, 7, 7, 7, 0]),
        torch.tensor([0, 0, 0, 0, 0]),
    ]
    gt_bboxes_3d = [
        torch.tensor(
            [
                [
                    -2.8387e00,
                    1.5213e00,
                    9.5143e00,
                    6.7500e-01,
                    1.8550e00,
                    8.1900e-01,
                    1.2177e00,
                    7.5275e-02,
                    -1.1432e-02,
                ],
                [
                    -4.7651e00,
                    1.3532e00,
                    1.5146e01,
                    5.4800e-01,
                    1.7140e00,
                    6.4300e-01,
                    3.4775e00,
                    -1.1014e00,
                    -1.5866e-01,
                ],
                [
                    2.4810e00,
                    1.1186e00,
                    2.4577e01,
                    4.4750e00,
                    1.6220e00,
                    1.8210e00,
                    1.5738e00,
                    -0.0000e00,
                    0.0000e00,
                ],
                [
                    -2.6716e00,
                    1.2003e00,
                    1.5090e01,
                    7.4800e-01,
                    1.7140e00,
                    6.4300e-01,
                    2.9164e00,
                    -1.2215e00,
                    -1.4070e-01,
                ],
                [
                    -3.7822e00,
                    1.3577e00,
                    9.9386e00,
                    5.4800e-01,
                    1.6430e00,
                    6.4300e-01,
                    1.2002e00,
                    9.2232e-02,
                    -4.6483e-02,
                ],
                [
                    5.8846e00,
                    1.0082e00,
                    3.6206e01,
                    1.3124e01,
                    4.5930e00,
                    3.1650e00,
                    1.6638e00,
                    -1.0693e-01,
                    -1.6223e00,
                ],
                [
                    5.4529e00,
                    1.0955e00,
                    2.5179e01,
                    4.8050e00,
                    1.5770e00,
                    1.9130e00,
                    1.5738e00,
                    -3.3606e-02,
                    -2.6023e-04,
                ],
                [
                    -2.6459e00,
                    9.6996e-01,
                    2.5203e01,
                    5.4800e-01,
                    1.7140e00,
                    6.4300e-01,
                    4.8654e00,
                    3.5884e-01,
                    1.0511e00,
                ],
                [
                    -1.0584e01,
                    5.3078e-02,
                    3.8399e01,
                    6.8600e-01,
                    1.7450e00,
                    8.6900e-01,
                    1.8373e00,
                    -1.5215e-01,
                    -1.3401e00,
                ],
                [
                    -1.1855e01,
                    8.1862e-01,
                    1.8199e01,
                    5.4800e-01,
                    1.8550e00,
                    6.4300e-01,
                    1.1584e00,
                    7.5085e-02,
                    9.1763e-03,
                ],
                [
                    1.3992e-01,
                    1.2594e00,
                    1.5857e01,
                    6.1000e-01,
                    1.8880e00,
                    6.4300e-01,
                    2.8233e00,
                    -8.5593e-01,
                    -4.0848e-01,
                ],
                [
                    2.2170e00,
                    7.8872e-01,
                    3.3644e01,
                    4.8050e00,
                    1.6070e00,
                    1.9130e00,
                    1.6577e00,
                    -4.3353e-02,
                    1.0222e-02,
                ],
            ]
        ),
        torch.tensor(
            [
                [
                    12.7790,
                    1.9499,
                    23.8330,
                    4.7440,
                    1.6140,
                    2.1840,
                    -0.3300,
                    13.9767,
                    5.2950,
                ],
                [
                    6.7291,
                    1.5664,
                    14.6661,
                    4.8630,
                    1.6150,
                    2.0370,
                    -0.3126,
                    7.4883,
                    2.3221,
                ],
                [
                    -8.8199,
                    1.9060,
                    15.7363,
                    4.7080,
                    1.6870,
                    2.2010,
                    -0.3475,
                    13.5555,
                    5.0413,
                ],
                [
                    -11.3158,
                    2.2203,
                    18.5137,
                    4.6290,
                    1.7900,
                    2.0090,
                    -0.9139,
                    15.4702,
                    9.0491,
                ],
                [
                    18.6438,
                    1.7889,
                    29.4585,
                    4.7270,
                    1.9180,
                    2.0000,
                    -0.3300,
                    10.4562,
                    3.8390,
                ],
            ]
        ),
    ]
    gt_labels_3d = [
        torch.tensor([7, 7, 0, 7, 7, 3, 0, 7, 7, 7, 7, 0]),
        torch.tensor([0, 0, 0, 0, 0]),
    ]
    centers2d = [
        torch.tensor(
            [
                [416.9552, 579.8104],
                [396.4944, 542.4586],
                [919.8904, 516.9593],
                [569.7523, 529.8461],
                [313.4307, 569.1484],
                [997.4542, 456.3780],
                [1065.5105, 516.5503],
                [660.5201, 506.8408],
                [445.5770, 474.3182],
                [-27.7396, 493.6605],
                [803.8611, 526.2501],
                [875.7449, 500.6425],
            ]
        ),
        torch.tensor(
            [
                [1482.5924, 561.5969],
                [1385.1434, 566.3702],
                [101.3158, 586.2394],
                [37.4251, 591.3552],
                [1604.3783, 536.6792],
            ]
        ),
    ]
    depths = [
        torch.tensor(
            [
                9.5143,
                15.1462,
                24.5773,
                15.0899,
                9.9386,
                36.2064,
                25.1794,
                25.2031,
                38.3989,
                18.1988,
                15.8569,
                33.6442,
            ]
        ),
        torch.tensor([23.8330, 14.6661, 15.7363, 18.5137, 29.4585]),
    ]
    attr_labels = [
        torch.tensor([3, 2, 7, 2, 3, 5, 7, 2, 2, 3, 2, 7]),
        torch.tensor([5, 5, 5, 5, 5]),
    ]

    (
        concat_lvl_labels_3d,
        concat_lvl_bbox_targets_3d,
        concat_lvl_centerness_targets,
        concat_lvl_attr_targets,
    ) = module(
        cls_scores=cls_scores,
        bbox_preds=bbox_preds,
        gt_bboxes_list=gt_bboxes,
        gt_labels_list=gt_labels,
        gt_bboxes_3d_list=gt_bboxes_3d,
        gt_labels_3d_list=gt_labels_3d,
        centers2d_list=centers2d,
        depths_list=depths,
        attr_labels_list=attr_labels,
    )
    assert (
        len(concat_lvl_labels_3d)
        == len(concat_lvl_bbox_targets_3d)
        == len(concat_lvl_centerness_targets)
        == len(concat_lvl_attr_targets)
        == 5
    )

    assert (
        concat_lvl_labels_3d[0].size(0)
        == concat_lvl_bbox_targets_3d[0].size(0)
        == concat_lvl_centerness_targets[0].size(0)
        == concat_lvl_attr_targets[0].size(0)
        == 46400
    )

    assert (
        concat_lvl_labels_3d[1].size(0)
        == concat_lvl_bbox_targets_3d[1].size(0)
        == concat_lvl_centerness_targets[1].size(0)
        == concat_lvl_attr_targets[1].size(0)
        == 11600
    )

    assert (
        concat_lvl_labels_3d[2].size(0)
        == concat_lvl_bbox_targets_3d[2].size(0)
        == concat_lvl_centerness_targets[2].size(0)
        == concat_lvl_attr_targets[2].size(0)
        == 2900
    )

    assert (
        concat_lvl_labels_3d[3].size(0)
        == concat_lvl_bbox_targets_3d[3].size(0)
        == concat_lvl_centerness_targets[3].size(0)
        == concat_lvl_attr_targets[3].size(0)
        == 750
    )

    assert (
        concat_lvl_labels_3d[4].size(0)
        == concat_lvl_bbox_targets_3d[4].size(0)
        == concat_lvl_centerness_targets[4].size(0)
        == concat_lvl_attr_targets[4].size(0)
        == 208
    )
