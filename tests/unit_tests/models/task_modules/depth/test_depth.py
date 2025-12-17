import torch

from hat.models.task_modules.depth import MultiStrideDepthLoss


def test_multi_stride_depth_loss():

    multi_scale_weight = [2.0, 1.0]
    depth_scale = 4
    max_depth = 8
    min_depth = 0.0001
    avoid_zero = 0.00001
    out_strides = [2, 4]

    pred_stage1 = torch.Tensor(
        [
            [
                [0.9140, 0.3939, 0.8436, 0.8709],
                [0.7222, 0.1991, 0.7227, 0.0187],
                [0.5644, 0.7513, 0.0774, 0.3974],
                [0.9640, 1.9106, 0.5067, 1.7544],
            ],
            [
                [0.8459, 0.2419, 0.1675, 0.3813],
                [0.2227, 0.8864, 0.8276, 0.9101],
                [0.2367, 0.6235, 0.2568, 0.9452],
                [0.9753, 0.3675, 0.5496, 0.2769],
            ],
        ]
    ).unsqueeze(0)

    pred_stage2 = torch.Tensor(
        [
            [[0.1075, -0.7404], [-0.0510, -0.1983]],
            [[-0.9819, -0.0738], [0.0911, -1.7037]],
        ]
    ).unsqueeze(0)

    target = torch.Tensor(
        [
            [
                [
                    -0.1332,
                    0.0224,
                    0.5097,
                    0.1895,
                    1.2923,
                    -1.2423,
                    0.7420,
                    1.0529,
                ],
                [
                    0.6418,
                    0.3959,
                    0.1529,
                    0.8543,
                    0.4105,
                    1.7808,
                    0.8711,
                    0.2725,
                ],
                [
                    1.9618,
                    -0.1793,
                    1.7589,
                    -0.2414,
                    0.6474,
                    1.2391,
                    -1.4800,
                    1.5674,
                ],
                [
                    -1.0816,
                    -2.2401,
                    1.6576,
                    -0.5589,
                    0.9981,
                    0.3467,
                    0.3410,
                    1.2890,
                ],
                [
                    2.5720,
                    0.8274,
                    0.1517,
                    0.2225,
                    1.1171,
                    0.9075,
                    0.1385,
                    1.2343,
                ],
                [
                    -0.3367,
                    0.3247,
                    1.8712,
                    -0.8863,
                    0.9770,
                    1.5657,
                    1.4273,
                    0.2204,
                ],
                [
                    0.3428,
                    0.3306,
                    0.4395,
                    -1.2534,
                    0.9280,
                    0.1556,
                    1.5055,
                    0.4188,
                ],
                [
                    1.0417,
                    -0.1422,
                    1.2299,
                    1.0013,
                    -1.3308,
                    -0.6434,
                    1.7245,
                    1.5007,
                ],
            ]
        ]
    ).unsqueeze(0)

    target_loss = {
        "loss_stride_2": torch.Tensor([1.4921]),
        "conf_loss_stride_2": torch.Tensor([0.5951]),
        "loss_stride_4": torch.Tensor([1.0807]),
        "conf_loss_stride_4": torch.Tensor([1.0598]),
    }

    pred_list = [pred_stage1, pred_stage2]
    multi_stride_depth_loss = MultiStrideDepthLoss(
        multi_scale_weight=multi_scale_weight,
        depth_scale=depth_scale,
        max_depth=max_depth,
        min_depth=min_depth,
        avoid_zero=avoid_zero,
        out_strides=out_strides,
    )

    loss = multi_stride_depth_loss(pred_list, target)

    for stride in out_strides:
        loss_stride = loss.get(f"loss_stride_{stride}", None)
        conf_loss_stride = loss.get(f"conf_loss_stride_{stride}", None)
        assert loss_stride
        assert conf_loss_stride

        assert (
            torch.abs(loss_stride - target_loss[f"loss_stride_{stride}"])
            < 1e-4
        )
        assert (
            torch.abs(
                conf_loss_stride - target_loss[f"conf_loss_stride_{stride}"]
            )
            < 1e-4
        )
