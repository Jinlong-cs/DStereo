import copy

import numpy as np
import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


def test_ganet_task():
    radius = 2  # gaussian circle radius
    alpha = 0.25

    config = dict(
        type="GaNet",
        backbone=dict(
            type="MobileNetV1",
            bn_kwargs={},
            num_classes=1000,
            include_top=False,
            alpha=alpha,
        ),
        neck=dict(
            type="GaNetNeck",
            fpn_module=dict(
                type="FPN",
                in_strides=[8, 16, 32],
                in_channels=[int(256 * alpha), int(512 * alpha), 64],
                out_strides=[8, 16, 32],
                out_channels=[64, 64, 64],
            ),
            attn_in_channels=[int(1024 * alpha), 64],
            attn_out_channels=[64, 64],
            attn_ratios=[4, 4],
            pos_shape=(1, 10, 25),
        ),
        head=dict(
            type="GaNetHead",
            in_channel=64,
        ),
        targets=dict(
            type="GaNetTarget",
            hm_down_scale=8,
            radius=radius,
        ),
        post_process=dict(
            type="GaNetDecoder",
            root_thr=1,
            kpt_thr=0.4,
            cluster_thr=5,
            downscale=8,
        ),
        losses=dict(
            type="GaNetLoss",
            loss_kpts_cls=dict(
                type="LaneFastFocalLoss",
                loss_weight=1.0,
            ),
            loss_pts_offset_reg=dict(
                type="L1Loss",
                loss_weight=0.5,
            ),
            loss_int_offset_reg=dict(
                type="L1Loss",
                loss_weight=1.0,
            ),
        ),
    )

    ganet_train = build_from_registry(config)
    ganet_train.train()
    x = {
        "img": torch.rand(1, 3, 320, 800),
        "gt_lines": [
            np.random.randint(0, 320, size=(12, 2)).astype(np.float32)
        ],
        "img_shape": [np.array([3, 320, 800]).astype(np.int64)],
        "scale_factor": torch.rand(1, 4),
        "crop_offset": [[0, 100, 0, 100]],
    }
    results_train = ganet_train(x)

    assert "kpts_cls_loss" in results_train
    assert "offset_reg_loss" in results_train
    assert "int_offset_reg_loss" in results_train

    qat_test(ganet_train, x, with_quantized=False)

    # deploy model
    config_deploy_model = copy.deepcopy(config)
    config_deploy_model["targets"] = None
    config_deploy_model["losses"] = None
    config_deploy_model["post_process"] = None

    ganet_deploy = build_from_registry(config_deploy_model)
    results_deploy = ganet_deploy(x)
    assert results_deploy[0].shape == (1, 1, 40, 100)
    assert results_deploy[1].shape == (1, 2, 40, 100)
    assert results_deploy[2].shape == (1, 2, 40, 100)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
