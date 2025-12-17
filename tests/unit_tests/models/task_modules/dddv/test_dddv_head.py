import pytest
import torch

from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels
from hat.registry import build_from_registry


@pytest.mark.parametrize(
    ["is_train", "out_strides", "out_nums", "output_name"],
    [
        pytest.param(True, [4], 1, "pred_depths"),
        pytest.param(False, [4], 1, "pred_depths"),
        pytest.param(True, [4, 8], 1, "pred_depths"),
        pytest.param(False, [4, 8], 1, "pred_depths"),
        pytest.param(False, [4, 8], [1, 1], ["pred_depths", "pred_om"]),
    ],
)
def test_pixel_head(is_train, out_strides, out_nums, output_name):
    alpha = 0.5
    feature_name = "feats"
    in_strides = [4, 8, 16, 32]
    stride2channels = get_vargnetv2_stride2channels(alpha)
    forward_frame_idxs = [0]
    batchnorm_output = True

    config = dict(
        type="PixelHead",
        feature_name=feature_name,
        in_strides=in_strides,
        out_strides=out_strides,
        stride2channels=stride2channels,
        forward_frame_idxs=forward_frame_idxs,
        out_nums=out_nums,
        output_name=output_name,
        bn_kwargs={},
        batchnorm_output=batchnorm_output,
    )

    pixel_head = build_from_registry(config)
    if not is_train:
        pixel_head.eval()
    input_size = 256
    feats = [
        [
            torch.randn((1, 16, input_size // 4, input_size // 4)),
            torch.randn((1, 32, input_size // 8, input_size // 8)),
            torch.randn((1, 64, input_size // 16, input_size // 16)),
            torch.randn((1, 128, input_size // 32, input_size // 32)),
        ]
    ]

    data = dict()
    data[feature_name] = feats
    out = pixel_head(data)

    if is_train:
        assert len(out["pred_depths_frame0"]) == len(out_strides)
    else:
        assert len(out["pred_depths_frame0"]) == 1
    for out, stride in zip(out["pred_depths_frame0"], out_strides):
        assert out.shape == (
            1,
            out_nums[0] if isinstance(out_nums, list) else out_nums,
            input_size // stride,
            input_size // stride,
        )


@pytest.mark.parametrize(
    ["is_train", "has_resflow"],
    [
        pytest.param(True, True),
        pytest.param(False, True),
        pytest.param(True, False),
        pytest.param(False, False),
    ],
)
def test_depth_pose_resflow_head(is_train, has_resflow):
    alpha = 0.5
    feature_name = "feats"
    in_strides = [4, 8, 16, 32]
    neck_stride2channels = {4: 16, 8: 32, 16: 64, 32: 128}
    out_strides = [4]
    stride2channels = get_vargnetv2_stride2channels(alpha)
    forward_frame_idxs = [0]
    out_nums = 1
    output_name = "pred_depths"
    input_w, input_h = 960, 512

    depth_config = dict(
        type="PixelHead",
        feature_name=feature_name,
        in_strides=in_strides,
        out_strides=out_strides,
        stride2channels=stride2channels,
        forward_frame_idxs=forward_frame_idxs,
        out_nums=out_nums,
        output_name=output_name,
        bn_kwargs={},
    )

    pose_resflow_config = dict(
        type="ResidualFlowPoseHead",
        input_shape=(input_h, input_w),
        in_strides=[4, 8, 16, 32],
        stride2channels=get_vargnetv2_stride2channels(alpha),
        has_resflow=has_resflow,
        bn_kwargs={},
    )

    depth_pose_resflow_config = dict(
        type="DepthPoseResflowHead",
        neck_out_stride2channels=neck_stride2channels,
        head_in_stride2channels=get_vargnetv2_stride2channels(alpha),
        depth_head=depth_config,
        pose_resflow_head=pose_resflow_config,
        feature_name=feature_name,
    )

    head = build_from_registry(depth_pose_resflow_config)
    if not is_train:
        head.eval()
    feats = [
        [
            torch.randn((1, 16, input_h // 4, input_w // 4)),
            torch.randn((1, 32, input_h // 8, input_w // 8)),
            torch.randn((1, 64, input_h // 16, input_w // 16)),
            torch.randn((1, 128, input_h // 32, input_w // 32)),
        ],
        [
            torch.randn((1, 16, input_h // 4, input_w // 4)),
            torch.randn((1, 32, input_h // 8, input_w // 8)),
            torch.randn((1, 64, input_h // 16, input_w // 16)),
            torch.randn((1, 128, input_h // 32, input_w // 32)),
        ],
    ]

    data = dict()
    data[feature_name] = feats
    out = head(data)

    assert "axisangle" in out
    assert "translation" in out

    if is_train:
        assert len(out["axisangle"]) == 2
        assert len(out["translation"]) == 2

        for tensor in out["axisangle"] + out["translation"]:
            assert tensor.shape == (1, 3, 1, 1)

        if has_resflow:
            assert len(out["residual_flow"]) == 2
            for tensor in out["residual_flow"]:
                assert tensor.shape == (1, 3, input_h // 4, input_w // 4)
    else:
        assert out["axisangle"].shape == (1, 3, 1, 1)
        assert out["translation"].shape == (1, 3, 1, 1)
        if has_resflow:
            assert out["residual_flow"].shape == (
                1,
                3,
                input_h // 4,
                input_w // 4,
            )
        else:
            assert "residual_flow" not in out
