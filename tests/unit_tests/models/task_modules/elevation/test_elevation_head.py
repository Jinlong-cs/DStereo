import pytest
import torch

from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels
from hat.models.task_modules.dddv import PixelHead
from hat.models.task_modules.elevation import ElevationHead, GroundHead


@pytest.mark.parametrize(
    ["is_train", "out_strides"],
    [
        pytest.param(True, [4]),
        pytest.param(False, [4]),
        pytest.param(True, [4, 8]),
        pytest.param(False, [4, 8]),
    ],
)
def test_elevation_head(is_train, out_strides):

    alpha = 0.5
    feature_name = "feats"
    in_strides = [4, 8, 16, 32]
    take_strides = (4, 8, 16, 32)
    input_w, input_h = 256, 256
    stride2channels = get_vargnetv2_stride2channels(alpha)
    forward_frame_idxs = [0]
    out_nums = 1
    output_name = "pred_gammas"
    bn_kwargs = {}
    group_base = 8

    gamma_head = PixelHead(
        in_strides=in_strides,
        out_strides=out_strides,
        stride2channels=stride2channels,
        forward_frame_idxs=forward_frame_idxs,
        out_nums=out_nums,
        output_name=output_name,
        bn_kwargs=bn_kwargs,
        group_base=group_base,
    )

    ground_head = GroundHead(
        in_strides=in_strides,
        stride2channels=stride2channels,
        bn_kwargs=bn_kwargs,
        input_shape=(input_h, input_w),
        take_strides=take_strides,
        feature_name=feature_name,
        forward_frame_idxs=forward_frame_idxs,
    )
    elevation_head = ElevationHead(
        feature_name=feature_name,
        gamma_head=gamma_head,
        ground_head=ground_head,
    )

    if not is_train:
        elevation_head.eval()
    input_size = 256
    feats = [
        [
            torch.randn((1, 16, input_size // 4, input_size // 4)),
            torch.randn((1, 32, input_size // 8, input_size // 8)),
            torch.randn((1, 64, input_size // 16, input_size // 16)),
            torch.randn((1, 128, input_size // 32, input_size // 32)),
        ],
        [
            torch.randn((1, 16, input_size // 4, input_size // 4)),
            torch.randn((1, 32, input_size // 8, input_size // 8)),
            torch.randn((1, 64, input_size // 16, input_size // 16)),
            torch.randn((1, 128, input_size // 32, input_size // 32)),
        ],
    ]

    data = dict()
    data[feature_name] = feats
    out = elevation_head(data)

    assert "pred_gammas_frame0" in out
    if is_train:
        assert len(out["pred_gammas_frame0"]) == len(out_strides)
    else:
        assert len(out["pred_gammas_frame0"]) == 1
    for gamma, stride in zip(out["pred_gammas_frame0"], out_strides):
        assert gamma.shape == (
            1,
            out_nums,
            input_size // stride,
            input_size // stride,
        )
    for ground_norm in out["pred_ground"]:
        assert ground_norm.shape == (1, 3, 1, 1)
