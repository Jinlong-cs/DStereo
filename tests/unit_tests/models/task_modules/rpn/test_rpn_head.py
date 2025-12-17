import pytest
import torch

from hat.models.task_modules.rpn import RPNVarGNetHead

im_hw = [128, 128]
batch_size = 2


@pytest.mark.parametrize(
    [
        "in_channels",
        "num_channels",
        "num_anchors",
        "feat_strides",
        "num_classes",
    ],
    [
        pytest.param(32, [32, 32, 64, 64], [4, 6, 8, 8], [4, 8, 16, 32], 2),
        pytest.param(
            [16, 32, 16, 32],
            [16, 32, 64, 64],
            [2, 4, 6, 8],
            [8, 16, 32, 64],
            1,
        ),
    ],
)
def test_rpn_vargnet_head(
    in_channels, num_channels, num_anchors, feat_strides, num_classes
):
    rpn_head = RPNVarGNetHead(
        in_channels,
        num_channels,
        num_anchors,
        feat_strides,
        num_classes,
        False,
        dict(momentum=0.1, eps=1e-5),
        1.0,
        8,
    )

    if isinstance(in_channels, int):
        in_channels = [in_channels] * len(num_channels)

    featmaps = [
        torch.rand(
            (batch_size, in_channel, im_hw[0] // stride, im_hw[1] // stride)
        )
        for in_channel, stride in zip(in_channels, feat_strides)
    ]
    rpn_head_out = rpn_head(featmaps)

    # 1. make sure lengths of all outputs are identical
    assert (
        len(rpn_head_out["rpn_head_out"])
        == len(rpn_head_out["rpn_cls_pred"])
        == len(rpn_head_out["rpn_reg_pred"])
        == len(in_channels)
    )

    # 2. check the shape
    for i, out in enumerate(rpn_head_out["rpn_head_out"]):
        assert out.shape[1] == (4 + num_classes) * num_anchors[i]
        assert out.shape[2] == im_hw[0] // feat_strides[i]
        assert out.shape[3] == im_hw[1] // feat_strides[i]

    for i, out in enumerate(rpn_head_out["rpn_cls_pred"]):
        assert out.shape[1] == num_classes * num_anchors[i]
        assert out.shape[2] == im_hw[0] // feat_strides[i]
        assert out.shape[3] == im_hw[1] // feat_strides[i]

    for i, out in enumerate(rpn_head_out["rpn_reg_pred"]):
        assert out.shape[1] == 4 * num_anchors[i]
        assert out.shape[2] == im_hw[0] // feat_strides[i]
        assert out.shape[3] == im_hw[1] // feat_strides[i]
