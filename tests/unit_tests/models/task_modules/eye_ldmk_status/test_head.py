# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.models.task_modules.eye_ldmk_status import (
    EyeClsHead,
    EyeLdmkHead,
    EyeLdmkHeatmapHead,
    EyeLdmkVectorHead,
    EyeMixVarGEBinClsHead,
    EyeMixVarGEClsHead,
    EyeMultiBinBranchHead,
    EyeMultiBranchHead,
    EyeMultiBranchSingleHead,
)


@pytest.mark.parametrize(
    [
        "in_channels",
        "out_channels",
        "fc_channels",
        "classifier_num",
        "dropout_rate",
        "feat_h",
        "feat_w",
        "flat_output",
    ],
    [
        pytest.param(128, 160, 160, 5, 0.2, 3, 5, True),
    ],
)
def test_multibranch_single_cls(
    in_channels,
    out_channels,
    fc_channels,
    classifier_num,
    dropout_rate,
    feat_h,
    feat_w,
    flat_output,
):
    net_config = [
        [
            MixVarGENetConfig(
                in_channels=in_channels,
                out_channels=out_channels,
                head_op="mixvarge_f2_gb16",
                stack_ops=["mixvarge_f2_gb16"],
                stack_factor=1,
                stride=1,
                fusion_strides=[],
                extra_downsample_num=0,
            ),
        ],
    ]
    batch_size = 32
    dummy_data = torch.randn(batch_size, in_channels, feat_h, feat_w)

    left_single_head = EyeMixVarGEClsHead(
        net_config=net_config,
        num_classes=classifier_num,
        bn_kwargs={},
        in_channels=out_channels,
        out_channels=fc_channels,
        pool_size=(feat_h, feat_w),
        bias=False,
        dropout_rate=dropout_rate,
        include_top=False,
    )

    right_single_head = EyeMixVarGEClsHead(
        net_config=net_config,
        num_classes=classifier_num,
        bn_kwargs={},
        in_channels=out_channels,
        out_channels=fc_channels,
        pool_size=(feat_h, feat_w),
        bias=False,
        dropout_rate=dropout_rate,
        include_top=False,
        flat_output=flat_output,
    )

    model = EyeMultiBranchSingleHead(
        l_cls_head=left_single_head,
        r_cls_head=right_single_head,
        num_classes=classifier_num,
        bn_kwargs={},
        in_channels=out_channels,
        out_channels=fc_channels,
        pool_size=(feat_h, feat_w),
    )

    y = model(dummy_data)

    if flat_output:
        assert y.shape == (batch_size * 2, classifier_num)
    else:
        assert y.shape == (batch_size * 2, classifier_num, 1, 1)


@pytest.mark.parametrize(
    [
        "in_channels",
        "out_channels",
        "fc_channels",
        "classifier_num",
        "dropout_rate",
        "feat_h",
        "feat_w",
        "flat_output",
    ],
    [
        pytest.param(128, 160, 160, 5, 0.2, 3, 5, True),
    ],
)
def test_multibranch(
    in_channels,
    out_channels,
    fc_channels,
    classifier_num,
    dropout_rate,
    feat_h,
    feat_w,
    flat_output,
):
    net_config = [
        [
            MixVarGENetConfig(
                in_channels=in_channels,
                out_channels=out_channels,
                head_op="mixvarge_f2_gb16",
                stack_ops=["mixvarge_f2_gb16"],
                stack_factor=1,
                stride=1,
                fusion_strides=[],
                extra_downsample_num=0,
            ),
        ],
    ]
    batch_size = 32
    dummy_data = torch.randn(batch_size, in_channels, feat_h, feat_w)

    left_single_head = EyeMixVarGEClsHead(
        net_config=net_config,
        num_classes=classifier_num,
        bn_kwargs={},
        in_channels=out_channels,
        out_channels=fc_channels,
        pool_size=(feat_h, feat_w),
        bias=False,
        dropout_rate=dropout_rate,
        include_top=True,
        flat_output=flat_output,
    )

    right_single_head = EyeMixVarGEClsHead(
        net_config=net_config,
        num_classes=classifier_num,
        bn_kwargs={},
        in_channels=out_channels,
        out_channels=fc_channels,
        pool_size=(feat_h, feat_w),
        bias=False,
        dropout_rate=dropout_rate,
        include_top=True,
        flat_output=flat_output,
    )

    model = EyeMultiBranchHead(
        l_cls_head=left_single_head,
        r_cls_head=right_single_head,
    )

    left_y, right_y = model(dummy_data)

    if flat_output:
        assert left_y.shape == (batch_size, classifier_num)
        assert right_y.shape == (batch_size, classifier_num)
    else:
        assert left_y.shape == (batch_size, classifier_num, 1, 1)
        assert right_y.shape == (batch_size, classifier_num, 1, 1)


@pytest.mark.parametrize(
    [
        "in_channels",
        "classifier_num",
        "dropout_rate",
        "alpha",
        "feat_h",
        "feat_w",
        "flat_output",
    ],
    [
        pytest.param(128, 5, 0.2, 0.5, 3, 5, True),
    ],
)
def test_single_mobilev2_cls(
    in_channels,
    classifier_num,
    dropout_rate,
    alpha,
    feat_h,
    feat_w,
    flat_output,
):
    batch_size = 32
    dummy_data = torch.randn(
        batch_size, int(in_channels * alpha), feat_h, feat_w
    )

    model = EyeClsHead(
        num_classes=classifier_num,
        bn_kwargs={},
        in_channels=in_channels,
        alpha=alpha,
        pool_size=(feat_h, feat_w),
        bias=False,
        dropout_rate=dropout_rate,
        include_top=True,
        flat_output=flat_output,
        use_dw_as_avgpool=False,
    )

    y = model(dummy_data)

    if flat_output:
        assert y.shape == (batch_size, classifier_num)
    else:
        assert y.shape == (batch_size, classifier_num, 1, 1)


@pytest.mark.parametrize(
    [
        "in_channels",
        "out_channels",
        "dropout_rate",
        "feat_h",
        "feat_w",
        "flat_output",
    ],
    [
        pytest.param(128, 160, 0.2, 3, 5, True),
    ],
)
def test_binary_multibranch(
    in_channels,
    out_channels,
    dropout_rate,
    feat_h,
    feat_w,
    flat_output,
):
    batch_size = 32
    dummy_data = torch.randn(batch_size, in_channels, feat_h, feat_w)

    left_single_head = EyeMixVarGEBinClsHead(
        bn_kwargs={},
        in_channels=in_channels,
        out_channels=out_channels,
        pool_size=(feat_h, feat_w),
        bias=False,
        dropout_rate=dropout_rate,
        flat_output=flat_output,
    )

    right_single_head = EyeMixVarGEBinClsHead(
        bn_kwargs={},
        in_channels=in_channels,
        out_channels=out_channels,
        pool_size=(feat_h, feat_w),
        bias=False,
        dropout_rate=dropout_rate,
        flat_output=flat_output,
    )

    model = EyeMultiBinBranchHead(
        l_cls_head=left_single_head,
        r_cls_head=right_single_head,
    )

    left_y, right_y = model(dummy_data)

    if flat_output:
        assert left_y.shape == (batch_size, 1)
        assert right_y.shape == (batch_size, 1)
    else:
        assert left_y.shape == (batch_size, 1, 1, 1)
        assert right_y.shape == (batch_size, 1, 1, 1)


@pytest.mark.parametrize(
    [
        "in_channels",
        "num_ldmk",
        "feat_h",
        "feat_w",
    ],
    [
        pytest.param(16, 8, 24, 40),
    ],
)
def test_ldmk_head(
    in_channels,
    num_ldmk,
    feat_h,
    feat_w,
):
    batch_size = 32
    dummy_data = torch.randn(batch_size, in_channels, feat_h, feat_w)

    single_eye_ldmk_num = num_ldmk
    double_eye_ldmk_num = single_eye_ldmk_num * 2
    vector_head = EyeLdmkVectorHead(
        num_ldmk=single_eye_ldmk_num,
        in_channels=in_channels,
        bias=False,
    )
    heatmap_head = EyeLdmkHeatmapHead(
        num_ldmk=double_eye_ldmk_num,
        in_channels=in_channels,
    )
    model = EyeLdmkHead(
        heatmap_head=heatmap_head,
        vector_head=vector_head,
    )

    pred_x, pred_y, heatmap = model(dummy_data)

    assert pred_x.shape == (batch_size, num_ldmk, 1, feat_w)
    assert pred_y.shape == (batch_size, num_ldmk, feat_h, 1)
    assert heatmap.shape == (batch_size, num_ldmk * 2, feat_h, feat_w)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
