from collections import OrderedDict

import pytest
import torch

from hat.models.backbones.vargnetv2 import (
    TinyVargNetV2,
    get_vargnetv2_stride2channels,
)
from hat.models.losses.toll_gate_loss import TollGateLoss
from hat.models.necks.unet import Unet
from hat.models.structures.toll_gate import TollGageModel
from hat.models.task_modules.toll_gate import TollGateHead
from tests.unit_tests.models.base import qat_test


@pytest.mark.parametrize(
    ["losses"],
    [
        pytest.param(
            TollGateLoss(
                task="tollgate",
                loss_weights=dict(hm=100.0, offset=1.0),
            )
        )
    ],
)
def test_tollgate(losses):
    bn_kwargs = {"eps": 1e-05, "momentum": 0.1}
    NUM_LDMK = 2
    STRIDE_LDMK = 4

    data = dict(
        img=torch.randn((1, 3, 512, 960)),
        gt_heatmap=torch.zeros(
            (1, NUM_LDMK, 512 // STRIDE_LDMK, 960 // STRIDE_LDMK)
        ),
        gt_heatmap_weight=torch.ones(
            (1, NUM_LDMK, 512 // STRIDE_LDMK, 960 // STRIDE_LDMK)
        ),
        gt_offset=torch.zeros(
            (1, NUM_LDMK, 512 // STRIDE_LDMK, 960 // STRIDE_LDMK)
        ),
        gt_offset_weight=torch.ones(
            (1, 2, 512 // STRIDE_LDMK, 960 // STRIDE_LDMK)
        ),
    )
    model = TollGageModel(
        backbone=TinyVargNetV2(
            num_classes=1000,
            alpha=0.5,
            group_base=8,
            bn_kwargs=bn_kwargs,
            include_top=False,
            extend_features=False,
            input_resize_scale=None,
            channel_list=[32, 32, 64, 128, 256],
        ),
        neck=Unet(
            in_strides=[2, 4, 8, 16, 32],
            out_strides=[4],
            stride2channels=get_vargnetv2_stride2channels(0.5),
            factor=2,
            bn_kwargs=bn_kwargs,
            group_base=8,
        ),
        head=TollGateHead(
            in_channels=16,
            hm_channels=2,
            offset_channels=2,
            use_bias=False,
        ),
        loss=losses,
    )
    y = model(data)
    assert isinstance(y, OrderedDict)
    assert "tollgate_hm_loss" in y
    assert "tollgate_off_hm_l1_loss" in y

    qat_test(model, data, with_quantized=False)
