from collections import OrderedDict

import pytest
import torch

from hat.models.backbones.vargnetv2 import TinyVargNetV2
from hat.models.base_modules.basic_vargnet_module import ExtendVarGNetFeatures
from hat.models.losses.cross_entropy_loss import CEWithLabelSmooth
from hat.models.structures.reid import ReIDModule
from hat.models.task_modules.reid import ReIDClsOutputBlock
from tests.unit_tests.models.base import qat_test


@pytest.mark.parametrize(
    [
        "losses",
    ],
    [
        pytest.param(
            CEWithLabelSmooth(
                smooth_alpha=0.1,
            )
        ),
        pytest.param(None),
    ],
)
def test_reid(losses):

    data = dict(
        img=torch.randn([256, 3, 128, 128]),
        labels=torch.zeros(256, dtype=torch.int64),
    )
    model = ReIDModule(
        backbone=TinyVargNetV2(
            num_classes=2,
            alpha=0.5,
            group_base=8,
            include_top=False,
            extend_features=False,
            bn_kwargs={},
        ),
        neck=ExtendVarGNetFeatures(
            prev_channel=128,
            channels=256,
            num_units=2,
            group_base=8,
            bn_kwargs={},
        ),
        head=ReIDClsOutputBlock(
            num_classes=2,
            include_top=True,
            in_channels=256,
            bn_kwargs={},
            feat_channels=128,
            pool_kernel_size=2,
            int8_output=False,
        ),
        losses=losses,
    )
    y = model(data)

    if losses is not None:
        assert isinstance(y, OrderedDict)
        assert "cls_loss" in y
        assert "cls_softmax_output" in y
        assert y["cls_softmax_output"].shape == (256, 2)
    else:
        assert y.shape == (256, 2)
        assert isinstance(y, torch.Tensor)

    qat_test(model, data, with_quantized=False)
