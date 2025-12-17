import pytest
import torch
from torch import nn

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


def test_pwcnet_task():

    loss_weights = [0.005, 0.01, 0.02, 0.08, 0.32]
    out_channels = [16, 32, 64, 96, 128, 196]
    flow_pred_lvl = 2
    pyr_lvls = 6
    train_scales = [2 ** x for x in range(flow_pred_lvl, pyr_lvls + 1)]
    use_bn = True
    bn_kwargs = {}
    use_res = True
    use_dense = True
    input_size = (256, 448)
    use_bias = True

    config = dict(
        type="PwcNet",
        backbone=dict(
            type="PwcNetNeck",
            out_channels=out_channels,
            use_bn=use_bn,
            bn_kwargs=bn_kwargs,
            bias=use_bias,
            pyr_lvls=pyr_lvls,
            flow_pred_lvl=flow_pred_lvl,
            act_type=nn.ReLU(),
        ),
        head=dict(
            type="PwcNetHead",
            in_channels=out_channels,
            bn_kwargs=bn_kwargs,
            use_bn=use_bn,
            md=4,
            use_res=use_res,
            use_dense=use_dense,
            pyr_lvls=pyr_lvls,
            flow_pred_lvl=flow_pred_lvl,
            act_type=nn.ReLU(),
        ),
        loss=dict(type="LnNormLoss", norm_order=2, power=1, reduction="mean"),
        loss_weights=loss_weights,
    )

    pwcnet_flow_pre = build_from_registry(config)

    x = {
        "img": torch.rand(1, 6, input_size[0], input_size[1]),
        "gt_flow": [
            torch.rand(
                1,
                2,
                int(input_size[0] / train_scale),
                int(input_size[1] / train_scale),
            )
            for train_scale in train_scales
        ],
    }

    y = pwcnet_flow_pre(x)

    assert len(y["losses"]) == len(loss_weights)
    assert y["pred_flows"].ndim == 4
    assert y["pred_flows"].shape[1] == 2
    assert y["pred_flows"].shape[2] == input_size[0] / (2 ** flow_pred_lvl)
    assert y["pred_flows"].shape[3] == input_size[1] / (2 ** flow_pred_lvl)

    qat_test(pwcnet_flow_pre, x, with_quantized=False)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
