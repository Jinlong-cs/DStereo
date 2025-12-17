import pytest
import torch
import torch.nn as nn

from hat.models.task_modules.roi_module import RoIModule
from .test_anchor_module import (
    ToyHead,
    ToyLoss,
    ToyNeck,
    ToyPostProcess,
    ToyTarget,
)


class ToyRoIFeatExt(nn.Module):
    def forward(self, x, rois):
        return x


@pytest.mark.parametrize(
    ["ext_feat", "target", "loss", "postprocess", "output_target", "rpn_pred"],
    [
        pytest.param(ToyNeck(), None, None, ToyPostProcess(), False, None),
        pytest.param(None, None, None, ToyPostProcess(), False, None),
        pytest.param(
            None,
            ToyTarget(),
            ToyLoss(),
            None,
            True,
            None,
            marks=pytest.mark.xfail(reason="Target should not be provided"),
        ),
        pytest.param(
            None,
            ToyTarget(),
            ToyLoss(),
            None,
            True,
            {"pred_boxes": torch.Tensor()},
        ),
        pytest.param(
            None,
            None,
            ToyLoss(),
            ToyPostProcess(),
            True,
            None,
            marks=pytest.mark.xfail(reason="Label keys not provided"),
        ),
    ],
)
def test_roi_module(
    ext_feat, target, loss, postprocess, output_target, rpn_pred
):

    mod = RoIModule(
        roi_feat_extractor=ToyRoIFeatExt(),
        head=ToyHead(),
        ext_feat=ext_feat,
        target=target,
        loss=loss,
        postprocess=postprocess,
        output_target=output_target,
        target_opt_keys=[],
    )

    features = torch.randn((1, 512, 7, 7))
    label = {
        "gt_boxes": torch.randint(0, 1000, (1,)),
        "gt_boxes_num": torch.zeros(0),
    }

    result = mod(features, rpn_pred=rpn_pred, y=label)

    assert isinstance(result, dict)
    if postprocess is not None:
        assert "toy_predict" in result
    if loss is not None:
        assert "toy_loss" in result
    if target is not None and output_target:
        assert "toy_target" in result
