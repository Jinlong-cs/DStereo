from typing import Sequence

import pytest
import torch
import torch.nn as nn

from hat.models.task_modules.anchor_module import AnchorModule
from tests.data.toy_modules import ToyHeadParser  # noqa: F403,F401


class ToyAG(nn.Module):
    def forward(self, feat):
        return []


class ToyHead(nn.Module):
    def forward(self, feats):
        return {"toy_head": feats}


class ToyNeck(nn.Module):
    def forward(self, feature_maps: Sequence[torch.Tensor]):
        return feature_maps


class ToyTarget(nn.Module):
    def forward(self, anchors, target, *args, **kwargs):
        return anchors, {"toy_target": target}


class ToyLoss(nn.Module):
    """
    ToyLoss stands for algorithm-related loss, such as Real3dLoss.
    """

    def forward(self, pred, label, *args):
        return {"toy_loss": pred}


class ToyPostProcess(nn.Module):
    def forward(self, boxes, pred, *args, **kwargs):
        # return tuple instead of list, cos constant container is recommended
        # by `torch.jit.trace()`
        # return multi tensors just for test
        pred = list(pred.values())
        cat_pred = torch.cat(pred, dim=1)
        pred1, pred2 = torch.split(
            cat_pred, split_size_or_sections=cat_pred.shape[1] // 2, dim=1
        )
        return {"toy_predict": [pred1, pred2]}


@pytest.mark.parametrize(
    ["ext_feat", "target", "loss", "postprocess", "desc", "output_target"],
    [
        pytest.param(ToyNeck(), None, None, ToyPostProcess(), None, False),
        pytest.param(
            None, None, None, ToyPostProcess(), ToyHeadParser(), False
        ),
        pytest.param(None, ToyTarget(), ToyLoss(), None, None, True),
        pytest.param(
            None, ToyTarget(), ToyLoss(), ToyPostProcess(), None, False
        ),
        pytest.param(
            None,
            None,
            ToyLoss(),
            ToyPostProcess(),
            None,
            True,
            marks=pytest.mark.xfail(reason="Label keys not provided"),
        ),
    ],
)
def test_anchor_module(
    ext_feat, target, loss, postprocess, desc, output_target
):

    mod = AnchorModule(
        anchor_generator=ToyAG(),
        head=ToyHead(),
        ext_feat=ext_feat,
        target=target,
        loss=loss,
        postprocess=postprocess,
        desc=desc,
        output_target=output_target,
        target_opt_keys=[],
    )

    features = torch.randn((1, 512, 7, 7))
    label = {
        "gt_boxes": torch.randint(0, 1000, (1,)),
        "gt_boxes_num": torch.zeros(0),
    }

    result = mod(features, y=label)

    assert isinstance(result, dict)
    if postprocess is not None:
        assert "toy_predict" in result
    if loss is not None:
        assert "toy_loss" in result
    if target is not None and output_target:
        assert "toy_target" in result
