from typing import Sequence

import pytest
import torch
import torch.nn as nn

from hat.models.task_modules.anchorfree_module import AnchorFreeModule
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
    def forward(self, pred, *args, **kwargs):
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
    ["ext_feat", "target", "loss", "postprocess", "desc"],
    [
        pytest.param(ToyNeck(), None, None, ToyPostProcess(), None),
        pytest.param(None, None, None, ToyPostProcess(), ToyHeadParser()),
        pytest.param(None, ToyTarget(), ToyLoss(), None, None),
        pytest.param(None, ToyTarget(), ToyLoss(), ToyPostProcess(), None),
    ],
)
def test_anchorfree_module(ext_feat, target, loss, postprocess, desc):

    mod = AnchorFreeModule(
        head=ToyHead(),
        ext_feat=ext_feat,
        target=target,
        loss=loss,
        postprocess=postprocess,
        desc=desc,
    )

    features = torch.randn((1, 512, 7, 7))
    h, w = 576, 704
    label = {}
    label["gt_bboxes"] = [
        torch.Tensor([[w // 4, h // 4, w // 2, h // 2]]),
        torch.Tensor(
            [
                [w // 8, h // 8, w // 3, h // 3],
                [w // 2, h // 2, w * 3 // 4, h * 3 // 4],
            ]
        ),
    ]
    label["gt_classes"] = [
        torch.Tensor([0]).long(),
        torch.Tensor([1, 1]).long(),
    ]

    result = mod(features, y=label)

    assert isinstance(result, dict)
    if postprocess is not None:
        assert "toy_predict" in result
    if loss is not None:
        assert "toy_loss" in result
