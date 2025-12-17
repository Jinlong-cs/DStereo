import copy

import pytest
import torch
import torch.nn as nn

from hat.models.task_modules.mtfcos3d_output_module import MTFCOS3DOutputModule
from tests.utils import gen_fake_det_label_pred_data


class ToyHead(nn.Module):
    def forward(self, feats):
        return {"toy_head": feats}


features = torch.randn((1, 512, 7, 7))
h, w = 576, 704
strides = [8, 16, 32, 64]
num_classes = 1
label, _ = gen_fake_det_label_pred_data(h, w, strides, num_classes)
label_with_roikey = copy.deepcopy(label)
label_with_roikey.update(gt_boxes_num=1)


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


class ToyRoiDecoder(nn.Module):
    def forward(self, pred, *args, **kwargs):
        # return tuple instead of list, cos constant container is recommended
        # by `torch.jit.trace()`
        # return multi tensors just for test

        pred = list(pred.values())
        cat_pred = torch.cat(pred, dim=1)
        pred1, pred2 = torch.split(
            cat_pred, split_size_or_sections=cat_pred.shape[1] // 2, dim=1
        )
        return {"toy_roi_predict": [pred1, pred2]}


@pytest.mark.parametrize(
    ["target", "loss", "postprocess", "roi_decoder", "label", "roi_task_key"],
    [
        pytest.param(
            ToyTarget(),
            ToyLoss(),
            ToyPostProcess(),
            ToyRoiDecoder(),
            label,
            "gt_boxes_num",
        ),
        pytest.param(
            ToyTarget(),
            ToyLoss(),
            ToyPostProcess(),
            ToyRoiDecoder(),
            label_with_roikey,
            "gt_boxes_num",
        ),
        pytest.param(
            ToyTarget(),
            ToyLoss(),
            ToyPostProcess(),
            ToyRoiDecoder(),
            label_with_roikey,
            "aaaa",
        ),
        pytest.param(
            None,
            ToyLoss(),
            ToyPostProcess(),
            ToyRoiDecoder(),
            label,
            "gt_boxes_num",
        ),
        pytest.param(
            None,
            None,
            ToyPostProcess(),
            ToyRoiDecoder(),
            label,
            "gt_boxes_num",
        ),
    ],
)
def test_mtfcos3d_output_module(
    target, loss, postprocess, roi_decoder, label, roi_task_key
):
    mod = MTFCOS3DOutputModule(
        head=ToyHead(),
        head_parser=None,
        target=target,
        loss=loss,
        postprocess=postprocess,
        roi_decoder=roi_decoder,
        roi_task_key=roi_task_key,
    )

    result = mod(features, label=label)

    if roi_decoder is not None and roi_task_key in label:
        assert "toy_roi_predict" in result
        return
    if postprocess is not None:
        assert "toy_predict" in result
    if loss is not None:
        assert "toy_loss" in result
