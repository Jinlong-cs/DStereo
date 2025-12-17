# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch
import torch.nn as nn

from hat.models.task_modules.roi_module import RoIModulePE
from tests.unit_tests.models.task_modules.test_anchor_module import (
    ToyHead,
    ToyLoss,
    ToyPostProcess,
    ToyTarget,
)


def generate_fake_inputs(input_size, in_strides, in_channels, output_strides):
    inputs = [
        torch.randn(1, in_ch, input_size[1] // in_st, input_size[0] // in_st)
        for (in_ch, in_st) in zip(in_channels, in_strides)
    ]

    out_stride = output_strides[0]
    pe_feat_shape = (
        1,
        3,
        input_size[1] // out_stride,
        input_size[0] // out_stride,
    )
    coordinate_map = torch.randn(pe_feat_shape, dtype=torch.float32)

    return inputs, coordinate_map


class ToyRoIFeatExt(nn.Module):
    def __init__(self, crop_area, output_index):
        super().__init__()
        self.crop_area = crop_area
        self.output_index = output_index

    def fake_roi_process(self, feat_maps):
        feat_map = feat_maps[self.output_index]
        feat_map = feat_map[:, :, : self.crop_area[0], : self.crop_area[1]]

        return feat_map

    def forward(self, x, rois):
        return self.fake_roi_process(x)


class ToyNeck(nn.Module):
    def forward(self, feature_maps, coordinate_map=None):
        return feature_maps


@pytest.mark.parametrize(
    ["target", "loss", "postprocess", "output_target", "rpn_pred"],
    [
        pytest.param(None, None, ToyPostProcess(), False, None),
        pytest.param(
            ToyTarget(),
            ToyLoss(),
            None,
            True,
            None,
            marks=pytest.mark.xfail(reason="Target should not be provided"),
        ),
        pytest.param(
            ToyTarget(),
            ToyLoss(),
            None,
            True,
            {"pred_boxes": torch.Tensor()},
        ),
        pytest.param(
            None,
            ToyLoss(),
            ToyPostProcess(),
            True,
            None,
            marks=pytest.mark.xfail(reason="Label keys not provided"),
        ),
    ],
)
def test_roi_module(target, loss, postprocess, output_target, rpn_pred):

    model = RoIModulePE(
        roi_feat_extractor=ToyRoIFeatExt(crop_area=(8, 8), output_index=0),
        head=ToyHead(),
        ext_feat=ToyNeck(),
        target=target,
        loss=loss,
        postprocess=postprocess,
        output_target=output_target,
        target_opt_keys=[],
        postprocess_keys=("calib", "distCoeffs"),
    )

    features, coordinate_map = generate_fake_inputs(
        input_size=(1920 // 2, 1280 // 2),
        in_strides=[4, 8, 16, 32, 64],
        in_channels=[16, 32, 64, 128, 128],
        output_strides=[4],
    )

    label = {
        "gt_boxes": torch.randint(0, 1000, (1,)),
        "gt_boxes_num": torch.zeros(0),
    }

    result = model(features, coordinate_map, rpn_pred=rpn_pred, y=label)

    assert isinstance(result, dict)
    if postprocess is not None:
        assert "toy_predict" in result
    if loss is not None:
        assert "toy_loss" in result
    if target is not None and output_target:
        assert "toy_target" in result


if __name__ == "__main__":
    pytest.main(["-s", __file__])
