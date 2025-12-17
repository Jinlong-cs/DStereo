import pytest
import torch
from horizon_plugin_pytorch.quantization import prepare_qat

from hat.models.base_modules.roi_feat_extractors import (
    Cropper,
    CropperQAT,
    MultiScaleRoIAlign,
    RoiResize,
)
from hat.utils.package_helper import check_packages_available


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.parametrize(
    ["size", "strides", "feature_maps", "pixel", "batch_index", "rois"],
    [
        pytest.param(
            11,
            [2, 4, 8, 16],
            [
                torch.randn((4, 32, int(512 / x), int(512 / x)))
                for x in [2, 4, 8, 16]
            ],
            torch.ones([10, 2]) * 256,
            torch.Tensor([0, 0, 1, 1, 2, 2, 3, 3, 3, 3]),
            [
                [torch.ones([y, 4]) for y in [2, 2, 2, 4]]
                for x in [2, 4, 8, 16]
            ],
        ),
    ],
)
def test_cropper_qat(size, strides, feature_maps, pixel, batch_index, rois):
    model = CropperQAT(
        size=size,
        strides=strides,
    )
    output = model(feature_maps, pixel, batch_index, None)
    output_2 = model(feature_maps, pixel, batch_index, rois)
    assert output.shape[0] == pixel.shape[0]
    assert output.shape[1] == sum([x.shape[1] for x in feature_maps])
    assert output.shape[2] == size
    assert output.shape[3] == size
    assert output.shape == output_2.shape


@pytest.mark.parametrize(
    [
        "size",
        "strides",
        "heading_norm",
        "feature_maps",
        "pixel",
        "batch_index",
        "angle",
    ],
    [
        pytest.param(
            11,
            [2, 4, 8, 16],
            False,
            [
                torch.randn((4, 32, int(512 / x), int(512 / x)))
                for x in [2, 4, 8, 16]
            ],
            torch.ones([10, 2]) * 256,
            torch.Tensor([0, 0, 1, 1, 2, 2, 3, 3, 3, 3]),
            None,
        ),
        pytest.param(
            11,
            [2, 4, 8, 16],
            True,
            [
                torch.randn((4, 32, int(512 / x), int(512 / x)))
                for x in [2, 4, 8, 16]
            ],
            torch.ones([10, 2]) * 256,
            torch.Tensor([0, 0, 1, 1, 2, 2, 3, 3, 3, 3]),
            torch.randn([10]),
        ),
    ],
)
def test_cropper(
    size, strides, heading_norm, feature_maps, pixel, batch_index, angle
):
    model = Cropper(size, strides, heading_norm)
    output = model(feature_maps, pixel, batch_index, angle)
    assert output.shape[0] == pixel.shape[0]
    assert output.shape[1] == sum([x.shape[1] for x in feature_maps])
    assert output.shape[2] == size
    assert output.shape[3] == size


@pytest.mark.parametrize(
    ["in_strides", "roi_resize_cfgs", "resize_mode", "feature_maps"],
    [
        [
            [2, 4, 8, 16],
            [
                {
                    "in_stride": 2,
                    "output_size": (512, 512),
                    "roi_box": [128, 128, 384, 384],
                    "roi_feat_name": "roi_feat_0",
                },
                {
                    "in_stride": 2,
                    "output_size": (512, 512),
                    "roi_box": [0, 0, 384, 384],
                    "roi_feat_name": "roi_feat_1",
                },
            ],
            "bilinear",
            [
                torch.randn((4, 32, int(1024 / x), int(1024 / x)))
                for x in [2, 4, 8, 16]
            ],
        ],
        [
            [2, 4, 8, 16],
            [
                {
                    "in_stride": 2,
                    "output_size": (512, 512),
                    "roi_box": [128, 128, 384, 384],
                    "roi_feat_name": "roi_feat_0",
                },
                {
                    "in_stride": 2,
                    "output_size": (512, 512),
                    "roi_box": [0, 0, 384, 384],
                    "roi_feat_name": "roi_feat_1",
                },
            ],
            "nearest",
            [
                torch.randn((4, 32, int(1024 / x), int(1024 / x)))
                for x in [2, 4, 8, 16]
            ],
        ],
    ],
)
def test_roiresize(in_strides, roi_resize_cfgs, resize_mode, feature_maps):
    roi_resize = RoiResize(in_strides, roi_resize_cfgs, resize_mode)
    output = roi_resize(feature_maps)

    assert len(output) == len(roi_resize_cfgs)
    for roi_resize_cfg in roi_resize_cfgs:
        assert (
            output[roi_resize_cfg["roi_feat_name"]].shape[2:]
            == roi_resize_cfg["output_size"]
        )


def test_multi_scale_roialign_qat_compatibility():
    model1 = MultiScaleRoIAlign(
        output_size=(8, 8),
        feature_strides=[2, 4],
    )

    model2 = MultiScaleRoIAlign(
        output_size=(8, 8),
        feature_strides=[2, 4],
    )

    class Pass(torch.nn.Module):
        def forward(self, x):
            return x

    model1.quant_roi = Pass()

    model1.set_qconfig()
    model2.set_qconfig()

    prepare_qat(model1, inplace=True)
    prepare_qat(model2, inplace=True)

    model2.load_state_dict(model1.state_dict(), strict=True)
