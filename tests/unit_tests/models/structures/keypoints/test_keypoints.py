import horizon_plugin_pytorch as horizon
import numpy as np
import pytest
import torch

from hat.registry import build_from_registry
from hat.utils import qconfig_manager

NUM_LDMK = 12

config = dict(
    type="HeatmapKeypointModel",
    backbone=dict(
        type="efficientnet",
        model_type="b0",
        num_classes=1,
        bn_kwargs={},
        activation="relu",
        use_se_block=False,
        include_top=False,
    ),
    decode_head=dict(
        type="DeconvDecoder",
        in_channels=320,
        out_channels=NUM_LDMK,
        input_index=4,
        num_conv_layers=3,
        num_deconv_filters=[128, 128, 128],
        num_deconv_kernels=[4, 4, 4],
        final_conv_kernel=3,
    ),
    deploy=True,
)


def gen_data():
    imgs = torch.from_numpy(np.random.randn(6, 3, 128, 128)).float()
    ldmk = torch.from_numpy(np.random.randn(6, NUM_LDMK, 2)).float() * 128

    ldmk_attr = torch.ones([6, NUM_LDMK]) * 2.0

    heatmap = torch.from_numpy(np.random.randn(6, NUM_LDMK, 32, 32)).float()
    heatmap_w = torch.from_numpy(np.random.randn(6, NUM_LDMK, 32, 32)).float()
    data = {
        "img": imgs,
        "gt_ldmk": ldmk,
        "gt_ldmk_attr": ldmk_attr,
        "layout": "chw",
        "gt_heatmap": heatmap,
        "gt_heatmap_weight": heatmap_w,
    }
    return data


def qat_test(model, input, with_quantized=True):
    horizon.march.set_march(horizon.march.March.BAYES)
    # model.fuse_model()
    qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.QAT)
    model.qconfig = qconfig_manager.get_default_qat_qconfig()
    if hasattr(model, "set_qconfig"):
        model.set_qconfig()
    qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.COMPATIBLE)
    qat_model = horizon.quantization.prepare_qat(model, inplace=False)

    qat_preds = qat_model(input)
    assert qat_preds is not None

    quantized_preds = None
    if with_quantized:
        quantized_model = horizon.quantization.convert(
            qat_model.eval(), inplace=False
        )
        quantized_preds = quantized_model(input)
        assert quantized_preds is not None

    return qat_preds, quantized_preds


@pytest.mark.parametrize(
    ["mode"],
    [
        pytest.param("train"),
        pytest.param("val"),
    ],
)
def test_keypoint_model(mode):
    model = build_from_registry(config)

    data = gen_data()

    if mode == "train":
        model(data)

        qat_test(model, data, with_quantized=False)

    if mode == "val" or mode == "test":
        model(data)
