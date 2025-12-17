import numpy as np
import torch
from horizon_plugin_pytorch.quantization import QTensor

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test_with_multi_inputs

bn_kwargs = dict(eps=2e-5, momentum=0.1)


def gen_data():
    feat = torch.from_numpy(np.random.randn(1, 160, 128, 128)).float()
    bev_seg = torch.from_numpy(np.random.randn(1, 10, 512, 512)).float()
    bev_bboxes = [np.random.randn(10, 10), np.random.randn(11, 10)]
    data = {"bev_bboxes_labels": bev_bboxes, "bev_seg_indices": bev_seg}
    return feat, data


def test_encocde():

    config = dict(
        type="BevEncoder",
        backbone=dict(
            type="VargBevBackbone",
            in_channels=160,
            feat_channels=[160],
            out_channels=64,
            bn_kwargs=bn_kwargs,
        ),
    )

    encoder = build_from_registry(config)

    feat, data = gen_data()

    encoder(feat, data)
    q_feat = QTensor(feat, scale=torch.tensor([0.78]), dtype="qint8")
    qat_test_with_multi_inputs(encoder, (q_feat, data), with_quantized=False)
