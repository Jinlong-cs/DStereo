import numpy as np
import torch
from horizon_plugin_pytorch.quantization import QTensor

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test_with_multi_inputs

bn_kwargs = dict(eps=2e-5, momentum=0.1)

bev_size = (51.2, 51.2, 0.8)
grid_size = (128, 128)


def gen_data():
    feat = torch.from_numpy(np.random.randn(3, 64, 128, 128)).float()
    ego2global = [np.random.randn(3, 4, 4).astype(dtype=np.float32)]

    data = {
        "ego2global": ego2global,
    }
    return feat, data


def test_temporal_fusion():

    config = dict(
        type="AddTemporalFusion",
        in_channels=64,
        out_channels=64,
        num_seq=3,
        bev_size=bev_size,
        grid_size=grid_size,
        num_encoder=2,
        num_project=1,
    )

    fusion = build_from_registry(config)

    feat, data = gen_data()

    fusion(feat, data, False)
    q_feat = QTensor(feat, scale=torch.tensor([0.78]), dtype="qint8")
    qat_test_with_multi_inputs(
        fusion,
        (q_feat, data, False),
        with_quantized=False,
    )
