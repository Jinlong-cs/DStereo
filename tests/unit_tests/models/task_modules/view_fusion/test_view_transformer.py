import numpy as np
import torch
from horizon_plugin_pytorch.quantization import QTensor

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test_with_multi_inputs

bn_kwargs = dict(eps=2e-5, momentum=0.1)

bev_size = (51.2, 51.2, 0.8)
grid_size = (128, 128)


def gen_data():
    feat = torch.from_numpy(np.random.randn(6, 64, 15, 32)).float()
    imgs = torch.from_numpy(np.random.randn(6, 3, 512, 960)).float()
    ego2imgs = torch.from_numpy(np.random.randn(6, 4, 4)).float()
    data = {"img": imgs, "ego2img": ego2imgs}
    return feat, data


def test_ipm():
    config = dict(
        type="WrappingTransformer",
        bev_size=bev_size,
        num_views=6,
        grid_size=grid_size,
    )

    transformer = build_from_registry(config)

    feat, data = gen_data()

    transformer(feat, data, False)
    q_feat = QTensor(feat, scale=torch.tensor([0.78]), dtype="qint8")
    qat_test_with_multi_inputs(
        transformer,
        (q_feat, data, False),
        with_quantized=False,
    )


def test_lss():
    config = dict(
        type="LSSTransformer",
        in_channels=64,
        feat_channels=64,
        z_range=(-10.0, 10.0),
        depth=60,
        num_points=10,
        bev_size=bev_size,
        grid_size=grid_size,
        num_views=6,
    )

    transformer = build_from_registry(config)

    feat, data = gen_data()

    transformer(feat, data, False)
    q_feat = QTensor(feat, scale=torch.tensor([0.78]), dtype="qint8")
    qat_test_with_multi_inputs(
        transformer,
        (q_feat, data, False),
        with_quantized=False,
    )


def test_gkt():
    config = dict(
        type="GKTTransformer",
        bev_size=bev_size,
        grid_size=grid_size,
        embed_dims=64,
        num_views=6,
    )

    transformer = build_from_registry(config)

    feat, data = gen_data()

    transformer(feat, data, False)
    q_feat = QTensor(feat, scale=torch.tensor([0.78]), dtype="qint8")
    qat_test_with_multi_inputs(
        transformer,
        (q_feat, data, False),
        with_quantized=False,
    )
