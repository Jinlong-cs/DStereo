import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test_fx

config = dict(
    type="MotionForecasting",
    encoder=dict(
        type="Vectornet",
        depth=3,
        traj_in_channels=9,
        traj_num_vec=9,
        lane_in_channels=11,
        lane_num_vec=19,
        hidden_size=128,
    ),
    decoder=dict(
        type="Densetnt",
        in_channels=128,
        hidden_size=128,
        num_traj=32,
        target_graph_depth=2,
        pred_steps=30,
        top_k=150,
    ),
    target=dict(
        type="DensetntTarget",
    ),
    loss=dict(
        type="DensetntLoss",
    ),
    postprocess=dict(
        type="DensetntPostprocess", threshold=2.0, pred_steps=30, mode_num=6
    ),
)


def gen_data():
    data = {
        "traj_feat": torch.randn((1, 9, 19, 32)),
        "lane_feat": torch.randn((1, 11, 9, 64)),
        "instance_mask": torch.randn((1, 1, 1, 96)),
        "goals_2d": torch.randn((1, 2, 1, 2048)),
        "goals_2d_mask": torch.randn((1, 1, 1, 2048)),
        "traj_labels": torch.randn((1, 30, 2)),
        "goals_2d_labels": torch.ones((1)).long(),
        "end_points": torch.randn((1, 1, 2)),
    }
    return data


@pytest.mark.parametrize(
    ["mode"],
    [
        pytest.param("train"),
        pytest.param("val"),
    ],
)
def test_motion_forecasting(mode):
    model = build_from_registry(config)

    data = gen_data()

    if mode == "train":
        model(data)

        qat_test_fx(model, data, with_quantized=False)

    if mode == "val" or mode == "test":
        model.eval()
        model(data)

        qat_test_fx(model, data, with_quantized=False)
