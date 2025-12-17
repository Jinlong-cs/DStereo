import torch

from hat.models.task_modules.traj_pred.heads.multipathpp_decoder import (
    MTPPlusDecoder,
    MultiHeadMTPPlusDecoder,
)


def test_MTPPlusDecoder():
    model = MTPPlusDecoder(
        num_modes=6,
        use_variance=False,
        op_len=80,
        hidden_size=1024,
        encoding_size=4096,
    )

    test_agg_encoding = torch.randn(1, 6, 4096)
    output = model(test_agg_encoding)

    assert output["traj"].shape == (1, 6, 80, 2)
    assert output["probs"].shape == (1, 6)


def test_MultiHeadMTPPlusDecoder():
    model = MultiHeadMTPPlusDecoder(
        num_header=2,
        num_modes=6,
        use_variance=False,
        op_len=80,
        hidden_size=1024,
        encoding_size=4096,
    )

    test_agg_encoding = [
        torch.randn(1, 6, 4096),
        torch.randn(1, 6, 4096),
    ]

    output = model(test_agg_encoding)

    for i in range(2):
        assert output[i]["traj"].shape == (1, 6, 80, 2)
        assert output[i]["probs"].shape == (1, 6)
