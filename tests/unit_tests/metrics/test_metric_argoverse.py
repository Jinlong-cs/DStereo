import pytest
import torch

from hat.metrics.argoverse_metric import ArgoverseMetric


def test_argoverse_metric():
    metric = ArgoverseMetric(max_guesses=6, horizon=30, miss_threshold=2.0)

    meta = {"file_name": ["test"], "traj_labels": torch.randn((1, 30, 2))}
    preds = (
        torch.randn((1, 6, 30, 2)),
        torch.randn((1, 6, 1)),
    )
    metric.update(meta, preds)
    metric.get()
    pass


if __name__ == "__main__":
    pytest.main(["-s", __file__])
