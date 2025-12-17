import pytest
import torch

from hat.registry import build_from_registry


def gen_bevformerprocess_data():
    preds_dicts = {
        "all_cls_scores": torch.randn(6, 1, 900, 10),
        "all_bbox_preds": torch.randn(6, 1, 900, 10),
    }
    return preds_dicts


@pytest.mark.serial_task
def test_bevformerprocess():
    config = dict(
        type="BevFormerProcess",
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_num=300,
        num_classes=10,
    )
    model = build_from_registry(config)
    data = gen_bevformerprocess_data()
    model(data)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
