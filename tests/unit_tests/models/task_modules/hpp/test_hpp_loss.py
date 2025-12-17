import pytest
import torch

from hat.registry import build_from_registry


def test_hpp_loss_task():
    config = dict(
        type="HppLoss",
        ins_embedding_channel=4,
        weight_offset=0.2,
        weight_exist=1.0,
        weight_nonexist=1.0,
        weight_attention=0.1,
        weight_sisc=0.5,
    )
    hpp_loss = build_from_registry(config)
    feat_stride = 8
    img_h, img_w = (256, 512)
    grid_h, grid_w = img_h // feat_stride, img_w // feat_stride
    pred = {
        "offsets": [torch.rand(2, 2, grid_h, grid_w) for i in range(4)],
        "confidences": [torch.rand(2, 1, grid_h, grid_w) for i in range(4)],
        "instances": [torch.rand(2, 4, grid_h, grid_w) for i in range(4)],
        "att_feats": [torch.rand(2, 32, 2, 4) for i in range(4)],
    }

    target = {
        "img": torch.rand(2, 3, img_h, img_w),
        "labels": (
            torch.rand(2, 3, grid_h, grid_w),
            torch.randint(0, 3, (2, 1, grid_h * grid_w, grid_h * grid_w)),
        ),
    }
    hpp_loss_result = hpp_loss(pred, target)
    assert "hpp_exist_loss" in hpp_loss_result
    assert "hpp_nonexist_loss" in hpp_loss_result
    assert "hpp_offset_loss" in hpp_loss_result
    assert "hpp_sisc_loss" in hpp_loss_result
    assert "hpp_attention_loss" in hpp_loss_result


if __name__ == "__main__":
    pytest.main(["-s", __file__])
