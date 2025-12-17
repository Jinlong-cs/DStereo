import pytest
import torch

from hat.registry import build_from_registry


def test_hpp_decoder():
    config = dict(
        type="HPPDecoder",
        feat_stride=8,
        img_shape=(512, 256),
        point_thresh=(0.7),
    )
    hpp_decoder = build_from_registry(config)
    grid_h = 32
    grid_w = 64
    pred = {
        "offsets": [torch.rand(1, 2, grid_h, grid_w) for i in range(4)],
        "confidences": [torch.rand(1, 1, grid_h, grid_w) for i in range(4)],
        "instances": [torch.rand(1, 4, grid_h, grid_w) for i in range(4)],
        "att_feats": [torch.rand(1, 32, 2, 4) for i in range(4)],
    }
    target = {"img_name": ["first_img.png"]}
    hpp_decoder_result = hpp_decoder(pred, target)
    assert "coordinate" in hpp_decoder_result
    assert "conf" in hpp_decoder_result


if __name__ == "__main__":
    pytest.main(["-s", __file__])
