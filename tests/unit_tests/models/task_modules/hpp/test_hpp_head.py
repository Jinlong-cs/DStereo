import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test, qtensor_test


def test_hpp_head():
    alpha = 0.5
    channel_list_ = [32, 32, 64, 128, 256]
    out_channels_ = int(channel_list_[2] * alpha)
    config = dict(
        type="HPPDecodeHead",
        block_num=4,
        in_channels=out_channels_,
        out_channels=out_channels_,
    )
    hpp_head = build_from_registry(config)
    feats = torch.randn(1, 32, 32, 64)
    hpp_head_out = hpp_head(feats)

    assert "offsets" in hpp_head_out
    assert "confidences" in hpp_head_out
    assert "instances" in hpp_head_out
    assert "att_feats" in hpp_head_out

    # test qat
    feats_qat = qtensor_test(feats)
    qat_test(hpp_head, feats_qat, with_quantized=False)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
