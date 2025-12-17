import pytest
import torch.nn as nn

from hat.registry import build_from_registry

try:
    from mmdet import models
except ImportError:
    MMDET_AVAILABLE = False
else:
    MMDET_AVAILABLE = True


@pytest.mark.skipif(not MMDET_AVAILABLE, reason="require mmdet")
def test_contrib_mmdet():
    cfg = dict(
        type=models.necks.FPN,
        in_channels=[32, 64, 128, 1024],
        out_channels=128,
        start_level=1,
        num_outs=5,
        relu_before_extra_convs=True,
    )
    model = build_from_registry(cfg)
    assert isinstance(model, nn.Module)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
