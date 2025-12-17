import pytest
import torch

from hat.data.transforms.avspeech.denoise import (
    LocalHisfDenoiserV103,
    LocalHisfDenoiserV200,
)
from tests import HAT_BUCKET_PATH

try:
    import pyhisf
    import soundfile as sf
except ImportError:
    pyhisf = None
    sf = None


@pytest.mark.skipif(pyhisf is None, reason="need pyhisf")
@pytest.mark.skipif(sf is None, reason="need soundfile")
def test_denoise():
    hisf_root_v103 = f"{HAT_BUCKET_PATH}/unit_test_data/avspeech/data/transforms/hisf_loader_v103"  # noqa: E501
    hisf_root_v200 = f"{HAT_BUCKET_PATH}/unit_test_data/avspeech/data/transforms/hisf_loader_v200"  # noqa: E501

    # 测试LocalHisfDenoiserV103
    waveform = torch.randn((10000, 2), dtype=torch.float32)
    data = {
        "waveform": (waveform, 16000),
        "channel_dim": 1,
        "role": "pilot",
    }

    data = LocalHisfDenoiserV103(
        hisf_root=hisf_root_v103,
        channel_select=True,
        pre_pad=True,
        p=1.0,
    )(data)

    assert data["waveform"][0].shape == (10000, 1)
    assert "source_audio" not in data
    assert "role" not in data

    # 测试LocalHisfDenoiserV200
    waveform = torch.randn((10000, 2), dtype=torch.float32)
    data = {
        "waveform": (waveform, 16000),
        "channel_dim": 1,
        "role": "pilot",
    }

    data = LocalHisfDenoiserV200(
        hisf_root=hisf_root_v200,
        channel_select=True,
        pre_pad=True,
        p=1.0,
    )(data)

    assert data["waveform"][0].shape == (10000, 1)
    assert "source_audio" not in data
    assert "role" not in data
