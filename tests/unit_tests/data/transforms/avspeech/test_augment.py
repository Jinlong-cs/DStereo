import pytest
import torch
import yaml

from hat.data.transforms.avspeech.augment import (
    OnLineAugmentPipeline,
    OnLineJsonAugmentPipeline,
)
from tests import HAT_BUCKET_PATH

try:
    import soundfile
except ImportError:
    soundfile = None

try:
    import pyroomacoustics as pra
except ImportError:
    pra = None

try:
    from gpuRIR_bind import gpuRIR_bind
except ImportError:
    gpuRIR_bind = None

try:
    import mxnet as mx
except ImportError:
    mx = None


@pytest.mark.skipif(soundfile is None, reason="need soundfile")
@pytest.mark.skipif(pra is None, reason="need pyroomacoustics")
@pytest.mark.skipif(gpuRIR_bind is None, reason="need gpuRIR_bind")
@pytest.mark.skipif(mx is None, reason="need mxnet")
def test_augment():
    noise_hdf5 = f"{HAT_BUCKET_PATH}/unit_test_data/avspeech/data/transforms/noise_unit_test.hdf5"  # noqa: E501
    noise_info = f"{HAT_BUCKET_PATH}/unit_test_data/avspeech/data/transforms/noise_unit_test_info.txt"  # noqa: E501
    augment_config = f"{HAT_BUCKET_PATH}/unit_test_data/avspeech/data/transforms/augment_unit_test.yaml"  # noqa: E501
    with open(augment_config) as fr:
        aug_config = yaml.load(fr, Loader=yaml.FullLoader)

    # 测试OnLineAugmentPipeline
    # 测试线性加噪的情况
    waveform = torch.randn((10000, 4), dtype=torch.float32)
    data = {
        "key": "J2MM_develop_speech_batch08-04619_41_02_mic1_1",
        "waveform": (waveform, 16000),
        "channel_dim": 1,
        "role": "pilot",
    }
    data = OnLineAugmentPipeline(
        config=aug_config,
        noise_hdf5_file=noise_hdf5,
        ret_dtype="float64",
        local_rir=False,
        noise_info=noise_info,
        p=1.0,
    )(data)

    assert data["waveform"][0].shape == (10000, 2)
    assert data["waveform"][1] == 16000
    assert "source_audio" in data and data["source_audio"].shape == (10000, 2)
    assert "role" in data

    # 测试单通道音频仿真的情况
    waveform = torch.randn((10000, 1), dtype=torch.float32)
    data = {
        "key": "FYZ_mmasr_batch01-F0011_1",
        "waveform": (waveform, 16000),
        "channel_dim": 1,
        "role": "pilot",
    }
    data = OnLineAugmentPipeline(
        config=aug_config,
        noise_hdf5_file=noise_hdf5,
        ret_dtype="float64",
        local_rir=False,
        noise_info=noise_info,
        p=1.0,
    )(data)

    assert data["waveform"][0].shape == (10000, 2)
    assert data["waveform"][1] == 16000
    assert "source_audio" in data and data["source_audio"].shape == (10000, 1)
    assert "role" in data

    # 测试OnLineJsonAugmentPipeline
    waveform = torch.randn((10000, 2), dtype=torch.float32)
    data = {
        "waveform": (waveform, 16000),
        "channel_dim": 1,
    }
    data = OnLineJsonAugmentPipeline(
        config=aug_config,
        noise_hdf5_file=noise_hdf5,
        ret_dtype="float64",
        local_rir=False,
        noise_info=noise_info,
        p=1.0,
    )(data)

    assert data["waveform"][0].shape == (10000, 2)
    assert data["waveform"][1] == 16000
    assert "source_audio" in data and data["source_audio"].shape == (10000, 2)
