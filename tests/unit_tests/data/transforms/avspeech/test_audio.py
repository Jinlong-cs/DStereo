import pathlib
import tempfile
from copy import deepcopy

import numpy as np
import torch

from hat.data.transforms.avspeech.audio import (
    MFCC,
    ChannelSelect,
    FBank,
    PAFFBank,
    RandomSpeedPerturb,
    ReSample,
    SpecAug,
    SpecSub,
    WaveformToTensor,
)

try:
    import py_audio_feature
except ImportError:
    py_audio_feature = None

try:
    import torchaudio
except ImportError:
    torchaudio = None

try:
    import soundfile as sf
except ImportError:
    sf = None


def check_data(path):
    if torchaudio is None or py_audio_feature is None or sf is None:
        return

    waveform, samplerate = sf.read(path)
    src_data = {
        "waveform": (waveform, samplerate),
        "channel_dim": 1,
    }

    # 测试WaveformToTensor
    src_data = WaveformToTensor()(src_data)
    waveform = src_data["waveform"][0]
    assert isinstance(waveform, torch.Tensor)
    assert len(waveform.shape) == 2 and waveform.shape[1] == 2

    # 测试ReSample
    # 测试音频采样率和目标采样率相同的情况
    tgt_sr = 48000
    data = ReSample(new_freq=tgt_sr)(src_data)
    waveform = data["waveform"][0]
    assert len(waveform.shape) == 2 and waveform.shape[1] == 2
    assert data["waveform"][1] == tgt_sr
    # 测试音频采样率和目标采样率不同的情况
    tgt_sr = 16000
    data = ReSample(new_freq=tgt_sr)(src_data)
    waveform = data["waveform"][0]
    assert len(waveform.shape) == 2 and waveform.shape[1] == 2
    assert data["waveform"][1] == tgt_sr

    # 测试ChannelSelect
    data = ChannelSelect(random=False, channel=0)(data)
    waveform = data["waveform"][0]
    assert len(waveform.shape) == 2 and waveform.shape[1] == 1
    assert data["waveform"][1] == 16000

    # 测试RandomSpeedPerturb
    data = RandomSpeedPerturb(speeds=[0.9, 1.0, 1.1], channel_first=True)(data)
    waveform = data["waveform"][0]
    assert len(waveform.shape) == 2 and waveform.shape[1] == 1
    assert data["waveform"][1] == 16000

    # 测试FBank
    data_fbank = deepcopy(data)
    data_fbank = FBank(
        num_mel_bins=80, frame_shift=10, frame_length=25, dither=1.0
    )(data_fbank)
    assert "audio" in data_fbank and "audio_lens" in data_fbank
    assert "waveform" not in data_fbank

    # 测试 PAFFBank
    if py_audio_feature is not None:
        data_paf_fbank = deepcopy(data)
        data_paf_fbank = PAFFBank()(data_paf_fbank)
        assert "audio" in data_fbank and "audio_lens" in data_fbank
        assert "waveform" not in data_fbank
        assert data_fbank["audio"].size() == data_paf_fbank["audio"].size()

    # 测试MFCC
    data_mfcc = deepcopy(data)
    data_mfcc = MFCC(
        num_mel_bins=80, frame_shift=10, frame_length=25, dither=1.0
    )(data_mfcc)
    assert "audio" in data_mfcc and "audio_lens" in data_mfcc
    assert "waveform" not in data_mfcc

    # 测试SpecAug
    data = SpecAug(num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, p=1.0)(
        data_fbank
    )
    assert "audio" in data and "audio_lens" in data

    # 测试SpecSub
    data = SpecSub(max_t=20, num_t_sub=3)(data)
    assert "audio" in data and "audio_lens" in data


def test_audio():
    if sf is None:
        return

    wav_data = np.random.random((10000, 2))
    samplerate = 48000
    with tempfile.TemporaryDirectory() as dtmp:
        path = pathlib.Path(dtmp).joinpath("test.wav")
        sf.write(
            path,
            wav_data,
            samplerate,
        )
        check_data(path)
