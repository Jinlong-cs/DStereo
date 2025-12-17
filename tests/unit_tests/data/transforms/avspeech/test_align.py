from copy import deepcopy

import numpy as np
import torch

from hat.data.transforms.avspeech import AVWindowAlign, ForceAlign


def test_align():
    # 构造数据
    audio = torch.rand((186, 80), dtype=torch.float32)
    images = [
        np.random.randint(0, 256, (96, 96, 3), dtype="uint8")
        for _ in range(47)
    ]
    sr_data = {"audio": audio, "images": images}

    # 测试ForceAlign
    data = deepcopy(sr_data)
    data = ForceAlign(
        v2a_factor=4,
        audio_left_extend=0,
    )(data)
    assert "audio" in data
    assert "images" in data
    assert "images_lens" in data and data["images_lens"] == 46
    assert "audio_lens" in data and data["audio_lens"] == 184

    data = deepcopy(sr_data)
    data = ForceAlign(
        v2a_factor=4,
        audio_left_extend=7,
    )(data)
    assert "audio" in data
    assert "images" in data
    assert "images_lens" in data and data["images_lens"] == 44
    assert "audio_lens" in data and data["audio_lens"] == 183

    # 构造数据
    audio = torch.rand((157, 1, 3, 40), dtype=torch.float32)
    images = [
        np.random.randint(0, 256, (96, 96, 3), dtype="uint8")
        for _ in range(52)
    ]
    data = {"audio": audio, "images": images}

    # 测试AVWindowAlign
    data = AVWindowAlign(
        video_window_size=10,
        v2a_factor=3,
        audio_pre_padding_mode="zero",
    )(data)
    assert "audio" in data
    assert "images" in data
    assert "images_lens" in data and data["images_lens"] == 52
    assert "audio_lens" in data and data["audio_lens"] == 52
