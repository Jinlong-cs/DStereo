import torch

from hat.models.task_modules.stereonet import (
    StereoNetPostProcess,
    StereoNetPostProcessPlus,
)


def test_stereonet_post_process():

    stereonet_post_process = StereoNetPostProcess(maxdisp=192)
    inputs = [
        torch.randn((1, 1, 16, 32), dtype=torch.float),
        torch.randn((1, 1, 32, 64), dtype=torch.float),
        torch.randn((1, 1, 64, 128), dtype=torch.float),
        torch.randn((1, 1, 128, 256), dtype=torch.float),
        torch.randn((1, 1, 256, 512), dtype=torch.float),
    ]

    stereonet_post_process.eval()

    outputs = stereonet_post_process(inputs)

    assert outputs.shape == (1, 256, 512)


def test_stereonet_post_process_plus():

    stereonet_post_process = StereoNetPostProcessPlus(maxdisp=192)
    inputs = [
        torch.randn((1, 1, 32, 64), dtype=torch.float),
        torch.randn((1, 4, 32, 64), dtype=torch.float),
        torch.randn((1, 4, 256, 512), dtype=torch.float),
    ]

    stereonet_post_process.eval()

    outputs = stereonet_post_process(inputs)

    assert outputs.shape == (1, 256, 512)
