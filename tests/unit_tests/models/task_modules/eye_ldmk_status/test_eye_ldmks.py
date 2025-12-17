# Copyright (c) Horizon Robotics. All rights reserved.

import torch

from hat.models.task_modules.eye_ldmk_status import FPEM, FPEM_FFM


def test_fpem():
    bn_kwargs = dict(eps=1e-3, momentum=0.01)
    fpem = FPEM(bn_kwargs, act_type="relu", in_channels=128, inplace=True)
    c2 = torch.rand(4, 128, 24, 40)
    c3 = torch.rand(4, 128, 12, 20)
    c4 = torch.rand(4, 128, 6, 10)
    c5 = torch.rand(4, 128, 3, 5)
    c2, c3, c4, c5 = fpem(c2, c3, c4, c5)
    assert c2.shape == (4, 128, 24, 40)
    assert c3.shape == (4, 128, 12, 20)
    assert c4.shape == (4, 128, 6, 10)
    assert c5.shape == (4, 128, 3, 5)


def test_fpem_ffm():
    bn_kwargs = dict(eps=1e-3, momentum=0.01)
    in_channels_list = [16, 32, 64, 128]
    fpem_ffm = FPEM_FFM(bn_kwargs, in_channels_list)
    c2 = torch.rand(4, 16, 24, 40)
    c3 = torch.rand(4, 32, 12, 20)
    c4 = torch.rand(4, 64, 6, 10)
    c5 = torch.rand(4, 128, 3, 5)
    x = [c2, c3, c4, c5]
    y = fpem_ffm(x)
    assert y.shape == (4, 16, 24, 40)
