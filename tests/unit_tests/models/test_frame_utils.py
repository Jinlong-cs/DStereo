import torch

from hat.models.frame_utils import framewise_operation, pad_and_split

batchsize = 8
seqlen = 80
num_cached_frames = 15


def test_pad_and_split():
    channel_num = 512
    x = torch.rand(batchsize, channel_num, 1, seqlen)
    x = pad_and_split(x, num_cached_frames)
    assert x.shape == (batchsize * seqlen, channel_num, 1, num_cached_frames)


def test_framewise_operation():
    channel_num = 256
    x = torch.rand(batchsize, seqlen, channel_num, 1)
    x = framewise_operation(x, "fdd")
    assert x.shape == (batchsize, seqlen, channel_num * 2, 1)
