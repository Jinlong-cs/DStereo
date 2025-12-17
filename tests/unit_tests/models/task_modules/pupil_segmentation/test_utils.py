import numpy as np
import torch

from hat.models.task_modules.pupil_segmentation.utils import (
    ConvBlock,
    get_sizes,
)


def test_get_size():
    chz, growth = 32, 1.5
    gt_enc_inter = np.array([32, 64, 96, 128])
    gt_enc_ip = np.array([32, 48, 96, 144])
    gt_enc_op = np.array([48, 96, 144, 192])
    gt_dec_skip = np.array([272, 192, 112, 64])
    gt_dec_ip = np.array([192, 144, 96, 48])
    gt_dec_op = np.array([144, 96, 48, 32])
    size = get_sizes(chz, growth)

    assert (size["enc"]["inter"] - gt_enc_inter).sum() == 0
    assert (size["enc"]["ip"] - gt_enc_ip).sum() == 0
    assert (size["enc"]["op"] - gt_enc_op).sum() == 0
    assert (size["dec"]["skip"] - gt_dec_skip).sum() == 0
    assert (size["dec"]["ip"] - gt_dec_ip).sum() == 0
    assert (size["dec"]["op"] - gt_dec_op).sum() == 0


def test_conv_block():
    in_c, inter_c, out_c = 3, 32, 32
    img = torch.randn(2, 3, 64, 64)
    conv_block = ConvBlock(in_c, inter_c, out_c)
    output = conv_block(img)
    assert output.data.numpy().shape == (2, 32, 64, 64)
