import torch

from hat.models.task_modules.pupil_segmentation.encoder import (
    PupilSegDownBlock,
    TransitionDown,
)
from hat.registry import build_from_registry


def test_trans_down():
    in_c, out_c, down_size = 64, 48, 2
    trans_down = TransitionDown(in_c, out_c, down_size)
    feat = torch.randn(2, 64, 32, 32)
    output = trans_down(feat)
    assert output.data.numpy().shape == (2, 48, 16, 16)


def test_pupil_seg_down_block():
    in_c, inter_c, op_c, down_size = 32, 32, 48, 2
    pupil_seg_down_block = PupilSegDownBlock(in_c, inter_c, op_c, down_size)
    feat = torch.randn(2, 32, 64, 64)
    out1, out2 = pupil_seg_down_block(feat)
    assert out1.data.numpy().shape == (2, 64, 64, 64)
    assert out2.data.numpy().shape == (2, 48, 32, 32)


def test_pupil_encoder():
    img_shape = (2, 3, 64, 64)
    config = dict(
        type="PupilSegEncoder",
        in_c=3,
        chz=32,
        growth=1.5,
    )
    pupil_seg_encoder = build_from_registry(config)
    img = torch.randn(img_shape)
    feat_4, feat_3, feat_2, feat_1, feat = pupil_seg_encoder(img)
    assert feat_4.data.numpy().shape == (2, 272, 8, 8)
    assert feat_3.data.numpy().shape == (2, 192, 16, 16)
    assert feat_2.data.numpy().shape == (2, 112, 32, 32)
    assert feat_1.data.numpy().shape == (2, 64, 64, 64)
    assert feat.data.numpy().shape == (2, 192, 4, 4)
