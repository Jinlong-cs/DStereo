import torch

from hat.models.task_modules.pupil_segmentation.decoder import PupilSegUpBlock
from hat.registry import build_from_registry


def test_pupil_seg_up_block():
    skip_c, in_c, out_c, up_stride = 272, 192, 144, 2
    pupil_seg_up_block = PupilSegUpBlock(skip_c, in_c, out_c, up_stride)
    feat = torch.randn(2, 192, 4, 4)
    feat_4 = torch.randn(2, 272, 8, 8)
    output = pupil_seg_up_block(feat_4, feat)
    assert output.data.numpy().shape == (2, 144, 8, 8)


def test_pupil_encoder():
    feat_4 = torch.randn(2, 272, 8, 8)
    feat_3 = torch.randn(2, 192, 16, 16)
    feat_2 = torch.randn(2, 112, 32, 32)
    feat_1 = torch.randn(2, 64, 64, 64)
    feat = torch.randn(2, 192, 4, 4)
    config = dict(
        type="PupilSegDecoder",
        chz=32,
        out_c=1,
        growth=1.5,
    )
    pupil_seg_decoder = build_from_registry(config)
    output = pupil_seg_decoder(feat_4, feat_3, feat_2, feat_1, feat)
    assert output.data.numpy().shape == (2, 1, 64, 64)
