import torch

from hat.models.task_modules.ganet.decoder import GaNetDecoder


def test_ganet_decoder():
    kpts_hm = torch.randn((1, 1, 40, 100))
    pts_offset = torch.randn((1, 2, 40, 100))
    int_offset = torch.randn((1, 2, 40, 100))
    decoder = GaNetDecoder(
        root_thr=1,
        kpt_thr=0.4,
        cluster_thr=5,
        downscale=8,
    )
    meta_data = dict(
        scale_factor=torch.rand(1, 4), crop_offset=[[0, 100, 0, 100]]
    )

    decoder(kpts_hm, pts_offset, int_offset, meta_data)
