import torch

from hat.models.task_modules.detr3d.post_process import Detr3dPostProcess


class test_centerpoint_decoder:
    bev_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
    ref_p = torch.randn(2, 4, 256, 3)
    cls_pred = torch.randn((2, 10, 4, 256))
    reg_pred = torch.randn((2, 10, 4, 256))
    decoder = Detr3dPostProcess(
        max_num=300, score_threshold=-1, bev_range=bev_range
    )
    decoder(cls_pred, reg_pred, ref_p)
