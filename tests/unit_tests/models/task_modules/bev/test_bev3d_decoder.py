import numpy as np
import torch

from hat.models.task_modules.bev.postprocess import ANCBEV3Decoder


def gen_fake_data_pred():
    return dict(
        bev3d_hm=torch.zeros(1, 3, 256, 256, dtype=torch.float32),
        bev3d_dim=torch.zeros(1, 3, 256, 256, dtype=torch.float32),
        bev3d_rot=torch.zeros(1, 2, 256, 256, dtype=torch.float32),
        bev3d_ct_offset=torch.zeros(1, 2, 256, 256, dtype=torch.float32),
        bev3d_loc_z=torch.zeros(1, 1, 256, 256, dtype=torch.float32),
    )


def test_bev3d_decoder():
    output = gen_fake_data_pred()

    cls_dimension = np.array(
        [
            [1.6383535, 0.60629284, 0.5058226],
            [1.7761209, 1.8237954, 4.79461],
            [1.5518998, 0.73560804, 1.7083853],
        ]
    )

    bev3d_decoder = ANCBEV3Decoder(
        cls_dimension=cls_dimension,
        topk=100,
        max_pool_kernel=9,
    )

    _ = bev3d_decoder(output)
