import numpy as np
import torch

from hat.models.task_modules.ganet.target import GaNetTarget


def test_ganet_target():
    target = GaNetTarget(
        hm_down_scale=8,
        radius=2,
    )
    data = dict(
        img=torch.rand(1, 3, 320, 800),
        img_shape=[np.array([3, 320, 800]).astype(np.int64)],
        gt_lines=[np.random.randint(0, 320, size=(12, 2)).astype(np.float32)],
    )
    results = target(data)
    assert "gt_kpts_hm" in results
    assert "int_offset" in results
    assert "pts_offset" in results
    assert "int_offset_mask" in results
    assert "pts_offset_mask" in results
