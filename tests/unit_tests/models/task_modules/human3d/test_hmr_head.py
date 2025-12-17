import torch

from hat.models.task_modules.human3d.head import HMRHead


def test_hmr_head():
    inputs = torch.rand(1, 2048, 7, 7)
    model = HMRHead(
        smpl_mean_params="./tmp_orig_data/human3d/spin_params/data/smpl_mean_params.npz"  # noqa
    )

    outputs = model(inputs)
    assert isinstance(outputs, tuple)
    assert len(outputs) == 3
