import torch

from hat.core.hand3d.hand3d_func import posenc, proj_func

BATCH_SIZE = 16


def test_proj_func():
    xyz = torch.rand(BATCH_SIZE, 21, 3)
    K = torch.rand(BATCH_SIZE, 3, 3)

    out_uv = proj_func(xyz, K)
    assert isinstance(out_uv, torch.Tensor)
    assert out_uv.shape == (BATCH_SIZE, 21, 2)


def test_posenc():
    input_tensor = torch.rand(BATCH_SIZE, 21, 3)
    num_encodings = 4

    out = posenc(input_tensor, num_encodings)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (BATCH_SIZE, 21, 27)
