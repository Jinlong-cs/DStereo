import pytest
import torch

from hat.core.face3d import batch_rodrigues, perspective_camera

try:
    import kornia
except ImportError:
    kornia = None


@pytest.mark.skipif(kornia is None, reason="need kornia")
def test_perspective_camera():
    batch_size = 7
    rvec = torch.randn((batch_size, 3))
    rot_mat = batch_rodrigues(rvec)
    transl = torch.randn((batch_size, 3))
    cam_ldmk = torch.randn((batch_size, 68, 3))
    cam_verts = torch.randn((batch_size, 5023, 3))
    intri = torch.eye(3).reshape(1, 3, 3).repeat(batch_size, 1, 1)

    img_ldmk, cam_ldmk, img_verts = perspective_camera(
        rot_mat, transl, cam_ldmk, cam_verts, intri
    )
    assert img_ldmk.shape == (batch_size, 68, 2)
    assert cam_ldmk.shape == (batch_size, 68, 3)
    assert img_verts.shape == (batch_size, 5023, 3)
