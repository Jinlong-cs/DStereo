import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


@pytest.mark.parametrize(
    "face3d_structure",
    ["Face3dModel", "FaceMtlFace3dModel"],
)
def test_face3d_task(face3d_structure):
    config = dict(
        type=face3d_structure,
        backbone=dict(
            type="VargNetV2",
            num_classes=1000,
            include_top=False,
            bn_kwargs={},
        ),
        head=dict(
            type="Face3dHead",
            kernel_size=4,
            in_channels=256,
        ),
        deploy=True,
    )
    face3d_model = build_from_registry(config).eval()
    x = {"img": torch.rand(4, 3, 128, 128)}
    if face3d_structure == "Face3dModel":
        global_pose, jaw_pose, camera, shape, expression = face3d_model(x)
        assert global_pose.shape == (4, 3, 1, 1)
        assert jaw_pose.shape == (4, 3, 1, 1)
        assert camera.shape == (4, 3, 1, 1)
        assert shape.shape == (4, 100, 1, 1)
        assert expression.shape == (4, 50, 1, 1)
    else:
        pose = face3d_model(x)
        assert pose["pred_pose"].shape == (4, 3, 1, 1)
    qat_test(face3d_model, x, with_quantized=False)
