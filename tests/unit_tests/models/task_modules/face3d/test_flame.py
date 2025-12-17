import os

import pytest
import torch

try:
    import kornia
except ImportError:
    kornia = None

from hat.models.task_modules.face3d.flame import (
    FLAME,
    FLAMEProcess,
    FLAMETex,
    SimpleFLAME,
)

NUM_VERTICES = 5023
NUM_LDMK = 68
BATCH_SIZE = 2
PREFIX = "./tmp_orig_data/face/face3d/hat_test/face3d_params"
# fmt: off
# basic indices for flame J_regressor.
BASIC_INDICES = [
    184, 231, 232, 262, 530, 868, 957, 1037, 1087, 1149, 1306,
    1447, 1504, 1867, 1902, 1983, 2029, 2104, 2260, 2374, 2664, 2778,
    2980, 3248, 3260, 3283, 3284, 3535, 3536, 3772, 4051, 4065, 4169,
    4273, 4346, 4378, 4597, 4611, 4715, 4819, 4892, 4924
]
# fmt: on


def test_flame():
    flame_model_path = os.path.join(PREFIX, "new_generic_model.pkl")
    flame_lmk_embedding_path = os.path.join(
        PREFIX, "new_landmark_embedding.npy"
    )
    flame_model = FLAME(flame_model_path, flame_lmk_embedding_path)
    full_pose = torch.rand(BATCH_SIZE, 15)
    shape_params = torch.rand(BATCH_SIZE, 100)
    expression_params = torch.rand(BATCH_SIZE, 50)
    vertices, ldmk_2d, ldmk_3d = flame_model(
        full_pose=full_pose,
        shape_params=shape_params,
        expression_params=expression_params,
        pose2rot=True,
    )
    assert isinstance(vertices, torch.Tensor)
    assert vertices.shape == (BATCH_SIZE, NUM_VERTICES, 3)
    assert isinstance(ldmk_2d, torch.Tensor)
    assert ldmk_2d.shape == (BATCH_SIZE, NUM_LDMK, 3)
    assert isinstance(ldmk_3d, torch.Tensor)
    assert ldmk_3d.shape == (BATCH_SIZE, NUM_LDMK, 3)
    assert (ldmk_2d[:, 17:] == ldmk_3d[:, 17:]).all()


def test_flame_texture():
    tex_path = os.path.join(PREFIX, "FLAME_albedo_from_BFM.npz")
    flame_tex = FLAMETex(tex_path)
    inputs = torch.rand(BATCH_SIZE, 50)
    texture = flame_tex(inputs)
    assert isinstance(texture, torch.Tensor)
    assert texture.shape == (BATCH_SIZE, 512, 512, 3)


def test_simple_flame():
    indices = BASIC_INDICES + list(range(1000))
    indices = list(set(indices))

    flame_model_path = os.path.join(PREFIX, "new_generic_model.pkl")
    flame_lmk_embedding_path = os.path.join(
        PREFIX, "new_landmark_embedding.npy"
    )
    flame_model = FLAME(flame_model_path, flame_lmk_embedding_path)
    full_pose = torch.rand(BATCH_SIZE, 15)
    shape_params = torch.rand(BATCH_SIZE, 100)
    expression_params = torch.rand(BATCH_SIZE, 50)
    vertices, _, _ = flame_model(
        full_pose=full_pose,
        shape_params=shape_params,
        expression_params=expression_params,
        pose2rot=True,
    )
    selected_verts = vertices[:, indices]

    simple_flame = SimpleFLAME(
        flame_model_path, flame_lmk_embedding_path, indices
    )
    simple_verts = simple_flame(
        full_pose, shape_params, expression_params, True
    )

    assert torch.abs(simple_verts - selected_verts).max() < 1e-6


@pytest.mark.skipif(kornia is None, reason="require kornia")
def test_flame_process():
    flame_model_path = os.path.join(PREFIX, "new_generic_model.pkl")
    flame_lmk_embedding_path = os.path.join(
        PREFIX, "new_landmark_embedding.npy"
    )
    flame_model = FLAME(flame_model_path, flame_lmk_embedding_path)

    tex_path = os.path.join(PREFIX, "FLAME_albedo_from_BFM.npz")
    flame_tex = FLAMETex(tex_path)

    # Ignore renderer
    batch_size = 7
    data = {}
    data["pred"] = {
        "global_pose": torch.randn(batch_size, 3),
        "transl": torch.randn(batch_size, 3),
        "jaw": torch.randn(batch_size, 3),
        "shape": torch.randn(batch_size, 100),
        "exp": torch.randn(batch_size, 50),
    }
    meta = torch.tensor([[1, 0, 200], [0, 1, 200]]).to(torch.float32)
    data["meta"] = meta.reshape((1, 2, 3)).repeat(batch_size, 1, 1)
    data["normed_img"] = torch.randn((batch_size, 3, 128, 128))
    data["net_input_size"] = torch.ones((batch_size, 2)) * 128
    intri = torch.tensor([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]])
    data["intrinsic"] = intri.reshape((1, 3, 3)).repeat(batch_size, 1, 1)
    data["distortion"] = torch.zeros(batch_size, 5)
    data["vir2real_rotmat"] = (
        torch.eye(3).reshape(1, 3, 3).repeat(batch_size, 1, 1)
    )
    flame_process = FLAMEProcess(flame_model, flame_tex, None, "pp")
    output = flame_process(data)
    pred = output["pred"]
    assert pred["eye3d_left"].shape == (batch_size, 3)
    assert pred["eye3d_right"].shape == (batch_size, 3)
    assert pred["real_eye3d_left"].shape == (batch_size, 3)
    assert pred["real_eye3d_right"].shape == (batch_size, 3)
    assert pred["real_verts"].shape == (batch_size, NUM_VERTICES, 3)
    assert pred["cam_verts"].shape == (batch_size, NUM_VERTICES, 3)
    assert pred["img_verts"].shape == (batch_size, NUM_VERTICES, 3)
    assert pred["img_ldmk"].shape == (batch_size, NUM_LDMK, 2)
    assert pred["cam_ldmk"].shape == (batch_size, NUM_LDMK, 3)

    flame_process = FLAMEProcess(flame_model, flame_tex, None, "wpp")
    output = flame_process(data)
    pred = output["pred"]
    assert pred["cam_verts"].shape == (batch_size, NUM_VERTICES, 3)
    assert pred["img_verts"].shape == (batch_size, NUM_VERTICES, 3)
    assert pred["img_ldmk"].shape == (batch_size, NUM_LDMK, 2)
    assert pred["cam_ldmk"].shape == (batch_size, NUM_LDMK, 3)
