import torch

from hat.models.task_modules.face3d.geometry import (
    batch_orth_proj,
    vertex_normals,
)

BATCH_SIZE = 4
NUM_VERTICES = 5023
NUM_TRIANGLES = 4000
NUM_LDMK = 68


def test_vertex_normals():
    vertices = torch.rand(BATCH_SIZE, NUM_VERTICES, 3)
    faces = torch.randint(
        0, NUM_VERTICES - 1, size=(BATCH_SIZE, NUM_TRIANGLES, 3)
    )
    normals = vertex_normals(vertices=vertices, faces=faces)
    assert isinstance(normals, torch.Tensor)
    assert normals.shape == vertices.shape


def test_batch_orth_proj():
    x = torch.rand(BATCH_SIZE, NUM_LDMK, 3)
    camera = torch.rand(BATCH_SIZE, 3)
    output = batch_orth_proj(x, camera)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (BATCH_SIZE, NUM_LDMK, 3)
