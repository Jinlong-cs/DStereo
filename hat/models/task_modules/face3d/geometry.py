import torch
import torch.nn.functional as F


def batch_orth_proj(x: torch.Tensor, camera: torch.Tensor):
    """Weak Perspective Projection.

    Project 3D landmark coodinates to 2D plane.

    Args:
        x: input tensor, (batch_size, num_ldmk, 3)
        camera: camera matrix, (batch_size, 3), the 2nd dimention is
            [scale, offset_x, offset_y].

    Returns:
        xn: projected tensor, (batch_size, num_ldmk, 3)
    """
    camera = camera.clone().view(-1, 1, 3)
    xn = camera[:, :, 0:1] * x
    xn[:, :, :2] = xn[:, :, :2] + camera[:, :, 1:]
    return xn


def vertex_normals(vertices: torch.tensor, faces: torch.Tensor):
    """Get normals form vertices.

    Args:
        vertices: face vertices, [batch_size, num_vertices, 3].
        faces: triangles, [batch_size, num_triangles, 3].

    Returns:
        normals: normals of every vertices. [batch_size, num_vertices, 3].
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    # expanded faces
    faces = (
        faces
        + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    )
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(
        0,
        faces[:, 1].long(),
        torch.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1],
            vertices_faces[:, 0] - vertices_faces[:, 1],
        ),
    )
    normals.index_add_(
        0,
        faces[:, 2].long(),
        torch.cross(
            vertices_faces[:, 0] - vertices_faces[:, 2],
            vertices_faces[:, 1] - vertices_faces[:, 2],
        ),
    )
    normals.index_add_(
        0,
        faces[:, 0].long(),
        torch.cross(
            vertices_faces[:, 1] - vertices_faces[:, 0],
            vertices_faces[:, 2] - vertices_faces[:, 0],
        ),
    )

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals
