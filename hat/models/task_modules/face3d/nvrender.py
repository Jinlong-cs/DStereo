import numpy as np
import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

try:
    import nvdiffrast.torch as dr
except ImportError:
    dr = None
try:
    from pytorch3d.io import load_obj
except ImportError:
    load_obj = None

__all__ = ["NVRenderer"]


class SH(object):
    def __init__(self):
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.0), 2 * np.pi / np.sqrt(8.0)]
        self.c = [
            1 / np.sqrt(4 * np.pi),
            np.sqrt(3.0) / np.sqrt(4 * np.pi),
            3 * np.sqrt(5.0) / np.sqrt(12 * np.pi),
        ]


@OBJECT_REGISTRY.register
class NVRenderer(nn.Module):
    """Differential renderer with nvdiffrast.

    Feed vertices in image space (with z in camera space), normals, albedos
    and lights and render depth image or color image.

    Args:
        obj_filename: .obj file with certain topology structure. i.e. FLAME.
    """

    @require_packages("nvdiffrast", "pytorch3d")
    def __init__(self, obj_filename):
        super(NVRenderer, self).__init__()

        self.glctx = None
        pi = np.pi
        constant_factor = torch.tensor(
            [
                1 / np.sqrt(4 * pi),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi))),
            ],
            dtype=torch.float32,
        )
        self.SH = SH()
        init_lights = torch.tensor(
            [0.8, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32
        )
        verts, faces, aux = load_obj(obj_filename)
        self.register_buffer("constant_factor", constant_factor)
        self.register_buffer("init_lights", init_lights)
        self.register_buffer("uv_coords", aux.verts_uvs[None, ...])
        self.register_buffer("uvFaces", faces.textures_idx.int())
        self.register_buffer("vertsFaces", faces.verts_idx.int())

    def sh_lights(self, lights, normal_images):
        batch_size = lights.size(0)
        a, c = self.SH.a, self.SH.c
        lights = lights.reshape([batch_size, 3, 9])
        init_lights = (
            self.init_lights.reshape(1, 1, 9)
            .repeat(batch_size, 3, 1)
            .to(lights.device)
        )
        lights = lights + init_lights
        lights = lights.permute(0, 2, 1)
        Y = torch.cat(
            [
                a[0]
                * c[0]
                * torch.ones_like(normal_images[..., :1]).to(lights.device),
                -a[1] * c[1] * normal_images[..., 1:2],
                a[1] * c[1] * normal_images[..., 2:],
                -a[1] * c[1] * normal_images[..., :1],
                a[2] * c[2] * normal_images[..., :1] * normal_images[..., 1:2],
                -a[2]
                * c[2]
                * normal_images[..., 1:2]
                * normal_images[..., 2:],
                0.5
                * a[2]
                * c[2]
                / np.sqrt(3.0)
                * (3 * normal_images[..., 2:] ** 2 - 1),
                -a[2] * c[2] * normal_images[..., :1] * normal_images[..., 2:],
                0.5
                * a[2]
                * c[2]
                * (normal_images[..., :1] ** 2 - normal_images[..., 1:2] ** 2),
            ],
            dim=-1,
        )

        r = Y * lights[..., :1].reshape(batch_size, 1, 1, 9)
        g = Y * lights[..., 1:2].reshape(batch_size, 1, 1, 9)
        b = Y * lights[..., 2:].reshape(batch_size, 1, 1, 9)
        r = r.sum(-1)
        g = g.sum(-1)
        b = b.sum(-1)
        shading_images = torch.stack([r, g, b], dim=-1)
        return shading_images

    def image2clip(self, trans_vertices):
        # normalize [-1, 1]
        b, v, _ = trans_vertices.size()
        z_min = torch.min(trans_vertices[..., 2], 1)[0].view(b, 1, 1)
        z_max = torch.max(trans_vertices[..., 2], 1)[0].view(b, 1, 1)
        vertices_normed = trans_vertices.clone()
        vertices_normed[..., 2:] = (vertices_normed[..., 2:] - z_min) / (
            z_max - z_min + 1e-4
        )
        vertices_normed = vertices_normed * 2 - 1
        # homogeneous coordinates
        homogen_coord = torch.ones(
            [b, v, 1], dtype=trans_vertices.dtype, device=trans_vertices.device
        )
        vertices_homo = torch.cat((vertices_normed, homogen_coord), dim=2)

        return vertices_homo

    def forward(
        self, trans_vertices, trans_normals, albedos, lights, height, width
    ):
        device = trans_vertices.device
        render_outputs = {}
        if self.glctx is None:
            self.glctx = dr.RasterizeCudaContext(device=device)

        vertices_homo = self.image2clip(trans_vertices)
        # rasterize [b, height, width, 4], (u, v, z/w, triangle_id)
        rast_out, _ = dr.rasterize(
            self.glctx,
            vertices_homo,
            self.vertsFaces.to(device),
            resolution=(height, width),
        )
        # background triangle_id=0
        render_mask = (rast_out[..., 3] > 0).float().unsqueeze(-1)
        depth_images, _ = dr.interpolate(
            trans_vertices[..., 2:3].contiguous(),
            rast_out,
            self.vertsFaces.to(device),
        )
        if albedos is not None:
            if len(albedos.shape) == 4:
                uvcoords_images, _ = dr.interpolate(
                    self.uv_coords, rast_out, self.uvFaces.to(device)
                )
                uvcoords_images[..., 1] = 1 - uvcoords_images[..., 1]

                albedo_images = dr.texture(
                    albedos, uvcoords_images, filter_mode="linear"
                )
            elif len(albedos.shape) == 3:
                albedo_images, _ = dr.interpolate(
                    albedos.contiguous(), rast_out, self.vertsFaces.to(device)
                )
            # shading
            normal_images, _ = dr.interpolate(
                trans_normals, rast_out, self.vertsFaces.to(device)
            )
            shading_images = self.sh_lights(lights, normal_images)
            predicted_images = albedo_images * shading_images
            render_outputs["render_img"] = predicted_images
            render_outputs["albedo_img"] = albedos

        render_outputs["render_depth"] = depth_images[..., 0]
        render_outputs["render_mask"] = render_mask

        return render_outputs
