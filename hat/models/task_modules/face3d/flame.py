import pickle
from typing import Dict, List, Optional

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from hat.core.face3d import (
    batch_rodrigues,
    lbs,
    perspective_camera,
    rot_mat_to_euler,
    vertices2landmarks,
)
from hat.registry import OBJECT_REGISTRY
from .geometry import batch_orth_proj, vertex_normals

__all__ = ["FLAME", "FLAMETex", "FLAMEProcess"]


def to_tensor(array, dtype=torch.float32):
    if not isinstance(array, torch.Tensor):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if isinstance(array, scipy.sparse.csc.csc_matrix):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


@OBJECT_REGISTRY.register
class FLAME(nn.Module):
    """FLAME: Articulated Expressive 3D Head Model.

    Refer to `https://flame.is.tue.mpg.de/`

    Given FLAME parameters this class generates a differentiable FLAME
    function which outputs the a mesh and 2D/3D facial landmarks.
    """

    SHAPE_SPACE_DIM = 300
    EXPRESSION_SPACE_DIM = 100

    def __init__(
        self,
        flame_model_path: str,
        flame_lmk_embedding_path: str,
        num_shape_coeffs=100,
        num_expr_coeffs=50,
    ):
        super(FLAME, self).__init__()
        with open(flame_model_path, "rb") as f:
            # flame_model = Struct(**pickle.load(f, encoding='latin1'))
            ss = pickle.load(f, encoding="latin1")
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        self.register_buffer(
            "faces_tensor",
            to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long),
        )
        # The vertices of the template model
        self.register_buffer(
            "v_template",
            to_tensor(to_np(flame_model.v_template), dtype=self.dtype),
        )
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        if (
            shapedirs.shape[-1]
            == self.SHAPE_SPACE_DIM + self.EXPRESSION_SPACE_DIM
        ):
            shape_basis = shapedirs[..., :num_shape_coeffs]
            expr_basis = shapedirs[
                ...,
                self.SHAPE_SPACE_DIM : self.SHAPE_SPACE_DIM + num_expr_coeffs,
            ]
            shapedirs = torch.cat(
                (shape_basis, expr_basis),
                dim=-1,
            )
        assert shapedirs.shape[-1] == num_shape_coeffs + num_expr_coeffs
        self.register_buffer("shapedirs", shapedirs)
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer(
            "posedirs", to_tensor(to_np(posedirs), dtype=self.dtype)
        )
        self.register_buffer(
            "J_regressor",
            to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype),
        )
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer(
            "lbs_weights",
            to_tensor(to_np(flame_model.weights), dtype=self.dtype),
        )

        # Static and Dynamic Landmark embeddings for FLAME
        lmk_embeddings = np.load(
            flame_lmk_embedding_path, allow_pickle=True, encoding="latin1"
        )
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer(
            "lmk_faces_idx",
            torch.tensor(
                lmk_embeddings["static_lmk_faces_idx"], dtype=torch.long
            ),
        )
        self.register_buffer(
            "lmk_bary_coords",
            torch.tensor(
                lmk_embeddings["static_lmk_bary_coords"], dtype=self.dtype
            ),
        )
        self.register_buffer(
            "dynamic_lmk_faces_idx",
            torch.tensor(
                lmk_embeddings["dynamic_lmk_faces_idx"], dtype=torch.long
            ),
        )
        self.register_buffer(
            "dynamic_lmk_bary_coords",
            torch.tensor(
                lmk_embeddings["dynamic_lmk_bary_coords"], dtype=self.dtype
            ),
        )
        self.register_buffer(
            "full_lmk_faces_idx",
            torch.tensor(
                lmk_embeddings["full_lmk_faces_idx"], dtype=torch.long
            ),
        )
        self.register_buffer(
            "full_lmk_bary_coords",
            torch.tensor(
                lmk_embeddings["full_lmk_bary_coords"], dtype=self.dtype
            ),
        )

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))

    def _find_dynamic_lmk_idx_and_bcoords(
        self,
        pose,
        dynamic_lmk_faces_idx,
        dynamic_lmk_b_coords,
        neck_kin_chain,
        dtype=torch.float32,
    ):
        """
        Select the face contour depending on the reletive position of the head.

        Args:
            vertices: N X num_of_vertices X 3
            pose: N X full pose
            dynamic_lmk_faces_idx: The list of contour face indexes
            dynamic_lmk_b_coords: The list of contour barycentric weights
            neck_kin_chain: The tree to consider for the relative rotation
            dtype: Data type
        return:
            The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(
            pose.view(batch_size, -1, 3), 1, neck_kin_chain
        )
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3), dtype=dtype).view(
            batch_size, -1, 3, 3
        )

        rel_rot_mat = (
            torch.eye(3, device=pose.device, dtype=dtype)
            .unsqueeze_(dim=0)
            .expand(batch_size, -1, -1)
        )
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat)[:, 1], max=39)
        ).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle

        dyn_lmk_faces_idx = torch.index_select(
            dynamic_lmk_faces_idx, 0, y_rot_angle
        )
        dyn_lmk_b_coords = torch.index_select(
            dynamic_lmk_b_coords, 0, y_rot_angle
        )
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def _vertices2landmarks(
        self, vertices, faces, lmk_faces_idx, lmk_bary_coords
    ):
        """Calculate landmarks by barycentric interpolation.

        Args:
            vertices: torch.tensor NxVx3, dtype = torch.float32
                The tensor of input vertices
            faces: torch.tensor (N*F)x3, dtype = torch.long
                The faces of the mesh
            lmk_faces_idx: torch.tensor N X L, dtype = torch.long
                The tensor with the indices of the faces used to calculate the
                landmarks.
            lmk_bary_coords: torch.tensor N X L X 3, dtype = torch.float32
                The tensor of barycentric coordinates that are used to
                interpolate the landmarks

        Returns:
            landmarks: torch.tensor NxLx3, dtype = torch.float32
                The coordinates of the landmarks for each mesh in the batch
        """
        # Extract the indices of the vertices for each face
        # NxLx3
        batch_size, num_verts = vertices.shape[:2]
        lmk_faces = (
            torch.index_select(faces, 0, lmk_faces_idx.view(-1))
            .view(1, -1, 3)
            .view(batch_size, lmk_faces_idx.shape[1], -1)
        )
        lmk_faces += (
            torch.arange(batch_size, dtype=torch.long)
            .view(-1, 1, 1)
            .to(device=vertices.device)
            * num_verts
        )
        lmk_vertices = vertices.view(-1, 3)[lmk_faces]
        landmarks = torch.einsum(
            "blfi,blf->bli", [lmk_vertices, lmk_bary_coords]
        )
        return landmarks

    def seletec_3d68(self, vertices):
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1),
        )
        return landmarks3d

    def forward(
        self, full_pose, shape_params, expression_params, pose2rot=False
    ):
        """Forward.

        Args:
            shape_params: N X number of shape parameters
            expression_params: N X number of expression parameters
            pose_params: N X number of pose parameters (6)

        return:d
            vertices: N X V X 3
            landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]
        betas = torch.cat([shape_params, expression_params], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        vertices, _ = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            dtype=self.dtype,
            pose2rot=pose2rot,
        )

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(
            batch_size, -1
        )
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(
            batch_size, -1, -1
        )

        (
            dyn_lmk_faces_idx,
            dyn_lmk_bary_coords,
        ) = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose,
            self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain,
            dtype=self.dtype,
        )
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = vertices2landmarks(
            vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords
        )
        bz = vertices.shape[0]
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(bz, 1),
            self.full_lmk_bary_coords.repeat(bz, 1, 1),
        )

        return vertices, landmarks2d, landmarks3d


@OBJECT_REGISTRY.register
class FLAMETex(nn.Module):
    def __init__(self, tex_path, num_components=50):
        super(FLAMETex, self).__init__()
        data = np.load(tex_path)
        self.register_buffer("texture_mean", to_tensor(to_np(data["MU"])))
        self.register_buffer(
            "texture_basis",
            to_tensor(to_np(data["PC"][..., :num_components]).T),
        )

    def forward(self, texcode):
        """Forward.

        Args:
            texcode: [batchsize, n_tex]
            texture: [bz, 512, 512, 3], rgb order, range: 0-1
        """
        batchsize = texcode.size(0)
        texture_mean = self.texture_mean.unsqueeze(0).repeat(batchsize, 1)
        texture_basis = torch.mm(texcode, self.texture_basis)
        texture = texture_mean + texture_basis
        texture = texture.reshape(texcode.shape[0], 512, 512, 3)
        texture = texture[:, :, :, [2, 1, 0]]
        return texture


class SimpleFLAME(FLAME):
    """A simple FLAME model with less vertices.

    This model is just used as an example of simplified post-processing.

    Refer to https://horizonrobotics.feishu.cn/wiki/wikcn752a47F8m3RvEt7t86mdwh

    Args:
        flame_model_path: refer to FLAME
        flame_lmk_embedding_path: refer to FLAME
        indices: user-defined vertices indices
    """

    def __init__(
        self,
        flame_model_path: str,
        flame_lmk_embedding_path: str,
        indices: List[int],
    ):
        super().__init__(flame_model_path, flame_lmk_embedding_path)
        self.v_template = self.v_template[indices]
        self.shapedirs = self.shapedirs[indices]
        self.posedirs = self.posedirs.reshape(36, 5023, 3)[:, indices].reshape(
            36, -1
        )
        self.J_regressor = self.J_regressor[:, indices]
        self.lbs_weights = self.lbs_weights[indices]

    def _check_indices(self, indices):
        num_verts = self.J_regressor.shape[1]
        basic_indices = torch.arange(num_verts)
        basic_indices = basic_indices[self.J_regressor.sum(axis=0) > 0]
        for idx in basic_indices:
            assert idx in indices, (
                "Basic indices are required. Please refer to "
                "https://horizonrobotics.feishu.cn/wiki/wikcn752a47F8m3RvEt7t86mdwh"  # noqa
            )

    def forward(
        self, full_pose, shape_params, expression_params, pose2rot=False
    ):
        batch_size = shape_params.shape[0]
        betas = torch.cat([shape_params, expression_params], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        vertices, _ = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            dtype=self.dtype,
            pose2rot=pose2rot,
        )
        return vertices


@OBJECT_REGISTRY.register
class FLAMEProcess(nn.Module):
    """FLAME forward process.

    Args:
        flame: FLAME module.
        flame_tex: FLAME texture module.
            Defaults to None.
        renderer: Differentiable renderer.
            Defaults to None.
        model_type: perspective(pp) or weak perspective(wpp).
            Defaults to "pp".
        max_depth_meter: maximum of face depth.
            Defaults to 1.0.
        use_dist: use distortion or not. Defaults to False.
        render_depth: render depth map or not.
            Defaults to False.
        render_img: render image or not. Defaults to False.
    """

    def __init__(
        self,
        flame: FLAME,
        flame_tex: Optional[FLAMETex] = None,
        renderer: Optional[nn.Module] = None,
        model_type: str = "pp",
        max_depth_meter: float = 1.0,
        use_dist: bool = False,
        render_depth: bool = False,
        render_img: bool = False,
    ):
        super().__init__()
        self.flame = flame
        self.flame_tex = flame_tex
        self.model_type = model_type
        self.max_depth_meter = max_depth_meter
        self.renderer = renderer
        self.left_eyeldmk_idx = [37, 38, 40, 41]
        self.right_eyeldmk_idx = [43, 44, 46, 47]
        self.use_dist = use_dist
        self.render_depth = render_depth
        self.render_img = render_img
        if self.flame is not None:
            for param in self.flame.parameters():
                param.requires_grad = False
        if self.flame_tex is not None:
            for param in self.flame_tex.parameters():
                param.requires_grad = False

    def forward(self, data):
        data = self.flame_forward(data)
        if self.render_depth or self.render_img:
            data = self.render(data)
        return data

    def render(self, data: Dict):
        vertices = data["pred"]["cam_verts"]
        img_verts = data["pred"]["img_verts"]
        global_pose = data["pred"]["global_pose"]
        light = data["pred"]["light"]
        gt_img = data["label"]["gt_img"]
        gt_mask = data["label"]["gt_mask"]
        batch_size = global_pose.size(0)
        # TODO: why detach
        if self.render_img:
            normal = vertex_normals(
                vertices,
                self.flame.faces_tensor.unsqueeze(0).repeat(batch_size, 1, 1),
            )
            global_rot = batch_rodrigues(global_pose)
            normal = torch.bmm(normal, global_rot.permute(0, 2, 1)).detach()
            albedo = self.flame_tex(data["pred"]["tex"])
        else:
            albedo, normal = None, None
        # renderer forward
        height, width = gt_img.shape[-2:]
        render_results = self.renderer(
            img_verts,
            normal,
            albedo,
            light,
            height,
            width,
        )
        predicted_img = render_results.get("render_img", None)
        predicted_depth = render_results.get("render_depth", None)
        predicted_mask = render_results.get("render_mask", None)

        predicted_mask = predicted_mask.permute(0, 3, 1, 2).contiguous()
        mask = predicted_mask * gt_mask
        data["pred"]["pr_mask"] = mask

        if predicted_img is not None:
            # bhwc to bchw
            predicted_img = (
                predicted_img.permute(0, 3, 1, 2).clamp(0, 1).contiguous()
            )
            predicted_img = predicted_img * mask + gt_img * (1 - mask)
            data["pred"]["pr_img"] = predicted_img
        if predicted_mask is not None:
            # TODO: process
            data["pred"]["pr_depth"] = predicted_depth

        return data

    def m2mm(self, meter):
        return meter * 1000

    def flame_forward(self, data):
        # flame forward
        # full_pose = head_pose + neck_pose + jaw_pose + left/right eyes
        global_pose = data["pred"]["global_pose"]
        camera = data["pred"]["transl"]
        jaw_pose = data["pred"]["jaw"]
        shape = data["pred"]["shape"]
        exp = data["pred"]["exp"]
        zeros = torch.zeros_like(global_pose)
        if self.model_type == "pp":
            if self.max_depth_meter > 0:
                camera[..., -1] = (
                    F.sigmoid(camera[..., -1]) * self.max_depth_meter
                )
            full_pose = torch.cat(
                (zeros, zeros, jaw_pose, zeros, zeros), dim=1
            )
        else:
            full_pose = torch.cat(
                (global_pose, zeros, jaw_pose, zeros, zeros), dim=1
            )
        vertices, img_ldmk, cam_ldmk = self.flame(
            full_pose=full_pose,
            shape_params=shape,
            expression_params=exp,
            pose2rot=True,
        )

        if self.model_type == "pp":
            # perspective projection
            global_rot = batch_rodrigues(global_pose)
            meta = data["meta"]
            intri = data.get("virtual_intrinsic", data["intrinsic"])
            net_input_size = data["net_input_size"]
            dist = data["distrotion"] if self.use_dist else None
            (img_ldmk, cam_ldmk, img_verts,) = perspective_camera(
                global_rot, camera, cam_ldmk, vertices, intri, dist
            )
            # crop for render
            img_verts[..., :2] = torch.bmm(
                img_verts[..., :2], meta[..., :2]
            ) + meta[..., 2:].reshape(-1, 1, 2)
            img_verts[..., 0] /= net_input_size[:, 1:2]  # height
            img_verts[..., 1] /= net_input_size[:, 0:1]  # width

            # eye 3d position
            left_eye_cam = cam_ldmk[:, self.left_eyeldmk_idx, :3].mean(axis=1)
            right_eye_cam = cam_ldmk[:, self.right_eyeldmk_idx, :3].mean(
                axis=1
            )

            # meter to millimeter
            left_eye_cam = self.m2mm(left_eye_cam)
            right_eye_cam = self.m2mm(right_eye_cam)
            vertices = self.m2mm(vertices)

            # TODO: Unifying keys
            # convert preds rot mat from virtual camera to real camera.
            vir2real_rotmat = data.get(
                "vir2real_rotmat", torch.eye(3, device=left_eye_cam.device)
            )
            data["pred"]["eye3d_left"] = left_eye_cam
            data["pred"]["eye3d_right"] = right_eye_cam
            data["pred"]["real_eye3d_left"] = (
                vir2real_rotmat @ left_eye_cam[..., None]
            ).reshape(-1, 3)
            data["pred"]["real_eye3d_right"] = (
                vir2real_rotmat @ right_eye_cam[..., None]
            ).reshape(-1, 3)
            data["pred"]["real_verts"] = (
                vir2real_rotmat @ vertices.permute(0, 2, 1)
            ).permute(0, 2, 1)
        elif self.model_type == "wpp":
            # weak perspective projection
            img_ldmk = batch_orth_proj(cam_ldmk, camera)
            img_verts = batch_orth_proj(vertices, camera)
            img_ldmk[..., 1:] *= -1
            img_verts[..., 1:] *= -1
        data["pred"]["cam_verts"] = vertices
        data["pred"]["img_verts"] = img_verts
        data["pred"]["img_ldmk"] = img_ldmk[..., :2]
        data["pred"]["cam_ldmk"] = cam_ldmk[..., :3]
        return data
