import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "rot_mat_to_euler",
    "euler_to_rot_mat",
    "vertices2landmarks",
    "lbs",
    "batch_rodrigues",
    "transform_mat",
    "rot6d_to_rotmat",
]


def rot_mat_to_euler(rot_mats: torch.Tensor, seq: str = "xyz") -> torch.Tensor:
    """Transfer rotation matrix to euler angles.

    Be careful with extreme cases of eular angles like [0.0, pi, 0.0].

    Args:
        rot_mats: rotation matrix, size (B, 3, 3).
        seq: transform sequential. Only "xyz" and "zxy" are supported.
            Defualts to "xyz".

    Returns:
        euler_angles: size (B, 3).
    """
    if seq == "xyz":
        sy = torch.sqrt(
            rot_mats[:, 0, 0] * rot_mats[:, 0, 0]
            + rot_mats[:, 1, 0] * rot_mats[:, 1, 0]
        )
        x = torch.atan2(rot_mats[:, 2, 1], rot_mats[:, 2, 2])
        y = torch.atan2(-rot_mats[:, 2, 0], sy)
        z = torch.atan2(rot_mats[:, 1, 0], rot_mats[:, 0, 0])
        euler_angles = torch.stack([x, y, z], dim=1)
        euler_angles = torch.rad2deg(euler_angles)
    elif seq == "zxy":
        sx = torch.sqrt(
            rot_mats[:, 1, 0] * rot_mats[:, 1, 0]
            + rot_mats[:, 1, 1] * rot_mats[:, 1, 1]
        )
        x = torch.atan2(-rot_mats[:, 1, 2], sx)
        y = torch.atan2(rot_mats[:, 0, 2], rot_mats[:, 2, 2])
        z = torch.atan2(rot_mats[:, 1, 0], rot_mats[:, 1, 1])
        euler_angles = torch.stack([z, x, y], dim=1)
        euler_angles = torch.rad2deg(euler_angles)
    else:
        raise TypeError("invalid type seq")

    return euler_angles


def euler_to_rot_mat(angles: torch.Tensor) -> torch.Tensor:
    """Transfer euler angles to rotation matrix.

    Args:
        angles: euler angles, size (B, 3)

    Returns:
        rot: rotation matrix, size (B, 3, 3)
    """

    batch_size = angles.shape[0]
    ones = torch.ones([batch_size, 1]).to(angles.device)
    zeros = torch.zeros([batch_size, 1]).to(angles.device)
    radians = torch.deg2rad(angles)
    x, y, z = torch.chunk(radians, 3, dim=1)

    rot_x = torch.cat(
        [
            ones,
            zeros,
            zeros,
            zeros,
            torch.cos(x),
            -torch.sin(x),
            zeros,
            torch.sin(x),
            torch.cos(x),
        ],
        dim=1,
    ).reshape([batch_size, 3, 3])

    rot_y = torch.cat(
        [
            torch.cos(y),
            zeros,
            torch.sin(y),
            zeros,
            ones,
            zeros,
            -torch.sin(y),
            zeros,
            torch.cos(y),
        ],
        dim=1,
    ).reshape([batch_size, 3, 3])

    rot_z = torch.cat(
        [
            torch.cos(z),
            -torch.sin(z),
            zeros,
            torch.sin(z),
            torch.cos(z),
            zeros,
            zeros,
            zeros,
            ones,
        ],
        dim=1,
    ).reshape([batch_size, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    rot = rot.permute(0, 2, 1)
    return rot


def find_dynamic_lmk_idx_and_bcoords(
    vertices: torch.Tensor,
    pose: torch.Tensor,
    dynamic_lmk_faces_idx: torch.Tensor,
    dynamic_lmk_b_coords: torch.Tensor,
    neck_kin_chain: list,
    dtype: torch.dtype = torch.float32,
):
    """Compute the faces, barycentric coordinates for the dynamic landmarks.

    To do so, we first compute the rotation of the neck around the y-axis
    and then use a pre-computed look-up table to find the faces and the
    barycentric coordinates that will be used.

    Special thanks to Soubhik Sanyal (soubhik.sanyal@tuebingen.mpg.de)
    for providing the original TensorFlow implementation and for the LUT.

    Args:
        vertices: input vertices, size (B, V, 3).
        pose: the current pose of the body model, size (B, Jx3).
        dynamic_lmk_faces_idx: the look-up table from neck rotation to
            faces, size (L, ).
        dynamic_lmk_b_coords: the loock-up table from neck rotation to
            barycentric coordinates, size (L , 3).
        neck_kin_chain: a python list that contains the indices of the
            joints that form the kinematic chain of the neck.
        dtype: data type. Defaults to torch.float32.

    Returns:
        dyn_lmk_faces_idx: a tensor of size BxL that contains the indices of
            the faces that will be used to compute the current dynamic
            landmarks.
        dyn_lmk_b_coords: barycentric coordinates.
    """

    batch_size = vertices.shape[0]

    aa_pose = torch.index_select(
        pose.view(batch_size, -1, 3), 1, neck_kin_chain
    )
    rot_mats = batch_rodrigues(aa_pose.view(-1, 3), dtype=dtype).view(
        batch_size, -1, 3, 3
    )

    rel_rot_mat = torch.eye(3, device=vertices.device, dtype=dtype).unsqueeze_(
        dim=0
    )
    for idx in range(len(neck_kin_chain)):
        rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

    y_rot_angle = torch.round(
        torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)
    ).to(dtype=torch.long)
    neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
    mask = y_rot_angle.lt(-39).to(dtype=torch.long)
    neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
    y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle

    dyn_lmk_faces_idx = torch.index_select(
        dynamic_lmk_faces_idx, 0, y_rot_angle
    )
    dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)

    return dyn_lmk_faces_idx, dyn_lmk_b_coords


def vertices2landmarks(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    lmk_faces_idx: torch.Tensor,
    lmk_bary_coords: torch.Tensor,
):
    """Calculate landmarks by barycentric interpolation.

    Args:
        vertices: input vertices, size (B, V, 3).
        faces: the faces of the mesh, size (F, 3).
        lmk_faces_idx: the tensor with the indices of the faces used to
            calculate the landmarks, size (L,).
        lmk_bary_coords: the tensor of barycentric coordinates that are used
            to interpolate the landmarks, size (L, 3).

    Returns:
        landmarks: the coordinates of the landmarks for each mesh in the
            batch, size (B, L, 3).
    """
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        batch_size, -1, 3
    )

    lmk_faces += (
        torch.arange(batch_size, dtype=torch.long, device=device).view(
            -1, 1, 1
        )
        * num_verts
    )

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(batch_size, -1, 3, 3)

    landmarks = torch.einsum("blfi,blf->bli", [lmk_vertices, lmk_bary_coords])
    return landmarks


def lbs(
    betas: torch.Tensor,
    pose: torch.Tensor,
    v_template: torch.Tensor,
    shapedirs: torch.Tensor,
    posedirs: torch.Tensor,
    J_regressor: torch.Tensor,
    parents: torch.Tensor,
    lbs_weights: torch.Tensor,
    pose2rot: bool = True,
    dtype: torch.dtype = torch.float32,
):
    """Perform Linear Blend Skinning with the given shape and pose parameters.

    Args:
        betas: the tensor of shape parameters, size (B, NxB).
        pose: the pose paramters in axis-angle format, size (B, (J+1)x3).
        v_template: the template mesh that will be deformed, size (B, V, 3).
        shapedirs: the tensor of PCA shape displacements, size (1, NxB).
        posedirs: the pose PCA coefficients, size (P, Vx3).
        J_regressor: the regressor array that is used to calculate the joints
            from the position of the vertices, size (J, V).
        parents: the array that describes the kinematic tree for the model,
            size (J,).
        lbs_weights: the linear blend skinning weights that represent how much
            the rotation matrix of each part affects each vertex,
            size (N, V, J+1).
        pose2rot: flag on whether to convert the input pose tensor to rotation
        matrices. The default value is True. If False, then the pose tensor
        should already contain rotation matrices and have a size of
        (B, (J + 1), 9). Defaults to True.
        dtype: data type. Defaults to torch.float32.

    Returns:
        verts: the vertices of the mesh after applying the shape and pose
            displacements, size (B, V, 3).
        J_transformed: the joints of the model, size (B, J, 3).
    """

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device
    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3), dtype=dtype).view(
            [batch_size, -1, 3, 3]
        )

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs).view(
            batch_size, -1, 3
        )
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(
            pose_feature.view(batch_size, -1), posedirs
        ).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(
        batch_size, -1, 4, 4
    )

    homogen_coord = torch.ones(
        [batch_size, v_posed.shape[1], 1], dtype=dtype, device=device
    )
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed


def vertices2joints(
    J_regressor: torch.Tensor, vertices: torch.Tensor
) -> torch.Tensor:
    """Calculate the 3D joint locations from the vertices.

    Args:
        J_regressor: the regressor array that is used to calculate the
            joints from the position of the vertices, size (J, V).
        vertices: the tensor of mesh vertices, size (B, V, 3).

    Returns:
        The location of the joints.
    """
    return torch.einsum("bik,ji->bjk", [vertices, J_regressor])


def blend_shapes(
    betas: torch.Tensor, shape_disps: torch.Tensor
) -> torch.Tensor:
    """Calculate the per vertex displacement due to the blend shapes.

    Args:
        betas: blend shape coefficients, size (B, num_betas).
        shape_disps: blend shapes, size (V, 3, num_betas).

    Returns:
        The per-vertex displacement due to shape deformation.
    """

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum("bl,mkl->bmk", [betas, shape_disps])
    return blend_shape


def batch_rodrigues(
    rot_vecs: torch.Tensor,
    epsilon: float = 1e-10,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Calculate the rotation matrices for a batch of rotation vectors.

    Args:
        rot_vecs: array of N axis-angle vectors, size (N, 3).
        epsilon: Defaults to 1e-10.
        dtype: Defaults to torch.float32.

    Returns:
        rot_mat: the rotation matices for the given axis-angle parameters,
            size (N, 3, 3).
    """

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1
    ).view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Create a batch of transformation matrices.

    Args:
        R: a batch of rotation matrices, size (B, 3, 3).
        t: a batch of translation vectors, size (B, 3, 1).

    Returns:
        Transformation matrix, size (B, 4, 4).
    """
    # No padding left or right, only add an extra row
    return torch.cat(
        [F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2
    )


def batch_rigid_transform(
    rot_mats: torch.Tensor,
    joints: torch.Tensor,
    parents: torch.Tensor,
    dtype=torch.float32,
):
    """Apply a batch of rigid transformations to the joints.

    Args:
        rot_mats: rotation matrices, (B, N, 3, 3).
        joints: locations of joints, (B, N, 3).
        parents: the kinematic tree of each object, (B, N).
        dtype: data type. Defaults to torch.float32.

    Returns:
        posed_joints : the locations of the joints after applying the
            pose rotations, (B, N, 3).
        rel_transforms : the relative (with respect to the root joint)
            rigid transformations for all the joints, (B, N, 4, 4).
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)
    ).view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(
            transform_chain[parents[i]], transforms_mat[:, i]
        )
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]
    )

    return posed_joints, rel_transforms


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Based on Zhou et al.,
    "On the Continuity of Rotation Representations in Neural Networks",
    CVPR 2019.

    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.contiguous().view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)
