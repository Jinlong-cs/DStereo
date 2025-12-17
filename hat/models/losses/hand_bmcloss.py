import pickle
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as torch_f
from torch import Tensor

LP_PATH = "./tmp_orig_data/hand3d/mano_params/bmc_limit_angles.pkl"

SNAP_PARENT = [
    0,  # 0's parent
    0,  # 1's parent
    1,
    2,
    3,
    0,  # 5's parent
    5,
    6,
    7,
    0,  # 9's parent
    9,
    10,
    11,
    0,  # 13's parent
    13,
    14,
    15,
    0,  # 17's parent
    17,
    18,
    19,
]


# bone indexes in 20 bones setting
ID_ROOT_bone = np.array([0, 4, 8, 12, 16])  # ROOT_bone from wrist to MCP
ID_PIP_bone = np.array([1, 5, 9, 13, 17])  # PIP_bone from MCP to PIP
ID_DIP_bone = np.array([2, 6, 10, 14, 18])  # DIP_bone from  PIP to DIP
ID_TIP_bone = np.array([3, 7, 11, 15, 19])  # TIP_bone from DIP to TIP


def two_norm(a: Tensor) -> Tensor:
    """Compute the two norm of a tensor along the last dimension.

    Args:
        a: A tensor of shape B*M*2 or B*M*3, where B is the batch size, \
            M is the number of elements in each batch, \
            and 2 or 3 represents the dimensionality of each element.

    Returns:
        A tensor of shape B*M, where each element in the output represents \
        the two norm of the corresponding element in the input tensor.
    """
    return torch.norm(a, dim=-1)


def one_norm(a: Tensor) -> Tensor:
    """Compute the one norm of a tensor along the last dimension.

    Args:
        a: A tensor of shape B*M*2 or B*M*3, where B is the batch size, \
            M is the number of elements in each batch, and 2 or 3 \
            represents the dimensionality of each element.

    Returns:
        A tensor of shape B*M, where each element in the output represents \
        the one norm of the corresponding element in the input tensor.
    """
    return torch.norm(a, dim=-1, p=1)


def calculate_joint_angle_loss(thetas: Tensor, hulls: List) -> Tensor:
    """Calculate the joint angle loss.

    Args:
        thetas: A tensor of shape B*15*2, where B is the batch size and 15 \
            represents the number of joints, each with 2 degrees of freedom.
        hulls: A list of 15 tensors, where each tensor has shape M*2, \
            representing the boundary of the region in which the \
            corresponding joint can move.

    Returns:
        A scalar tensor representing the joint angle loss.
    """
    device = thetas.device
    loss = torch.Tensor([0]).to(device)
    for i in range(15):
        hull = hulls[i]  # (M*2)
        theta = thetas[:, i].to(device)  # (B*2)
        hull = torch.cat((hull, hull[0].unsqueeze(0)), dim=0).to(device)

        v = (hull[1:] - hull[:-1]).unsqueeze(0)  # (M-1)*2
        w = -hull[:-1].unsqueeze(0) + theta.unsqueeze(1).repeat(
            1, hull[:-1].shape[0], 1
        )

        cross_product_2d = w[:, :, 0] * v[:, :, 1] - w[:, :, 1] * v[:, :, 0]
        tmp = torch.sum(cross_product_2d < 1e-6, dim=-1)

        is_outside = tmp != (hull.shape[0] - 1)
        if not torch.sum(is_outside):
            sub_loss = torch.Tensor([0]).to(device)
        else:
            outside_theta = theta[is_outside]
            outside_theta = outside_theta.unsqueeze(1).repeat(
                1, hull[:-1].shape[0], 1
            )
            w_outside = -hull[:-1].unsqueeze(0) + outside_theta  # B*(M-1)*2
            t = torch.clamp(
                inner_product(w_outside, v) / (two_norm(v) ** 2), min=0, max=1
            ).unsqueeze(2)
            p = hull[:-1] + t * v

            D = one_norm(torch.cos(outside_theta) - torch.cos(p)) + one_norm(
                torch.sin(outside_theta) - torch.sin(p)
            )
            sub_loss = torch.sum(torch.min(D, dim=-1)[0])

        loss += sub_loss.to(device)

    loss /= 15 * thetas.shape[0]

    return loss


def angle_between(v1: Tensor, v2: Tensor) -> Tensor:
    epsilon = 1e-7
    cos = torch_f.cosine_similarity(v1, v2, dim=-1).clamp(
        -1 + epsilon, 1 - epsilon
    )  # (B)
    theta = torch.acos(cos)  # (B)
    return theta


def normalize(vec: Tensor) -> Tensor:
    return torch_f.normalize(vec, p=2, dim=-1)


def inner_product(x1: Tensor, x2: Tensor) -> Tensor:
    return torch.sum(x1 * x2, dim=-1)


def cross_product(x1: Tensor, x2: Tensor) -> Tensor:
    return torch.cross(x1, x2, dim=-1)


def axangle2mat_torch(
    axis: Tensor, angle: Tensor, is_normalized: bool = False
) -> Tensor:
    """
    Return the rotation matrix for rotating `angle` radians around the `axis`.

    Args:
        axis: A tensor of shape B*M*3, where B is batch size, M is the \
            number of axes and each axis is represented by a 3D vector.
        angle: A tensor of shape B*M, representing the angle of rotation \
            in radians.
        is_normalized: A boolean indicating whether `axis` is already \
            normalized (has norm of 1).

    Returns:
        A tensor of shape B*M*3*3 representing the rotation matrix for \
        the specified rotation.

    Notes:
        This function is based on the formula from \
        http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    B = axis.shape[0]
    M = axis.shape[1]

    if not is_normalized:
        norm_axis = axis.norm(p=2, dim=-1, keepdim=True)
        normed_axis = axis / norm_axis
    else:
        normed_axis = axis
    x, y, z = normed_axis[:, :, 0], normed_axis[:, :, 1], normed_axis[:, :, 2]
    c = torch.cos(angle)
    s = torch.sin(angle)
    C = 1 - c

    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    TMP = torch.stack(
        [
            x * xC + c,
            xyC - zs,
            zxC + ys,
            xyC + zs,
            y * yC + c,
            yzC - xs,
            zxC - ys,
            yzC + xs,
            z * zC + c,
        ],
        dim=-1,
    )
    return TMP.reshape(B, M, 3, 3)


def interval_loss(value: Tensor, min: Tensor, max: Tensor) -> Tensor:
    """
    Calculate the interval loss for a tensor `value` given the \
    minimum and maximum values.

    Args:
        value: A tensor of shape B*M representing the values to \
            be compared with the interval.
        min: A tensor of shape M representing the minimum values \
            of the interval.
        max: A tensor of shape M representing the maximum values \
            of the interval.

    Returns:
        A scalar tensor representing the interval loss.

    Notes:
        The interval loss is the sum of the positive differences between \
        the values and the interval boundaries.
        This implementation assumes that `value`, `min`, and `max` have \
        the same number of dimensions.
    """
    device = value.device

    batch_3d_size = value.shape[0]

    min = min.repeat(value.shape[0], 1)
    max = max.repeat(value.shape[0], 1)

    loss1 = torch.relu(min.to(device) - value)
    loss2 = torch.relu(value - max.to(device))

    loss = (loss1 + loss2).sum()

    loss /= batch_3d_size * value.shape[1]

    return loss


class H3DBMCLoss:
    """
    Loss function for enforcing physiological constraints \
    on 3D human pose predictions.

    Args:
        lambda_bl:
            A float representing the loss weight of bone length.
            Defaults to 0.0.
        lambda_rb:
            A float representing the loss weight of root bones.
            Defaults to 0.0.
        lambda_a:
            A float representing the loss weight of joint angles.
            Defaults to 0.0.
        path_bmc_lp:
            A string representing the path to the bmc weight file.
            Defaults to LP_PATH.

    Attributes:
        lambda_bl:
            A float representing the loss weight of bone length.
        lambda_rb:
            A float representing the loss weight of root bones.
        lambda_a:
            A float representing the loss weight of joint angles.
        lp:
            A string representing the path to the bmc weight file.
        bone_len_max:
            A numpy array containing the maximum bone lengths.
        bone_len_min:
            A numpy array containing the minimum bone lengths.
        rb_curvatures_max:
            A numpy array containing the maximum root bone curvatures.
        rb_curvatures_min:
            A numpy array containing the minimum root bone curvatures.
        rb_PHI_max:
            A numpy array containing the maximum root bone PHI angles.
        rb_PHI_min:
            A numpy array containing the minimum root bone PHI angles.
        joint_angle_limit:
            A list containing torch tensors representing the \
            joint angle limits.
    """

    def __init__(
        self,
        lambda_bl: float = 0.0,
        lambda_rb: float = 0.0,
        lambda_a: float = 0.0,
        path_bmc_lp: str = LP_PATH,
    ):

        self.lambda_bl = lambda_bl
        self.lambda_rb = lambda_rb
        self.lambda_a = lambda_a
        self.lp = path_bmc_lp

        bmc_limit_dict = pickle.load(open(self.lp, "rb"))
        self.bone_len_max = bmc_limit_dict["bone_len_max"]
        self.bone_len_min = bmc_limit_dict["bone_len_min"]
        self.rb_curvatures_max = bmc_limit_dict["curvatures_max"]
        self.rb_curvatures_min = bmc_limit_dict["curvatures_min"]
        self.rb_PHI_max = bmc_limit_dict["PHI_max"]
        self.rb_PHI_min = bmc_limit_dict["PHI_min"]
        self.joint_angle_limit = bmc_limit_dict["CONVEX_HULLS"]
        LEN_joint_angle_limit = len(self.joint_angle_limit)

        # numpy to torch
        self.bl_max = torch.from_numpy(self.bone_len_max).float()
        self.bl_min = torch.from_numpy(self.bone_len_min).float()
        self.rb_curvatures_max = torch.from_numpy(
            self.rb_curvatures_max
        ).float()
        self.rb_curvatures_min = torch.from_numpy(
            self.rb_curvatures_min
        ).float()
        self.rb_PHI_max = torch.from_numpy(self.rb_PHI_max).float()
        self.rb_PHI_min = torch.from_numpy(self.rb_PHI_min).float()
        self.joint_angle_limit = [
            torch.from_numpy(self.joint_angle_limit[i]).float()
            for i in range(LEN_joint_angle_limit)
        ]

    def compute_loss(self, joints: Tensor) -> Tuple:
        """Compute loss.

        Args:
            joints: B*21*3
        """
        device = joints.device
        batch_size = joints.shape[0]
        final_loss = torch.Tensor([0]).to(device)

        # to device
        self.bl_max = self.bl_max.to(device)
        self.bl_min = self.bl_min.to(device)
        self.rb_curvatures_max = self.rb_curvatures_max.to(device)
        self.rb_curvatures_min = self.rb_curvatures_min.to(device)
        self.rb_PHI_max = self.rb_PHI_max.to(device)
        self.rb_PHI_min = self.rb_PHI_min.to(device)
        self.joint_angle_limit = [
            joint_limit.to(device) for joint_limit in self.joint_angle_limit
        ]

        BMC_losses = {
            "l_bmc_bl": torch.Tensor([0]).to(device),
            "l_bmc_rb": torch.Tensor([0]).to(device),
            "l_bmc_a": torch.Tensor([0]).to(device),
        }

        if (
            (self.lambda_bl < 1e-6)
            and (self.lambda_rb < 1e-6)
            and (self.lambda_a < 1e-6)
        ):
            return final_loss, BMC_losses

        ALL_bones = [
            (joints[:, i, :] - joints[:, SNAP_PARENT[i], :]) for i in range(21)
        ]
        ALL_bones = torch.stack(ALL_bones[1:], dim=1)  # (B,20,3)
        ROOT_bones = ALL_bones[:, ID_ROOT_bone]  # 1, 5, 9, 13, 17
        PIP_bones = ALL_bones[:, ID_PIP_bone]
        DIP_bones = ALL_bones[:, ID_DIP_bone]
        TIP_bones = ALL_bones[:, ID_TIP_bone]

        ALL_Z_axis = normalize(ALL_bones)
        PIP_Z_axis = ALL_Z_axis[:, ID_ROOT_bone]
        DIP_Z_axis = ALL_Z_axis[:, ID_PIP_bone]
        TIP_Z_axis = ALL_Z_axis[:, ID_DIP_bone]

        # normal of 01-05, 05-09, ...
        normals = normalize(
            cross_product(ROOT_bones[:, 1:], ROOT_bones[:, :-1])
        )

        # compute loss of bone length (√)
        bl_loss = torch.Tensor([0])
        if self.lambda_bl:
            bls = two_norm(ALL_bones)  # (B,20,1)
            bl_loss = interval_loss(
                value=bls, min=self.bl_min, max=self.bl_max
            )
            final_loss += self.lambda_bl * bl_loss
        BMC_losses["l_bmc_bl"] = self.lambda_bl * bl_loss.to(device)

        # compute loss of Root bones (√)
        rb_loss = torch.Tensor([0]).to(device)
        if self.lambda_rb:
            # edge normal at bone bi
            edge_normals = torch.zeros_like(ROOT_bones).to(device)  # (B,5,3)
            edge_normals[:, [0, 4]] = normals[:, [0, 3]]
            edge_normals[:, 1:4] = normalize(normals[:, 1:4] + normals[:, :3])

            curvatures = (
                inner_product(
                    edge_normals[:, 1:] - edge_normals[:, :4],
                    # 1-5, 5-9, 9-13, 13-17
                    ROOT_bones[:, 1:] - ROOT_bones[:, :4],
                )
                / (two_norm(ROOT_bones[:, 1:] - ROOT_bones[:, :4]) ** 2 + 1e-6)
            )
            curvatures_abs = torch.abs(curvatures)
            PHI = angle_between(ROOT_bones[:, :4], ROOT_bones[:, 1:])  # (B)

            rb_loss = interval_loss(
                value=curvatures_abs,
                min=self.rb_curvatures_min,
                max=self.rb_curvatures_max,
            ) + interval_loss(
                value=PHI,
                min=self.rb_PHI_min,
                max=self.rb_PHI_max,
            )
            final_loss += self.lambda_rb * rb_loss
        BMC_losses["l_bmc_rb"] = self.lambda_rb * rb_loss.to(device)

        # compute loss of Joint angles
        a_loss = torch.Tensor([0]).to(device)
        if self.lambda_a:
            # PIP bones
            PIP_X_axis = torch.zeros([batch_size, 5, 3]).to(device)  # (B,5,3)
            PIP_X_axis[:, [0, 1, 4], :] = -normals[:, [0, 1, 3]]
            PIP_X_axis[:, 2:4] = -normalize(
                normals[:, 2:4] + normals[:, 1:3]
            )  # (B,2,3)
            PIP_Y_axis = normalize(
                cross_product(PIP_Z_axis, PIP_X_axis)
            )  # (B,5,3)

            PIP_bones_xz = (
                PIP_bones
                - inner_product(PIP_bones, PIP_Y_axis).unsqueeze(2)
                * PIP_Y_axis
            )
            PIP_theta_flexion = angle_between(
                PIP_bones_xz, PIP_Z_axis
            )  # in global coordinate (B)
            PIP_theta_abduction = angle_between(
                PIP_bones_xz, PIP_bones
            )  # in global coordinate (B)
            # x-component of the bone vector
            tmp = inner_product(PIP_bones, PIP_X_axis)
            PIP_theta_flexion = torch.where(
                tmp < 1e-6, -PIP_theta_flexion, PIP_theta_flexion
            )
            # y-component of the bone vector
            tmp = inner_product(PIP_bones, PIP_Y_axis)
            PIP_theta_abduction = torch.where(
                tmp < 1e-6, -PIP_theta_abduction, PIP_theta_abduction
            )

            temp_axis = normalize(cross_product(PIP_Z_axis, PIP_bones))
            temp_alpha = angle_between(
                PIP_Z_axis, PIP_bones
            )  # alpha belongs to [pi/2, pi]
            temp_R = axangle2mat_torch(
                axis=temp_axis, angle=temp_alpha, is_normalized=True
            )

            # DIP bones
            DIP_X_axis = torch.matmul(
                temp_R, PIP_X_axis.unsqueeze(3)
            ).squeeze()
            DIP_Y_axis = torch.matmul(
                temp_R, PIP_Y_axis.unsqueeze(3)
            ).squeeze()

            DIP_bones_xz = (
                DIP_bones
                - inner_product(DIP_bones, DIP_Y_axis).unsqueeze(2)
                * DIP_Y_axis
            )
            DIP_theta_flexion = angle_between(
                DIP_bones_xz, DIP_Z_axis
            )  # in global coordinate
            DIP_theta_abduction = angle_between(
                DIP_bones_xz, DIP_bones
            )  # in global coordinate
            # x-component of the bone vector
            tmp = inner_product(DIP_bones, DIP_X_axis)
            DIP_theta_flexion = torch.where(
                tmp < 1e-6, -DIP_theta_flexion, DIP_theta_flexion
            )
            # y-component of the bone vector
            tmp = inner_product(DIP_bones, DIP_Y_axis)
            DIP_theta_abduction = torch.where(
                tmp < 1e-6, -DIP_theta_abduction, DIP_theta_abduction
            )

            temp_axis = normalize(cross_product(DIP_Z_axis, DIP_bones))
            temp_alpha = angle_between(
                DIP_Z_axis, DIP_bones
            )  # alpha belongs to [pi/2, pi]
            temp_R = axangle2mat_torch(
                axis=temp_axis, angle=temp_alpha, is_normalized=True
            )

            # TIP bones
            TIP_X_axis = torch.matmul(
                temp_R, DIP_X_axis.unsqueeze(3)
            ).squeeze()
            TIP_Y_axis = torch.matmul(
                temp_R, DIP_Y_axis.unsqueeze(3)
            ).squeeze()
            TIP_bones_xz = (
                TIP_bones
                - inner_product(TIP_bones, TIP_Y_axis).unsqueeze(2)
                * TIP_Y_axis
            )

            TIP_theta_flexion = angle_between(
                TIP_bones_xz, TIP_Z_axis
            )  # in global coordinate
            TIP_theta_abduction = angle_between(
                TIP_bones_xz, TIP_bones
            )  # in global coordinate
            # x-component of the bone vector
            tmp = inner_product(TIP_bones, TIP_X_axis)
            TIP_theta_flexion = torch.where(
                tmp < 1e-6, -TIP_theta_flexion, TIP_theta_flexion
            )
            # y-component of the bone vector
            tmp = inner_product(TIP_bones, TIP_Y_axis)
            TIP_theta_abduction = torch.where(
                tmp < 1e-6, -TIP_theta_abduction, TIP_theta_abduction
            )

            # ALL
            ALL_theta_flexion = torch.cat(
                (PIP_theta_flexion, DIP_theta_flexion, TIP_theta_flexion),
                dim=-1,
            )
            ALL_theta_abduction = torch.cat(
                (
                    PIP_theta_abduction,
                    DIP_theta_abduction,
                    TIP_theta_abduction,
                ),
                dim=-1,
            )
            ALL_theta = torch.stack(
                (ALL_theta_flexion, ALL_theta_abduction), dim=-1
            )

            a_loss = calculate_joint_angle_loss(
                ALL_theta.to(device), self.joint_angle_limit
            )
            final_loss += self.lambda_a * a_loss

        BMC_losses["l_bmc_a"] = self.lambda_a * a_loss.to(device)

        return final_loss.to(device), BMC_losses
