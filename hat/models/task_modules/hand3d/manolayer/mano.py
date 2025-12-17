# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import os
import os.path as osp
import pickle
from typing import NewType, Optional

import numpy as np
import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

try:
    from smplx.body_models import MANO
    from smplx.lbs import batch_rodrigues, lbs
    from smplx.utils import MANOOutput, Struct, to_tensor
    from smplx.vertex_ids import vertex_ids as VERTEX_IDS
except Exception:
    lbs = None
    batch_rodrigues = None
    to_tensor = None
    MANO = nn.Module
    MANOOutput = nn.Module
    Struct = nn.Module
    VERTEX_IDS = None

__all__ = ["MANOLayer"]
Tensor = NewType("Tensor", torch.Tensor)


@OBJECT_REGISTRY.register
class MANOLayer(MANO):
    def __init__(
        self,
        model_path: str,
        is_rhand: bool = True,
        data_struct: Optional[Struct] = None,
        hand_pose: Optional[Tensor] = None,
        use_pca: bool = True,
        num_pca_comps: int = 6,
        center_idx: int = 0,
        flat_hand_mean: bool = False,
        smpl_hands_mean: Optional[list] = None,
        batch_size: int = 1,
        dtype=torch.float32,
        vertex_ids=None,
        use_compressed: bool = True,
        ext: str = "pkl",
        **kwargs,
    ) -> None:
        """MANO as a layer model constructor.

        Reference: https://github.com/vchoutas/smplx/tree/main.
        1. Add base pose parameters (smpl_hands_mean) to replace T-pose \
        when flat_hand_mean is False.
        2. Map the 16 mano joints to 21 hand joints, adding 5 hand tips and \
        changing the indexes of joints sequence.

        Args:
            model_path: Path to the pre-trained MANO model.
            is_rhand: Whether to use the right hand. Defaults to True.
            data_struct: Data structure to use. Defaults to None.
            hand_pose: Hand pose tensor. Defaults to None.
            use_pca: Whether to use PCA decomposition. Defaults to True.
            num_pca_comps: Number of PCA components to use. Defaults to 6.
            center_idx: Index of the center joint. Defaults to 0.
            flat_hand_mean:
                Whether to use a flat hand mean pose. Defaults to False.
            smpl_hands_mean:
                Base pose parameters to replace T-pose \
                when flat_hand_mean is False. Defaults to None.
            batch_size: Batch size. Defaults to 1.
            dtype: Data type for tensors. Defaults to torch.float32.
            vertex_ids: Vertex IDs. Defaults to None.
            use_compressed:
                Whether to use compressed format. Defaults to True.
            ext: File extension for the model. Defaults to "pkl".
            **kwargs: Additional keyword arguments.
        """
        create_hand_pose = False
        self.num_pca_comps = num_pca_comps
        self.is_rhand = is_rhand
        # If no data structure is passed, then load the data from the given
        # model folder
        if data_struct is None:
            # Load the model
            if osp.isdir(model_path):
                model_fn = "MANO_{}.{ext}".format(
                    "RIGHT" if is_rhand else "LEFT", ext=ext
                )
                mano_path = os.path.join(model_path, model_fn)
            else:
                mano_path = model_path
                self.is_rhand = (
                    True if "RIGHT" in os.path.basename(model_path) else False
                )
            assert osp.exists(mano_path), "Path {} does not exist!".format(
                mano_path
            )

            if ext == "pkl":
                with open(mano_path, "rb") as mano_file:
                    model_data = pickle.load(mano_file, encoding="latin1")
            elif ext == "npz":
                model_data = np.load(mano_path, allow_pickle=True)
            else:
                raise ValueError("Unknown extension: {}".format(ext))
            data_struct = Struct(**model_data)

        if vertex_ids is None:
            vertex_ids = VERTEX_IDS["smplh"]

        super(MANOLayer, self).__init__(
            model_path=model_path,
            data_struct=data_struct,
            batch_size=batch_size,
            vertex_ids=vertex_ids,
            use_compressed=use_compressed,
            dtype=dtype,
            ext=ext,
            **kwargs,
        )

        # add only MANO tips to the extra joints
        self.vertex_joint_selector.extra_joints_idxs = to_tensor(
            list(VERTEX_IDS["mano"].values()), dtype=torch.long
        )

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        if self.num_pca_comps == 45:
            self.use_pca = False

        self.center_idx = center_idx
        self.flat_hand_mean = flat_hand_mean

        hand_components = data_struct.hands_components[:num_pca_comps]
        self.np_hand_components = hand_components
        if self.use_pca:
            self.register_buffer(
                "hand_components", torch.tensor(hand_components, dtype=dtype)
            )

        if self.flat_hand_mean:
            hand_mean = data_struct.hands_mean
        else:
            hand_mean = (
                np.array(smpl_hands_mean)
                if smpl_hands_mean is not None
                else np.zeros_like(data_struct.hands_mean)
            )

        self.register_buffer(
            "hand_mean", to_tensor(hand_mean, dtype=self.dtype)
        )

        # Create the buffers for the pose of the left hand
        hand_pose_dim = num_pca_comps if use_pca else 3 * self.NUM_HAND_JOINTS
        if create_hand_pose:
            if hand_pose is None:
                default_hand_pose = torch.zeros(
                    [batch_size, hand_pose_dim], dtype=dtype
                )
            else:
                default_hand_pose = torch.tensor(hand_pose, dtype=dtype)

            hand_pose_param = nn.Parameter(
                default_hand_pose, requires_grad=True
            )
            self.register_parameter("hand_pose", hand_pose_param)

        # Create the buffer for the mean pose.
        global_orient_mean = torch.zeros([3], dtype=dtype)
        pose_mean = torch.cat(
            [global_orient_mean, self.hand_mean], dim=0
        ).unsqueeze(0)
        pose_mean_tensor = pose_mean.clone().to(dtype)
        self.register_buffer("pose_mean", pose_mean_tensor)

        self.right_handtips_indexes = [745, 317, 444, 556, 673]
        self.left_handtips_indexes = [745, 317, 445, 556, 673]
        self.sequential_mapping_of_mano_joints = [
            0,
            13,
            14,
            15,
            16,
            1,
            2,
            3,
            17,
            4,
            5,
            6,
            18,
            10,
            11,
            12,
            19,
            7,
            8,
            9,
            20,
        ]

    def name(self) -> str:
        return "MANOLayer"

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        pose2rot: bool = False,
        **kwargs,
    ) -> MANOOutput:
        """Forward pass for the MANO model."""
        device, dtype = self.shapedirs.device, self.shapedirs.dtype
        if global_orient is None:
            batch_size = 1
            global_orient = (
                torch.eye(3, device=device, dtype=dtype)
                .view(1, 1, 3, 3)
                .expand(batch_size, -1, -1, -1)
                .contiguous()
            )
        else:
            batch_size = global_orient.shape[0]
        if hand_pose is None:
            hand_pose = (
                torch.eye(3, device=device, dtype=dtype)
                .view(1, 1, 3, 3)
                .expand(batch_size, 15, -1, -1)
                .contiguous()
            )
        if betas is None:
            betas = torch.zeros(
                [batch_size, self.num_betas], dtype=dtype, device=device
            )
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        full_pose = torch.cat([global_orient, hand_pose], dim=1)
        if pose2rot:
            full_pose += self.pose_mean.repeat((batch_size, 1))
        else:
            pose_mean = batch_rodrigues(
                self.pose_mean.repeat((batch_size, 1)).view(-1, 3)
            ).view([batch_size, -1, 3, 3])
            full_pose = torch.matmul(full_pose, pose_mean)

        vertices, joints = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=pose2rot,
        )

        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if transl is not None:
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)
        else:
            if self.center_idx is not None:
                center_joint = joints[:, self.center_idx].unsqueeze(1)
                joints = joints - center_joint
                vertices = vertices - center_joint

        # mapping 16 mano joints to 21 hand joints
        if self.is_rhand:
            tips = vertices[:, self.right_handtips_indexes]
        else:
            tips = vertices[:, self.left_handtips_indexes]
        joints = torch.cat([joints, tips], 1)
        joints = joints[
            :,
            self.sequential_mapping_of_mano_joints,
        ]

        return vertices, joints
