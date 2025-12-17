import logging
import os
from typing import List

import cv2
import numpy as np
import torch

import hat.models.task_modules.human3d.utils.human3d_constants as constants
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

try:
    from kornia.geometry.conversions import angle_axis_to_rotation_matrix
except ImportError:
    pass


@OBJECT_REGISTRY.register
class FitsDict:
    """Dictionary keeping track of the best fit per image in the training set.

    Args:
        load_dir: Directory to load initial fitting result.
        checkpoint_dir: Directory to save checkpoint and fitting result.
        dataset_list: Dataset name list.
        age: Age for choose which model used, kid or adult. Defaults to kid.
    """

    @require_packages("kornia")
    def __init__(
        self,
        load_dir: str,
        checkpoint_dir: str,
        dataset_list: List[str],
        age: str = "kid",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.dataset_list = dataset_list
        self.fits_dict = {}
        self.age = age
        # array used to flip SMPL pose parameters
        self.flipped_parts = torch.tensor(
            constants.SMPL_POSE_FLIP_PERM, dtype=torch.int64
        )

        # Load dictionary state
        for ds_name in dataset_list:
            dict_file = os.path.join(load_dir, ds_name + "_fits.npy")
            if os.path.exists(dict_file):
                self.fits_dict[ds_name] = torch.from_numpy(np.load(dict_file))
                logging.info(">> Loading dictionary: {}".format(dict_file))
            else:
                # Dictionary does not exist.
                logging.info("cannot find dict: {}".format(dict_file))
                self.fits_dict[ds_name] = None

    def save(self):
        """Save dictionary state to disk."""
        for ds_name in self.dataset_list:
            dict_file = os.path.join(
                self.checkpoint_dir, ds_name + "_fits.npy"
            )
            np.save(dict_file, self.fits_dict[ds_name].cpu().numpy())

    def __getitem__(self, x):
        """Retrieve dictionary entries."""
        dataset_name, ind, rot, is_flipped = x
        batch_size = len(dataset_name)
        pose = torch.zeros((batch_size, 72))
        betas_len = 11 if self.age == "kid" else 10
        if self.age == "kid":
            betas = torch.zeros((batch_size, betas_len))
        else:
            betas = torch.zeros((batch_size, betas_len))
        validity = torch.zeros((batch_size))
        for ds, i, n in zip(dataset_name, ind, range(batch_size)):
            if self.fits_dict[ds] is not None:
                params = self.fits_dict[ds][i]
                pose[n, :] = params[:72]

                betas[n, :] = params[72 : 72 + betas_len]
                validity[n] = 1
            else:
                validity[n] = 0
        pose = pose.clone()
        # Apply flipping and rotation
        pose = self.flip_pose(self.rotate_pose(pose, rot), is_flipped)
        betas = betas.clone()
        return pose, betas, validity

    def __setitem__(self, x, val):
        """Update dictionary entries."""
        dataset_name, ind, rot, is_flipped, update = x
        pose, betas = val
        batch_size = len(dataset_name)
        # Undo flipping and rotation
        pose = self.rotate_pose(self.flip_pose(pose, is_flipped), -rot)
        params = torch.cat((pose, betas), dim=-1).cpu()
        for ds, i, n in zip(dataset_name, ind, range(batch_size)):
            if update[n]:
                self.fits_dict[ds][i] = params[n]

    def flip_pose(self, pose, is_flipped):
        """Flip SMPL pose parameters."""
        is_flipped = is_flipped.byte()
        pose_f = pose.clone()
        pose_f[is_flipped, :] = pose[is_flipped][:, self.flipped_parts]
        # we also negate the second and the third dimension
        # of the axis-angle representation
        pose_f[is_flipped, 1::3] *= -1
        pose_f[is_flipped, 2::3] *= -1
        return pose_f

    def rotate_pose(self, pose, rot):
        """Rotate SMPL pose parameters by rot degrees."""
        pose = pose.clone()
        cos = torch.cos(-np.pi * rot / 180.0)
        sin = torch.sin(-np.pi * rot / 180.0)
        zeros = torch.zeros_like(cos)
        r3 = torch.zeros(cos.shape[0], 1, 3, device=cos.device)
        r3[:, 0, -1] = 1
        R = torch.cat(
            [
                torch.stack([cos, -sin, zeros], dim=-1).unsqueeze(1),
                torch.stack([sin, cos, zeros], dim=-1).unsqueeze(1),
                r3,
            ],
            dim=1,
        )
        global_pose = pose[:, :3]
        global_pose_rotmat = angle_axis_to_rotation_matrix(global_pose)
        global_pose_rotmat_3b3 = global_pose_rotmat[:, :3, :3]
        global_pose_rotmat_3b3 = torch.matmul(R, global_pose_rotmat_3b3)
        global_pose_rotmat[:, :3, :3] = global_pose_rotmat_3b3
        global_pose_rotmat = global_pose_rotmat[:, :3, :3].cpu().numpy()
        global_pose_np = np.zeros((global_pose.shape[0], 3))
        for i in range(global_pose.shape[0]):
            aa, _ = cv2.Rodrigues(global_pose_rotmat[i])
            global_pose_np[i, :] = aa.squeeze()
        pose[:, :3] = torch.from_numpy(global_pose_np).to(pose.device)
        return pose
