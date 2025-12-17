import logging
import os.path as osp

import numpy as np
import torch

from hat.core.face3d.lbs import rot6d_to_rotmat
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages
from .metric import EvalMetric

try:
    from smplx import SMPL
except ImportError:
    SMPL = None

logger = logging.getLogger(__name__)

__all__ = ["Human3dMetric"]


@OBJECT_REGISTRY.register
class Human3dMetric(EvalMetric):
    """Human3d metric, output mpjpe or pa-mpjpe.

    Args:
        metric_name: Name of metirc.
        dataset_name: Name of dataset.
        smpl_model_dir: SMPL model directory path.
        use_pa: Whether use procrustes analysis. Defaults to True.
    """

    @require_packages("smplx")
    def __init__(
        self,
        metric_name: str,
        dataset_name: str,
        smpl_model_dir: str,
        use_pa: bool = True,
    ):
        super(Human3dMetric, self).__init__(metric_name)
        self.dataset_name = dataset_name
        self.use_pa = use_pa

        # Load SMPL model
        self.smpl_neutral = SMPL(
            osp.join(smpl_model_dir, "data/smpl"),
            create_transl=False,
            age="kid",
            joint_regressor_train_extra=osp.join(
                smpl_model_dir, "data/J_regressor_extra.npy"
            ),
            kid_template_path=osp.join(
                smpl_model_dir, "data/smpl_kid_template.npy"
            ),
        )
        self.J_regressor = torch.from_numpy(
            np.load(osp.join(smpl_model_dir, "data/J_regressor_h36m.npy"))
        ).float()

    def _compute_similarity_transform_batch(self, S1, S2):
        S1_hat = torch.zeros_like(S1).to(S1.device)
        for i in range(S1.size()[0]):
            S1_hat[i] = self._compute_similarity_transform(S1[i], S2[i])
        return S1_hat

    def _compute_similarity_transform(self, S1, S2):
        transposed = False
        if S1.size()[0] != 3 and S1.size()[0] != 2:
            S1 = S1.T
            S2 = S2.T
            transposed = True
        assert S2.shape[1] == S1.shape[1]

        # 1. Remove mean.
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = torch.sum(X1 ** 2)

        # 3. The outer product of X1 and X2.
        K = X1 @ X2.T

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, Vh = torch.linalg.svd(K)
        V = Vh.T
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[0]).to(S1.device)
        Z[-1, -1] *= torch.sign(torch.linalg.det(U @ V.T))
        # Construct R.
        R = V @ (Z @ U.T)

        # 5. Recover scale.
        scale = torch.trace(R @ K) / var1

        # 6. Recover translation.
        t = mu2 - scale * (R @ mu1)

        # 7. Error:
        S1_hat = scale * (R @ S1) + t

        if transposed:
            S1_hat = S1_hat.T

        return S1_hat

    def pa_mpjpe(self, S1, S2, reduction="mean"):
        S1_hat = self._compute_similarity_transform_batch(S1, S2)
        re = torch.sqrt(((S1_hat - S2) ** 2).sum(axis=-1)).mean(axis=-1)
        if reduction == "mean":
            re = re.mean()
        elif reduction == "sum":
            re = re.sum()
        return re

    def update(self, labels, preds):
        gt_pose = labels["gt_smpl_pose"]
        gt_betas = labels["gt_smpl_betas"]
        img = labels["img"]
        curr_batch_size = img.size()[0]
        self.smpl_neutral = self.smpl_neutral.to(img.device)
        gt_vertices = self.smpl_neutral(
            betas=gt_betas,
            body_pose=gt_pose[:, 3:],
            global_orient=gt_pose[:, :3],
        ).vertices

        pred_pose = preds["pred_pose"]
        pred_betas = preds["pred_betas"]
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(
            curr_batch_size, 24, 3, 3
        )
        pred_smpl = self.smpl_neutral(
            betas=pred_betas,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False,
        )
        pred_vertices = pred_smpl.vertices

        self.J_regressor_batch = (
            self.J_regressor[None, :]
            .expand(pred_vertices.shape[0], -1, -1)
            .to(img.device)
        )

        gt_keypoints_3d = torch.matmul(self.J_regressor_batch, gt_vertices)
        gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

        pred_keypoints_3d = torch.matmul(self.J_regressor_batch, pred_vertices)
        pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

        if self.use_pa:
            error = self.pa_mpjpe(
                pred_keypoints_3d, gt_keypoints_3d, reduction="sum"
            )
        else:
            error = (
                torch.sqrt(
                    ((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)
                )
                .mean(dim=-1)
                .sum()
            )

        self.sum_metric += 1000 * error
        self.num_inst += curr_batch_size
