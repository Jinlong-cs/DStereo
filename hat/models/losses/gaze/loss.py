import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from .wing_loss import GazeWingLoss, WingLoss


@OBJECT_REGISTRY.register
class GazeEyeldmkLoss(nn.Module):
    """loss for gaze and eye_ldmk."""

    def __init__(
        self,
        gaze_loss_weight: float = 1.0,
        eye_ldmk_loss_weight: float = 1.0,
        gaze_loss_name: str = "wing",
        gaze_loss_params: dict = None,
        eye_ldmk_loss_name: str = "wing",
        eye_ldmk_loss_params: dict = None,
        **kwargs
    ):
        super(GazeEyeldmkLoss, self).__init__()
        self.gaze_loss_weight = gaze_loss_weight
        self.eye_ldmk_loss_weight = eye_ldmk_loss_weight
        if gaze_loss_name == "wing":
            self.gaze_loss = GazeWingLoss(
                pitch_w=gaze_loss_params["pitch_w"],
                yaw_w=gaze_loss_params["yaw_w"],
                pitch_e=gaze_loss_params["pitch_e"],
                yaw_e=gaze_loss_params["yaw_e"],
                is_averaged_output=gaze_loss_params["averaged_output"],
                batch_axis=gaze_loss_params["batch_axis"],
            )
        if eye_ldmk_loss_name == "wing":
            self.eye_ldmk_loss = WingLoss(
                w=eye_ldmk_loss_params["w"],
                epsilon=eye_ldmk_loss_params["epsilon"],
                is_averaged_output=gaze_loss_params["averaged_output"],
                batch_axis=gaze_loss_params["batch_axis"],
                adaptive_weight=True,
            )
        self.eye_ldmk_num = kwargs.get("eye_ldmk_num", 21)

    def preprocess(self, pred, target):
        # for gaze
        gaze_mask = target["gt_gaze"] > -1000
        gt_gaze = target["gt_gaze"] * gaze_mask
        pred_gaze = pred["gaze"].squeeze() * gaze_mask

        # for eye_ldmk
        eye_ldmk_mask = target["gt_normed_eye_ldmk"] > -1000
        gt_eye_ldmk = target["gt_normed_eye_ldmk"] * eye_ldmk_mask
        pred_eye_ldmk = (
            torch.concat(pred["eye_ldmk"], dim=1)
            .squeeze()
            .reshape((-1, self.eye_ldmk_num * 2, 2))
        )
        pred_eye_ldmk = pred_eye_ldmk * eye_ldmk_mask

        preprocess_res = {
            "preds": {"pred_gaze": pred_gaze, "pred_eye_ldmks": pred_eye_ldmk},
            "gts": {
                "gt_gaze": gt_gaze,
                "gt_eye_ldmks": gt_eye_ldmk,
            },
        }

        return preprocess_res

    def forward(self, pred, target):
        preprocess_res = self.preprocess(pred, target)
        pred_gaze = preprocess_res["preds"]["pred_gaze"]
        pred_eye_ldmk = preprocess_res["preds"]["pred_eye_ldmks"]
        gt_gaze = preprocess_res["gts"]["gt_gaze"]
        gt_eye_ldmk = preprocess_res["gts"]["gt_eye_ldmks"]

        gaze_loss = self.gaze_loss(pred_gaze, gt_gaze)
        eye_ldmk_loss = self.eye_ldmk_loss(pred_eye_ldmk, gt_eye_ldmk)

        loss_dict = {
            "gaze_loss": gaze_loss * self.gaze_loss_weight,
            "eye_ldmk_loss": eye_ldmk_loss * self.eye_ldmk_loss_weight,
        }

        return loss_dict
