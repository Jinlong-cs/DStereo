from collections import OrderedDict
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from hat.core.gaze.transform import pitchyaw_to_unit_vector
from hat.registry import OBJECT_REGISTRY

__all__ = ["PCCRLoss"]


KQ_NORM = 1


@OBJECT_REGISTRY.register
class PCCRLoss(nn.Module):
    def __init__(
        self,
        params_range: Dict,
        eye_params_typical: Dict,
        loss_weights: Dict,
        id_batch_size: int,
    ):
        super().__init__()
        self.loss_weights = loss_weights
        self.batch_size = id_batch_size
        self.params_range = params_range
        self._check_range_valid()
        self.eye_params_typical = eye_params_typical

    def _check_range_valid(self):
        if (
            "R" in self.params_range.keys()
            and "K" in self.params_range.keys()
            and "alpha" in self.params_range.keys()
            and "beta" in self.params_range.keys()
        ):
            self.params_range_valid = True
        else:
            self.params_range_valid = False

        if self.params_range_valid:
            if (
                len(self.params_range["R"]) != 2
                or len(self.params_range["K"]) != 2
                or len(self.params_range["alpha"]) != 2
                or len(self.params_range["beta"]) != 2
            ):
                self.params_range_valid = False

        if self.params_range_valid:
            if self.params_range["R"][1] <= self.params_range["R"][0]:
                self.params_range_valid = False

            if self.params_range["K"][1] <= self.params_range["K"][0]:
                self.params_range_valid = False

            if self.params_range["alpha"][1] <= self.params_range["alpha"][0]:
                self.params_range_valid = False

            if self.params_range["beta"][1] <= self.params_range["beta"][0]:
                self.params_range_valid = False

    def params_consistency_loss(self, pred: Dict, loss_weight: Dict):
        ret = {}
        cur_lw = loss_weight.get("params_consistency")
        if cur_lw is not None:
            loss = 0
            if cur_lw.get("R", -1) > 0:
                pred_R = pred["R"].reshape(
                    [self.batch_size, pred["R"].shape[0] // self.batch_size]
                    + list(pred["R"].shape[1:])
                )
                loss += (
                    F.l1_loss(pred_R, pred_R.mean(1, keepdims=True))
                    * cur_lw["R"]
                )
            if cur_lw.get("K", -1) > 0:
                pred_K = pred["K"].reshape(
                    [self.batch_size, pred["K"].shape[0] // self.batch_size]
                    + list(pred["K"].shape[1:])
                )
                loss += (
                    F.l1_loss(pred_K, pred_K.mean(1, keepdims=True))
                    * cur_lw["K"]
                )
            if cur_lw.get("alpha", -1) > 0:
                pred_alpha = pred["alpha"].reshape(
                    [
                        self.batch_size,
                        pred["alpha"].shape[0] // self.batch_size,
                    ]
                    + list(pred["alpha"].shape[1:])
                )
                loss += (
                    F.l1_loss(pred_alpha, pred_alpha.mean(1, keepdims=True))
                    * cur_lw["alpha"]
                )
            if cur_lw.get("beta", -1) > 0:
                pred_beta = pred["beta"].reshape(
                    [self.batch_size, pred["beta"].shape[0] // self.batch_size]
                    + list(pred["beta"].shape[1:])
                )
                loss += (
                    F.l1_loss(pred_beta, pred_beta.mean(1, keepdims=True))
                    * cur_lw["beta"]
                )
            ret["params_consistency"] = loss
        return ret

    def regularization_loss(self, pred: Dict, loss_weight: Dict):
        ret = {}
        cur_lw = loss_weight.get("regularization")
        if cur_lw is not None:
            loss = 0
            if cur_lw.get("R", 0) > 0:
                loss += (
                    torch.linalg.norm(
                        pred["R"] - self.eye_params_typical.get("R", 7.8)
                    )
                    * cur_lw["R"]
                )
            if cur_lw.get("K", 0) > 0:
                loss += (
                    torch.linalg.norm(
                        pred["K"] - self.eye_params_typical.get("K", 4.75)
                    )
                    * cur_lw["K"]
                )
            if cur_lw.get("alpha", 0) > 0:
                loss += (
                    torch.linalg.norm(
                        pred["alpha"] - self.eye_params_typical.get("alpha", 5)
                    )
                    * cur_lw["alpha"]
                )
            if cur_lw.get("beta", 0) > 0:
                loss += (
                    torch.linalg.norm(
                        pred["beta"] - self.eye_params_typical.get("beta", 1.5)
                    )
                    * cur_lw["beta"]
                )

            if cur_lw.get("pitch", 0) > 0:
                loss += torch.linalg.norm(pred["pitch"]) * cur_lw["pitch"]
            if cur_lw.get("yaw", 0) > 0:
                loss += torch.linalg.norm(pred["yaw"]) * cur_lw["yaw"]

            ret["regularization"] = loss
        return ret

    def in_range_loss(self, pred: Dict, loss_weight: Dict):
        # TODO: 偏离越远lw越大
        ret = {}
        cur_lw = loss_weight.get("in_range")
        if cur_lw is not None:
            assert self.params_range_valid
            loss = 0
            dis_min_R = torch.clip(self.params_range["R"][0] - pred["R"], 0)
            dis_max_R = torch.clip(pred["R"] - self.params_range["R"][1], 0)
            loss += cur_lw["R"] * (dis_min_R + dis_max_R).mean()

            dis_min_K = torch.clip(self.params_range["K"][0] - pred["K"], 0)
            dis_max_K = torch.clip(pred["K"] - self.params_range["K"][1], 0)
            loss += cur_lw["K"] * (dis_min_K + dis_max_K).mean()

            dis_min_a = torch.clip(
                self.params_range["alpha"][0] - pred["alpha"], 0
            )
            dis_max_a = torch.clip(
                pred["alpha"] - self.params_range["alpha"][1], 0
            )
            loss += cur_lw["alpha"] * (dis_min_a + dis_max_a).mean()

            dis_min_b = torch.clip(
                self.params_range["beta"][0] - pred["beta"], 0
            )
            dis_max_b = torch.clip(
                pred["beta"] - self.params_range["beta"][1], 0
            )
            loss += cur_lw["beta"] * (dis_min_b + dis_max_b).mean()

            dis_min_p = torch.clip(
                self.params_range["pitch"][0] - pred["pitch"], 0
            )
            dis_max_p = torch.clip(
                pred["pitch"] - self.params_range["pitch"][1], 0
            )
            loss += cur_lw["pitch"] * (dis_min_p + dis_max_p).mean()
            dis_min_y = torch.clip(
                self.params_range["yaw"][0] - pred["yaw"], 0
            )
            dis_max_y = torch.clip(
                pred["yaw"] - self.params_range["yaw"][1], 0
            )
            loss += cur_lw["yaw"] * (dis_min_y + dis_max_y).mean()
            # 两个kq相差不超过0.5mm
            kq_dis = torch.abs(
                pred["kq_result"][:, 0] - pred["kq_result"][:, 1]
            )
            kq_dis[kq_dis < 0.5 * KQ_NORM] = 0
            loss += cur_lw["kq"] * kq_dis.mean()
            # kq大于300mm
            dis_min_kq = torch.clip(300 * KQ_NORM - pred["kq_result"], min=0)
            loss += cur_lw["kq"] * dis_min_kq.mean()

            ret["in_range"] = loss
        return ret

    def pupil_distance_loss(self, pred: Dict, loss_weight: Dict):
        # 约束所有瞳孔边缘点到中心距离相等
        # TODO: 后续可以加上对瞳孔半径范围/典型值的约束
        ret = {}
        cur_lw = loss_weight.get("pupil_dis")
        if cur_lw is not None:
            r_pupil = torch.linalg.norm(
                pred["pb"] - pred["p"].reshape(-1, 1, 3), 2, -1
            )  # imgn, pbn
            # 每张图有超过3个有效瞳孔点认为有效
            valid_m = (
                pred["valid_mask"] * pred["valid_mask"].sum(-1, keepdims=True)
                > 3
            )  # valid img, imgn,pbn
            r_pupil = r_pupil * valid_m  # imgn,pbn,you quan0 hang
            r_pupil_mean = r_pupil.sum(-1, keepdims=True) / (
                1e-6 + valid_m.sum(-1, keepdims=True)
            )  # imgn,1,you quan0 hang
            abs_dis = (
                torch.abs(r_pupil - r_pupil_mean) * valid_m
            )  # imgn,pbn,you quan0 hang
            abs_dis[abs_dis < 0.1] = 0
            loss = (abs_dis.sum(-1) / (1e-6 + valid_m.sum(-1))).mean() * cur_lw
            # 同一环境下瞳孔变化小
            valid_img_mask = pred["valid_mask"].sum(-1, keepdims=True) > 10
            valid_img_n = valid_img_mask.sum()
            r_pupil_mean_mean = (r_pupil_mean * valid_img_mask).sum() / (
                1e-6 + valid_img_n
            )
            same_loss = (
                torch.abs(r_pupil_mean - r_pupil_mean_mean) * valid_img_mask
            ).sum() / (1e-6 + valid_img_n)
            same_loss = torch.clip(same_loss - 0.5, min=0) * cur_lw * 0.2
            ret = {"pupil_dis": loss, "pupil_same_dis": same_loss}
            # 正常成人瞳孔直径 2-5(稍暗光线)
            if (
                loss_weight.get("in_range") is not None
                and loss_weight["in_range"].get("r_pb", 0) > 0
            ):
                dis_min_rpb = torch.clip(1 - r_pupil_mean, 0) * valid_img_mask
                dis_max_rpb = (
                    torch.clip(r_pupil_mean - 2.5, 0) * valid_img_mask
                )
                loss_rpb_range = (
                    loss_weight["in_range"]["r_pb"]
                    * (dis_min_rpb + dis_max_rpb).sum()
                    / (1e-6 + valid_img_n)
                )
                ret["pupil_in_range"] = loss_rpb_range
        return ret

    def angle_loss(self, pred: Dict, label: Dict, loss_weight: Dict):
        ret = {}
        cur_lw = loss_weight.get("angle")
        if cur_lw is not None:
            pred_x, pred_y, pred_z = pitchyaw_to_unit_vector(
                pred["angle"][:, 0:1], pred["angle"][:, 1:2], angle_rad=False
            )
            label_x, label_y, label_z = pitchyaw_to_unit_vector(
                label["angle"][:, 0:1], label["angle"][:, 1:2], angle_rad=False
            )
            angle_v = pred_x * label_x + pred_y * label_y + pred_z * label_z
            angle = (
                torch.acos(torch.clip(angle_v, -1 + 1e-6, 1 - 1e-6))
                / torch.pi
                * 180
            )
            loss = angle.mean() * cur_lw

            ret = {"angle": loss}
        return ret

    def gaze_point_loss(self, pred: Dict, label: Dict, loss_weight: Dict):
        ret = {}
        cur_lw = loss_weight.get("gaze_point_dis")
        if cur_lw is not None:
            allow_dis = loss_weight.get("gaze_point_dis_thresh", 0)
            dis = torch.abs(pred["screen_coords"] - label["gaze_point"])
            dis_mask = dis < allow_dis
            pred_screen = (~dis_mask) * pred[
                "screen_coords"
            ] + dis_mask * label["gaze_point"]
            loss = (
                torch.linalg.norm(
                    pred_screen - label["gaze_point"], 2, -1
                ).mean()
                * cur_lw
            )
            ret = {"gaze_point_dis": loss}
        return ret

    def corneal_centers_distance(self, pred: Dict, loss_weight: Dict):
        ret = {}
        cur_lw = loss_weight.get("center_dis")
        if cur_lw is not None:
            loss = (
                torch.linalg.norm(
                    pred["center_1"] - pred["center_2"], 2, -1
                ).mean()
                * cur_lw
            )
            ret = {"center_dis": loss}
        return ret

    def eye3d_center_dis_loss(
        self, pred: Dict, label: Dict, loss_weight: Dict
    ):
        ret = {}
        cur_lw = loss_weight.get("eye3d_center_dis")
        if cur_lw is not None:
            allow_dis = loss_weight.get("eye3d_allow_dis", 0)
            dis = torch.abs(pred["center"] - label["eye3d"])
            dis[dis < allow_dis] = 0
            loss = torch.linalg.norm(dis, 2, -1).mean() * cur_lw

            loss_2 = label["eye3d"][:, -1] - pred["center"][:, -1]
            loss_2[loss_2 < allow_dis] = 0
            loss2 = loss_2.mean() * cur_lw

            ret = {"eye3d_center_dis": loss, "eye3d_center_z_dis": loss2}
        return ret

    def forward(self, data: Dict):
        pred = data["pred"]
        label = data["label"]

        loss = OrderedDict()
        loss.update(self.params_consistency_loss(pred, self.loss_weights))
        loss.update(self.regularization_loss(pred, self.loss_weights))
        loss.update(self.in_range_loss(pred, self.loss_weights))
        loss.update(self.pupil_distance_loss(pred, self.loss_weights))
        loss.update(self.angle_loss(pred, label, self.loss_weights))
        loss.update(self.gaze_point_loss(pred, label, self.loss_weights))
        loss.update(self.corneal_centers_distance(pred, self.loss_weights))
        loss.update(self.eye3d_center_dis_loss(pred, label, self.loss_weights))

        total_loss = 0
        for _, v in loss.items():
            total_loss += v
        loss["total_loss"] = total_loss
        return loss
