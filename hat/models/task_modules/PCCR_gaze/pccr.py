import logging

import torch
import torch.nn as nn

from hat.core.gaze.transform import (
    pitchyaw_to_unit_vector,
    unit_vector_to_pitchyaw,
)
from hat.models.task_modules.PCCR_gaze.screencoord import ScreenCoordinate
from hat.registry import OBJECT_REGISTRY

__all__ = ["PCCRProcess"]

logger = logging.getLogger(__name__)

KQ_NORM = 1


@OBJECT_REGISTRY.register
class PCCRProcess(nn.Module):
    def __init__(self):
        super().__init__()
        self.screen = ScreenCoordinate()

    def calc_center_corneal(self, kq, glint, light, R, cam_o):
        # 已知在 z=-1 平面的反光点 $$glint$$, 相机光心$$o$$, 角膜反光点为$$q$$
        # 则反光点在相机坐标系下的3d位置:
        q = cam_o + kq * (cam_o - glint)
        # 已知光源的位置$$light$$, 则入射光线的反方向为$$iq=light-q$$, 反射光线方向为$$oq=o-q$$
        # 在角膜上q点的入射光单位向量
        iq_norm = (light - q) / torch.linalg.norm(
            light - q, ord=2, dim=-1, keepdim=True
        )
        # 在角膜上q点的反射光单位向量
        oq_norm = (cam_o - q) / torch.linalg.norm(
            cam_o - q, ord=2, dim=-1, keepdim=True
        )
        # 由于角膜上的点$$q$$与角膜中心$$c$$的距离为角膜半径$$R$$
        # 所以角膜中心:
        center_corneal = q - R * (iq_norm + oq_norm) / torch.linalg.norm(
            iq_norm + oq_norm, ord=2, dim=-1, keepdim=True
        )
        return center_corneal

    def calc_pupil_rb_lb(
        self, pupil_boundary_ccs, center_corneal, R, cam_o, pb_real_mask
    ):
        # 计算角膜折射点
        # 根据式(3.48)，先求解比例系数k_b,jk，再根据k_b,jk求解角膜表面折射点r_b,jk
        # 求解比例系数k_b,jk
        OC = cam_o - center_corneal  # 角膜中心c指向相机光心o, imgn,3
        OV = (
            cam_o.reshape(-1, 1, 3) - pupil_boundary_ccs
        )  # 瞳孔边缘成像点v指向相机光心o, imgn, pbn, 3
        OCV = torch.einsum("bij,bj->bi", [OV, OC])  # imgn, pbn

        # tmp为式(3.48)根号下内容
        tmp = torch.pow(OCV, 2) - torch.pow(
            torch.linalg.norm(OV, 2, dim=-1), 2.0  # imgn, pbn
        ) * (
            torch.pow(
                torch.linalg.norm(OC, 2, dim=-1, keepdim=True), 2.0
            )  # imgn, 1
            - torch.pow(R, 2.0)  # imgn, 1
        )  # imgn, pbn

        # 求根号需要大于0
        valid_mask = tmp > 0  # imgn, pbn
        if pb_real_mask is not None:
            valid_mask = valid_mask * pb_real_mask
        if valid_mask.sum(-1).min() < 1:
            logger.warning("img in batch has no valid pupil boundary point.")
            # import pdb;pdb.set_trace()
        tmp = torch.clip(tmp, 0)  # imgn, pbn

        # k_b,jk的分子
        krb_fz = -OCV - torch.sqrt(tmp)  # imgn, pbn
        # k_b,jk的分母
        krb_fm = torch.pow(torch.linalg.norm(OV, 2, dim=-1), 2.0)  # imgn, pbn
        # 式(3.48)最终的k_b,jk
        krb = krb_fz / krb_fm  # imgn, pbn

        # 求解角膜表面折射点r_b,jk, 式(3.48)
        rb = (
            cam_o.reshape(-1, 1, 3)
            + (cam_o.reshape(-1, 1, 3) - pupil_boundary_ccs) * krb[..., None]
        )  # imgn, pbn, 3

        # 计算瞳孔边缘点 3D 位置，需要知道两个信息，一是瞳孔边缘点的入射方向，二是
        # 这个入射方向方向可以通过折射定律计算得到
        # 式(3.49) 瞳孔边界点的入射光线方向上的单位矢量
        tmp1 = (
            pupil_boundary_ccs - cam_o.reshape(-1, 1, 3)
        ) / torch.linalg.norm(
            pupil_boundary_ccs - cam_o.reshape(-1, 1, 3),
            2,
            axis=-1,
            keepdims=True,
        )  # imgn, pbn, 3
        # 式(3.51) 折射点垂直于角膜表面的单位向量
        tmp2 = (rb - center_corneal.reshape(-1, 1, 3)) / R.reshape(
            -1, 1, 1
        )  # imgn, pbn, 3
        # 式(3.52) 瞳孔边界点的入射光线方向上的单位矢量
        n1 = 1.3375
        n2 = 1
        n1dn2_sq = (n1 / n2) ** 2
        lb = (
            n2
            / n1
            * (
                (
                    torch.einsum("bij, bij->bi", tmp1, tmp2)  # imgn, pbn
                    - torch.sqrt(
                        n1dn2_sq
                        - 1
                        + torch.pow(
                            torch.einsum("bij, bij->bi", tmp1, tmp2), 2.0
                        )  # imgn, pbn
                    )
                )[
                    ..., None
                ]  # imgn, pbn, 1
                * tmp2
                - tmp1
            )
        )  # imgn, pbn, 3
        return rb, lb, valid_mask

    def calc_pupil_points(
        self, list_rb, list_lb, center_corneal, K, pitch, yaw, angle_rad=True
    ):
        # 式2.26, 光轴方向
        cor_x, cor_y, cor_z = pitchyaw_to_unit_vector(
            pitch, yaw, angle_rad=angle_rad
        )
        w = torch.cat([cor_x, cor_y, cor_z], -1)  # imgn, 3

        # 式(2.28), 瞳孔中心p
        p = center_corneal + K * w  # imgn, 3
        # 式(3.55)
        kpb = (
            K
            - torch.einsum(
                "bj,bij->bi", [w, list_rb - center_corneal.reshape(-1, 1, 3)]
            )  # imgn, pbn
        ) / (
            torch.einsum("bj,bij->bi", [w, list_lb])  # imgn, pbn
        )  # imgn, pbn
        # 式(3.53)
        pb = list_rb + kpb[..., None] * list_lb  # imgn, pbn, 3

        return pb, p

    def optical_to_visual_axis(self, pitch, yaw, R_opt2vis, angle_rad=True):
        R_c2opt = self.get_optical_to_visual_rot_mat(
            pitch, yaw, angle_rad=angle_rad
        )
        R_c2vis = R_opt2vis @ R_c2opt  # imgn,3,3

        R_c2vis_inv = torch.linalg.inv(R_c2vis)

        visual_vector = -R_c2vis_inv[:, :, 2]  # imgn,3

        visual_vector_norm = visual_vector / torch.linalg.norm(
            visual_vector, 2, dim=-1, keepdim=True
        )  # imgn,3

        gaze_angle = unit_vector_to_pitchyaw(
            visual_vector_norm, angle_rad=False
        )  # degree

        return gaze_angle, visual_vector_norm

    def get_optical_to_visual_rot_mat(self, alpha, beta, angle_rad=True):
        if not angle_rad:
            alpha = alpha * torch.pi / 180.0
            beta = beta * torch.pi / 180.0

        cos_neg_alpha = torch.cos(-alpha)
        sin_neg_alpha = torch.sin(-alpha)
        cos_beta = torch.cos(beta)
        sin_beta = torch.sin(beta)
        zeros = torch.zeros_like(alpha).to(alpha.device)
        ones = torch.ones_like(alpha).to(alpha.device)

        R_pitch_r1 = torch.stack([ones, zeros, zeros], dim=2)  # imgn, 1, 3
        R_pitch_r2 = torch.stack(
            [zeros, cos_beta, sin_beta], dim=2
        )  # imgn, 1, 3
        R_pitch_r3 = torch.stack(
            [zeros, -sin_beta, cos_beta], dim=2
        )  # imgn, 1, 3
        R_pitch = torch.cat(
            [R_pitch_r1, R_pitch_r2, R_pitch_r3], dim=1
        )  # imgn, 3, 3

        R_yaw_r1 = torch.stack(
            [cos_neg_alpha, zeros, -sin_neg_alpha], dim=2
        )  # imgn, 1, 3
        R_yaw_r2 = torch.stack([zeros, ones, zeros], dim=2)  # imgn, 1, 3
        R_yaw_r3 = torch.stack(
            [sin_neg_alpha, zeros, cos_neg_alpha], dim=2
        )  # imgn, 1, 3
        R_yaw = torch.cat([R_yaw_r1, R_yaw_r2, R_yaw_r3], dim=1)  # imgn, 3, 3

        R_opt2vis = R_pitch @ R_yaw
        return R_opt2vis

    def forward(self, data):
        glint_ccs_list = data["glint_ccs_list"]  # imgn, 2, 3
        pupil_boundary_ccs = data["pupil_boundary_ccs"]  # imgn, pbn, 3
        kq_result = data["kq_result"] / KQ_NORM  # imgn, 2
        light_left = data["light_left"]  # imgn, 3
        light_right = data["light_right"]  # imgn, 3
        cam_o = data["cam_o"]  # imgn, 3
        R = data["R"]  # imgn, 1
        K = data["K"]  # imgn, 1
        alpha = data["alpha"]  # imgn, 1
        beta = data["beta"]  # imgn, 1
        pitch = data["pitch"]  # imgn, 1
        yaw = data["yaw"]  # imgn, 1

        pccr_result = {}
        # 计算视轴 alpha, beta, pitch, yaw都是 deg
        R_opt2vis = self.get_optical_to_visual_rot_mat(
            alpha, beta, angle_rad=False
        )

        gaze_angle, visual_vector_norm = self.optical_to_visual_axis(
            pitch, yaw, R_opt2vis, angle_rad=False
        )
        pccr_result["angle"] = gaze_angle

        # calculate corneal center求解两个反光点下的两个角膜中心
        center1 = self.calc_center_corneal(
            kq_result[:, 0:1], glint_ccs_list[:, 0, :], light_left, R, cam_o
        )
        center2 = self.calc_center_corneal(
            kq_result[:, 1:2], glint_ccs_list[:, 1, :], light_right, R, cam_o
        )
        center = (center1 + center2) / 2.0
        pccr_result["center_1"] = center1
        pccr_result["center_2"] = center2
        pccr_result["center"] = center

        # calc screen coords计算视点
        pixel_width = data["pixel_width"]  # imgn,1
        pixel_height = data["pixel_height"]  # imgn,1
        screen_lt = data["scr_left_top_3d"]  # imgn,3
        screen_lb = data["scr_left_bottom_3d"]  # imgn,3
        screen_rb = data["scr_right_bottom_3d"]  # imgn,3
        screen_coords = self.screen(
            gaze=visual_vector_norm,
            origin=center,
            lb=screen_lb,
            lt=screen_lt,
            rb=screen_rb,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
        )
        pccr_result["screen_coords"] = screen_coords

        # calculate boundary of pupil计算瞳孔边缘
        # 瞳孔边缘点是从瞳孔边缘开始的光线, 经过角膜折射, 经过相机成像, 投影到图像平面
        # 计算瞳孔边缘分为两步，一个是计算角膜折射点
        # 计算角膜表面折射点rb，顺便把瞳孔边缘点的入射方向lb也一起算出来
        rb, lb, valid_mask = self.calc_pupil_rb_lb(
            pupil_boundary_ccs, center, R, cam_o, data.get("pb_real_mask")
        )
        # 二是计算瞳孔边缘点的3D位置
        # nn预测光轴方向，光轴是垂直于瞳孔平面的
        pb, p = self.calc_pupil_points(
            rb, lb, center, K, pitch, yaw, angle_rad=False
        )

        pccr_result["valid_mask"] = valid_mask
        pccr_result["pb"] = pb
        pccr_result["p"] = p

        return pccr_result
