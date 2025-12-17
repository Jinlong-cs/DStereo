import math

import numpy as np
import torch

from hat.registry import OBJECT_REGISTRY

__all__ = [
    "OvalHmTargetGenerator",
]


@OBJECT_REGISTRY.register
class OvalHmTargetGenerator(object):
    """Generate oval/normal heatmap and offset for toll gate.

    Args:
        img_scale: Images scales for generating hm GT.
        oval: Whether to use oval heatmap.
        sigma: Sigma of gaussian distribution. Defaults to 2.
        encode_lmks: Number of heatmaps.
        ng_weights: loss weights for negative pixels.
        stride: Down stride of heatmap.
        dynamic_sigma: Whether to use dynamic sigma. Defaults to False.

    """

    def __init__(
        self,
        img_scale: tuple = None,
        oval: bool = False,
        sigma: float = 2.0,
        encode_lmks: int = 2,
        ng_weights: float = 1.0,
        stride: int = 4,
        dynamic_sigma: bool = False,
    ):
        self.sigma = sigma
        self.oval = oval
        self.encode_lmks = encode_lmks
        self.ng_weights = ng_weights
        self.stride = stride
        self.dynamic_sigma = dynamic_sigma
        self.h, self.w = [s // self.stride for s in list(img_scale)]

    def preprocessLine_Pole_lmks(self, lines: dict):
        for line, data in lines.items():
            if len(data.shape) == 2:
                data = data[np.newaxis, :]
            lines[line] = torch.tensor(data)

    def _normal_encode(self, landmark: np.ndarray, tmp_size: float):
        target = np.zeros((self.h, self.w), dtype=np.float32)
        mu_x, mu_y, _ = [int(_ / self.stride) for _ in landmark]
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= self.w or ul[1] >= self.h or br[0] < 0 or br[1] < 0:
            return target
        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], self.w) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], self.h) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], self.w)
        img_y = max(0, ul[1]), min(br[1], self.h)

        target[img_y[0] : img_y[1], img_x[0] : img_x[1]] = self.kernel[
            g_y[0] : g_y[1], g_x[0] : g_x[1]
        ]
        return target

    def heatmapGenerator(self, top_pt: np.ndarray, down_pt: np.ndarray):
        num_joints = len(top_pt)
        heatmap = np.zeros(
            (self.encode_lmks, max(1, num_joints), self.h, self.w),
            dtype=np.float32,
        )
        offset_labels = np.zeros((2, self.h, self.w), dtype=np.float32)
        heatmap_weights = np.zeros(
            (self.encode_lmks, self.h, self.w), dtype=np.float32
        )
        offset_weights = np.zeros((2, self.h, self.w), dtype=np.float32)
        pt = [
            math.sqrt(
                (top_pt[i][0] - down_pt[i][0]) ** 2
                + (top_pt[i][1] - down_pt[i][1]) ** 2
            )
            for i in range(num_joints)
        ]
        sig = [
            min(3, (top_pt[i][2] + down_pt[i][2]) / 10 / 2)
            for i in range(num_joints)
        ]
        sinn = [
            (top_pt[i][1] - down_pt[i][1]) / pt[i] for i in range(num_joints)
        ]
        coss = [
            (top_pt[i][0] - down_pt[i][0]) / pt[i] for i in range(num_joints)
        ]
        joints = np.array([top_pt, down_pt])

        for index in range(self.encode_lmks):
            for joint_id in range(num_joints):
                sigma = sig[joint_id] if self.dynamic_sigma else self.sigma
                tmp_size = sigma * 3
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                if self.oval:
                    k1 = (x - x0) * coss[joint_id] + (y - y0) * sinn[joint_id]
                    k2 = (x0 - x) * sinn[joint_id] + (y - y0) * coss[joint_id]
                    self.kernel = np.exp(
                        -(
                            (k1 ** 2) / (2 * (sigma * 1.5) ** 2)
                            + (k2 ** 2) / (2 * sigma ** 2)
                        )
                    )
                else:
                    self.kernel = np.exp(
                        -((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)
                    )
                heatmap[index][joint_id] = self._normal_encode(
                    joints[index][joint_id], tmp_size
                )
                mu_x = int((joints[index][joint_id][0]) / self.stride)
                mu_y = int((joints[index][joint_id][1]) / self.stride)
                # second
                if index == 0:
                    # down point
                    off_x = (down_pt[joint_id][0]) / self.stride
                    off_y = (down_pt[joint_id][1]) / self.stride
                    point_map = heatmap[index][joint_id].copy()
                    center_point_index = [mu_y, mu_x]
                    # center_point_index = np.where(point_map == 1)
                    offset_add_x = np.zeros(point_map.shape)
                    offset_add_y = np.zeros(point_map.shape)
                    point_map[point_map > 0.1] = 1
                    center_points_index = np.where(point_map == 1)
                    for i, x in enumerate(center_points_index[0]):
                        offset_add_x[x][center_points_index[1][i]] = (
                            center_point_index[0] - x
                        )
                    for i, y in enumerate(center_points_index[1]):
                        offset_add_y[center_points_index[0][i]][y] = (
                            center_point_index[1] - y
                        )

                    offset_labels[0] += (
                        offset_add_x + np.round(point_map) * (off_y - mu_y)
                    ) / (self.h / 2)
                    offset_labels[1] += (
                        offset_add_y + np.round(point_map) * (off_x - mu_x)
                    ) / (self.w / 2)
                    offset_weights += np.round(point_map)
        offset_weights[offset_weights > 1] = 1.0

        heatmap = np.max(heatmap, axis=1)
        heatmap = np.clip(heatmap, 0, 1)
        heatmap_weights[:2][heatmap == 0] = self.ng_weights
        heatmap_weights[:2][heatmap > 0] = 1
        return {
            "gt_heatmap": heatmap,
            "gt_offset": offset_labels,
            "gt_offset_weight": offset_weights,
            "gt_heatmap_weight": heatmap_weights,
        }

    def __call__(self, data: dict):
        if len(data["gt_lines"]) != 0:
            joints_top, joints_down = data["gt_lines"]
        else:
            joints_top, joints_down = [], []
        out = self.heatmapGenerator(joints_top, joints_down)
        self.preprocessLine_Pole_lmks(out)
        img = data["img"].copy()
        out["img"] = np.transpose(img, (2, 0, 1))
        out["img_name"] = data["img_name"]
        return out
