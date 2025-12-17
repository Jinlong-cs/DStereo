# Copyright (c) Horizon Robotics. All rights reserved.
import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["SuperPSDGlobalTarget", "SuperPSDLocalTarget", "SuperPSDTarget"]


@OBJECT_REGISTRY.register
class SuperPSDTarget(nn.Module):
    """Target module for ipm psd task.

    Args:
        global_target: Global target module for ipm psd task.
        local_target: Local target module for ipm psd task.
        is_val: Is validation stage or not.
    """

    def __init__(
        self,
        global_target: nn.Module = None,
        local_target: nn.Module = None,
        is_val: bool = False,
    ):
        super().__init__()
        self.global_target = global_target
        self.local_target = local_target
        self.is_val = is_val

    def forward(self, label, preds):
        # for psd multi-task post-process
        # in OutputModule post-process, label here is preds
        if self.is_val:
            return {"preds": label}

        if isinstance(label, dict):
            label = label.get("label", None)
        result_dict = {}
        total_target_dict, total_grad_dict = {}, {}
        global_preds, local_preds = preds[0:5], preds[5:]
        global_target = [i[0] for i in label]
        local_target = [i[1] for i in label]
        global_target_dict, global_grad_dict = self.global_target(
            global_preds, global_target
        )
        local_target_dict, local_grad_dict = self.local_target(
            local_preds, local_target
        )
        total_target_dict.update(global_target_dict)
        total_target_dict.update(local_target_dict)
        total_grad_dict.update(global_grad_dict)
        total_grad_dict.update(local_grad_dict)

        result_dict["preds"] = preds
        result_dict["targets"] = [total_target_dict]
        result_dict["grads"] = [total_grad_dict]
        return [result_dict]


@OBJECT_REGISTRY.register
class SuperPSDLocalTarget(nn.Module):
    """Local target module for super psd task.

    Args:
        radius: The radius of guassian kernel.
        feature_size: The size of local feature.
        input_size: The size of input image.
        far_weight: The loss weight of the third and forth corner point.  # noqa
        slot_weight_dict: The loss weight for different slot type.
    """

    def __init__(
        self,
        radius: int,
        feature_size: int = 112,
        input_size: int = 448,
        far_weight: float = 1.0,
        slot_weight_dict: dict = {0: 1.0, 1: 1.0, 2: 1.0, -1: 0.0},  # noqa
    ):
        super().__init__()
        self.radius = radius
        self.feature_size = feature_size
        self.input_size = input_size
        self.far_weight = far_weight
        self.slot_weight_dict = slot_weight_dict

    def gaussian2D(self, radius, sigma=1, dtype=torch.float32, device="cpu"):

        x = torch.arange(-radius, radius + 1, dtype=dtype, device=device).view(
            1, -1
        )
        y = torch.arange(-radius, radius + 1, dtype=dtype, device=device).view(
            -1, 1
        )

        h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()

        h[h < torch.finfo(h.dtype).eps * h.max()] = 0
        return h

    def gen_label_area(self, target_map, center, radius, target):

        device = target_map.device
        x, y = int(center[0]), int(center[1])
        target = torch.tensor(target)
        target_kernel = target.repeat(2 * radius + 1, 2 * radius + 1).to(
            device
        )
        height, width = target_map.shape[:2]
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        masked_target_map = target_map[
            y - top : y + bottom, x - left : x + right
        ]
        masked_kernel = target_kernel[
            radius - top : radius + bottom, radius - left : radius + right
        ]
        out_map = target_map
        torch.max(
            masked_target_map,
            masked_kernel,
            out=out_map[y - top : y + bottom, x - left : x + right],
        )
        return out_map

    def gen_gaussian_target(self, heatmap, center, radius, k=1):
        """Generate 2D gaussian heatmap."""
        diameter = 2 * radius + 1
        gaussian_kernel = self.gaussian2D(
            radius,
            sigma=diameter / 6,
            dtype=heatmap.dtype,
            device=heatmap.device,
        )

        # x, y = center
        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[:2]
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
        masked_gaussian = gaussian_kernel[
            radius - top : radius + bottom, radius - left : radius + right
        ]
        out_heatmap = heatmap
        torch.max(
            masked_heatmap,
            masked_gaussian * k,
            out=out_heatmap[y - top : y + bottom, x - left : x + right],
        )

        return out_heatmap

    def forward(self, local_preds, local_labels):
        grid_stride = self.input_size // self.feature_size
        # local_preds: [batch_size, 24, 112, 112]
        # local_labels: [[[x1, y1, x2, y2, x3, y3, x4, y4,
        #                   p1_vec_x, p1_vec_y, p2_vec_x, p2_vec_y,
        #                   p3_vec_x, p3_vec_y, p4_vec_x, p4_vec_y,
        #                  p1_type, p2_type, p3_type, p4_type, slot_type], []]]
        classification, offset, sline_angle, point_type = local_preds
        classification_obj, offset_obj, sline_angle_obj, point_type_obj = (
            torch.zeros_like(classification),
            torch.zeros_like(offset),
            torch.zeros_like(sline_angle),
            torch.zeros_like(point_type),
        )
        classification_grad, offset_grad, sline_angle_grad, point_type_grad = (
            torch.zeros_like(classification),
            torch.zeros_like(offset),
            torch.zeros_like(sline_angle),
            torch.zeros_like(point_type),
        )
        classification_grad[:, 0:2].fill_(1.0)
        classification_grad[:, 2:].fill_(self.far_weight)
        for batch_idx, local_target in enumerate(local_labels):
            for point_info in local_target:
                for i in range(4):
                    if point_info[i + 16] == 0:
                        # print('skipping sheter points')
                        continue
                    scale_x = point_info[2 * i]
                    scale_y = point_info[2 * i + 1]
                    ct_x = scale_x / grid_stride
                    ct_y = scale_y / grid_stride
                    col, row = int(ct_x), int(ct_y)
                    col = (
                        col
                        if col < self.feature_size
                        else self.feature_size - 1
                    )
                    row = (
                        row
                        if row < self.feature_size
                        else self.feature_size - 1
                    )
                    self.gen_gaussian_target(
                        classification_obj[batch_idx, i],
                        [ct_x, ct_y],
                        self.radius,
                    )
                    offset_obj[batch_idx, 2 * i, row, col] = ct_x - col
                    offset_obj[batch_idx, 2 * i + 1, row, col] = ct_y - row

                    sline_angle_obj[batch_idx, 2 * i, row, col] = point_info[
                        2 * i + 8
                    ]
                    sline_angle_obj[
                        batch_idx, 2 * i + 1, row, col
                    ] = point_info[2 * i + 9]
                    point_type_obj[batch_idx, i, row, col] = point_info[16 + i]
                    # if slot_type is Ignore, this slot shall not be calculated
                    self.gen_label_area(
                        classification_grad[batch_idx, i],
                        [ct_x, ct_y],
                        self.radius,
                        self.slot_weight_dict[point_info[-1]],
                    )
                    if point_info[-1] == -1:
                        continue
                    sline_angle_grad[batch_idx, i : i + 2, row, col].fill_(1)
                    offset_grad[batch_idx, i : i + 2, row, col].fill_(1)
                    point_type_grad[batch_idx, i, row, col].fill_(1)
        local_target_dict = {
            "local_classification_obj": classification_obj,
            "local_offset_obj": offset_obj,
            "local_sline_angle_obj": sline_angle_obj,
            "local_point_type_obj": point_type_obj,
        }
        local_grad_dict = {
            "local_classification_grad": classification_grad,
            "local_offset_grad": offset_grad,
            "local_sline_angle_grad": sline_angle_grad,
            "local_point_type_grad": point_type_grad,
        }
        return local_target_dict, local_grad_dict


@OBJECT_REGISTRY.register
class SuperPSDGlobalTarget(nn.Module):
    """The global target module for super psd task.

    Args:
        input_size: The size of input image.
        feature_size: The size of global feature.
        slot_weight_dict: The loss weight dict for different slot type.
    """

    def __init__(
        self,
        input_size: int,
        feature_size: int,
        slot_weight_dict: dict = {0: 1.0, 1: 1.0, 2: 1.0, -1: 0.0},  # noqa
    ):
        super().__init__()
        self.input_size = input_size
        self.feature_size = feature_size
        self.slot_weight_dict = slot_weight_dict

    def forward(self, global_preds, global_labels):
        grid_stride = self.input_size // self.feature_size
        # global_preds: batch_size*15*28*28
        # global_labels: [[[centerx, centery,
        #               slot_type, slot_vec_x, slot_vec_y, occupancey,
        #               x1, y1, x2, y2, x3, y3, x4, y4], []],[[]]]
        classification, offset, occupancy, slot_type, direction = global_preds
        (
            classification_obj,
            offset_obj,
            occupancy_obj,
            slot_type_obj,
            direction_obj,
        ) = (
            torch.zeros_like(classification),
            torch.zeros_like(offset),
            torch.zeros_like(occupancy),
            torch.zeros_like(slot_type),
            torch.zeros_like(direction),
        )
        (
            classification_grad,
            offset_grad,
            occupancy_grad,
            slot_type_grad,
            direction_grad,
        ) = (
            torch.zeros_like(classification),
            torch.zeros_like(offset),
            torch.zeros_like(occupancy),
            torch.zeros_like(slot_type),
            torch.zeros_like(direction),
        )
        classification_grad.fill_(1.0)
        for batch_idx, global_target in enumerate(global_labels):
            for slot_info in global_target:
                scale_x = slot_info[0]
                scale_y = slot_info[1]
                ct_x = scale_x / grid_stride
                ct_y = scale_y / grid_stride
                col, row = int(ct_x), int(ct_y)
                col = col if col < self.feature_size else self.feature_size - 1
                row = row if row < self.feature_size else self.feature_size - 1
                # confidence regression
                classification_obj[batch_idx, 0, row, col] = 1.0
                for i in range(4):
                    offset_obj[batch_idx, 2 * i, row, col] = (
                        slot_info[2 * i + 6] / grid_stride - col
                    )
                    offset_obj[batch_idx, 2 * i + 1, row, col] = (
                        slot_info[2 * i + 7] / grid_stride - row
                    )
                # point offset regression
                occupancy_obj[batch_idx, 0, row, col] = slot_info[5]
                # occupancey classifiaction
                slot_type_obj[batch_idx, int(slot_info[2]), row, col] = 1
                # slot type classification
                direction_obj[batch_idx, 0, row, col] = slot_info[3]
                direction_obj[batch_idx, 1, row, col] = slot_info[4]
                # direction regression
                # if slot_type is Ignore, this slot shall not be calculated.
                classification_grad[batch_idx, :, row, col].fill_(
                    self.slot_weight_dict[slot_info[2]]
                )
                if slot_info[2] == -1:
                    continue
                offset_grad[batch_idx, :, row, col].fill_(1.0)
                occupancy_grad[batch_idx, :, row, col].fill_(1.0)
                slot_type_grad[batch_idx, :, row, col].fill_(1.0)
                direction_grad[batch_idx, :, row, col].fill_(1.0)
                # assign grad
        global_target_dict = {
            "global_classification_obj": classification_obj,
            "global_offset_obj": offset_obj,
            "global_occupancy_obj": occupancy_obj,
            "global_slot_type_obj": slot_type_obj,
            "global_direction_obj": direction_obj,
        }
        global_grad_dict = {
            "global_classification_grad": classification_grad,
            "global_offset_grad": offset_grad,
            "global_occupancy_grad": occupancy_grad,
            "global_slot_type_grad": slot_type_grad,
            "global_direction_grad": direction_grad,
        }
        return global_target_dict, global_grad_dict
