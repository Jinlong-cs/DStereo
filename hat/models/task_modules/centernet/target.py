import copy
from math import sqrt
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["CenterNetTarget"]


@OBJECT_REGISTRY.register
class CenterNetTarget(nn.Module):
    """CenterNetTarget.

    Args:
        num_classes: Number of categories excluding the background
            category.
        strides2levels: A stride to levels of feature map dict.
        in_strides: The stride of used feature maps from neck.
        img_shape: Image shape in hwc/hw format.
        max_num_targets: Max number of targets in one sample.
        use_ignore: A flag too control whether to use the ignored
            instances in one sample as target.
        gaussian_radius_alpha: This value decide the gaussian radius
            of the weight mask inside the instance bboxes.
    """

    def __init__(
        self,
        num_classes: int,
        strides2levels: Dict[int, int],
        in_strides: int,
        img_shape: Tuple[int, int],
        max_num_targets: int = 200,
        use_ignore: bool = False,
        gaussian_radius_alpha: float = 0.2,
    ):
        super(CenterNetTarget, self).__init__()
        self.num_classes = num_classes
        self.in_strides = in_strides
        self.img_shape = img_shape
        self.strides2levels = strides2levels
        self.max_num_targets = max_num_targets
        self.use_ignore = use_ignore
        self.gaussian_radius_alpha = gaussian_radius_alpha

    def forward(
        self, label: Dict[str, Any], feats: Tuple[torch.Tensor, ...]
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """Generate regression and classification targets in batched images.

        Args:
            label: A dict contains ground truth of each bboxes and information
                of images.
            feats: Feature map shape with value [B, _, H, W].

        Returns:
            The float value is avg_factor, the dict has components below:
                center_heatmap_target: Targets of center heatmap,
                shape (B, num_classes, H, W).
                wh_target: Targets of wh predict, shape (B, 2, H, W).
                offset_target: Targets of offset predict, shape (B, 2, H, W).
                wh_offset_target_weight: Weights of wh and offset predict,
                shape (B, 2, H, W).
                valid_classes_list: Valid class ids for each instance.
                bbox_target: Decoded bboxes of each grid cell.
                shape (B, 4, H, W).
                heatmap_hard_mask: Loss mask for hard instances.
                shape (B, 1, H, W).

        """

        gt_bboxes = label["gt_bboxes"]
        gt_classes = label["gt_classes"]
        gt_labels = label.get("gt_labels", None)
        bs = len(gt_bboxes)
        img_h, img_w = self.img_shape[:2]
        feat_h = img_h // self.in_strides
        feat_w = img_w // self.in_strides
        if self.use_ignore:
            ig_bboxes = label["ig_bboxes"]

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w]
        )
        heatmap_hard_mask = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w]
        )
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 1, feat_h, feat_w]
        )
        bbox_target = gt_bboxes[-1].new_zeros([bs, 4, feat_h, feat_w])
        ig_bboxes_mask = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w]
        )

        valid_classes_list = None
        if gt_labels is not None:
            valid_classes_list = []
            for batch_id in range(bs):
                # fix single-class labeled data in multiclass head loss
                valid_classes = []
                for cls in gt_labels[batch_id].unique():
                    valid_classes.append(cls.item())
                valid_classes_list.append(valid_classes)

        for batch_id in range(bs):
            if gt_bboxes[batch_id].numel() == 0:
                continue
            gt_bbox = gt_bboxes[batch_id]
            gt_class = gt_classes[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)
            if self.use_ignore:
                ig_bbox = ig_bboxes[batch_id]
                for igb in ig_bbox:
                    ig_regions_x = torch.round(igb[[0, 2]] * width_ratio).to(
                        int
                    )
                    ig_regions_y = torch.round(igb[[1, 3]] * height_ratio).to(
                        int
                    )
                    ig_bboxes_mask[
                        batch_id,
                        valid_classes_list[batch_id],
                        ig_regions_y[0] : ig_regions_y[1],
                        ig_regions_x[0] : ig_regions_x[1],
                    ] = 1

            # process hard
            for j in range(len(gt_centers)):
                ind = gt_class[j]
                regions_x = torch.round(gt_bbox[j][[0, 2]] * width_ratio).to(
                    int
                )
                regions_y = torch.round(gt_bbox[j][[1, 3]] * height_ratio).to(
                    int
                )
                if ind >= 0:
                    continue
                real_ind = int(-ind - 1)  # recover real class
                ig_bboxes_mask[
                    batch_id,
                    real_ind,
                    regions_y[0] : regions_y[1],
                    regions_x[0] : regions_x[1],
                ] = 1

            # process normal
            for j, ct in enumerate(gt_centers):
                ind = gt_class[j]
                if ind < 0:
                    continue
                ctx_int, cty_int = ct.int()

                gt_bbbox_j = copy.deepcopy(gt_bbox[j])
                gt_bbbox_j[[0, 2]] *= width_ratio
                gt_bbbox_j[[1, 3]] *= height_ratio
                gt_bbbox_j[:2] = torch.floor(gt_bbbox_j[:2])
                gt_bbbox_j[2:] = torch.ceil(gt_bbbox_j[2:])
                gt_bbbox_j_int = gt_bbbox_j.to(int)
                ig_bboxes_mask[
                    batch_id,
                    valid_classes_list[batch_id],
                    gt_bbbox_j_int[1] : gt_bbbox_j_int[3],
                    gt_bbbox_j_int[0] : gt_bbbox_j_int[2],
                ] = 0

                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius_v2(
                    [scale_box_w, scale_box_h],
                    alpha=self.gaussian_radius_alpha,
                )
                ind = ind.type(torch.int64)
                center_heatmap_target[batch_id, ind] = gen_gaussian_target(
                    center_heatmap_target[batch_id, ind],
                    [ctx_int, cty_int],
                    radius,
                )

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                bbox_target[batch_id, :, cty_int, ctx_int] = torch.Tensor(
                    [
                        ctx - scale_box_w / 2,
                        cty - scale_box_h / 2,
                        ctx + scale_box_w / 2,
                        cty + scale_box_h / 2,
                    ]
                )

                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = {
            "center_heatmap_target": center_heatmap_target,
            "heatmap_hard_mask": heatmap_hard_mask,
            "wh_target": wh_target,
            "offset_target": offset_target,
            "wh_offset_target_weight": wh_offset_target_weight,
            "valid_classes_list": valid_classes_list,
            "bbox_target": bbox_target,
        }
        if self.use_ignore:
            target_result["ig_bboxes_mask"] = ig_bboxes_mask
        return target_result, avg_factor


def gaussian_radius(
    det_size: Tuple[int], min_overlap: float
) -> Tuple[int, int]:
    """Orignal gaussian_radius calculation of centernet.

    To ensure IoU of generated box and gt box is larger than ``min_overlap``:

    Args:
        det_size: Shape of object.
        min_overlap: Min IoU with ground truth for boxes.

    Returns:
        radius: Radius of gaussian kernel.
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    r = min(r1, r2, r3)
    return (r, r)


def gaussian_radius_v2(det_size: Tuple[int, int], alpha: float):
    """SimpleGaussianRadius.

    Refer to https://arxiv.org/abs/1909.00700.

    Args:
        det_size: Shape of target.
        alpha: a scale ratio.

    Returns:
        radius (Tuple[int]): Radius of gaussian kernel in h, w.
    """
    return (max(det_size[0] / 2 * alpha, 1), max(det_size[1] / 2 * alpha, 1))


def gen_gaussian_target(
    heatmap: torch.Tensor,
    center: Tuple[int, int],
    radius: Tuple[int, int],
    k: int = 1,
) -> torch.Tensor:
    """Generate 2D gaussian heatmap.

    Args:
        heatmap: Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center: Coord of gaussian kernel's center.
        radius: Radius of gaussian kernel.
        k: Coefficient of gaussian kernel. Default is 1.

    Returns:
        out_heatmap: Updated heatmap covered by gaussian kernel.
    """
    diameter = [2 * r + 1 for r in radius]
    sigma = [d / 6 for d in diameter]
    gaussian_kernel = gaussian2D(
        radius, sigma=sigma, dtype=heatmap.dtype, device=heatmap.device
    )

    x, y = center

    height, width = heatmap.shape[:2]
    rx, ry = int(radius[0]), int(radius[1])

    left, right = min(x, rx), min(width - x, rx + 1)
    top, bottom = min(y, ry), min(height - y, ry + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian_kernel[
        ry - top : ry + bottom, rx - left : rx + right
    ]
    out_heatmap = heatmap
    torch.max(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top : y + bottom, x - left : x + right],
    )
    return out_heatmap


def gaussian2D(
    radius: Tuple[int, int],
    sigma: Tuple[int, int],
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cpu",
) -> torch.Tensor:
    """Generate 2D gaussian kernel.

    Args:
        radius: Radius of gaussian kernel in x and y direction.
        sigma: Sigma of gaussian function in x and y direction.
        dtype: Dtype of gaussian tensor. Default is torch.float32.
        device: Device of gaussian tensor. Default is 'cpu'.

    Returns:
        h: Gaussian kernel with a
            ``(2 * radius + 1) * (2 * radius + 1)`` shape.
    """
    rx, ry = list(map(int, radius))
    x = torch.arange(-rx, rx + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(-ry, ry + 1, dtype=dtype, device=device).view(-1, 1)

    s1, s2 = sigma
    h = (-(x * x / s1 ** 2 + y * y / s2 ** 2) / 2).exp()

    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h
