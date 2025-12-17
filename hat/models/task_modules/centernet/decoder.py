from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

try:
    from torchvision.ops.boxes import nms
except ImportError:
    nms = None

__all__ = ["CenterNetDecoder"]


@OBJECT_REGISTRY.register
class CenterNetDecoder(nn.Module):
    """CenterNet decoder.

    Args:
        num_classes: Number of categories excluding the background category.
        img_shape: Image shape in [h, w] format.
        test_cfg: Testing config of CenterNet. Default is None.
            If given, the content should be like a dict below:
            ```
            "topk": 100,
            "local_maximum_kernel": 3,
            "max_per_img": 100,
            "nms": "nms",
            "score_thresholds": [0] * num_classes,
            "iou_thresholds": [0.5] * num_classes,
            ```
        with_nms: If True, do nms before return boxes. Default is False.
        class_map: A dict or list to map bbox output labels to new labels.
    """

    def __init__(
        self,
        num_classes: int,
        img_shape: Optional[Tuple[int, int]] = None,
        test_cfg: Optional[dict] = None,
        with_nms: bool = True,
        class_map: Optional[Union[List, Dict[int, int]]] = None,
    ):
        super(CenterNetDecoder, self).__init__()
        self.num_classes = num_classes
        self.img_shape = img_shape
        self.with_nms = with_nms
        if test_cfg is None:
            test_cfg = {
                "topk": 100,
                "local_maximum_kernel": 3,
                "max_per_img": 100,
                "nms": "nms",
                "score_thresholds": [0] * num_classes,
                "iou_thresholds": [0.5] * num_classes,
            }
        self.test_cfg = test_cfg
        self.class_map = class_map

    def forward(
        self,
        preds: Union[
            Dict[str, torch.Tensor],
            Tuple[torch.Tensor, ...],
            Tuple[Tuple[torch.Tensor, ...]],
        ],
        img_meta: dict,
    ):
        """Decode head outputs in a batch into bbox predictions.

        Args:
            preds: Head output tensors with shape (B, C, H, W).
            img_meta: Meta information of each image. If self.input_shape
                is None, "image_height" and "image_width" must be included.

        Returns:
            A dict containing names that map to:
                pred_bboxes: A list of tensor in shape [n_bboxes, 6].
                    List length is the batch size. 6 indicate
                    [lt_x, lt_y, rb_x, rb_y, conf, cls].
        """
        # -----------------------------------------------------------------------
        if isinstance(preds, dict):
            assert isinstance(preds, OrderedDict)
            center_heatmap_pred, wh_pred, offset_pred, _ = preds.values()
        else:
            assert isinstance(preds, (list, tuple))
            if len(preds) == 1:
                preds = preds[0]
            center_heatmap_pred, wh_pred, offset_pred, _ = preds
        center_heatmap_pred = center_heatmap_pred.sigmoid()
        # -----------------------------------------------------------------------
        results = {}
        batch_bboxes_list = []
        center_heatmap_pred = get_local_maximum(center_heatmap_pred, 3)
        image_shape = (
            self.img_shape
            if self.img_shape is not None
            else (int(img_meta["img_height"]), int(img_meta["img_width"]))
        )
        batch_det_bboxes = self._decode_heatmap(
            center_heatmap_pred,
            wh_pred,
            offset_pred,
            image_shape,
            self.test_cfg.get("topk"),
        )  # [bs, topk, 6] : (tlx, tly, brx, bry, score, label)

        if self.with_nms:
            score_thresh = torch.Tensor(
                self.test_cfg.get("score_thresholds")
            ).to(wh_pred.device)
            for bboxes in batch_det_bboxes:
                bboxes = multiclass_nms(
                    bboxes[:, :4],
                    bboxes[:, -2],
                    bboxes[:, -1],
                    score_thresh,
                    self.test_cfg.get("iou_thresholds"),
                    self.test_cfg.get("nms"),
                )
                if self.class_map is not None:
                    bbox_labels = bboxes[:-1]
                    for ori_id, after_id in enumerate(self.class_map):
                        bboxes[:-1] = torch.where(
                            bbox_labels == ori_id,
                            torch.ones_like(bbox_labels) * after_id,
                            bbox_labels,
                        )
                batch_bboxes_list.append(bboxes)
        results["pred_bboxes"] = batch_bboxes_list
        return results

    def _decode_heatmap(
        self,
        center_heatmap_pred: torch.Tensor,
        wh_pred: torch.Tensor,
        offset_pred: torch.Tensor,
        img_shape: Tuple[int, int],
        k: int = 100,
    ) -> torch.Tensor:
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred: center predict heatmap, with shape
                (1, num_classes, H, W).
            wh_pred: wh predict, with shape (1, 2, H, W).
            offset_pred: offset predict, shape (1, 2, H, W).
            img_shape: image shape in [h, w] format.
            k: Get top k center keypoints from heatmap. Default is 100.

        Returns:
            batch_bboxes: Infomation of bboxes with shape (1, k, 6).
        """

        height, width = center_heatmap_pred.shape[-2:]
        inp_h, inp_w = img_shape[:2]

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k
        )
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        tl_x = (topk_xs - (wh[..., 0] ** 2) / 2) * (inp_w / width)
        tl_y = (topk_ys - (wh[..., 1] ** 2) / 2) * (inp_h / height)
        br_x = (topk_xs + (wh[..., 0] ** 2) / 2) * (inp_w / width)
        br_y = (topk_ys + (wh[..., 1] ** 2) / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)  # 1 K 4
        batch_bboxes = torch.cat(
            (
                batch_bboxes,
                batch_scores[..., None],
                batch_topk_labels[..., None],
            ),  # 1 K 6
            dim=-1,
        )

        return batch_bboxes


def get_local_maximum(heat: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Extract local maximum pixel with given kernel.

    Args:
        heat: Target heatmap.
        kernel: Kernel size of max pooling. Default is 3.

    Returns:
        heat: A heatmap where local maximum pixels maintain its
            own value and other positions are 0.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def get_topk_from_heatmap(
    scores: torch.Tensor, k: int = 20
) -> Tuple[torch.Tensor]:
    """Get top k positions from heatmap.

    Args:
        scores: Target heatmap with shape
            [batch, num_classes, height, width].
        k: Target number. Default is 20.

    Returns:
        tuple: Scores, indexes, categories and coords of
            topk keypoint. Containing following Tensors:

        - topk_scores: Max scores of each topk keypoint.
        - topk_inds: Indexes of each topk keypoint.
        - topk_clses: Categories of each topk keypoint.
        - topk_ys: Y-coord of each topk keypoint.
        - topk_xs: X-coord of each topk keypoint.

    """
    batch, _, height, width = scores.size()
    max_scores, max_score_cls = scores.max(dim=1)
    topk_scores, topk_inds = torch.topk(max_scores.view(batch, -1), k)
    topk_clses = max_score_cls.view(batch, -1).gather(1, topk_inds)
    topk_inds = topk_inds % (height * width)
    topk_ys = torch.div(topk_inds, width, rounding_mode="floor")
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def transpose_and_gather_feat(feat, ind):
    """Transpose and gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.

    Returns:
        feat (Tensor): Transposed and gathered feature.
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat


def gather_feat(feat, ind, mask=None):
    """Gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.
        mask (Tensor | None): Mask of feature map. Default: None.

    Returns:
        feat (Tensor): Gathered feature.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).repeat(1, 1, dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


@require_packages("torchvision")
def multiclass_nms(
    bboxes,
    scores,
    labels,
    score_thresh,
    iou_threshold,
    nms_method,
):
    """NMS for multi-class bboxes.

    Args:
        bboxes (Tensor): shape (n, 4)
        scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thresh (Tensor): score threshold for each class
                shape (num_classes,)
        iou_threshold (List[float]): NMS IoU thresholds for each class
        nms_method (str): nms type, candidate values are ['nms', 'soft_nms'].
    """

    # filter bboxes using score threshold for each class
    score_thresh = (
        score_thresh.repeat(labels.shape[0], 1)
        .gather(-1, labels.unsqueeze(-1).to(int))
        .squeeze(-1)
    )
    valid_mask = scores > score_thresh

    bboxes = torch.masked_select(
        bboxes,
        valid_mask[..., None],
    ).view(-1, 4)

    scores = torch.masked_select(scores, valid_mask)
    labels = torch.masked_select(labels, valid_mask)

    if bboxes.numel() == 0:
        bboxes = bboxes.new_zeros((0, 6))
        return bboxes

    if nms_method == "nms":
        """Modify from torchvision.ops.boxes._batched_nms_vanilla
        do nms using iou thresh for each cls
        """
        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        for class_id in torch.unique(labels):
            curr_indices = torch.where(labels == class_id)[0]
            curr_keep_indices = nms(
                bboxes[curr_indices],
                scores[curr_indices],
                iou_threshold[
                    int(class_id.item())
                ],  # using iou thresh for each cls
            )
            keep_mask[curr_indices[curr_keep_indices]] = True
        keep_indices = torch.where(keep_mask)[0]
        keep = keep_indices[scores[keep_indices].sort(descending=True)[1]]
    else:
        raise NotImplementedError

    bboxes = bboxes[keep]
    scores = scores[keep][:, None]
    labels = labels[keep][:, None]
    det = torch.cat(
        [bboxes, scores, labels], dim=-1
    )  # need to contain lebels for voc validation
    return det
