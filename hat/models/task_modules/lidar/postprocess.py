# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__file__)

__all__ = ["AfdetPredict", "AfsegPredict"]


def maxpool_nms(x: torch.Tensor, k: int = 3):
    """Use op maxpool do nms.

    Args:
        k:the size of maxpool.
    """
    pad = math.ceil((k - 1) / 2)
    maxp = torch.nn.functional.max_pool2d(
        x, (k, k), stride=1, padding=(pad, pad)
    )
    _, _, H, W = x.shape
    maxp = maxp[:, :, :H, :W]
    keep = (maxp == x).float()
    return x * keep


class _Sigmoid_clamp(nn.Module):
    def __init__(self, min: float = 1e-4, max: float = 1 - 1e-4):
        """Calculate input with sigmoid and clamp.

        Args:
            min:min value of clamp.
            max:max value of clamp.
        Returns:
            y:output after sigmoid and clamp.
        """
        super().__init__()

        self.min = min
        self.max = max

    def forward(self, x):
        temp = torch.sigmoid(x)
        y = torch.clamp(temp, min=self.min, max=self.max)
        return y


@OBJECT_REGISTRY.register_module
class AfdetPredict(nn.Module):
    """Prediction procedure for anchor free det head."""

    def __init__(
        self,
        num_classes: List[int],
        maxpool_dict: Dict,
        with_hm_pred: bool = False,
        use_bev_format: bool = False,
    ):
        """Init AfdetPredict.

        Args:
            num_classes: number of classes for each task.
            maxpool_dict: maxpool_nms parameters.
            with_hm_pred: whether to return heatmap prediction.
            use_bev_format: whether to return bev format prediction.
        """

        super(AfdetPredict, self).__init__()
        self.num_classes = num_classes
        self.post_processing = AfdetPostprocess(maxpool_dict)
        self.with_hm_pred = with_hm_pred
        self.use_bev_format = use_bev_format

    def forward(
        self,
        example: Dict,
        src_preds_dicts: Dict,
        pc_range: List[float],
        alpha: List[float],
        fb_head_cfg: Dict = None,
        per_class_nms: bool = False,
        split_hm_head: bool = False,
        return_box_preds_all: bool = False,
    ):
        """Decode, nms, then return the detection result.

        Additionaly support double flip testing.
        """
        # get loss info
        rets = []
        box_preds_all = []

        # 复制tensor,防止predict对feature修改后对后续loss计算使用的feature产生影响
        preds_dicts = []
        src_preds_dicts_tmp = {}
        for k, v in src_preds_dicts.items():
            src_preds_dicts_tmp[k] = v.clone()
        preds_dicts.append(src_preds_dicts_tmp)

        for task_id, preds_dict in enumerate(preds_dicts):
            # convert N C H W to N H W C
            if "rhd" in preds_dict:
                (
                    preds_dict["reg"],
                    preds_dict["height"],
                    preds_dict["dim"],
                ) = preds_dict["rhd"].split([2, 1, 3], dim=1)

            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()

            if split_hm_head:
                hms = [preds_dict[k] for k in self.hm_keys]
                preds_dict["hm"] = torch.cat(hms, dim=1)

            batch_hm = torch.sigmoid(preds_dict["hm"])
            batch_dim = torch.exp(preds_dict["dim"])
            batch_rots = preds_dict["rot"][..., 0:1]
            batch_rotc = preds_dict["rot"][..., 1:2]
            batch_reg = preds_dict["reg"]
            batch_hei = preds_dict["height"]
            batch_rot = torch.atan2(batch_rots, batch_rotc)
            batch, H, W, num_cls = batch_hm.size()
            feature_resolution = (
                (pc_range[3] - pc_range[0]) / H,
                (pc_range[4] - pc_range[1]) / W,
            )
            batch_iou = None
            if "iou" in preds_dict:
                batch_iou = preds_dict["iou"]
                batch_iou = batch_iou.reshape(batch, H * W, 1)
            batch_fb = None
            if "fb" in preds_dict:
                batch_fb = torch.sigmoid(preds_dict["fb"])
                batch_fb = batch_fb.reshape(
                    batch, H * W, fb_head_cfg.fb_channel
                )
                if fb_head_cfg.with_bg:
                    batch_fb = batch_fb[:, :, 1:]

            batch_reg = batch_reg.reshape(batch, H * W, 2)
            batch_hei = batch_hei.reshape(batch, H * W, 1)

            batch_rot = batch_rot.reshape(batch, H * W, 1)
            batch_dim = batch_dim.reshape(batch, H * W, 3)
            batch_hm = batch_hm.reshape(batch, H * W, num_cls)
            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)
            xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)
            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]
            xs_ = H - ys - 1
            ys_ = W - xs - 1
            xs = xs_ * feature_resolution[0] + pc_range[0]
            ys = ys_ * feature_resolution[1] + pc_range[1]

            if "vel" in preds_dict:
                batch_vel = preds_dict["vel"]
                batch_vel = batch_vel.reshape(batch, H * W, 2)
                batch_box_preds = torch.cat(
                    [xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=2
                )
            else:
                batch_box_preds = torch.cat(
                    [xs, ys, batch_hei, batch_dim, batch_rot], dim=2
                )

            if per_class_nms is not False:
                pass
            else:
                box_preds_all.append(batch_box_preds)
                rets.append(
                    self.post_processing(
                        batch_box_preds,
                        batch_hm,
                        task_id,
                        alpha,
                        batch_iou,
                        H=H,
                        W=W,
                        with_hm_pred=self.with_hm_pred,
                    )
                )

        # Merge branches results
        num_samples = len(rets[0])
        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                else:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
            ret["token"] = example["object_token"][i]
            ret_list.append(ret)

        if self.use_bev_format:
            bev_ret = {}
            bev_ret["predict_bev3d_cls_id"] = torch.stack([ret["label_preds"]])
            bev_ret["predict_bev3d_ct"] = torch.stack(
                [ret["box3d_lidar"][:, 0:2]]
            )
            bev_ret["predict_bev3d_loc_z"] = torch.stack(
                [ret["box3d_lidar"][:, 2]]
            )
            bev_ret["predict_bev3d_dim"] = torch.stack(
                [
                    torch.stack(
                        (
                            ret["box3d_lidar"][:, 5],
                            ret["box3d_lidar"][:, 3],
                            ret["box3d_lidar"][:, 4],
                        ),
                        dim=1,
                    )
                ]
            )  # afdet wlh > bev hwl
            bev_ret["predict_bev3d_rot"] = torch.stack(
                [0 - ret["box3d_lidar"][:, 6] - math.pi / 2]
            )
            bev_ret["predict_bev3d_score"] = torch.stack([ret["scores"]])
            return bev_ret

        if return_box_preds_all:
            return ret_list, box_preds_all
        else:
            return ret_list


class AfdetPostprocess(nn.Module):
    """Post-process module of afdet head. Called during prediction.

    Args:
        maxpool_dict: maxpool_nms parameters.
    """

    def __init__(self, maxpool_dict):
        super(AfdetPostprocess, self).__init__()
        self.maxpool_dict = maxpool_dict

    def forward(
        self,
        batch_box_preds: torch.Tensor,
        batch_hm: torch.Tensor,
        task_id: int,
        alpha: List[float],
        batch_iou: List[float] = None,
        H: int = 960,
        W: int = 1080,
        with_hm_pred: bool = False,
    ):
        batch_size = len(batch_hm)

        prediction_dicts = []

        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]
            num_cls = hm_preds.shape[-1]

            scores, labels = torch.max(hm_preds, dim=-1)

            alpha = alpha[task_id]
            if isinstance(alpha, float):
                alpha = [alpha] * num_cls
            maxpool_kernel_size = self.maxpool_dict["maxpool_kernel_size"][
                task_id
            ]
            if isinstance(maxpool_kernel_size, int):
                maxpool_kernel_size = [maxpool_kernel_size] * num_cls

            selected_scores_cls = []
            selected_boxes_cls = []
            selected_labels_cls = []
            if batch_iou is not None:
                selected_iou_cls = []
            if with_hm_pred:
                selected_hm_pred_cls = []
            for k in range(num_cls):
                scores_cls = scores.clone()
                labels_cls = labels.clone()

                if batch_iou is not None:
                    iou_preds = batch_iou[i].view(-1)
                    iou_preds = (iou_preds + 1) * 0.5
                    iou_preds = torch.clamp(iou_preds, 0, 1.0)
                    scores_cls = torch.pow(
                        torch.pow(iou_preds, alpha[k]) * scores_cls,
                        1 / (alpha[k] + 1),
                    )
                scores_cls = scores_cls.view((1, 1, H, W))
                labels_cls = labels_cls.view((1, 1, H, W))
                scores_cls = maxpool_nms(
                    scores_cls, k=int(maxpool_kernel_size[k])
                )
                scores_cls = scores_cls * (labels_cls == k)
                scores_cls = scores_cls.view(-1)
                topk = self.maxpool_dict.get("topk", 300)[k]
                argsort_inds = torch.argsort(scores_cls, descending=True)[
                    :topk
                ]

                selected_scores_cls.append(scores_cls[argsort_inds])
                selected_boxes_cls.append(box_preds[argsort_inds])
                selected_labels_cls.append(labels[argsort_inds])
                if batch_iou is not None:
                    selected_iou_cls.append(iou_preds[argsort_inds])

                if with_hm_pred:
                    selected_hm_pred_cls.append(hm_preds[argsort_inds])
            if len(selected_boxes_cls) > 0:
                selected_boxes = torch.cat(selected_boxes_cls, dim=0)
                selected_scores = torch.cat(selected_scores_cls, dim=0)
                selected_labels = torch.cat(selected_labels_cls, dim=0)

                if batch_iou is not None:
                    selected_iou = torch.cat(selected_iou_cls, dim=0)
                if with_hm_pred:
                    selected_hm_pred = torch.cat(selected_hm_pred_cls, dim=0)
            prediction_dict = {
                "box3d_lidar": selected_boxes,
                "scores": selected_scores,
                "label_preds": selected_labels,
            }

            if batch_iou is not None:
                prediction_dict.update({"iou_preds": selected_iou})

            if with_hm_pred:
                prediction_dict.update({"hm_preds": selected_hm_pred})
            prediction_dicts.append(prediction_dict)

        return prediction_dicts


class Argmax_qat(torch.nn.Module):
    """Apply argmax of data in pred_dict.

    Args:
        data_name (str): name of data to apply argmax.
        dim (int): the dimension to reduce.
        keepdim (bool): whether the output tensor has dim retained or not.

    """

    def __init__(self, data_name: str, dim: int, keepdim: bool = True):
        super(Argmax_qat, self).__init__()
        self.data_name = data_name
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input, return_prob=False, *args):
        if return_prob:
            result_prob, result = input.max(self.dim, self.keepdim)
            return result, result_prob
        else:
            result = input.argmax(self.dim, self.keepdim)
            return result

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register_module
class AfsegPredict(nn.Module):
    """Prediction procedure for anchor free det head.

    Args:
        hm_key: key name of heatmap feature.
    """

    def __init__(
        self,
        hm_key: str = "map_seg_hm",
    ):
        super(AfsegPredict, self).__init__()
        self.sigmoid = _Sigmoid_clamp()
        self._hm_key = hm_key
        self.argmax = Argmax_qat(hm_key, 1)

    def forward(
        self,
        example: Dict,
        preds_dicts: List[Dict],
        grid: Optional[torch.Tensor] = None,
        is_compile: bool = False,
    ):
        """Decode, then return segmentation result."""
        if is_compile:
            ret_dicts = []
            for _, preds_dict in enumerate(preds_dicts):
                ret_dict = {}
                cls = preds_dict[self._hm_key]
                cls_argmax, cls_prob = self.argmax(cls, return_prob=True)

                ret_dict["cls_prob"] = cls_prob
                ret_dict["cls_argmax"] = cls_argmax

                ret_dicts.append(ret_dict)
            return ret_dicts
        else:
            rets = []
            metas = []
            for _, preds_dict in enumerate(preds_dicts):
                task_ret = []
                batch_size = preds_dict[self._hm_key].shape[0]
                preds_dict[self._hm_key] = self.sigmoid(
                    preds_dict[self._hm_key]
                )
                if "metadata" not in example or len(example["metadata"]) == 0:
                    meta_list = [None] * batch_size
                else:
                    meta_list = example["metadata"]

                metas.append(meta_list)

                for i in range(batch_size):
                    batch_ret = {}

                    batch_ret["map_seg_hm"] = torch.argmax(
                        preds_dict[self._hm_key][i], dim=0
                    )
                    if grid is not None:
                        batch_ret["grid"] = grid

                    task_ret.append(batch_ret)

                rets.append(task_ret)

            # Merge branches results
            num_samples = len(rets[0])
            ret_list = []
            for i in range(num_samples):
                ret = {}
                for k in rets[0][i].keys():
                    if k in ["metadata"]:
                        pass
                    elif k in ["map_seg_hm"]:
                        ret[k] = torch.cat(
                            [ret[i][k].unsqueeze(0) for ret in rets]
                        )
                    else:
                        ret[k] = torch.cat([ret[i][k] for ret in rets])
                ret["metadata"] = metas[0][i]
                ret_list.append(ret)

            return ret_list
