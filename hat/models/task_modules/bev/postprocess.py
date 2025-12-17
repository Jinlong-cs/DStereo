# Copyright (c) Horizon Robotics. All rights reserved.
import json
import os
import threading
from collections import OrderedDict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from hat.core.box3d_utils import bev3d_let_nms, bev3d_nms
from hat.metrics.bev.online_mapping_utils import (
    get_target_categorys,
    get_vcs_lane,
    pts_format_dict,
)
from hat.models.losses.utils import sigmoid_and_clip
from hat.models.task_modules.bev.utils import do_ct_dist_nms
from hat.models.task_modules.e2e_dynamic.e2e_dynamic_util import decode_psc_rot
from hat.registry import OBJECT_REGISTRY
from .crosspt_postprocess_utils import cal_vcs_pts, nms_pts, predict_preprocess
from .om_postprocess_utils import convert_pred, get_offset, pred_preprocess

__all__ = [
    "ANCBEV3Decoder",
    "ANCBEVDiscreteObjectDecoder",
    "ANCOnlineMappingDecoder",
    "ANCBEVCrosspointDecoder",
    "ANCBevPSDDecoder",
    "ANCBEVParkingrodDecoder",
]


@OBJECT_REGISTRY.register
class ANCBEV3Decoder(torch.nn.Module):
    """The Bev3D Decoder, convert the bev3d output to obj results.

    Args:
        cls_dimension: the average dimension of each category.
        topk: The maximum number of objects in multi-view bev defualt=100.
        use_maxpool: Whether to use maxpool or not, default True.
        max_pool_kernel: The max pooling kernel that is used
            to do the nms, default=9.
        vcs_range: vcs range. order is (bottom,right,top,left).
        bev_3d_out_size: output size of bev3d head.
        num_classes: num of classes.
        nms_setting: Dict contains information about which nms to use,
            it's threshold, and whether to do class-agnostic NMS.
            key "letnms" or "bevnms" is the type of nms, the value is
            it's param, "letnms" and "bevnms" can not use simultaneously,
            "agnostic" specify whether do class-agnostic NMS or not,
            default is False if not specified, "topk_deploy" specify the
            topk for deploy, it's for compile, default is 120 if not specified.
            for instance.

            .. code-block:: none

                {
                    "letnms": [
                        {
                            "p_t": 0.15,
                            "min_t": 4.0,
                            "max_t": 6.0,
                            "radius": 0.4,
                            "e_loc_threshold": 0.1,

                        }

                    ],

                    "agnostic" : True,
                    "topk_deploy": 120,

                }

                # or

                {
                    "bevnms": {
                        "nms_thresh": 0.5,

                    },
                    "agnostic" : False,
                    "topk_deploy": 120,

                }

        add_hm_eps: whether to add eps for heatmap, default False.

            .. note:: since the quantize of heatmap will cause the same score
                of points in the maxpooling kernel region, will resulting high
                score FP objs in eval. Add eps to make them difference.

        eps_channels: number of channels of eps, if None, use num_classes,
            default None.
        use_category_decouple: If True, foreground/background
            classification(1-channel) with category classification(c-channel)
            is used to distinguish the object category.
        score_threshold: score threshold, can be scalar or list of scaler for
            each category.
        use_psc_rot: whether use psc to encode the heading angle.
        N_steps_PSC_rot: nums of steps to decode the heading angle by PSC.
            Refer to
            https://horizonrobotics.feishu.cn/docx/EwYEdysIBodB69xn6Mdc7HkSnif.
        rot_mod_threshold: threshold to filter decoded phase of the heading angle.
        use_dual_freq: wether to use dual frequency for heading angle,
            True not implemented yet.
        use_norm_rot: whether norm rot output to [-1,1].
        roi_filter_info: filter info for roi range, default None, such as.

            .. code-block:: none

                {
                    "roi_vcs_range": (-20, -8.0, 50, 8.0),
                    "roi_score_threshold": [0.16] * class_num,  # should > global score thresh  # noqa

                }

    """

    def __init__(
        self,
        cls_dimension: np.ndarray,
        topk: Optional[int] = 100,
        use_maxpool: bool = True,
        max_pool_kernel: Optional[int] = 3,
        vcs_range: Optional[Sequence] = (-30.0, -51.2, 72.4, 51.2),
        bev_3d_out_size: Optional[Sequence] = (256, 256),
        num_classes: int = 1,
        nms_setting: Dict = None,
        add_hm_eps: bool = False,
        eps_channels: int = None,
        use_category_decouple: bool = False,
        score_threshold: Union[float, List[float]] = 0.0,
        use_psc_rot: bool = False,
        N_steps_PSC_rot: int = None,
        rot_mod_threshold: float = 0.0001,
        use_dual_freq: bool = False,
        use_norm_rot: bool = False,
        roi_filter_info: Mapping = None,
    ):
        super(ANCBEV3Decoder, self).__init__()
        self.topk = topk
        self.use_maxpool = use_maxpool
        self.max_pool_kernel = max_pool_kernel
        self.cls_dimension = cls_dimension
        self.vcs_range = vcs_range
        self.num_classes = num_classes
        self.nms_setting = nms_setting
        self.add_hm_eps = add_hm_eps
        self.use_category_decouple = use_category_decouple
        # TODO: @jili.li unify setting score threshold in decoder,
        # including train_node, val_node
        self.score_threshold = torch.tensor(score_threshold)
        if self.add_hm_eps:
            torch.manual_seed(1)
            if eps_channels is None:
                eps_channels = num_classes
            self.hm_eps = (
                torch.rand((1, eps_channels, *bev_3d_out_size)) * 1e-3
            )  # (bs, cls, h, w)
        self.use_psc_rot = use_psc_rot
        self.N_steps_PSC_rot = N_steps_PSC_rot
        self.rot_mod_threshold = rot_mod_threshold
        self.use_dual_freq = use_dual_freq
        if self.use_psc_rot:
            assert self.N_steps_PSC_rot >= 3
            if self.use_dual_freq:
                raise NotImplementedError
            self.coef_sin = torch.tensor(
                tuple(
                    torch.sin(
                        torch.tensor(2 * k * torch.pi / self.N_steps_PSC_rot)
                    )
                    for k in range(self.N_steps_PSC_rot)
                )
            )
            self.coef_cos = torch.tensor(
                tuple(
                    torch.cos(
                        torch.tensor(2 * k * torch.pi / self.N_steps_PSC_rot)
                    )
                    for k in range(self.N_steps_PSC_rot)
                )
            )
        self.use_norm_rot = use_norm_rot
        self.roi_filter_info = roi_filter_info

    def coord_transform(
        self,
        bev_center: torch.Tensor,
        bev_size: Tuple = (256, 256),
        vcs_range: Tuple = (-30.0, -51.2, 72.4, 51.2),
    ) -> torch.Tensor:
        """Convert the bev image coordinate to vcs coordinate.

        Args:
            bev_center: the predict object center on bev image
                coordinate. shape: [bs, topk, 2], the "2" means image coord
                [x(u),y(v)].
            bev_size: the bev map size. Defaults to (256,256)
            vcs_range: vcs visiable range, (order:
                (bottom,right,up,left)), Defaults to (-30.0, -51.2, 72.4, 51.2)
        Returns:
            vcs center, shape:[bs, topk, 2], the "2" means
                vcs coord [x,y].
        """
        m_perpixel = (
            abs(vcs_range[2] - vcs_range[0]) / bev_size[0],
            abs(vcs_range[3] - vcs_range[1]) / bev_size[1],
        )  # bev coord [y, x]

        vcs_x = vcs_range[2] - (bev_center[:, :, 1]) * m_perpixel[0]
        vcs_y = vcs_range[3] - (bev_center[:, :, 0]) * m_perpixel[1]
        vcs_center = torch.stack([vcs_x, vcs_y], dim=-1)

        return vcs_center

    def forward(
        self,
        pred: Mapping,
        label: Dict = None,
    ) -> Dict:
        outputs = OrderedDict({k: v.clone() for k, v in pred.items()})
        bev3d_hm = sigmoid_and_clip(outputs["bev3d_hm"])
        if self.add_hm_eps:
            bev3d_hm += self.hm_eps.to(bev3d_hm.device)
        # max_pooling nms
        batch_size, _, height, width = bev3d_hm.shape
        if self.use_maxpool:
            kernel = self.max_pool_kernel
            pad = (kernel - 1) // 2
            max_bev3d_hm = F.max_pool2d(
                bev3d_hm, (kernel, kernel), stride=1, padding=(pad, pad)
            )
            pos_mask = max_bev3d_hm == bev3d_hm
            bev3d_hm *= pos_mask.float()
        outputs["bev3d_hm"] = bev3d_hm

        for key, out in outputs.items():
            channel_size = out.shape[1]
            outputs[key] = out.permute(0, 2, 3, 1).contiguous()
            outputs[key] = outputs[key].view(batch_size, -1, channel_size)

        # get the topk indices of heatmap
        bev3d_hm = outputs.pop("bev3d_hm")
        # TODO@zihao.lu unify the name of cls_hm/cls_id
        if "bev3d_cls_hm" in outputs:
            bev3d_cls_hm = outputs.pop("bev3d_cls_hm")
            classes = bev3d_cls_hm.squeeze(dim=-1)
            scores = bev3d_hm.squeeze(dim=-1)
        elif "bev3d_cls_id" in outputs:
            bev3d_cls_id = outputs.pop("bev3d_cls_id")
            scores = bev3d_hm.squeeze(dim=-1)
            classes = bev3d_cls_id.squeeze(dim=-1)
        else:
            scores, classes = bev3d_hm.max(dim=-1)

        score_threshold = self.score_threshold.to(scores.device)
        if score_threshold.numel() > 1:
            scores[scores < score_threshold[classes]] = 0
        else:
            scores[scores < score_threshold] = 0
        outputs["bev3d_cls_id"] = classes

        # use stable sorting
        scores, indices = torch.sort(
            scores, dim=-1, descending=True, stable=True
        )
        scores = scores[..., : self.topk]
        topk_indices = indices[..., : self.topk]

        # get the topk indices of regression items (dim, rot etc.)
        for key, out in outputs.items():
            outputs[key] = torch.cat(
                [_out[_inds][None] for _out, _inds in zip(out, topk_indices)],
                dim=0,
            )

        # convert topk indices value to the coordinate
        outputs["bev3d_score"] = scores
        u, v = topk_indices % width, torch.div(
            topk_indices, width, rounding_mode="floor"
        )
        # bev: u=x, v=y
        bev3d_center = torch.cat(
            [u[:, :, None], v[:, :, None]], axis=-1
        )  # (u,v) = (x,y)

        if "bev3d_occlusion_hm" in outputs:
            decode_occlusion = True
            bev3d_occlusion_id = outputs.pop("bev3d_occlusion_hm")
            bev3d_occlusion_id = bev3d_occlusion_id.squeeze(dim=-1)
            outputs["bev3d_occlusion_id"] = bev3d_occlusion_id
        else:
            decode_occlusion = False

        if "bev3d_ct_offset" in outputs:
            bev3d_ct_offset = outputs.pop("bev3d_ct_offset")
        else:
            bev3d_ct_offset = (
                torch.zeros_like(bev3d_center, dtype=torch.float32) + 0.1
            )
        bev3d_center = bev3d_center + bev3d_ct_offset  # (u,v)

        # convert the bev center in bev coor to vcs coor
        vcs_center = self.coord_transform(
            bev3d_center,
            bev_size=(height, width),
            vcs_range=self.vcs_range,
        )
        outputs["bev3d_ct"] = vcs_center

        # convert the rot to rad value (sin / cos), range=[-pi,pi]
        bev3d_rot = outputs.pop("bev3d_rot")
        if self.use_psc_rot:
            phase = decode_psc_rot(
                bev3d_rot=bev3d_rot,
                coef_sin=self.coef_sin,
                coef_cos=self.coef_cos,
                N_steps_PSC_rot=self.N_steps_PSC_rot,
                use_sigmoid=self.use_norm_rot,
            )
        else:
            if self.use_norm_rot:
                bev3d_rot = torch.add(
                    torch.mul(bev3d_rot.sigmoid(), 2), -1
                )  # norm rot
            phase = torch.atan2(bev3d_rot[:, :, 1], bev3d_rot[:, :, 0])

        outputs["bev3d_rot"] = phase

        # squeeze the shape of loc_z from [b,topk,1] -> [b,topk]
        outputs["bev3d_loc_z"] = outputs["bev3d_loc_z"].squeeze(-1)

        # if cls_dimension is not None, process the pred residual
        # dim to real dim.
        if self.cls_dimension is not None:
            cls_dimension = torch.from_numpy(self.cls_dimension).to(
                outputs["bev3d_dim"].device
            )
            average_dim = cls_dimension[outputs["bev3d_cls_id"]]
            outputs["bev3d_dim"] = (
                torch.exp(outputs["bev3d_dim"]) * average_dim
            )

        if self.nms_setting is not None:
            agnostic = self.nms_setting.get("agnostic", False)
            if "letnms" in self.nms_setting:
                let_nms_param = self.nms_setting["letnms"]
            elif "bevnms" in self.nms_setting:
                thresh = self.nms_setting["bevnms"]["nms_thresh"]
            else:
                raise ValueError(
                    f"nms_setting is not supported yet, {self.nms_setting}"
                )
            # using vcs rotate_3d iou for matching
            bev3d_bboxes = torch.cat(
                (
                    outputs["bev3d_ct"],
                    outputs["bev3d_loc_z"].unsqueeze(-1),
                    outputs["bev3d_dim"],
                    outputs["bev3d_rot"].unsqueeze(-1),
                ),
                dim=2,
            )
            bev3d_score = outputs["bev3d_score"]
            bev3d_cls_id = outputs["bev3d_cls_id"]

            bev3d_bboxes_nms = []
            bev3d_score_nms = []
            bev3d_cls_id_nms = []

            if decode_occlusion:
                bev3d_occlusion_id = outputs["bev3d_occlusion_id"]
                bev3d_occlusion_id_nms = []

            batch_size = bev3d_bboxes.shape[0]
            for bs in range(batch_size):
                _bev3d_bbox = bev3d_bboxes[bs]
                _bev3d_score = bev3d_score[bs]
                _bev3d_cls_id = bev3d_cls_id[bs]
                if decode_occlusion:
                    _bev3d_occlusion = bev3d_occlusion_id[bs]
                inds = _bev3d_score > 0.09
                _bev3d_bbox = _bev3d_bbox[inds, :]
                _bev3d_score = _bev3d_score[inds]
                _bev3d_cls_id = _bev3d_cls_id[inds]
                if decode_occlusion:
                    _bev3d_occlusion = _bev3d_occlusion[inds]
                if inds.any():
                    if "letnms" in self.nms_setting:
                        selected = bev3d_let_nms(
                            _bev3d_bbox,
                            _bev3d_score,
                            let_nms_param,
                            _bev3d_cls_id,
                            agnostic,
                        )
                    elif "bevnms" in self.nms_setting:
                        selected = bev3d_nms(
                            _bev3d_bbox,
                            _bev3d_score,
                            thresh,
                            _bev3d_cls_id,
                            agnostic,
                        )

                    _bev3d_bbox = _bev3d_bbox[selected, :]
                    _bev3d_score = _bev3d_score[selected]
                    _bev3d_cls_id = _bev3d_cls_id[selected]
                    if decode_occlusion:
                        _bev3d_occlusion = _bev3d_occlusion[selected]

                if len(_bev3d_bbox) < self.topk:
                    pad_length = self.topk - len(_bev3d_bbox)
                    pad_bbox = torch.zeros((pad_length, 7)).to(
                        _bev3d_bbox.device
                    )
                    pad_scores = torch.zeros((pad_length,)).to(
                        _bev3d_bbox.device
                    )
                    pad_labels = (
                        torch.ones((pad_length,)).to(_bev3d_bbox.device) * -99
                    )
                    if decode_occlusion:
                        pad_occlusions = (
                            torch.ones((pad_length,)).to(_bev3d_bbox.device)
                            * -99
                        )

                    bboxes = torch.cat((_bev3d_bbox, pad_bbox))
                    scores = torch.cat((_bev3d_score, pad_scores))
                    labels = torch.cat((_bev3d_cls_id, pad_labels))
                    if decode_occlusion:
                        occlusions = torch.cat(
                            (_bev3d_occlusion, pad_occlusions)
                        )
                else:
                    bboxes = _bev3d_bbox
                    scores = _bev3d_score
                    labels = _bev3d_cls_id
                    if decode_occlusion:
                        occlusions = _bev3d_occlusion

                bev3d_bboxes_nms.append(bboxes)
                bev3d_score_nms.append(scores)
                bev3d_cls_id_nms.append(labels)
                if decode_occlusion:
                    bev3d_occlusion_id_nms.append(occlusions)

            bev3d_bboxes_nms = torch.stack(bev3d_bboxes_nms)
            bev3d_score_nms = torch.stack(bev3d_score_nms)
            bev3d_cls_id_nms = torch.stack(bev3d_cls_id_nms)

            # parsing the results
            outputs["bev3d_ct"] = bev3d_bboxes_nms[:, :, :2]
            outputs["bev3d_loc_z"] = bev3d_bboxes_nms[:, :, 2]
            outputs["bev3d_dim"] = bev3d_bboxes_nms[:, :, 3:6]
            outputs["bev3d_rot"] = bev3d_bboxes_nms[:, :, -1]
            outputs["bev3d_score"] = bev3d_score_nms
            outputs["bev3d_cls_id"] = bev3d_cls_id_nms
            if decode_occlusion:
                bev3d_occlusion_id_nms = torch.stack(bev3d_occlusion_id_nms)
                outputs["bev3d_occlusion_id"] = bev3d_occlusion_id_nms

            if self.roi_filter_info:
                roi_vcs_range = self.roi_filter_info["roi_vcs_range"]
                roi_score_threshold = self.roi_filter_info[
                    "roi_score_threshold"
                ]
                bev3d_ct = outputs["bev3d_ct"]
                in_roi_range_idx = torch.logical_and(
                    torch.logical_and(
                        bev3d_ct[..., 0] > roi_vcs_range[0],
                        bev3d_ct[..., 0] < roi_vcs_range[2],
                    ),
                    torch.logical_and(
                        bev3d_ct[..., 1] > roi_vcs_range[1],
                        bev3d_ct[..., 1] < roi_vcs_range[3],
                    ),
                )
                bev3d_score = outputs["bev3d_score"]
                bev3d_cls_id = outputs["bev3d_cls_id"]
                for cls_i in range(len(roi_score_threshold)):
                    bev3d_cls_id_i = bev3d_cls_id == cls_i
                    invalid_score_idx = (
                        bev3d_score < roi_score_threshold[cls_i]
                    )
                    bev3d_score[
                        in_roi_range_idx & bev3d_cls_id_i & invalid_score_idx
                    ] = 0
                outputs["bev3d_score"] = bev3d_score
        return outputs


@OBJECT_REGISTRY.register
class ANCBEVDiscreteObjectDecoder(ANCBEV3Decoder):
    """Decoder, convert model output to object results.

    Args:
        ct_nms_config: include do_ct_nms and ct_dist_scale.
            do_ct_nms: whether use ct_nms for decodered bboxes.
            ct_dist_scale: The scale of wh used as threshold for ct_nms.
        psc_phase_factor: Factor number to align the rotation period
            of the taraget object to 2pi. Refer to https://arxiv.org/abs/2211.06368. # noqa
        res_key: result key of output.
    """

    def __init__(
        self,
        ct_nms_config: dict = None,
        psc_phase_factor: int = 1,
        res_key: str = "bev_discobj",
        **kwargs,
    ):
        super(ANCBEVDiscreteObjectDecoder, self).__init__(**kwargs)
        self.ct_nms_config = ct_nms_config
        self.res_key = res_key
        assert psc_phase_factor >= 1
        self.psc_phase_factor = psc_phase_factor

    def forward(
        self,
        pred: Mapping,
        label: Dict = None,
    ) -> Dict:
        bev_discobj_hm = sigmoid_and_clip(pred["pred_bev_discobj_hm"])
        # max_pooling nms
        batch_size, _, height, width = bev_discobj_hm.shape
        kernel = self.max_pool_kernel
        pad = (kernel - 1) // 2
        max_bev_discobj_hm = F.max_pool2d(
            bev_discobj_hm, (kernel, kernel), stride=1, padding=(pad, pad)
        )
        bev_discobj_hm *= (max_bev_discobj_hm == bev_discobj_hm).float()
        pred["pred_bev_discobj_hm"] = bev_discobj_hm

        for key, out in pred.items():
            channel_size = out.shape[1]
            pred[key] = out.permute(0, 2, 3, 1).contiguous()
            pred[key] = pred[key].view(batch_size, -1, channel_size)

        # get the topk indices of heatmap
        bev_discobj_hm = pred.pop("pred_bev_discobj_hm")
        bev_discobj_hm_cls = pred.pop("pred_bev_discobj_hm_cls", None)
        if bev_discobj_hm_cls is not None:
            # 1 channel heatmap to do location regression and
            # c channel cls maps to do classification.
            _, classes = bev_discobj_hm_cls.max(dim=-1)
            scores, topk_indices = bev_discobj_hm.squeeze(-1).topk(
                k=self.topk, dim=1
            )
        else:
            # c channel heatmaps to do location regression and do classification. # noqa
            scores, classes = bev_discobj_hm.max(dim=-1)
            scores, topk_indices = scores.topk(k=self.topk, dim=1)
        pred["pred_bev_discobj_cls_id"] = classes

        # get the topk indices of regression items, e.g., dim, rot etc.
        for key, out in pred.items():
            pred[key] = torch.cat(
                [_out[_inds][None] for _out, _inds in zip(out, topk_indices)],
                dim=0,
            )

        # convert topk indices value to the coordinate
        pred["pred_bev_discobj_score"] = scores
        u, v = topk_indices % width, topk_indices // width  # bev: u=x, v=y
        bev_discobj_center = torch.cat(
            [u[:, :, None], v[:, :, None]], axis=-1
        )  # (u, v) = (x, y)

        if "pred_bev_discobj_ct_offset" in pred:
            bev_discobj_ct_offset = pred.pop("pred_bev_discobj_ct_offset")
            bev_discobj_center = (
                bev_discobj_center + bev_discobj_ct_offset
            )  # (u, v)

        # convert the bev center in bev coord to vcs coord
        vcs_center = self.coord_transform(
            bev_discobj_center,
            bev_size=(height, width),
            vcs_range=self.vcs_range,
        )
        pred["pred_bev_discobj_ct"] = vcs_center

        # convert the rot to rad value (sin / cos), range=[-pi, pi]
        bev_discobj_rot = pred.pop("pred_bev_discobj_rot")
        if self.use_psc_rot:
            phase = decode_psc_rot(
                bev3d_rot=bev_discobj_rot,
                coef_sin=self.coef_sin,
                coef_cos=self.coef_cos,
                use_sigmoid=self.use_norm_rot,
                N_steps_PSC_rot=self.N_steps_PSC_rot,
                rot_mod_threshold=self.rot_mod_threshold,
            )
            phase /= self.psc_phase_factor
        else:
            if self.use_norm_rot:
                bev_discobj_rot = torch.add(
                    torch.mul(bev_discobj_rot.sigmoid(), 2), -1
                )  # norm rot
            phase = torch.atan2(
                bev_discobj_rot[:, :, 1], bev_discobj_rot[:, :, 0]
            )

        pred["pred_bev_discobj_rot"] = phase

        # clamp wh
        pred["pred_bev_discobj_wh"] = pred["pred_bev_discobj_wh"].clamp(
            0,
        )
        if self.ct_nms_config and self.ct_nms_config.get("do_ct_nms", False):

            self.ct_nms_config.pop("do_ct_nms")
            pred = do_ct_dist_nms(pred, **self.ct_nms_config)

            # recover do_ct_nms
            self.ct_nms_config["do_ct_nms"] = True

        # rename keys of pred.
        for key in list(pred.keys()):
            value = pred.pop(key)
            new_key = key.replace("bev_discobj", f"{self.res_key}")
            pred[new_key] = value

        return pred


@OBJECT_REGISTRY.register
class ANCBevPSDDecoder(object):
    """Postprocess for Super PSD algorithm.

        Include two main steps:
        1. Decode globalslot features and localjunction features;
        2. Fuse results from globalslot and localjunction

    Args:
        input_size: Input resolution of model
        downsample_factor_globalslot_feature: Downsample factor of
            global slot feature relative to input size
        downsample_factor_localjunction_feature: Downsample factor
            local junction feature  relative to input size
        threshold_globalslot_feature: Threshold to decode
            slot from global slot feature
        threshold_localjunction_feature: Threshold to decode
            junction from local junction feature
        topk_globalslot_feature: Number of slots
            to decode from global slot feature
        topk_localjunction_feature: Number of junctions
            to decode from local junction feature
        threshold_occupancy_feature: Threshold to classify
            slot occupancy from occupancy feature
        threshold_junction_type_feature: Threshold to classify
            junction type from junction type feature
        threshold_fuse_distance: Threshold of distance to fuse junctions
            decoded from global slot feature and local junction feature
        global_kernel_size: kernel_size of max-pooling
            to get topk from global-head output
        local_kernel_size: kernel_size of max-pooling
            to get topk from local-head output
        nms_distance_threshold: distance for nms
    """

    def __init__(
        self,
        input_size: List[int],
        downsample_factor_globalslot_feature: int,
        downsample_factor_localjunction_feature: int,
        threshold_globalslot_feature: float,
        threshold_localjunction_feature: float,
        topk_globalslot_feature: int,
        topk_localjunction_feature: int,
        threshold_occupancy_feature: float,
        threshold_junction_type_feature: float,
        threshold_fuse_distance: int,
        global_kernel_size: int,
        local_kernel_size: int,
        nms_distance_threshold: int,
    ):
        super().__init__()
        self.input_size = input_size
        self.downsample_factor_globalslot_feature = (
            downsample_factor_globalslot_feature
        )
        self.downsample_factor_localjunction_feature = (
            downsample_factor_localjunction_feature
        )

        self.globalslot_output_size = [
            i // self.downsample_factor_globalslot_feature
            for i in self.input_size
        ]
        self.localjunction_output_size = [
            i // self.downsample_factor_localjunction_feature
            for i in self.input_size
        ]

        self.threshold_globalslot_feature = threshold_globalslot_feature
        self.threshold_localjunction_feature = threshold_localjunction_feature
        self.topk_globalslot_feature = topk_globalslot_feature
        self.topk_localjunction_feature = topk_localjunction_feature

        self.threshold_occupancy_feature = threshold_occupancy_feature
        self.threshold_junction_type_feature = threshold_junction_type_feature

        self.threshold_fuse_distance = threshold_fuse_distance
        self.global_kernel_size = global_kernel_size
        self.local_kernel_size = local_kernel_size

        self.nms_distance_threshold = nms_distance_threshold

    def slot_nms(self, det_results, nms_distance_threshold):
        """Use to erase repeated detection results.

        Args:
            det_results: detection results after fusing global head \
                and local head.
            iou_threshold: nms iou threshold.
        """
        # det_results: [junction_locations*8, junction_types*4,
        # junction_visibilities*4, slot_occupancies*1,
        # slot_types*1, junction_orientations*8,
        # slot_orientations*2, slot_score, junction_score*4]
        if len(det_results) == 0:
            return det_results
        det_results = np.array(det_results)
        scores = det_results[:, 28]
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            idx = order[0]
            keep.append(det_results[idx].tolist())
            bbox1 = det_results[idx, :8].reshape(4, 2)
            bbox1_center = np.mean(bbox1, axis=0)
            rest_bbox = det_results[order[1:], :8].reshape(-1, 4, 2)
            rest_bbox_center = np.mean(rest_bbox, axis=1)
            distance = np.sqrt(
                np.sum((bbox1_center - rest_bbox_center) ** 2, axis=-1)
            )
            index = np.where(distance > nms_distance_threshold)[0]
            order = order[index + 1]
        return keep

    def decode_globalslot_feature(
        self, globalslot_feature
    ) -> Dict[str, List[torch.Tensor]]:
        """Decode slot information from global slot feature.

        Args:
            globalslot_feature: tensor of globalslot_feature from the model
        Returns:
            A dict mapping keys to the corresponding list
        """
        # Postprocess feature
        # objectness score of slot center
        classification_slot = torch.sigmoid(globalslot_feature[0]).squeeze()
        # (x,y) offset of four junctions relative to slot tile center
        offset_junctions = (
            globalslot_feature[1].squeeze()
            * self.downsample_factor_globalslot_feature
        )
        # occupancy of the slot by vehicles
        occupancy = torch.sigmoid(globalslot_feature[2]).squeeze()
        if globalslot_feature[3].shape[0] != 1:
            type_slot = torch.argmax(
                globalslot_feature[3].squeeze(), dim=0
            )  # Moved into BPU
        else:
            type_slot = globalslot_feature[3].squeeze()
        # sin(theta), cos(theta) of slot
        orientation = F.normalize(globalslot_feature[4].squeeze(), p=2, dim=0)

        # Decode result from feature
        slot_junctions_locations = []
        slot_scores = []
        slot_occupancies = []
        slot_types = []
        slot_orientations = []
        center_indices, center_values = self.decode_object(
            classification_slot,
            self.threshold_globalslot_feature,
            self.topk_globalslot_feature,
            self.global_kernel_size,
        )
        tile_center = self.generate_tile_centers(
            self.downsample_factor_globalslot_feature,
            self.globalslot_output_size,
        )
        tile_center = tile_center.to(offset_junctions.device)
        for i, center_index in enumerate(center_indices):
            offset_junctions_i = offset_junctions[
                :, center_index[0], center_index[1]
            ].reshape(4, 2)
            tile_center_i = tile_center[:, center_index[0], center_index[1]]
            slot_junctions_locations_i = (
                self.downsample_factor_globalslot_feature
                * tile_center_i[None, :]
                + offset_junctions_i
            )

            occupancy_i = occupancy[center_index[0], center_index[1]]
            type_slot_i = type_slot[center_index[0], center_index[1]]
            orientation_i = orientation[:, center_index[0], center_index[1]]

            slot_junctions_locations.append(slot_junctions_locations_i)
            slot_occupancies.append(occupancy_i)
            slot_types.append(type_slot_i)
            slot_orientations.append(orientation_i)
            slot_scores.append(center_values[i])

        return {
            "slot_junctions_locations_list": slot_junctions_locations,
            "slot_occupancy_list": slot_occupancies,
            "slot_type_list": slot_types,
            "slot_orientation_list": slot_orientations,
            "slot_score_list": slot_scores,
        }

    def decode_localjunction_feature(
        self, localjunction_feature: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Decode junction information from local junction feature.

        Args:
            localjunction_feature: tensor of localjunction_feature \
                from the model
        Returns:
            Dicts of junction0, junction1, junction2, junction3
            mapping keys to the corresponding list
        """
        # Postprocess feature
        junction_dict_list = []
        tile_center = self.generate_tile_centers(
            self.downsample_factor_globalslot_feature,
            self.localjunction_output_size,
        )
        tile_center = tile_center.to(localjunction_feature[0].device)
        for junction_idx in range(4):

            classification = torch.sigmoid(
                localjunction_feature[0][junction_idx]
            ).squeeze()
            offset = localjunction_feature[1][
                2 * junction_idx : 2 * junction_idx + 2
            ].squeeze()
            sline_angle = F.normalize(
                localjunction_feature[2][
                    2 * junction_idx : 2 * junction_idx + 2
                ].squeeze(),
                p=2,
                dim=0,
            )
            type_junction = torch.sigmoid(
                localjunction_feature[3][junction_idx]
            ).squeeze()
            j_indices, j_values = self.decode_object(
                classification,
                self.threshold_localjunction_feature,
                self.topk_localjunction_feature,
                self.local_kernel_size,
            )
            locations, sline_angles, junction_types, scores = [], [], [], []
            for index, junction_index in enumerate(j_indices):
                offset_junction_i = offset[
                    :, junction_index[0], junction_index[1]
                ]
                tile_center_i = tile_center[
                    :, junction_index[0], junction_index[1]
                ]
                junction_location_i = tile_center_i + offset_junction_i
                junction_location_i *= (
                    self.downsample_factor_localjunction_feature
                )  # Newly added
                sline_anlge_junction_i = sline_angle[
                    :, junction_index[0], junction_index[1]
                ]
                type_junction_i = type_junction[
                    junction_index[0], junction_index[1]
                ]
                locations.append(junction_location_i)
                sline_angles.append(sline_anlge_junction_i)
                junction_types.append(type_junction_i)
                scores.append(j_values[index])

            junction_dict = {
                f"j{junction_idx}_location_list": locations,
                f"j{junction_idx}_sline_angle_list": sline_angles,
                f"j{junction_idx}_type_list": junction_types,
                f"j{junction_idx}_score_list": scores,
            }
            junction_dict_list.append(junction_dict)
        return junction_dict_list

    def calculate_angle(self, point_dets: List[float]) -> List[float]:
        """Calculate the angles between four pairs coordinates.

        Args:
            point_dets: Containing eight float values representing four
            pairs the endpoints.

        Returns:
            angles: List containing the four angles (in degrees)
            between the pairs of vectors formed by the input coordinates.
            The Angle order depends on the input, clockwise from front to back.
        """
        # Unpack coordinates from input list
        x0, y0, x1, y1, x2, y2, x3, y3 = point_dets

        # Convert each pair of coordinates to a numpy array
        p0 = np.array([x0, y0])
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        p3 = np.array([x3, y3])

        # Calculate the angle between the pair of vectors
        j0_v1 = p0 - p1
        j0_v2 = p0 - p3
        angle0 = (
            np.arccos(
                np.dot(j0_v1, j0_v2)
                / (np.linalg.norm(j0_v1) * np.linalg.norm(j0_v2))
            )
            * 180
            / np.pi
        )

        j1_v1 = p1 - p0
        j1_v2 = p1 - p2
        angle1 = (
            np.arccos(
                np.dot(j1_v1, j1_v2)
                / (np.linalg.norm(j1_v1) * np.linalg.norm(j1_v2))
            )
            * 180
            / np.pi
        )

        j2_v1 = p2 - p1
        j2_v2 = p2 - p3
        angle2 = (
            np.arccos(
                np.dot(j2_v1, j2_v2)
                / (np.linalg.norm(j2_v1) * np.linalg.norm(j2_v2))
            )
            * 180
            / np.pi
        )

        j3_v1 = p3 - p2
        j3_v2 = p3 - p2
        angle3 = (
            np.arccos(
                np.dot(j3_v1, j3_v2)
                / (np.linalg.norm(j3_v1) * np.linalg.norm(j3_v2))
            )
            * 180
            / np.pi
        )

        return [angle0, angle1, angle2, angle3]

    def fuse_globalslot_localjunction(self, slot, junction_dict_list):
        """Fuse results from globalslot and localjunction.

        Args:
            slot: results of slot decoded from globalslot, i.e., \
            {"slot_junctions_locations_list": slot_junctions_locations, \
            "slot_occupancy_list": slot_occupancies,"slot_type_list": \
            slot_types, "slot_orientation_list": slot_orientations}
            j0: results of junction0 decoded from localjunction, i.e., \
            {"j0_location_list":j0_locations, "j0_sline_angle_list": \
            j0_sline_angles, "j0_type_list":j0_types}
            j1: results of junction1 decoded from localjunction, i.e., \
            {"j1_location_list":j1_locations, "j1_sline_angle_list": \
            j1_sline_angles, "j1_type_list":j1_types}
            j2: results of junction2 decoded from localjunction, i.e., \
            {"j2_location_list":j2_locations, "j2_sline_angle_list": \
            j2_sline_angles, "j2_type_list":j2_types}
            j3: results of junction3 decoded from localjunction, i.e., \
            {"j3_location_list":j3_locations, "j3_sline_angle_list": \
            j3_sline_angles, "j3_type_list":j3_types}
        Returns:
            det_results: the output slots locations and attributions, \
            shape (N, 28), N-represents slot numbers, 28 represents \
            [x0,y0,x1,y1,x2,y2,x3,y3, type_j0, type_j1, type_j2, \
            type_j3,visibility_j0, visibility_j1, visibility_j2, \
            visibility_j3, occupancy of slot, type of slot, \
            sin_j0, cos_j0, sin_j1, cos_j1, sin_j2, \
            cos_j2, sin_j3, cos_j3, sin_slot, cos_slot]
        """
        det_results = []
        slot_scores = slot["slot_score_list"]
        sorted_score_idx_list = sorted(
            range(len(slot_scores)), key=lambda i: slot_scores[i], reverse=True
        )
        chosen_point_list = [{}, {}, {}, {}]
        for idx in sorted_score_idx_list:
            slot_junctions = slot["slot_junctions_locations_list"][idx]
            det_result = []
            total_score = 0
            slot_occupancy = (
                1
                if slot["slot_occupancy_list"][idx]
                > self.threshold_occupancy_feature
                else 0
            )
            slot_type = slot["slot_type_list"][idx]
            slot_orientation = slot["slot_orientation_list"][idx]
            slot_score = slot["slot_score_list"][idx]
            visibility, type_junction, orientation, score = (
                0,
                0,
                torch.tensor([0, 0]),
                torch.tensor(0.1),
            )
            (
                coord_list,
                type_junction_list,
                visibility_list,
                orientation_list,
                score_list,
            ) = ([], [], [], [], [])
            for junction_idx in range(4):
                if (
                    len(
                        junction_dict_list[junction_idx][
                            f"j{junction_idx}_location_list"
                        ]
                    )
                    > 0
                ):
                    locations_tensor = torch.cat(
                        [
                            junction.unsqueeze(0)
                            for junction in junction_dict_list[junction_idx][
                                f"j{junction_idx}_location_list"
                            ]
                        ],
                        dim=0,
                    )
                    # distance of j0 decoded from globalslot and localjunction
                    slot_dist = torch.sqrt(
                        torch.sum(
                            (
                                slot_junctions[junction_idx][None, :]
                                - locations_tensor
                            )
                            ** 2,
                            dim=1,
                        )
                    )
                    min_slot_dist, selected_j = torch.min(slot_dist, 0)
                    if min_slot_dist < self.threshold_fuse_distance:
                        saved_flag = chosen_point_list[junction_idx].get(
                            selected_j.item(), False
                        )
                        if not saved_flag:
                            chosen_point_list[junction_idx][
                                selected_j.item()
                            ] = True
                            total_score += 1
                            # Substitute junction 0 locations
                            # by the locations decoded from localjunction
                            slot_junctions[junction_idx] = locations_tensor[
                                selected_j
                            ]
                            visibility = 1
                            score = junction_dict_list[junction_idx][
                                f"j{junction_idx}_score_list"
                            ][selected_j]
                            type_junction = (
                                1
                                if junction_dict_list[junction_idx][
                                    f"j{junction_idx}_type_list"
                                ][selected_j]
                                > self.threshold_junction_type_feature
                                else 0
                            )
                            orientation = junction_dict_list[junction_idx][
                                f"j{junction_idx}_sline_angle_list"
                            ][selected_j]
                coord_list.append(slot_junctions[junction_idx][0].item())
                coord_list.append(slot_junctions[junction_idx][1].item())
                type_junction_list.append(type_junction)
                visibility_list.append(visibility)
                orientation_list.append(orientation[0].item())
                orientation_list.append(orientation[1].item())
                score_list.append(score.item())
            det_result.extend(coord_list)
            det_result.extend(type_junction_list)
            det_result.extend(visibility_list)
            det_result.append(slot_occupancy)
            det_result.append(slot_type.item())
            det_result.extend(orientation_list)
            det_result.extend(
                [slot_orientation[0].item(), slot_orientation[1].item()]
            )
            det_result.append(slot_score.item())
            det_result.extend(score_list)
            if total_score > 0:
                ang0, ang1, ang2, ang3 = self.calculate_angle(det_result[:8])
                if (
                    # The default of 80 is a demarcation between the
                    # oblique and vertical slot, angle unit is degree(°).
                    min(ang0, ang1) > 80
                    and det_result[17] == 2
                ):  # 2 is oblique slot flag
                    # index 17 is slot type
                    det_result[17] = 0  # # 0 is vertical slot flag
                det_results.append(det_result)
                # [junction_locations*8, junction_types*4,
                # junction_visibilities*4, slot_occupancies*1,
                # slot_types*1, junction_orientations*8,
                # slot_orientations*2, slot_score, junction_score*4]
        return det_results

    @staticmethod
    def decode_object(
        feature: torch.Tensor,
        threshold: float,
        topk_number: int,
        kernel_size: int,
    ):
        """Decode object from the feature of objectness score."""
        maximum = F.max_pool2d(
            feature.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        maximum = torch.eq(feature, maximum)
        feature = feature * maximum
        feature = feature.squeeze()
        # feature = torch.where(feature>threshold,feature,0.)
        # in pytorch version >=1.8.0
        feature[feature < threshold] = 0
        H, W = feature.size()
        feature = feature.view(-1)
        values, indices = feature.topk(topk_number, dim=0)
        indices = indices[values > 0]
        values = values[values > 0]
        # [y1, y2, y3,...., x1, x2, x3,...]
        indices = torch.cat([indices // W, indices % W])
        indices = indices.view(2, -1)  # [[y1, y2, y3, ...], [x1, x2, x3,...]]
        indices = indices.transpose(1, 0)  # [[y1, x1], [y2, x2], [y3, x3],...]
        return indices, values

    @staticmethod
    def generate_tile_centers(
        downsample_factor: int, output_size: List[int]
    ) -> torch.Tensor:
        """Generate tile centers."""
        tile_center_x = [n for n in range(output_size[0])]  # noqa
        tile_center_y = [n for n in range(output_size[1])]  # noqa
        tile_center = torch.empty(
            size=[
                2,
            ]
            + output_size
        )
        tile_center[1, :, :] = torch.tensor(tile_center_x)[:, None]
        tile_center[0, :, :] = torch.tensor(tile_center_y)
        return tile_center

    @staticmethod
    def tensor2list(data):
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                result[key] = ANCBevPSDDecoder.tensor2list(value)
            return result
        elif isinstance(data, (list, tuple)):
            return [ANCBevPSDDecoder.tensor2list(d) for d in data]
        elif isinstance(data, (int, float, str)):
            return data
        elif isinstance(data, torch.Tensor):
            return data.tolist()

    def __call__(
        self,
        preds: Mapping,
        label: Dict = None,
    ) -> Dict:
        """Get target from label.

        Args:
            preds (Mapping):
                global_classification:     torch.randn((1, 1, 48, 32)),
                global_offset:             torch.randn((1, 8, 48, 32)),
                global_occupancy:          torch.randn((1, 1, 48, 32)),
                global_slot_type:          torch.randn((1, 1, 48, 32)),
                global_direction:          torch.randn((1, 2, 48, 32)),
                local_classification:     torch.randn((1, 4, 192, 128)),
                local_offset:             torch.randn((1, 8, 192, 128)),
                local_sline_angle:        torch.randn((1, 8, 192, 128)),
                local_point_type:         torch.randn((1, 4, 192, 128)),
            label (Dict, optional): _description_. Defaults to None.

        Returns:
            Dict: _description_
        """
        decode_global, decode_local, decode_slots = [], [], []
        for (
            global_classification,
            global_offset,
            global_occupancy,
            global_slot_type,
            global_direction,
            local_classification,
            local_offset,
            local_sline_angle,
            local_point_type,
        ) in zip(*preds):
            pred_global = (
                global_classification,
                global_offset,
                global_occupancy,
                global_slot_type,
                global_direction,
            )
            pred_local = (
                local_classification,
                local_offset,
                local_sline_angle,
                local_point_type,
            )
            d_global = self.decode_globalslot_feature(pred_global)
            d_local = self.decode_localjunction_feature(pred_local)
            d_slots = self.fuse_globalslot_localjunction(d_global, d_local)
            d_slots = self.slot_nms(d_slots, self.nms_distance_threshold)
            decode_global.append(d_global)
            decode_local.append(d_local)
            decode_slots.append(d_slots)
        preds_label = {
            "decode_global": self.tensor2list(decode_global),
            "decode_local": self.tensor2list(decode_local),
            "decode_slots": self.tensor2list(decode_slots),
        }
        return {
            "decode_label": preds_label,
        }


@OBJECT_REGISTRY.register
class ANCBEVParkingrodDecoder(object):
    """Postprocess for BEV parkingrod detection.

    Args:
        input size: Input resolution of head
        downsample_factor_feature: Downsample factor
            feature relative to input size
        score_threshold: threshold to decode rod from feature
        topk: number of parkingrods to decode from rod feature
        nms_distance_threshold: distance for nms
        kernel_size: kernel size of max-pooling to
            get topk from parkingrod head out
    """

    def __init__(
        self,
        input_size: List[int],
        downsample_factor_feature: int,
        score_threshold: float,
        topk: int,
        nms_distance_threshold: float,
        kernel_size: int,
    ):
        super().__init__()
        self.input_size = input_size
        self.downsample_factor_feature = downsample_factor_feature
        self.output_size = [
            i // self.downsample_factor_feature for i in self.input_size
        ]
        self.score_threshold = score_threshold
        self.topk = topk
        self.nms_distance_threshold = nms_distance_threshold
        self.kernel_size = kernel_size

    def decode_feature(self, feature) -> Dict[str, List[torch.Tensor]]:
        """Decode parkingrod information from head prediction.

        Args:
            feature: tensor of feature from model
                pred_cls: the pred of cls from head.
                pred_endpoint_offset: the pred of endpoint offset from head.
        Returns:
            A dict mapping keys to the corresponding list
        """
        # Postprocess feature
        pred_cls, pred_endpoint_offset = feature
        # objectness score of parkingrod center
        classification = torch.sigmoid(pred_cls).squeeze()
        # (x, y) offset of two endpoints relative to parkingrod tile center
        endpoint_offsets = (
            pred_endpoint_offset.squeeze() * self.downsample_factor_feature
        )

        # decode result from feature
        endpoint_locations = []
        rod_scores = []
        center_indices, center_values = ANCBevPSDDecoder.decode_object(
            feature=classification,
            threshold=self.score_threshold,
            topk_number=self.topk,
            kernel_size=self.kernel_size,
        )

        tile_center = ANCBevPSDDecoder.generate_tile_centers(
            self.downsample_factor_feature, self.output_size
        )
        tile_center = tile_center.to(endpoint_offsets.device)
        for i, center_index in enumerate(center_indices):
            endpoint_offset_i = endpoint_offsets[
                :, center_index[0], center_index[1]
            ].reshape(2, 2)
            tile_center_i = tile_center[:, center_index[0], center_index[1]]
            endpoint_locations_i = (
                self.downsample_factor_feature * tile_center_i[None, :]
                + endpoint_offset_i
            )
            endpoint_locations.append(endpoint_locations_i)
            rod_scores.append(center_values[i])

        # sort result append to scores
        det_results = []
        sorted_score_idx_list = sorted(
            range(len(rod_scores)),
            key=lambda i: rod_scores[i],
            reverse=True,
        )
        for idx in sorted_score_idx_list:
            det_result = []
            coord_list = []
            endpoint_location = endpoint_locations[idx]
            rod_score = rod_scores[idx]
            for index in range(2):
                coord_list.append(endpoint_location[index][0].item())
                coord_list.append(endpoint_location[index][1].item())
            det_result.extend(coord_list)
            det_result.append(rod_score.item())
            # det_result:
            #   [endpoint_location*4, score*1]
            det_results.append(det_result)
        return det_results

    def rod_nms(self, det_results, nms_distance_threshold):
        """Use to erase repeated detection results.

        Args:
            det_results: detection results after fusing global head
                and local head.
                [endpoint_locations*4, rod_score*1]
            distance_threshold: nms distance threshold.
        """
        if len(det_results) == 0:
            return det_results
        det_results = np.array(det_results)
        scores = det_results[:, -1]
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            cur_line_idx = order[0]
            rest_line_idx = order[1:]
            keep.append(det_results[cur_line_idx].tolist())
            cur_line = det_results[cur_line_idx, :4].reshape(2, 2)
            cur_line_center = np.mean(cur_line, axis=0)
            rest_line = det_results[rest_line_idx, :4].reshape(-1, 2, 2)
            rest_line_center = np.mean(rest_line, axis=1)
            distance = np.sqrt(
                np.sum((cur_line_center - rest_line_center) ** 2, axis=-1)
            )
            index = np.where(distance > nms_distance_threshold)[0]
            order = order[index + 1]
        return keep

    def __call__(
        self,
        preds: Mapping,
        label: Dict = None,
    ) -> Dict:
        """Get target from label.

        Args:
            preds:
                classification:  torch.randn((b, 1, h, w)),
                endpoint_offset: torch.randn((b, 4, h, w)),

        Returns:
            preds_label:
                [[x0, y0, x1, y1, rod_type],[],[],...]
        """
        preds_parkingrod = []
        for pred in zip(*preds):
            decode_parkingrod = self.decode_feature(pred)
            decode_parkingrod = self.rod_nms(
                decode_parkingrod, self.nms_distance_threshold
            )
            preds_parkingrod.append(decode_parkingrod)
        preds_label = ANCBevPSDDecoder.tensor2list(preds_parkingrod)
        return {"preds_parkingrod": preds_label}


@OBJECT_REGISTRY.register
class ANCOnlineMappingDecoder(object):
    """The OnlineMapping Decoder, convert the model output to lane stats.

    Args:
        head_groups: OM heads configs.
        embedding_dim: the dimension of the embedding feature.
        out_size: model output size.
        vcs_range: vcs range, in meter.(order is
            (bottom,right,top,left), e.g.(-30, -51.2, 72.4, 51.2))
        use_seq: if generate sequence results
        use_seq_post: if enable post lane sequence process.
        decoder_output_dir: decoder result save dir.
            If None not save decoder result
        cluster_alg: The algorithm of cluster.
        cluster_cfg: the configs of cluster.
        cluster_split_channel: if run cluster on different channel.
        cpts_decoder: Crosspoints decoder.

    """

    def __init__(
        self,
        head_groups: Dict,
        embedding_dim: int,
        out_size: Sequence[int],
        vcs_range: Sequence[float],
        use_seq: bool = False,
        use_seq_post: bool = False,
        decoder_output_dir: Optional[str] = None,
        cluster_alg: str = "dbscan",
        cluster_cfg: Optional[Dict] = None,
        cluster_split_channel: bool = False,
        cpts_decoder: Optional[torch.nn.Module] = None,
    ):
        self.head_groups = head_groups
        self.cluster_alg = cluster_alg
        self.cluster_cfg = cluster_cfg
        self.cluster_split_channel = cluster_split_channel
        self.embedding_dim = embedding_dim
        self.target_categorys = get_target_categorys(head_groups)
        self.use_seq = use_seq
        self.use_seq_post = use_seq_post
        self.out_h, self.out_w = out_size
        self.cpts_decoder = cpts_decoder
        self.vcs_range = vcs_range
        (
            self.bottom,
            self.right,
            self.top,
            self.left,
        ) = vcs_range
        scope_h = self.top - self.bottom
        scope_w = self.left - self.right
        self.res_h = scope_h / self.out_h
        self.res_w = scope_w / self.out_w
        self.decoder_output_dir = decoder_output_dir
        self.lane_pts_attr = {
            "vcs": 2,
            "vcs_origin": 2,
            "cls": 1,
            "prob": 1,
            "sin": 1,
            "cos": 1,
            "embedding": self.embedding_dim,
            "bev": 2,
            "offset_r": 1,
        }
        self.decode_config = {
            "config": {
                "vcs_range": vcs_range,
                "out_size": out_size,
                "out_res": [self.res_h, self.res_w],
            }
        }

        self.sub_head = []
        for head, head_infos in head_groups.items():
            multi_head = head_infos.get("multi", False)
            if not multi_head:
                self.lane_pts_attr[head] = 2
                self.sub_head.append(head)

        pts_attr_cum = 0
        self.lane_pts_attr_start_end = {}
        for key in self.lane_pts_attr:
            if key in self.sub_head:
                self.lane_pts_attr_start_end[key] = (
                    pts_attr_cum,
                    pts_attr_cum + 1,
                )
            else:
                self.lane_pts_attr_start_end[key] = (
                    pts_attr_cum,
                    pts_attr_cum + self.lane_pts_attr[key],
                )
            pts_attr_cum += self.lane_pts_attr[key]

    def decoder_thread(self, pred, thread_id, results):
        pred_stats = pred_preprocess(
            thread_id,
            pred,
            head_groups=self.head_groups,
        )
        pred_stats = convert_pred(
            pred_stats,
            head_groups=self.head_groups,
            vcs_range=self.vcs_range,
            cluster_alg=self.cluster_alg,
            cluster_cfg=self.cluster_cfg,
            cluster_split_channel=self.cluster_split_channel,
        )
        pred_stats = get_offset(pred_stats)
        pred_lanes_raw, pred_lanes_seq, group_thrs = get_vcs_lane(
            pred_stats,
            head_groups=self.head_groups,
            out_h=self.out_h,
            out_w=self.out_w,
            top=self.top,
            left=self.left,
            res_h=self.res_h,
            res_w=self.res_w,
            target_categorys=self.target_categorys,
            embedding_dim=self.embedding_dim,
            use_seq=self.use_seq,
            use_seq_post=self.use_seq_post,
        )
        aux_info = {
            "group_thrs": group_thrs,
            "head_groups": self.head_groups,
        }
        aux_info.update(self.decode_config["config"])
        results[thread_id] = {
            "pred_lanes_raw": pred_lanes_raw,
            "pred_lanes_seq": pred_lanes_seq,
            "pred_stats": pred_stats,
            "aux_info": aux_info,
        }

    def __call__(
        self,
        pred: Mapping,
        label: Dict = None,
    ) -> Dict:
        first_key = list(pred.keys())[0]
        batch_size = pred[first_key][0].shape[0]
        pred_stats_gather = [{}] * batch_size
        thread_list = []
        for batch_id in range(batch_size):
            sub_thread = threading.Thread(
                target=self.decoder_thread,
                args=(pred, batch_id, pred_stats_gather),
            )
            sub_thread.start()
            thread_list.append(sub_thread)
        for sub_thread in thread_list:
            sub_thread.join()
        if self.decoder_output_dir:
            timestamp = label["timestamp"]
            if len(timestamp.shape) == 1:
                pass
            elif len(timestamp.shape) == 2:
                timestamp = torch.squeeze(timestamp, dim=1)
            else:
                raise Exception(
                    f"timestamp dim must 1 or 2 but {timestamp.shape}"
                )
            os.makedirs(self.decoder_output_dir, exist_ok=True)
            for index, pred_stats_gather_single in enumerate(
                pred_stats_gather
            ):
                decoder_json_file = str(timestamp[index].item()) + ".json"
                decoder_file_path = os.path.join(
                    self.decoder_output_dir, decoder_json_file
                )
                pts_formatted_dict = {}
                for category, lanes in pred_stats_gather_single[
                    "pred_lanes_raw"
                ].items():
                    pts_formatted_dict[category] = [
                        pts_format_dict(
                            lane, self.lane_pts_attr_start_end, self.sub_head
                        )
                        for lane in lanes
                    ]
                pts_formatted_dict.update(self.decode_config)
                json.dump(pts_formatted_dict, open(decoder_file_path, "w"))

        result = {"om_predict": pred_stats_gather}
        if self.cpts_decoder is not None:
            cpts_result = self.cpts_decoder(pred, label=label)
            result.update(cpts_result)
        return result


@OBJECT_REGISTRY.register
class ANCBEVCrosspointDecoder(object):
    """The BEVCrosspoint Decoder, convert the model output to crosspts.

    Args:
        stride: Crosspoint output stride.
        cls_group_map: Output categories of each group.
        bev_size: Bev size, in pixel.(order is (h,w))
        vcs_range: Vcs range, in meter.(order is
            (bottom,right,top,left), e.g.(-30, -51.2, 72.4, 51.2)).
        ap_score_thresh: Score thresh used to cal AP.
        nms_thresh: Nms threshold, (x_thresh, y_thresh), unit: m.
        ignore_index: Ignored class id.
        gt_name: gt name.
    """

    def __init__(
        self,
        stride: int,
        cls_group_map: Dict[str, Sequence[int]],
        bev_size: Sequence[int],
        vcs_range: Sequence[float],
        ap_score_thresh: float,
        nms_thresh: Sequence[float] = (1.6, 1.6),
        ignore_index: int = 255,
        gt_name: str = "crosspoint_target",
    ):
        self.stride = stride
        self.cls_group_map = cls_group_map
        self.vcs_range = vcs_range
        self.bev_h, self.bev_w = bev_size
        self.out_h = self.bev_h // self.stride
        self.out_w = self.bev_w // self.stride
        _, _, self.top_offset, self.left_offset = vcs_range
        self.scope_h = vcs_range[2] - vcs_range[0]
        self.scope_w = vcs_range[3] - vcs_range[1]
        self.meter_per_out_pixel_h = self.scope_h / self.out_h
        self.meter_per_out_pixel_w = self.scope_w / self.out_w
        self.ignore_index = ignore_index
        self.nms_thresh = nms_thresh
        self.ap_score_thresh = ap_score_thresh
        self.gt_name = gt_name

    def __call__(
        self,
        pred: Mapping,
        label: Dict = None,
    ) -> Dict:
        first_key = list(pred.keys())[0]
        batch_size = pred[first_key][0].shape[0]
        pred_pts_gather = []
        for batch_id in range(batch_size):
            pred_pts = []
            cls_id_offset = 0
            for group in self.cls_group_map:
                pred_stats_group = predict_preprocess(batch_id, group, pred)
                if label is not None and self.gt_name in label:
                    label_cls = label[self.gt_name][group]["cls"][batch_id]
                    label_prob = torch.max(label_cls, dim=0, keepdim=True)[0]
                    label_prob = label_prob.cpu().numpy()
                    pred_stats_group["cls"][
                        label_prob == self.ignore_index
                    ] = self.ignore_index
                pred_stats_group["cls"][
                    (pred_stats_group["prob"] > self.ap_score_thresh)
                    * (pred_stats_group["cls"] != self.ignore_index)
                ] += cls_id_offset
                pred_pts_group = cal_vcs_pts(
                    pred_stats_group,
                    self.ap_score_thresh,
                    self.vcs_range,
                    self.meter_per_out_pixel_h,
                    self.meter_per_out_pixel_w,
                    self.ignore_index,
                    is_gt=False,
                )
                pred_pts_group = nms_pts(pred_pts_group, self.nms_thresh)
                pred_pts.extend(pred_pts_group)
                cls_id_offset += len(self.cls_group_map[group])
            pred_pts_gather.append(pred_pts)
        return {"crosspt_predict": pred_pts_gather}


@OBJECT_REGISTRY.register
class ANCPadPostprocess(torch.nn.Module):
    """Apply pad of data in pred_dict.

    Args:
        data_name: name of data to apply pad.
        padding: the padding size. If is int, uses
            the same padding in all boundaries. If a 4-tuple, uses
            (padding_left, padding_right, padding_top, padding_bottom).

    """

    def __init__(self, data_name: str, padding: Tuple[int, int, int, int]):
        super(ANCPadPostprocess, self).__init__()
        self.data_name = data_name
        self.padding = padding

    def forward(self, pred_dict: Mapping, *args):
        if isinstance(pred_dict[self.data_name], list):
            pad_datas = []
            for each_data in pred_dict[self.data_name]:
                pad_datas.append(F.pad(each_data, self.padding))
            pred_dict[self.data_name] = pad_datas
        elif isinstance(pred_dict[self.data_name], torch.Tensor):
            pred_dict[self.data_name] = F.pad(
                pred_dict[self.data_name], self.padding
            )
        else:
            raise TypeError("only support torch.tensor or list[torch.tensor]")
        return pred_dict

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
