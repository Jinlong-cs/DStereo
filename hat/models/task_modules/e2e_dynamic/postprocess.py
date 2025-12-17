import copy
from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torchvision.ops.boxes import nms

from hat.core.box3d_utils import bev3d_let_nms
from hat.core.box_utils import box_center_to_corner
from hat.models.task_modules.e2e_dynamic.e2e_dynamic_util import (  # noqa
    coord_bev2vcs,
    decode_psc_rot,
    dict_select_keep_dim,
    e2e_instance_bev2vcs,
    get_coef_of_psc,
    instance_convert_yaw,
)
from hat.registry import OBJECT_REGISTRY

__all__ = ["TrackerPostProcess", "TrackerManager"]


@OBJECT_REGISTRY.register
class TrackerPostProcess(nn.Module):
    """PostProcess module for track, traj, velocities task.

    Args:
        vcs_range: The range of vcs coordinate.
        decode_rot_setting: The setting of PSC.
    """

    def __init__(
        self,
        vcs_range: Sequence,
        decode_rot_setting: Optional[Dict] = None,
    ):
        super().__init__()
        self.vcs_range = vcs_range
        self.decode_rot_setting = None
        if decode_rot_setting is not None:
            if "psc_rot" in decode_rot_setting:
                psc_rot_setting = decode_rot_setting.pop("psc_rot")
                assert "N_steps_PSC_rot" in psc_rot_setting
                coef_sin, coef_cos = get_coef_of_psc(
                    psc_rot_setting["N_steps_PSC_rot"], return_tensor=True
                )
                psc_rot_setting["coef_sin"] = coef_sin
                psc_rot_setting["coef_cos"] = coef_cos
                decode_rot_setting["psc_rot"] = psc_rot_setting
            self.decode_rot_setting = decode_rot_setting

    @torch.no_grad()
    def forward(self, track_instances: dict):
        """Perform the computation."""

        out_logits = track_instances["pred_logits"]

        if out_logits.size(-1) == 3:
            prob = out_logits.sigmoid()
            # prob = out_logits[...,:1].sigmoid()
            scores, labels = prob.max(-1)
        else:
            prob = out_logits.softmax(dim=-1)[..., :-1]
            scores, labels = prob.max(-1)

        track_instances.update(
            {
                "boxes": track_instances["pred_boxes"],
                "scores": track_instances["scores"],
                "labels": labels,
                "yaws": track_instances["pred_yaws"],
                "bev_loc_z": track_instances["pred_zheights"][..., 0],
                "heights": track_instances["pred_zheights"][..., 1],
            }
        )
        if "pred_TrajLogits" in track_instances.keys():
            pred_TrajRegs = (
                track_instances["pred_TrajRegs"]
                .permute(0, 2, 1, 3, 4)
                .squeeze(0)
            )
            pred_TrajLogits = (
                track_instances["pred_TrajLogits"]
                .permute(0, 2, 1, 3)
                .squeeze(0)
                .squeeze(-1)
            )
            TrajScores = pred_TrajLogits.softmax(dim=-1)
            track_instances.update(
                {
                    "TrajRegs": pred_TrajRegs,
                    "TrajScores": TrajScores,
                }
            )
        if "pred_velocities" in track_instances.keys():
            track_instances.update(
                {"velocities": track_instances["pred_velocities"]}
            )
        if "pred_boxes_kalerman" in track_instances.keys():
            track_instances.update(
                {"boxes": track_instances["pred_boxes_kalerman"]}
            )
        if "pred_velocities_kalerman" in track_instances.keys():
            track_instances.update(
                {"velocities": track_instances["pred_velocities_kalerman"]}
            )

        track_instances = e2e_instance_bev2vcs(track_instances, self.vcs_range)

        if self.decode_rot_setting is not None:
            track_instances = instance_convert_yaw(
                track_instances,
                self.decode_rot_setting,
                use_sigmoid=True,
            )

        for key in [
            "pred_logits",
            "pred_boxes",
            "pred_yaws",
            "pred_zheights",
            "pred_track_scores",
            "pred_TrajRegs",
            "pred_TrajLogits",
            "pred_velocities",
            "pred_boxes_kalerman",
            "pred_velocities_kalerman",
            "ref_pts",
            "query_embed",
            "output_embedding",
            "matched_gt_idxes",
            "save_period",
            "mem_bank",
            "mem_padding_mask",
            "mem_coord",
            "mem_velo",
            "mem_acce",
        ]:
            if key in track_instances.keys():
                track_instances.pop(key)

        return track_instances


@OBJECT_REGISTRY.register
class TrackerManager(object):
    """Used to manage the tracks.

    Args:
        cls_score_thr: dict contains each cls's score_thresh and
            filter_score_thresh.

            .. code-block:: none

                score_thresh: score thr for newbone tracks.

                filter_score_thresh: score thr for previous tracks,
                    common lower than score_thresh.

        miss_tolerance: the tolerance frames for  tracks disappear time.
            Defaults to 5.
        nms_setting: Dict contains information about which nms to use,
            it's threshold, and whether to do class-agnostic NMS.
            key "letnms" or "bevnms" is the type of nms, the value is
            it's param, "letnms" and "bevnms" can not use simultaneously,
            "agnostic" specify whether do class-agnostic NMS or not,
            default is False if not specified, "topk_deploy" specify the
            topk for deploy, it's for compile, default is 120 if not specified.
            e.g.

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

                or

                {
                    "bevnms": {
                        "nms_thresh": 0.5,

                    },

                    "agnostic" : False,
                    "topk_deploy": 120,

                }

        decode_rot_setting: dict contains the param to decode rotation_yaw
            to (cosine, sine). Now only support 'psc_rot' where
            'N_steps_PSC_rot' represents the number of angular phases.
            An example, {'psc_rot':{'N_steps_PSC_rot': 3}}.

        vcs_range: (bottom, right, top, left).
        valid_range: (bottom, right, top, left) Valid rangeis used to filter
            out the targets in the boundary part, which can alleviate the
            case of target staying on the boundary caused by the trajectory
            survival time.
        traj_thickness: Used to highlight the trajectory confidence in nms.
            Specifically, the predicted target is filtered first, then the
            confidence of the retained trajectory is increased by
            traj_thickness, and finally sent to the nms module to ensure
            that the trk box is preferentially reserved when the trk box and
            det box overlap.
    """

    def __init__(
        self,
        cls_score_thr: dict,
        miss_tolerance: int = 5,
        nms_setting: Dict = None,
        decode_rot_setting: Mapping = None,
        vcs_range: List = None,
        valid_range: List = None,
        traj_thickness: float = 0,
    ):

        self.cls_score_thr = cls_score_thr
        self.score_thresh = self._get_threshold("score_threshold")
        self.filter_score_thresh = self._get_threshold("filter_score_thresh")
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0
        self.nms_setting = nms_setting
        self.traj_thickness = traj_thickness
        if decode_rot_setting is not None:
            if "psc_rot" in decode_rot_setting:
                coef_sin, coef_cos = get_coef_of_psc(
                    decode_rot_setting["psc_rot"]["N_steps_PSC_rot"],
                    return_tensor=True,
                )
                decode_rot_setting["psc_rot"].update(
                    {"coef_sin": coef_sin, "coef_cos": coef_cos}
                )

        self.decode_rot_setting = decode_rot_setting
        self.vcs_range = vcs_range
        self.vcs_wh = [
            vcs_range[3] - vcs_range[1],
            vcs_range[2] - vcs_range[0],
        ]
        self.valid_region = {
            "left": (vcs_range[3] - valid_range[3]) / self.vcs_wh[0],
            "top": (vcs_range[2] - valid_range[2]) / self.vcs_wh[1],
            "right": (vcs_range[3] - valid_range[1]) / self.vcs_wh[0],
            "bottom": (vcs_range[2] - valid_range[0]) / self.vcs_wh[1],
        }

    def _get_threshold(self, key):
        ret = {}
        for cls_id, cfg in self.cls_score_thr.items():
            assert key in cfg, f"{key} not in track_manager"
            ret[cls_id] = cfg[key]
        return ret

    def clear(self):
        self.max_obj_id = 0

    def __call__(self, track_instances: dict, out: dict):
        # score大于阈值表示当前帧更新成功
        assert "classes" in track_instances

        trk_mask = track_instances["obj_idxes"] >= 0
        for cls in self.score_thresh.keys():
            cls_mask = (track_instances["classes"] == cls) & (
                track_instances["scores"] >= self.score_thresh[cls]
            )
            track_instances["disappear_time"][cls_mask] = 0

        for i in range(len(track_instances["scores"])):
            obj_cls = track_instances["classes"][i].item()
            if (
                track_instances["obj_idxes"][i] == -1
                and track_instances["scores"][i] >= self.score_thresh[obj_cls]
            ):
                # 之前没有匹配过，则表示新出现的轨迹
                track_instances["obj_idxes"][i] = self.max_obj_id
                self.max_obj_id += 1
            elif (
                track_instances["obj_idxes"][i] >= 0
                and track_instances["scores"][i]
                < self.filter_score_thresh[obj_cls]
            ):
                # 之前出现的轨迹，但当前没有跟踪到
                track_instances["disappear_time"][i] += 1
                box_xy = track_instances["pred_boxes"][i][:2]
                if not (
                    box_xy[0] > self.valid_region["left"]
                    and box_xy[0] < self.valid_region["right"]
                    and box_xy[1] > self.valid_region["top"]
                    and box_xy[1] < self.valid_region["bottom"]
                ):
                    track_instances["obj_idxes"][i] = -1
                if track_instances["disappear_time"][i] >= self.miss_tolerance:
                    # Set the obj_id to -1.
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances["obj_idxes"][i] = -1
            if track_instances["obj_idxes"][i] >= 0:
                track_instances["appear_time"][i] += 1

        pred_scores = copy.deepcopy(track_instances["scores"])
        pred_scores[trk_mask] += self.traj_thickness

        # select valid instances to speed up nms
        active_idxes = track_instances["obj_idxes"] >= 0
        track_instances = dict_select_keep_dim(track_instances, active_idxes)
        pred_scores = pred_scores[active_idxes]
        out = dict_select_keep_dim(out, active_idxes)

        pred_yaws = track_instances["pred_yaws"]
        pred_boxes = track_instances["pred_boxes"]
        pred_labels = track_instances["classes"]

        if pred_yaws.ndim == 2:
            if pred_yaws.size(-1) > 2:
                assert (
                    self.decode_rot_setting is not None
                ), "Please check the shape of pred_yaw, decode rot setting should be provided."  # noqa
                psc_rot_setting = self.decode_rot_setting["psc_rot"]
                pred_yaws = decode_psc_rot(
                    pred_yaws,
                    psc_rot_setting["coef_sin"],
                    psc_rot_setting["coef_cos"],
                    psc_rot_setting["N_steps_PSC_rot"],
                    True,  # for pred, convert the yaw after norm by sigmoid.
                )
            elif pred_yaws.size(-1) == 2:
                pred_yaws = torch.atan2(pred_yaws[:, 1], pred_yaws[:, 0])
            else:
                pred_yaws = pred_yaws.squeeze(1)

        pred_boxes = coord_bev2vcs(pred_boxes, self.vcs_range)
        pred_boxes = torch.cat((pred_boxes, pred_yaws.unsqueeze(1)), dim=-1)

        if self.nms_setting is not None:
            agnostic = self.nms_setting.get("agnostic", True)
            if "let_nms" in self.nms_setting:
                let_nms_param = self.nms_setting["let_nms"]
                keep = bev3d_let_nms(
                    pred_boxes,
                    pred_scores,
                    let_nms_param,
                    pred_labels,
                    agnostic,
                )
            elif "nms" in self.nms_setting:
                thresh = self.nms_setting["nms"]["nms_thresh"]
                pred_boxes = box_center_to_corner(pred_boxes[..., :4])
                keep = nms(pred_boxes, pred_scores, thresh)
            else:
                raise ValueError(
                    f"nms_setting is not supported yet, {self.nms_setting}"
                )
            mask = torch.zeros_like(pred_scores)
            mask[keep] = 1
            track_instances["obj_idxes"][mask < 0.5] = -1

        track_instances.pop("classes")
        # pred yaws not needed.
        track_instances.pop("pred_yaws")
        return track_instances, out
