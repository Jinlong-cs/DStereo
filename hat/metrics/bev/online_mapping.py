# Copyright (c) Horizon Robotics. All rights reserved.
import json
import logging
import os
import time
from typing import Dict, List, Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from aidisdk.experiment import Table
except ImportError:
    Table = None
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from torch import distributed as dist

from hat.utils.aidi import EvalResult

try:
    from horizon_plugin_pytorch import om_chamfer_distance, om_confusion_matrix
except ImportError:
    om_chamfer_distance = None
    om_confusion_matrix = None

from hat.core.affine import get_vcs2bev_img_mat
from hat.metrics.bev.online_mapping_utils import (
    chamfer_distance,
    convert_gt,
    get_offset,
    get_smoothing,
    get_target_categorys,
    get_vcs_lane,
)
from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY
from hat.visualize.online_mapping import (
    arrange_imgs,
    draw_direction,
    draw_label,
    draw_online_mapping,
    draw_raw_img,
    draw_reg,
    draw_single_task,
    put_text_on_image,
)

__all__ = ["ANCOnlineMappingMetric"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ANCOnlineMappingMetric(EvalMetric):
    """Metric for online mapping task.

    # TODO(yueyu.wang, senyao.du): improve docstring

    Args:
        eval_result_file: eval result dir, if None not save eval result
        save_infer_output_dir: infer result dir, if None not save infer result
        metric_save_path: Path to save metric result.
        save_raw_model_output: if save raw model output as numpy data
        visualize_infer_output_dir: infer result visualization dir.
            If None not visualize infer result
        visualize_eval_output_dir: eval result visualization dir.
            If None not visualize eval result
        cls_loss_type: ce_loss or focal loss
        with_offset: with offset or not, default is True
        analysis: in analysis mode
        do_eval: do bench in metric
        smoothing: in smoothing mode
        embedding_dim: the dimension of the embedding feature
        out_size: model output size.
        view_bev_size: the size of view bev results.
        image_size: image size, in pixel.(order is (h,w))
        vcs_range: vcs range, in meter.(order is
            (bottom,right,top,left), e.g.(-30, -51.2, 72.4, 51.2))
        fine_metric_cfg: the config of fine metric, None means disable.
        view_cols: the column of view img.
        view_sub_head: if view sub task head.
        result_prefix: Prefix of aidi eval result.
    """

    def __init__(
        self,
        embedding_dim: int,
        head_groups: Dict,
        out_size: Sequence[int],
        view_bev_size: Sequence[int],
        image_size: Sequence[int],
        vcs_range: Sequence[float],
        cd_threshold: int,
        metric_state_names: List[str],
        name: str = "OnlineMappingMetric",
        eval_result_file: Optional[str] = None,
        metric_save_path: Optional[str] = None,
        save_infer_output_dir: Optional[str] = None,
        save_raw_model_output: bool = False,
        visualize_infer_output_dir: Optional[str] = None,
        visualize_eval_output_dir: Optional[str] = None,
        profile_path: Optional[str] = None,
        analysis: bool = False,
        do_eval: bool = True,
        smoothing: bool = False,
        fine_metric_cfg: Optional[Dict] = None,
        view_cols: int = 3,
        view_sub_head: bool = True,
        result_prefix: str = "",
    ):
        self.om_chamfer_distance_valid = True
        self.chamfer_distance = om_chamfer_distance
        if om_chamfer_distance is None or om_confusion_matrix is None:
            self.om_chamfer_distance_valid = False
            self.chamfer_distance = chamfer_distance
            logger.info(
                "Fail to find om_chamfer_distance or om_confusion_matrix, "
                "please update horizon_plugin_pytorch>=0.16.5. "
            )
        self.analysis = analysis
        self.do_eval = do_eval
        self.smoothing = smoothing
        self.embedding_dim = embedding_dim

        self.name = name
        self.metric_state_names = metric_state_names

        self.view_cols = view_cols
        self.view_sub_head = view_sub_head

        # the composition of each point in gt lane
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

        self.head_groups = head_groups
        # generate group attr list and group main cls dims
        self.group_cls_dims = {}
        self.group_max_patch = {}
        self.gt_attr_list = {}
        self.pred_attr_list = {}
        self.sub_heads = []
        for head, head_infos in head_groups.items():
            group = head_infos["group"]
            group_head = head_groups[group]
            max_patch_segment = group_head.get("max_patch_segment", 1)
            multi_head = head_infos.get("multi", False)
            if group not in self.gt_attr_list:
                self.gt_attr_list[group] = []
                self.pred_attr_list[group] = []
            if multi_head:
                self.group_max_patch[group] = max_patch_segment
                num_class = max(
                    [v for _, v in head_infos["cls_remap"].items()]
                )
                self.group_cls_dims[group] = num_class
                self.gt_attr_list[group] += [
                    "cls",
                    "instance",
                    "r",
                    "sin",
                    "cos",
                ]
                self.pred_attr_list[group] += [
                    "prob",
                    "cls",
                    "instance",
                    "r",
                    "sin",
                    "cos",
                    "embedding",
                    "offset",
                    "direction",
                ]
            else:
                self.sub_heads.append(head)
                self.gt_attr_list[group] += [head]
                self.pred_attr_list[group] += [head + "_cls", head + "_prob"]
                # update lane pts list
                self.lane_pts_attr[head] = 2

        # update lane pts info
        self.lane_pts_attr_dim = sum(self.lane_pts_attr.values())
        pts_attr_cum = 0
        self.lane_pts_attr_start = {}
        for key in self.lane_pts_attr:
            self.lane_pts_attr_start[key] = pts_attr_cum
            pts_attr_cum += self.lane_pts_attr[key]

        self.eval_result_file = eval_result_file
        self.save_infer_output_dir = save_infer_output_dir
        self.save_raw_model_output = save_raw_model_output
        self.visualize_infer_output_dir = visualize_infer_output_dir
        self.visualize_eval_output_dir = visualize_eval_output_dir
        self.profile_path = profile_path
        self.metric_save_path = metric_save_path
        self.fine_metric_cfg = fine_metric_cfg

        self.target_categorys = get_target_categorys(head_groups)
        self.cd_threshold = cd_threshold

        self.out_h, self.out_w = out_size
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

        self.view_bev_size = view_bev_size
        self.view_bev_h, self.view_bev_w = self.view_bev_size
        self.view_mat_vcs2bev = get_vcs2bev_img_mat(
            vcs_range=vcs_range,
            bev_size=self.view_bev_size,
        )

        self.image_size = image_size
        self.image_height, self.image_width = image_size

        self.x_min = self.bottom
        self.y_max = self.left
        self.x_max = self.top
        self.y_min = self.right

        self.cls_thr_bins = 40
        self.result_prefix = result_prefix
        self.generate_grids()

        super(ANCOnlineMappingMetric, self).__init__(name=name)

    def generate_grids(self):
        coord_ys, coord_xs = np.meshgrid(
            np.linspace(
                self.left - self.res_w / 2,
                self.right + self.res_w / 2,
                self.out_w,
            ),
            np.linspace(
                self.top - self.res_h / 2,
                self.bottom + self.res_h / 2,
                self.out_h,
            ),
        )

        self.map_grids = np.concatenate(
            [coord_xs[:, :, np.newaxis], coord_ys[:, :, np.newaxis]],
            axis=2,
        )
        self.unit_res = max(self.res_h, self.res_w)

    def update(
        self,
        image_files_batch,
        origin_imgs_batch,
        gt_stats_batch,
        pred_stats_batch,
    ):
        first_group = list(gt_stats_batch.keys())[0]
        first_attr = list(gt_stats_batch[first_group].keys())[0]
        batch_size = gt_stats_batch[first_group][first_attr].shape[0]
        for batch_id in range(batch_size):
            image_files = np.array(image_files_batch)[:, batch_id]
            origin_imgs = [imgs[batch_id] for imgs in origin_imgs_batch[0]]
            gt_stats = {}
            pred_stats = {}
            for group in list(self.group_cls_dims.keys()):
                gt_stats[group] = {}
                for attr in self.gt_attr_list[group]:
                    gt_stats[group][attr] = gt_stats_batch[group][attr][
                        batch_id : batch_id + 1, ...
                    ]
            pred_lanes_raw = pred_stats_batch[batch_id]["pred_lanes_raw"]
            pred_lanes_seq = pred_stats_batch[batch_id]["pred_lanes_seq"]
            pred_stats = pred_stats_batch[batch_id]["pred_stats"]
            # label
            gt_stats = convert_gt(gt_stats, self.head_groups)
            for group in gt_stats:
                if self.group_max_patch[group] != 2:
                    continue
                for attr in gt_stats[group]:
                    gt_stats[group][attr] = gt_stats[group][attr][0, ...]
            for group in gt_stats:
                for attr in gt_stats[group]:
                    if len(gt_stats[group][attr].shape) > 3:
                        gt_stats[group][attr] = gt_stats[group][attr][0, ...]
            gt_stats = get_offset(gt_stats)
            label_lanes, _, _ = get_vcs_lane(
                gt_stats,
                head_groups=self.head_groups,
                out_h=self.out_h,
                out_w=self.out_w,
                top=self.top,
                left=self.left,
                res_h=self.res_h,
                res_w=self.res_w,
                target_categorys=self.target_categorys,
                pred_stats=pred_stats,
                embedding_dim=self.embedding_dim,
            )

            gt_lanes = label_lanes
            if self.smoothing:
                pred_lanes = get_smoothing(pred_lanes_raw, deg=3)
            elif pred_lanes_seq is not None:
                pred_lanes = pred_lanes_seq
            else:
                pred_lanes = pred_lanes_raw
            # eval
            tmp = image_files[0].split("/")
            pack_name = tmp[-3]
            image_name = os.path.splitext(tmp[-1])[0]
            if self.do_eval:
                # update score distribution
                self.update_distribution(gt_stats, pred_stats)
                # update group statistics
                for category in self.target_categorys:
                    singleton_stats = self.bench(
                        pred_lanes.get(category, []),
                        gt_lanes.get(category, []),
                        pack_name,
                        image_name,
                        category,
                    )
                    if singleton_stats is not None:
                        for state_name in singleton_stats:
                            metric_state = getattr(
                                self, f"{category}_{state_name}"
                            )
                            metric_state.append(
                                torch.tensor(
                                    singleton_stats[state_name],
                                    device=self._device_key.device,
                                )
                            )
                            setattr(
                                self,
                                f"{category}_{state_name}",
                                metric_state,
                            )

            # visual
            if self.visualize_infer_output_dir:
                tic = time.time()
                # plot raw img as n * 3 arrangements
                view_raw_img = draw_raw_img(
                    image_files,
                    origin_imgs,
                    self.image_size,
                    self.view_cols,
                )
                # plot main and sub task img list
                view_img_list = []
                # view label instance
                view_gt_instance = draw_online_mapping(
                    label_lanes,
                    self.view_bev_h,
                    self.view_bev_w,
                    self.view_mat_vcs2bev,
                    text="Label instance",
                    draw_pt=True,
                    res_h=self.res_h,
                    res_w=self.res_w,
                )
                view_img_list.append(view_gt_instance)
                # view pred instance
                view_pred_instance = draw_online_mapping(
                    pred_lanes_raw,
                    self.view_bev_h,
                    self.view_bev_w,
                    self.view_mat_vcs2bev,
                    text="PRED instance",
                    draw_pt=True,
                    res_h=self.res_h,
                    res_w=self.res_w,
                )
                view_img_list.append(view_pred_instance)
                # view raw pred instance
                view_pred_segment = draw_online_mapping(
                    pred_lanes_raw,
                    self.view_bev_h,
                    self.view_bev_w,
                    self.view_mat_vcs2bev,
                    text="PRED instance segment",
                    draw_segment=True,
                    res_h=self.res_h,
                    res_w=self.res_w,
                )
                view_img_list.append(view_pred_segment)
                # view pred instance after sequence
                view_pred_sequence = draw_online_mapping(
                    pred_lanes_seq,
                    self.view_bev_h,
                    self.view_bev_w,
                    self.view_mat_vcs2bev,
                    text="PRED instance sequence",
                    draw_pt_crosswalk=True,
                    draw_line=True,
                    res_h=self.res_h,
                    res_w=self.res_w,
                )
                view_img_list.append(view_pred_sequence)
                # view gt category
                view_gt_category = draw_online_mapping(
                    label_lanes,
                    self.view_bev_h,
                    self.view_bev_w,
                    self.view_mat_vcs2bev,
                    text="Label catogory",
                    draw_pt=True,
                    color_by_category=True,
                    res_h=self.res_h,
                    res_w=self.res_w,
                )
                view_img_list.append(view_gt_category)
                # view pred category
                view_pred_category = draw_online_mapping(
                    pred_lanes_raw,
                    self.view_bev_h,
                    self.view_bev_w,
                    self.view_mat_vcs2bev,
                    text="PRED catogory",
                    draw_pt=True,
                    color_by_category=True,
                    res_h=self.res_h,
                    res_w=self.res_w,
                )
                view_img_list.append(view_pred_category)
                # view pred category after sequence
                view_pred_category_sequence = draw_online_mapping(
                    pred_lanes_seq,
                    self.view_bev_h,
                    self.view_bev_w,
                    self.view_mat_vcs2bev,
                    text="PRED catogory sequence",
                    draw_pt_crosswalk=True,
                    draw_line=True,
                    color_by_category=True,
                    res_h=self.res_h,
                    res_w=self.res_w,
                )
                view_img_list.append(view_pred_category_sequence)
                # plot sub tasks
                if self.view_sub_head:
                    for sub_head in self.sub_heads:
                        sub_index = self.lane_pts_attr_start[sub_head]
                        sub_text = "Label " + sub_head
                        view_gt_sub_head = draw_single_task(
                            label_lanes,
                            sub_index,
                            sub_head,
                            self.head_groups,
                            self.view_bev_size,
                            self.view_mat_vcs2bev,
                            sub_text,
                        )
                        view_img_list.append(view_gt_sub_head)
                        sub_text = "PRED " + sub_head
                        view_pred_sub_head = draw_single_task(
                            pred_lanes_raw,
                            sub_index,
                            sub_head,
                            self.head_groups,
                            self.view_bev_size,
                            self.view_mat_vcs2bev,
                            sub_text,
                        )
                        view_img_list.append(view_pred_sub_head)

                if self.analysis:
                    for group in self.group_max_patch:
                        for idx in range(pred_stats[group]["cls"].shape[0]):
                            # view pred cls with prob
                            pred_cls_img = draw_label(
                                pred_stats[group]["cls"][idx, ...],
                                prob=pred_stats[group]["prob"][idx, ...],
                                text=f"pred cls with prob_{idx}",
                                out_width=self.view_bev_w,
                                out_height=self.view_bev_h,
                                color_by_category=True,
                                target_categorys=self.target_categorys,
                            )
                            view_img_list.append(pred_cls_img)
                            # view label cls with prob
                            gt_cls_img = draw_label(
                                gt_stats[group]["cls"][idx, ...],
                                prob=gt_stats[group]["prob"][idx, ...],
                                text=f"label cls with prob_{idx}",
                                out_width=self.view_bev_w,
                                out_height=self.view_bev_h,
                                color_by_category=True,
                                target_categorys=self.target_categorys,
                            )
                            view_img_list.append(gt_cls_img)
                            # view pred instance
                            pred_instance_img = draw_label(
                                pred_stats[group]["instance"][idx, ...],
                                prob=pred_stats[group]["instance"][idx, ...],
                                text=f"pred instance with prob_{idx}",
                                out_width=self.view_bev_w,
                                out_height=self.view_bev_h,
                            )
                            view_img_list.append(pred_instance_img)
                            # view pred r
                            pred_r_img = draw_reg(
                                pred_stats[group]["r"][idx, ...],
                                prob=pred_stats[group]["prob"][idx, ...],
                                text=f"pred r with prob_{idx}",
                                out_width=self.view_bev_w,
                                out_height=self.view_bev_h,
                            )
                            view_img_list.append(pred_r_img)
                            # view label r
                            label_r_img = draw_reg(
                                gt_stats[group]["r"][idx, ...],
                                prob=gt_stats[group]["r"][idx, ...],
                                text=f"label r with prob_{idx}",
                                out_width=self.view_bev_w,
                                out_height=self.view_bev_h,
                            )
                            view_img_list.append(label_r_img)
                            # view pred angle
                            pred_direction_img = draw_direction(
                                pred_stats[group]["sin"][idx, ...],
                                pred_stats[group]["cos"][idx, ...],
                                prob=pred_stats[group]["prob"][idx, ...],
                                text=f"pred direction with prob_{idx}",
                                out_width=self.view_bev_w,
                                out_height=self.view_bev_h,
                                normalize=True,
                            )
                            view_img_list.append(pred_direction_img)
                            # view label angle
                            label_direction_img = draw_direction(
                                gt_stats[group]["sin"][idx, ...],
                                gt_stats[group]["cos"][idx, ...],
                                prob=gt_stats[group]["prob"][idx, ...],
                                text=f"label direction with prob_{idx}",
                                out_width=self.view_bev_w,
                                out_height=self.view_bev_h,
                            )
                            view_img_list.append(label_direction_img)

                base_w = view_raw_img.shape[1]
                new_h = (
                    self.view_bev_h
                    * base_w
                    // (self.view_bev_w * self.view_cols)
                )
                view_img = arrange_imgs(
                    view_img_list,
                    self.view_cols,
                    new_h,
                    base_w // self.view_cols,
                )
                view_img = np.vstack([view_raw_img, view_img])
                os.makedirs(
                    self.visualize_infer_output_dir,
                    exist_ok=True,
                )
                cv2.imwrite(
                    os.path.join(
                        self.visualize_infer_output_dir,
                        pack_name + "_" + image_name + ".jpg",
                    ),
                    view_img,
                )
                print("visualize metric time cost:", time.time() - tic)

            if self.save_infer_output_dir:
                os.makedirs(
                    os.path.join(self.save_infer_output_dir, pack_name),
                    exist_ok=True,
                )
                prediction_file = os.path.join(
                    self.save_infer_output_dir,
                    pack_name,
                    image_name + ".json",
                )
                json.dump(
                    {
                        category: [lane.tolist() for lane in lanes]
                        for category, lanes in pred_lanes.items()
                    },
                    open(prediction_file, "w"),
                )
                gt_file = os.path.join(
                    self.save_infer_output_dir,
                    pack_name,
                    image_name + "_gt.json",
                )
                json.dump(
                    {
                        category: [lane.tolist() for lane in lanes]
                        for category, lanes in gt_lanes.items()
                    },
                    open(gt_file, "w"),
                )

                # save raw model output map
                if self.save_raw_model_output:
                    for group in self.cls_group_map:
                        feature_file = os.path.join(
                            self.save_infer_output_dir,
                            pack_name,
                            image_name + f"_feature_{group}.npy",
                        )
                        pred_embedding_post = pred_stats[group][
                            "embedding"
                        ].reshape(
                            (-1,) + pred_stats[group]["embedding"].shape[-2:]
                        )
                        fuse_data = np.concatenate(
                            [
                                pred_stats[group]["cls"],
                                pred_stats[group]["prob"],
                                pred_stats[group]["r"],
                                pred_stats[group]["sin"],
                                pred_stats[group]["cos"],
                                pred_embedding_post,
                            ],
                            axis=0,
                        )
                        np.save(feature_file, fuse_data.astype(np.float32))

    def bench(self, pred_lanes, gt_lanes, pack_name, image_name, category):
        num_gt_lane = len(gt_lanes)
        num_pred_lane = len(pred_lanes)
        gt_all = (
            np.vstack(gt_lanes)[:, :2] if num_gt_lane > 0 else np.empty((0, 2))
        )
        pred_all = (
            np.vstack(pred_lanes)[:, :2]
            if num_pred_lane > 0
            else np.empty((0, 2))
        )
        num_gt_pt = gt_all.shape[0]
        num_pred_pt = pred_all.shape[0]
        if self.om_chamfer_distance_valid:
            gt_all = torch.tensor(gt_all, dtype=torch.float32, device="cuda")
            pred_all = torch.tensor(
                pred_all, dtype=torch.float32, device="cuda"
            )
        cd_g2p = (
            self.chamfer_distance(
                gt_all, pred_all, direction="x_to_y", dist_thresh=10
            )
            if num_gt_lane > 0
            else 0
        )
        cd_g2p = np.clip(cd_g2p, a_min=0, a_max=3.5)  # avoid abnormal value
        cd_p2g = (
            self.chamfer_distance(
                gt_all, pred_all, direction="y_to_x", dist_thresh=10
            )
            if num_pred_lane > 0
            else 0
        )
        cd_p2g = np.clip(cd_p2g, a_min=0, a_max=3.5)  # avoid abnormal value
        cost_mat = np.zeros((num_gt_lane, num_pred_lane))
        for i in range(num_gt_lane):
            gt_lane = gt_lanes[i][:, :2]
            if self.om_chamfer_distance_valid:
                gt_lane = torch.tensor(
                    gt_lane, dtype=torch.float32, device="cpu"
                )
            for j in range(num_pred_lane):
                pred_lane = pred_lanes[j][:, :2]
                if self.om_chamfer_distance_valid:
                    pred_lane = torch.tensor(
                        pred_lane, dtype=torch.float32, device="cpu"
                    )
                cost_mat[i, j] = (
                    self.chamfer_distance(gt_lane, pred_lane, direction="bi")
                    / 2.0
                )
        gt_inds, pred_inds = linear_sum_assignment(cost_mat)

        num_tp_lane = 0
        fn_gt_ids = list(range(num_gt_lane))
        fp_pred_ids = list(range(num_pred_lane))
        tp_gt_ids = []
        tp_pred_ids = []
        num_gt_pt_inst_list = []
        num_pred_pt_inst_list = []
        cd_g2p_inst_list = []
        cd_p2g_inst_list = []
        sub_task_cls_list = {}
        for head in self.sub_heads:
            sub_task_cls_list[f"gt_{head}_cls_inst"] = []
            sub_task_cls_list[f"pred_{head}_cls_inst"] = []
            sub_task_cls_list[f"gt_{head}_cls_pts"] = []
            sub_task_cls_list[f"pred_{head}_cls_pts"] = []
        # process sub task
        for gt_ind in range(num_gt_lane):
            for head in self.sub_heads:
                attr_index = self.lane_pts_attr_start[head]
                sub_task_cls_list[f"gt_{head}_cls_pts"] += gt_lanes[gt_ind][
                    :, attr_index
                ].tolist()
                sub_task_cls_list[f"pred_{head}_cls_pts"] += gt_lanes[gt_ind][
                    :, attr_index + 1
                ].tolist()
                gt_subtask_category = np.argmax(
                    np.bincount(
                        gt_lanes[gt_ind][:, attr_index].astype(np.uint8)
                    )
                )
                pred_subtask_category = np.argmax(
                    np.bincount(
                        gt_lanes[gt_ind][:, attr_index + 1].astype(np.uint8)
                    )
                )
                sub_task_cls_list[f"gt_{head}_cls_inst"].append(
                    gt_subtask_category
                )
                sub_task_cls_list[f"pred_{head}_cls_inst"].append(
                    pred_subtask_category
                )

        for gt_ind, pred_ind in zip(gt_inds, pred_inds):
            # process main task
            if cost_mat[gt_ind, pred_ind] < self.cd_threshold:
                num_tp_lane += 1
                fn_gt_ids.remove(gt_ind)
                fp_pred_ids.remove(pred_ind)
                tp_gt_ids.append(gt_ind)
                tp_pred_ids.append(pred_ind)
                gt_lane = gt_lanes[gt_ind][:, :2]
                pred_lane = pred_lanes[pred_ind][:, :2]
                num_gt_pt_inst = gt_lane.shape[0]
                num_pred_pt_inst = pred_lane.shape[0]
                if self.om_chamfer_distance_valid:
                    gt_lane = torch.tensor(
                        gt_lane,
                        dtype=torch.float32,
                        device="cpu",
                    )
                    pred_lane = torch.tensor(
                        pred_lane,
                        dtype=torch.float32,
                        device="cpu",
                    )
                cd_g2p_inst = self.chamfer_distance(
                    gt_lane, pred_lane, direction="x_to_y"
                )
                cd_p2g_inst = self.chamfer_distance(
                    gt_lane, pred_lane, direction="y_to_x"
                )
                num_gt_pt_inst_list.append(num_gt_pt_inst)
                num_pred_pt_inst_list.append(num_pred_pt_inst)
                cd_g2p_inst_list.append(cd_g2p_inst)
                cd_p2g_inst_list.append(cd_p2g_inst)

        if self.visualize_eval_output_dir:
            img_bev = np.zeros((self.view_bev_h, self.view_bev_w, 3))

            recall = (num_tp_lane / (num_gt_lane + 1e-6),)
            precision = (num_tp_lane / (num_pred_lane + 1e-6),)

            show_info = [
                {"Recall": recall, "Precision": precision},
                {"num_gt_lane": num_gt_lane, "num_pred_lane": num_pred_lane},
                {"FN_GT": num_gt_lane - num_tp_lane},
                {"FP_PRED": num_pred_lane - num_tp_lane},
                {"TP_GT": num_tp_lane},
                {"TP_PRED": num_tp_lane},
            ]
            put_text_on_image(img_bev, show_info, vertical_margin=20)

            fn_gt_lanes = [gt_lanes[i] for i in fn_gt_ids]
            fp_pred_lanes = [pred_lanes[i] for i in fp_pred_ids]
            tg_gt_lanes = [gt_lanes[i] for i in tp_gt_ids]
            tg_pred_lanes = [pred_lanes[i] for i in tp_pred_ids]
            img_bev = draw_online_mapping(
                {"FN_GT": fn_gt_lanes, "TP_GT": tg_gt_lanes},
                self.view_bev_h,
                self.view_bev_w,
                self.view_mat_vcs2bev,
                draw_polyline=False,
                draw_pt=True,
                color_by_category=True,
                img_bev=img_bev,
                res_h=self.res_h,
                res_w=self.res_w,
            )
            img_bev = draw_online_mapping(
                {"FP_PRED": fp_pred_lanes, "TP_PRED": tg_pred_lanes},
                self.view_bev_h,
                self.view_bev_w,
                self.view_mat_vcs2bev,
                draw_polyline=False,
                draw_pt=True,
                color_by_category=True,
                img_bev=img_bev,
                res_h=self.res_h,
                res_w=self.res_w,
            )
            img_bev_gt = draw_online_mapping(
                {category: gt_lanes},
                self.view_bev_h,
                self.view_bev_w,
                self.view_mat_vcs2bev,
                draw_polyline=False,
                draw_pt=True,
                res_h=self.res_h,
                res_w=self.res_w,
            )
            img_bev_pred = draw_online_mapping(
                {category: pred_lanes},
                self.view_bev_h,
                self.view_bev_w,
                self.view_mat_vcs2bev,
                draw_polyline=False,
                draw_pt=True,
                res_h=self.res_h,
                res_w=self.res_w,
            )

            os.makedirs(
                self.visualize_eval_output_dir,
                exist_ok=True,
            )
            image_stamp = os.path.basename(image_name)
            cv2.imwrite(
                os.path.join(
                    self.visualize_eval_output_dir,
                    f"{pack_name}_{image_stamp}_{category}.jpg",
                ),
                np.concatenate([img_bev_gt, img_bev_pred, img_bev], axis=1),
            )

        for key in sub_task_cls_list:
            sub_task_cls_list[key] = np.array(sub_task_cls_list[key])

        results = {
            "num_gt_pt": num_gt_pt,
            "num_pred_pt": num_pred_pt,
            "cd_g2p": cd_g2p,
            "cd_p2g": cd_p2g,
            "num_gt_lane": num_gt_lane,
            "num_pred_lane": num_pred_lane,
            "num_tp_lane": num_tp_lane,
            "num_gt_pt_inst_list": np.array(num_gt_pt_inst_list),
            "num_pred_pt_inst_list": np.array(num_pred_pt_inst_list),
            "cd_g2p_inst_list": np.array(cd_g2p_inst_list),
            "cd_p2g_inst_list": np.array(cd_p2g_inst_list),
        }
        results.update(sub_task_cls_list)
        return results

    def get_match_num(self, x, y, thr=0.5):
        if len(x) == 0 or len(y) == 0:
            return 0, 0

        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric="l2"
        ).fit(x)
        min_y2x = x_nn.kneighbors(y)[0][:, 0]
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric="l2"
        ).fit(y)
        min_x2y = y_nn.kneighbors(x)[0][:, 0]
        x_match_num = np.sum(min_x2y < thr)
        y_match_num = np.sum(min_y2x < thr)
        return x_match_num, y_match_num

    def update_distribution(self, gt_stats, pred_stats):
        for group in gt_stats:
            gt_foreground = gt_stats[group]["cls"] > 0
            gt_pos = self.map_grids + gt_stats[group]["offset"] * self.unit_res
            gt_foreground_pos = gt_pos[gt_foreground, :]
            tp_list = []
            fp_list = []
            fn_list = []
            pred_match_list = []
            gt_match_list = []
            pred_pos = (
                self.map_grids + pred_stats[group]["offset"] * self.unit_res
            )
            for i in range(1, self.cls_thr_bins, 1):
                thr = i / float(self.cls_thr_bins)
                pred_foreground = pred_stats[group]["prob"] > thr
                pred_foreground_pos = pred_pos[pred_foreground, :]
                pred_match_num, gt_match_num = self.get_match_num(
                    pred_foreground_pos, gt_foreground_pos
                )
                tp = np.logical_and(pred_foreground, gt_foreground)
                fp = np.logical_and(pred_foreground, ~gt_foreground)
                fn = np.logical_and(~pred_foreground, gt_foreground)
                tp_list.append(tp.sum())
                fp_list.append(fp.sum())
                fn_list.append(fn.sum())
                pred_match_list.append(pred_match_num)
                gt_match_list.append(gt_match_num)
            keys = ["tp", "fp", "fn", "pred_match", "gt_match"]
            values = [
                tp_list,
                fp_list,
                fn_list,
                pred_match_list,
                gt_match_list,
            ]
            for k, v in zip(keys, values):
                metric_state = getattr(self, f"{group}_{k}")
                metric_state += torch.tensor(
                    v,
                    device=self._device_key.device,
                )
                setattr(self, f"{group}_{k}", metric_state)
            # update tp score hist
            pred_scores = pred_stats[group]["prob"][gt_foreground][:]
            tp_hist = np.histogram(
                pred_scores, bins=self.cls_thr_bins, range=(0, 1)
            )[0]
            metric_state = getattr(self, f"{group}_tp_hist")
            metric_state += torch.tensor(
                tp_hist, device=self._device_key.device
            )
            setattr(self, f"{group}_tp_hist", metric_state)

    def _init_states(self):
        for category in self.target_categorys:
            for state_name in self.metric_state_names:
                metric_name = f"{category}_{state_name}"
                self.add_state(metric_name, default=[], dist_reduce_fx=None)

        # group score
        for group in self.group_max_patch:
            self.add_state(
                f"{group}_tp_hist",
                default=torch.zeros((self.cls_thr_bins)),
                dist_reduce_fx="sum",
            )
            for state_name in ["tp", "fp", "fn", "pred_match", "gt_match"]:
                self.add_state(
                    f"{group}_{state_name}",
                    default=torch.zeros((self.cls_thr_bins - 1)),
                    dist_reduce_fx="sum",
                )

        self.add_state(
            "_device_key",
            default=torch.tensor([0.0]),
            dist_reduce_fx="sum",
        )

    def compute_sub(
        self, sub_head, output_stats, names, values, metric_states
    ):
        # get main categories of sub_head
        category_list = []
        group = self.head_groups[sub_head]["group"]
        for head, head_infos in self.head_groups.items():
            multi_head = head_infos.get("multi", False)
            if multi_head and head_infos["group"] == group:
                if "cls_list" in head_infos:
                    category_list = head_infos["cls_list"]
                else:
                    category_list = list(head_infos["cls_remap"].keys())
                category_list = [head + "_" + k for k in category_list]

        gt_cls_inst_list = []
        pred_cls_inst_list = []
        gt_cls_pts_list = []
        pred_cls_pts_list = []
        for category in self.target_categorys:
            if category in category_list:
                gt_cls_inst_list += (
                    metric_states[category][f"gt_{sub_head}_cls_inst"]
                    .astype(np.int)
                    .tolist()
                )
                pred_cls_inst_list += (
                    metric_states[category][f"pred_{sub_head}_cls_inst"]
                    .astype(np.int)
                    .tolist()
                )
                gt_cls_pts_list += (
                    metric_states[category][f"gt_{sub_head}_cls_pts"]
                    .astype(np.int)
                    .tolist()
                )
                pred_cls_pts_list += (
                    metric_states[category][f"pred_{sub_head}_cls_pts"]
                    .astype(np.int)
                    .tolist()
                )

        # get cls dims
        head_infos = self.head_groups[sub_head]
        num_class = max([v for _, v in head_infos["cls_remap"].items()])
        # point level
        if pred_cls_pts_list:
            head_conf = confusion_matrix(
                gt_cls_pts_list,
                pred_cls_pts_list,
                labels=range(0, num_class + 2),
            )
            head_conf = head_conf[1:, 1:]
        else:
            head_conf = np.zeros((num_class + 1, num_class + 1))
        num_gt_pts_per_category = head_conf.sum(axis=1)
        num_pred_pts_per_category = head_conf.sum(axis=0)
        pts_recall = np.diag(head_conf) / (
            num_gt_pts_per_category.astype(np.float64) + 1e-6
        )
        pts_presion = np.diag(head_conf) / (
            num_pred_pts_per_category.astype(np.float64) + 1e-6
        )
        pts_f1_scores = (
            2 * pts_recall * pts_presion / (pts_recall + pts_presion + 1e-6)
        )
        # compute all cls pts metric
        pts_precision_all = np.sum(np.diag(head_conf)) / (
            np.sum(num_pred_pts_per_category.astype(np.float64)) + 1e-6
        )
        # instance level
        if pred_cls_inst_list:
            head_conf = confusion_matrix(
                gt_cls_inst_list,
                pred_cls_inst_list,
                labels=range(0, num_class + 2),
            )
            head_conf = head_conf[1:, 1:]
        else:
            head_conf = np.zeros((num_class + 1, num_class + 1))
        num_gt_inst_per_category = head_conf.sum(axis=1)
        num_pred_inst_per_category = head_conf.sum(axis=0)
        recall = np.diag(head_conf) / (
            num_gt_inst_per_category.astype(np.float64) + 1e-6
        )
        presion = np.diag(head_conf) / (
            num_pred_inst_per_category.astype(np.float64) + 1e-6
        )
        f1_scores = 2 * recall * presion / (recall + presion + 1e-6)
        # compute all cls instance metric
        inst_precision_all = np.sum(np.diag(head_conf)) / (
            np.sum(num_pred_inst_per_category.astype(np.float64)) + 1e-6
        )
        inst_num_all = np.sum(num_gt_inst_per_category)

        # get head categories
        head_categorys = []
        if "cls_list" in self.head_groups[sub_head]:
            head_categorys = self.head_groups[sub_head]["cls_list"]
        else:
            head_categorys = list(
                self.head_groups[sub_head]["cls_remap"].keys()
            )

        for i in range(len(head_categorys)):
            category = sub_head + "_" + head_categorys[i]
            output_stats.update({category: {}})
            summary_str = "~~~~ %s Summary metrics ~~~~\n" % (self.name)
            summary_str += "%s:\n" % (category)
            line_format = "{:>10} {:>10} {:>10} {:>10} {:>10} " + (
                "{:>14} {:>14} {:>14} {:>10} {:>10}\n"
            )
            summary_str += line_format.format(
                "Precision_pts",
                "Recall_pts",
                "F-score_pts",
                "GT_pts",
                "Pred_pts",
                "Precision_inst",
                "Recall_inst",
                "F-score_inst",
                "GT_inst",
                "Pred_inst",
            )
            summary_str += line_format.format(
                "{:.3f}".format(pts_presion[i]),
                "{:.3f}".format(pts_recall[i]),
                "{:.3f}".format(pts_f1_scores[i]),
                "{:.0f}".format(num_gt_pts_per_category[i]),
                "{:.0f}".format(num_pred_pts_per_category[i]),
                "{:.3f}".format(presion[i]),
                "{:.3f}".format(recall[i]),
                "{:.3f}".format(f1_scores[i]),
                "{:.0f}".format(num_gt_inst_per_category[i]),
                "{:.0f}".format(num_pred_inst_per_category[i]),
            )
            logger.info(summary_str)

            output_stats[category]["Precision_pts"] = round(pts_presion[i], 4)
            output_stats[category]["Recall_pts"] = round(pts_recall[i], 4)
            output_stats[category]["F-score_pts"] = round(pts_f1_scores[i], 4)
            output_stats[category]["GT_pts_num"] = round(
                num_gt_pts_per_category[i], 4
            )
            output_stats[category]["Pred_pts_num"] = round(
                num_pred_pts_per_category[i], 4
            )
            output_stats[category]["Precision_inst"] = round(presion[i], 4)
            output_stats[category]["Recall_inst"] = round(recall[i], 4)
            output_stats[category]["F-score_inst"] = round(f1_scores[i], 4)
            output_stats[category]["GT_inst_num"] = round(
                num_gt_inst_per_category[i], 4
            )
            output_stats[category]["Pred_inst_num"] = round(
                num_pred_inst_per_category[i], 4
            )

            names.append(f"{self.name}_{category}_pts_F-score")
            values.append(output_stats[category]["F-score_pts"])
            names.append(f"{self.name}_{category}_inst_F-score")
            values.append(output_stats[category]["F-score_inst"])

        # update all cls metric
        output_stats.update({sub_head: {}})
        summary_str = "~~~~ %s Summary All metrics ~~~~\n" % (self.name)
        summary_str += "%s all:\n" % (sub_head)
        line_format = "{:>10} {:>10} {:>10}\n"
        summary_str += line_format.format(
            "Precision_pts", "Precision_inst", "Inst_num"
        )
        summary_str += line_format.format(
            "{:.3f}".format(pts_precision_all),
            "{:.3f}".format(inst_precision_all),
            "{:.0f}".format(inst_num_all),
        )
        logger.info(summary_str)
        output_stats[sub_head]["Precision_pts"] = round(pts_precision_all, 4)
        output_stats[sub_head]["Precision_inst"] = round(inst_precision_all, 4)
        output_stats[sub_head]["Inst_num"] = int(inst_num_all)
        names.append(f"{self.name}_{sub_head}_Precision_pts")
        values.append(output_stats[sub_head]["Precision_pts"])
        names.append(f"{self.name}_{sub_head}_Precision_inst")
        values.append(output_stats[sub_head]["Precision_inst"])

        return output_stats, names, values, metric_states

    def compute(self):
        if dist.get_rank() != 0:
            return None
        output_stats = {}
        names = []
        values = []
        metric_states = {}

        for category in self.target_categorys:
            metric_states[category] = {}
            for state_name in self.metric_state_names:
                metric_state = getattr(self, f"{category}_{state_name}")
                metric_state = np.hstack([item.cpu() for item in metric_state])
                metric_states[category][state_name] = metric_state

        # compute main task
        for category in self.target_categorys:
            output_stats.update({category: {}})
            cd = (
                np.sum(
                    metric_states[category]["cd_g2p"]
                    * metric_states[category]["num_gt_pt"]
                )
                + np.sum(
                    metric_states[category]["cd_p2g"]
                    * metric_states[category]["num_pred_pt"]
                )
            ) / (
                np.sum(metric_states[category]["num_gt_pt"])
                + np.sum(metric_states[category]["num_pred_pt"])
                + 1e-6
            )
            cd_g2p = np.sum(
                metric_states[category]["cd_g2p"]
                * metric_states[category]["num_gt_pt"]
            ) / (np.sum(metric_states[category]["num_gt_pt"] + 1e-6))
            cd_p2g = np.sum(
                metric_states[category]["cd_p2g"]
                * metric_states[category]["num_pred_pt"]
            ) / (np.sum(metric_states[category]["num_pred_pt"] + 1e-6))
            num_gt_lane = np.sum(metric_states[category]["num_gt_lane"])
            num_pred_lane = np.sum(metric_states[category]["num_pred_lane"])
            recall = np.sum(metric_states[category]["num_tp_lane"]) / (
                num_gt_lane + 1e-6
            )
            precision = np.sum(metric_states[category]["num_tp_lane"]) / (
                num_pred_lane + 1e-6
            )
            f_score = 2 * recall * precision / (recall + precision + 1e-6)

            num_gt_pt_inst_list = metric_states[category][
                "num_gt_pt_inst_list"
            ]
            num_pred_pt_inst_list = metric_states[category][
                "num_pred_pt_inst_list"
            ]
            cd_g2p_inst_list = metric_states[category]["cd_g2p_inst_list"]
            cd_p2g_inst_list = metric_states[category]["cd_p2g_inst_list"]
            cd_inst = (
                np.sum(cd_g2p_inst_list * num_gt_pt_inst_list)
                + np.sum(cd_p2g_inst_list * num_pred_pt_inst_list)
            ) / (
                np.sum(num_gt_pt_inst_list)
                + np.sum(num_pred_pt_inst_list)
                + 1e-6
            )
            cd_g2p_inst = np.sum(cd_g2p_inst_list * num_gt_pt_inst_list) / (
                (np.sum(num_gt_pt_inst_list) + 1e-6)
            )
            cd_p2g_inst = np.sum(cd_p2g_inst_list * num_pred_pt_inst_list) / (
                (np.sum(num_pred_pt_inst_list) + 1e-6)
            )
            summary_str = "~~~~ %s Summary metrics ~~~~\n" % (self.name)
            summary_str += "%s:\n" % (category)
            line_format = (
                "{:>10} {:>10} {:>10} {:>10} {:>10} "
                + "{:>10} {:>10} {:>10} {:>10} {:>10} {:>10}\n"
            )
            summary_str += line_format.format(
                "CD",
                "CD_g2p",
                "CD_p2g",
                "CD_inst",
                "CD_g2p_inst",
                "CD_p2g_inst",
                "Precision",
                "Recall",
                "F-score",
                "num_gt_lane",
                "num_pred_lane",
            )
            summary_str += line_format.format(
                "{:.3f}".format(cd),
                "{:.3f}".format(cd_g2p),
                "{:.3f}".format(cd_p2g),
                "{:.3f}".format(cd_inst),
                "{:.3f}".format(cd_g2p_inst),
                "{:.3f}".format(cd_p2g_inst),
                "{:.3f}".format(precision),
                "{:.3f}".format(recall),
                "{:.3f}".format(f_score),
                "{:.0f}".format(num_gt_lane),
                "{:.0f}".format(num_pred_lane),
            )
            logger.info(summary_str)

            output_stats[category]["CD"] = round(cd, 4)
            output_stats[category]["CD_g2p"] = round(cd_g2p, 4)
            output_stats[category]["CD_p2g"] = round(cd_p2g, 4)
            output_stats[category]["CD_inst"] = round(cd_inst, 4)
            output_stats[category]["CD_g2p_inst"] = round(cd_g2p_inst, 4)
            output_stats[category]["CD_p2g_inst"] = round(cd_p2g_inst, 4)
            output_stats[category]["Precision"] = round(precision, 4)
            output_stats[category]["Recall"] = round(recall, 4)
            output_stats[category]["F-score"] = round(f_score, 4)
            output_stats[category]["num_gt_lane"] = int(num_gt_lane)
            output_stats[category]["num_pred_lane"] = int(num_pred_lane)

            names.append(f"{self.name}_{category}_F-score")
            values.append(output_stats[category]["F-score"])

        # compute sub task metric
        for sub_head in self.sub_heads:
            output_stats, names, values, metric_states = self.compute_sub(
                sub_head, output_stats, names, values, metric_states
            )

        # compute score distribution
        cls_thr_list = [
            i / float(self.cls_thr_bins)
            for i in range(1, self.cls_thr_bins, 1)
        ]
        logger.info("\ncls_thr_list:{}".format(cls_thr_list))
        for group in self.group_max_patch:
            # compute precision and recall
            tp = getattr(self, f"{group}_tp").cpu()
            fp = getattr(self, f"{group}_fp").cpu()
            fn = getattr(self, f"{group}_fn").cpu()
            pred_match = getattr(self, f"{group}_pred_match").cpu()
            gt_match = getattr(self, f"{group}_gt_match").cpu()
            precision = (tp / (tp + fp + 1e-6)).numpy()
            recall = (tp / (tp + fn + 1e-6)).numpy()
            match_precision = (pred_match / (tp + fp + 1e-6)).numpy()
            match_recall = (gt_match / (tp + fn + 1e-6)).numpy()
            group_str = "{}:\n\tprecision:{}\n\trecall:{}\n".format(
                group, precision, recall
            )
            group_str += "\n\tmatch precision:{}\n\tmatch recall:{}\n".format(
                match_precision, match_recall
            )
            logger.info(group_str)
            # get best threshold
            best_thr_index = np.argmin(np.abs(match_precision - match_recall))
            best_thr = cls_thr_list[best_thr_index]
            best_thr_str = "\n\n!!!!!!!!!!{}, equal thr = {}!!!!!!!!!!!!\n\n"
            logger.info(best_thr_str.format(group, best_thr))
            # compute gt corresponding score hist
            tp_hist = getattr(self, f"{group}_tp_hist").cpu().numpy()
            tp_hist = (tp_hist / (tp_hist.sum() + 1e-6)).cumsum()
            # plot
            ax1 = plt.subplot(141)
            ax1.set_title("precision vs recall", fontsize=10)
            [label.set_fontsize(4) for label in ax1.get_xticklabels()]
            [label.set_fontsize(5) for label in ax1.get_yticklabels()]
            ax1.grid(alpha=1, linestyle="--")
            ax1.set_xticks(np.arange(0, 1.1, 0.1))
            ax1.set_yticks(np.arange(0, 1.1, 0.1))
            ax1.set_xlim(0, 1.0)
            ax1.set_ylim(0, 1.0)
            ax1.set_yticks(np.arange(0, 1.1, 0.1))
            plt.plot(cls_thr_list, precision, ls="--", c="r", label="p")
            plt.plot(cls_thr_list, recall, ls="--", c="g", label="r")
            plt.legend()
            ax2 = plt.subplot(142)
            ax2.set_title("matched P vs R", fontsize=10)
            [label.set_fontsize(4) for label in ax2.get_xticklabels()]
            [label.set_fontsize(5) for label in ax2.get_yticklabels()]
            ax2.grid(alpha=1, linestyle="--")
            ax2.set_xticks(np.arange(0, 1.1, 0.1))
            ax2.set_yticks(np.arange(0, 1.1, 0.1))
            ax2.set_xlim(0, 1.0)
            ax2.set_ylim(0, 1.0)
            plt.plot(cls_thr_list, match_precision, ls="--", c="r", label="p")
            plt.plot(cls_thr_list, match_recall, ls="--", c="g", label="r")
            plt.legend()
            ax3 = plt.subplot(143)
            [label.set_fontsize(4) for label in ax3.get_xticklabels()]
            [label.set_fontsize(5) for label in ax3.get_yticklabels()]
            align_recall = [1] + match_recall.tolist() + [0]
            align_precision = [0] + match_precision.tolist() + [1]
            align_region = 0
            for bin in range(len(align_recall) - 1):
                r_bin = align_recall[bin] - align_recall[bin + 1]
                p_bin = (align_precision[bin] + align_precision[bin + 1]) * 0.5
                align_region += p_bin * r_bin
            logger.info(
                "\n!!!!!!{}, matched ROC={:.3f}!!!!!!!\n".format(
                    group, align_region
                )
            )
            ax3.set_title(
                "matched ROC={:.2f}".format(align_region), fontsize=10
            )
            plt.plot(align_recall, align_precision, c="b")
            ax3.grid(alpha=1, linestyle="--")
            ax3.set_xticks(np.arange(0, 1.1, 0.1))
            ax3.set_yticks(np.arange(0, 1.1, 0.1))
            ax3.set_xlim(0, 1.0)
            ax3.set_ylim(0, 1.0)
            ax4 = plt.subplot(144)
            ax4.set_title("gt_score_hist", fontsize=10)
            [label.set_fontsize(4) for label in ax4.get_xticklabels()]
            [label.set_fontsize(5) for label in ax4.get_yticklabels()]
            plt.plot(cls_thr_list + [1.0], tp_hist, ls="-", c="b")
            ax4.grid(alpha=1, linestyle="--")
            ax4.set_xticks(np.arange(0, 1.1, 0.1))
            ax4.set_yticks(np.arange(0, 1.1, 0.1))
            ax4.set_xlim(0, 1.0)
            ax4.set_ylim(0, 1.0)
            plt.suptitle(f"{group}")
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            save_name = self.save_infer_output_dir + f"/../{group}_hist.png"
            plt.savefig(save_name, dpi=300, bbox_inches="tight")
            plt.close()

        # cal region based metirc
        if self.fine_metric_cfg is not None and self.eval_result_file:
            logger.info("==============om region based metric==============")
            cmd = "python3 hat/metrics/bev/om_eval_metric.py"
            cmd += " --profile {} --infer_path {} --eval_path {} --metric_save_path {}".format(  # noqa
                self.profile_path,
                self.save_infer_output_dir,
                os.path.dirname(self.eval_result_file),
                self.metric_save_path,
            )
            # use current head groups config
            head_group_dict_str = eval(str(self.head_groups))
            cmd += ' --head_groups "{}"'.format(head_group_dict_str)
            # use current input fine metric config
            fine_metric_cfg_dict_str = eval(str(self.fine_metric_cfg))
            cmd += ' --metric_cfg "{}"'.format(fine_metric_cfg_dict_str)
            fine_metric_str = os.popen(cmd).read()
            logger.info(fine_metric_str)
            res = (
                fine_metric_str.split(
                    "!!!!!!!!!!!!!!!!!! summary !!!!!!!!!!!!!!!!!!!!!"
                )[1]
                .strip()
                .split("\n")
            )
            columns = ["items"] + res[0].split()
            data = []
            for sub_res in res[1:]:
                sub_data = {}
                for k, v in zip(columns, sub_res.strip().split()):
                    if k != "items":
                        try:
                            v = float(v)
                        except Exception:
                            pass
                    sub_data.update({k: v})
                data.append(sub_data)
            tables = [
                Table(
                    name=self.result_prefix + "_" + self.name,
                    columns=columns,
                    data=data,
                )
            ]

            return EvalResult(tables=tables)
