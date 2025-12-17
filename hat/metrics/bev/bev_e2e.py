# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import json
import logging
import os
import shutil
import time
from collections import defaultdict
from copy import deepcopy
from typing import Mapping, Optional, Sequence, Union

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from hat.metrics.bev.e2e_dynamic_metric import e2e_dynamic_bevtracking_eval
from hat.metrics.bev.e2e_dynamic_metric_utils import (
    MetricTrajpred,
    dict_select,
    e2e_dynamic_bbox_eval,
)
from hat.metrics.bev_3d import BEVDetEval
from hat.models.task_modules.e2e_dynamic.e2e_dynamic_util import (  # noqa
    e2e_instance_bev2vcs,
    get_coef_of_psc,
    instance_convert_yaw,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import get_dist_info
from hat.visualize.bev_e2e import ANCBevE2EVisualize

__all__ = ["BEVE2Eval"]

logger = logging.getLogger(__name__)
PACK_NAME_LENGTH = 16  # e.g. len("H3165_20220307_D")
VIDEO_FPS = 10


def collect_data_for_bev3d(
    batch: dict,
    output: dict,
    gt_cids: list,
    det_cids: list,
    timestamps: list,
    eval_all_category: bool = False,
):
    """Collect data for BEV3D metric calculate in update stage.

    The GT and network output formats for end-to-end perception and
    trajectory prediction tasks are different from BEV3D. The detection task
    is a sub-task that must be selected in the e2e task. In order to compare
    the detection indicators in BEV3D and E2E, we directly call the script
    for evaluating detection in BEV3D to evaluate the detection indicators
    in the E2E task. In order to achieve this goal, the data format needs to
    be converted.

    Args:
        batch: batch data of dataset.
        output: the model's output.
        gt_cids: eval_category_ids of ground truth.
        det_cids: eval_category_ids of pred's output.
        vcs_range: used to transfer img_coord to vcs_coord
        eval_all_category: whether eval all category,
            i.e., take all category as one class.

    """

    _gt_group_by_cid = {cid: [] for cid in gt_cids}
    _det_group_by_cid = {cid: [] for cid in det_cids}
    _timestamps = timestamps

    for bs, det in enumerate(output):
        new_det = {}
        new_det["appear_time"] = det["appear_time"]
        new_det["bev3d_score"] = det["scores"]
        new_det["bev3d_cls_id"] = (
            det["labels"] if (not eval_all_category) else det["labels"] * 0
        )
        new_det["bev3d_loc_z"] = det["bev_loc_z"]
        if "velocities" in det:
            new_det["bev3d_velocities"] = det["velocities"]
        new_det["bev3d_ct"] = det["boxes"][:, :2]
        new_det["bev3d_dim"] = np.concatenate(
            (det["heights"][:, None], det["boxes"][:, [3, 2]]), axis=-1
        )
        new_det["bev3d_rot"] = np.arctan2(
            det["yaws"][..., 1], det["yaws"][..., 0]
        )
        for obj_idx in range(new_det[list(new_det.keys())[0]].shape[0]):
            pred_items = {key: val[obj_idx] for key, val in new_det.items()}
            pred_items["timestamp"] = timestamps[bs]
            cid = pred_items["bev3d_cls_id"].tolist()
            if cid in _det_group_by_cid:
                _det_group_by_cid[cid] += [pred_items]

    for cid in det_cids:
        for bs, gt_instance in enumerate(batch):
            if eval_all_category:
                labels = gt_instance["labels"] * 0
            else:
                labels = gt_instance["labels"]

            cur_cid_idxs = np.where(gt_instance["labels"] == cid)[0]

            boxes = gt_instance["boxes"]
            vcs_loc_ = np.concatenate(
                (boxes[:, :2], gt_instance["bev_loc_z"][:, None]), axis=-1
            )
            vcs_dim_ = np.concatenate(
                (gt_instance["heights"][:, None], boxes[..., [3, 2]]), axis=-1
            )
            gt_cur_cid = {
                "vcs_loc_": vcs_loc_[cur_cid_idxs],
                "vcs_cls_": labels[cur_cid_idxs],
                "vcs_rot_z_": (
                    np.arctan2(
                        gt_instance["yaws"][cur_cid_idxs][..., 1],
                        gt_instance["yaws"][cur_cid_idxs][..., 0],
                    )
                ),
                "vcs_dim_": vcs_dim_[cur_cid_idxs],
                "vcs_ignore_": np.zeros(len(cur_cid_idxs)).astype(np.bool),
                "vcs_visible_": np.zeros(len(cur_cid_idxs)),
                "obj_idxes": gt_instance["obj_idxes"][cur_cid_idxs],
            }
            if "velocities" in gt_instance:
                gt_cur_cid.update(
                    {
                        "vcs_velocities": (
                            gt_instance["velocities"][cur_cid_idxs]
                        ),
                    }
                )
            num_objs_gt = gt_cur_cid["vcs_cls_"].shape[0]
            for gt_obj_idx in range(num_objs_gt):
                gt_items = {
                    key: val[gt_obj_idx] for key, val in gt_cur_cid.items()
                }
                gt_items["timestamp"] = timestamps[bs]
                _gt_group_by_cid[cid] += [gt_items]
    return _gt_group_by_cid, _det_group_by_cid, _timestamps


def collect_data_for_e2e(
    batch: dict,
    output: list,
    vcs_range: Sequence,
    gt_max_depth: float,
    score_threshold: float,
    filter_vcs_range: Union[Optional[Sequence[float]], Optional[dict]] = None,
    ego_ignore_range: Sequence = None,
    decode_rot_setting: Mapping = None,
    prefix: str = "e2e_dynamic_detection",
):
    """Prepare the data used for e2e evaluation.

    This function performs the following conversions:
    - filter the gt and outputs according to the valid vcs_range;
    - filter the ignored gts;
    - change the bevimg coord to vcs coord;
    - collect the predictions in one clip. As the output format in clip,
      the predicted values named in sequence, we rename the variables and
      collect them in batch format.

    Args:
        batch: batch data of dataset
        output_dict: the model's output.
        vcs_range: used to tansfer bevimg coord to vcs coord.
        gt_max_depth: max depth used to evaluate the results.
        score_threshold: score threshold used to filter the predictions.
        filter_vcs_range: valid vcs range in the validation. Could be a dict
            containing valid vcs range for every class, or a list containing
            valid vcs range for all classes.
        ego_ignore_range: ignore the nearest region around ego.
        decode_rot_setting: dict contains the param to decode rotation_yaw
            to (cosine, sine)
        prefix: task_name, used to filter out the outputs.

    """

    def _filter_instance(
        instance: dict,
        filter_with_scores: bool,
    ):
        """Filter object instance.

        Filter object in trk instance according to filter_vcs_range,
        ego_ignore_range and score.

        Args:
            instance: track instance, keys as below:
                boxes: instance vcs location and shape, shape: N * 4,
                    in the format of [x, y, w, l].
                scores: the confidence of instance, shape: N.
                labels: instance lalesl, shape: N.
                obj_idxes: the index of instance, optional, shape: N.
            filter_with_scores: whether filter instance according to scores.
        """

        device = instance["boxes"].device
        mask = torch.ones((len(instance["boxes"]),), dtype=bool).to(device)
        if filter_vcs_range is not None:
            if isinstance(filter_vcs_range, dict):
                mask = torch.zeros((len(instance["boxes"]),), dtype=bool).to(
                    device
                )
                for label, vcs_range_class in filter_vcs_range.items():
                    mask_range = torch.logical_and(
                        torch.logical_and(
                            instance["boxes"][:, 0] > vcs_range_class[0],
                            instance["boxes"][:, 0] < vcs_range_class[2],
                        ),
                        torch.logical_and(
                            instance["boxes"][:, 1] > vcs_range_class[1],
                            instance["boxes"][:, 1] < vcs_range_class[-1],
                        ),
                    )
                    mask_label = instance["labels"] == label
                    mask_class = torch.logical_and(mask_range, mask_label)
                    mask = torch.logical_or(mask, mask_class)
            else:
                mask_range = torch.logical_and(
                    torch.logical_and(
                        instance["boxes"][:, 0] > filter_vcs_range[0],
                        instance["boxes"][:, 0] < filter_vcs_range[2],
                    ),
                    torch.logical_and(
                        instance["boxes"][:, 1] > filter_vcs_range[1],
                        instance["boxes"][:, 1] < filter_vcs_range[-1],
                    ),
                )
                mask = torch.logical_and(mask, mask_range)
        if ego_ignore_range is not None:
            mask1 = torch.logical_and(
                torch.logical_and(
                    instance["boxes"][:, 0] >= ego_ignore_range[0],
                    instance["boxes"][:, 0] <= ego_ignore_range[2],
                ),
                torch.logical_and(
                    instance["boxes"][:, 1] >= ego_ignore_range[1],
                    instance["boxes"][:, 1] <= ego_ignore_range[-1],
                ),
            )
            mask = torch.logical_and(mask, ~mask1)
        if "obj_idxes" in instance:
            mask = torch.logical_and(mask, instance["obj_idxes"] >= 0)
        if filter_with_scores:
            if "scores" in instance:
                mask = torch.logical_and(
                    mask, instance["scores"] >= score_threshold
                )
        mask = torch.logical_and(
            mask, instance["boxes"][:, 0].abs() <= gt_max_depth
        )
        new_instance = {}
        for k, v in instance.items():
            new_instance[k] = v[mask].to(torch.device("cpu")).numpy()
        return new_instance

    num_frames_per_clip = len(batch["motr_targets"][0]["bev_tracking"])
    # filter the invalid instances in each timestamp
    batch_gt_instances, batch_det_instances, batch_timestamps = [], [], []
    for idx in range(num_frames_per_clip):
        timestamp = int(batch["timestamp"][0].cpu().numpy()[idx] * 1000)
        batch_timestamps.append(timestamp)
        gt_instance = batch["motr_targets"][0]["bev_tracking"][idx]
        gt_instance = e2e_instance_bev2vcs(
            copy.deepcopy(gt_instance), vcs_range
        )
        gt_instance = instance_convert_yaw(
            gt_instance, decode_rot_setting, use_sigmoid=False
        )
        gt_instance = _filter_instance(gt_instance, filter_with_scores=False)
        batch_gt_instances.append(gt_instance)
        dt_instance = _filter_instance(
            copy.deepcopy(output[idx]), filter_with_scores=True
        )
        batch_det_instances.append(dt_instance)
    return batch_gt_instances, batch_det_instances, batch_timestamps


@OBJECT_REGISTRY.register_module
class BEVE2Eval(BEVDetEval):
    """The BEV E2E tasks eval metrics.

    The BEV E2E tasks metric calculation is based on the BEV 3D eval metric,
    for more details please refer to BEVDetEval (HAT/hat/metrics/bev_3d.py)
    evaluate detection by default.

    Args:
        classes: class names for evaluation, such as ["car", "bicycle", "pedestrian"]  # noqa
        vcs_range: vcs_range
        e2e_predefined_classes: Support the storage of three types
            (car,bicycle,pedestrian) of target tracking results
        eval_dir_path: Dir path to save eval results.
        e2e_eval_tracking: evaluation on tracking.
        e2e_eval_velocity: evaluation on velocity prediction.
        e2e_eval_trajectory: evaluation on trajectory prediction.
        e2e_submit_evs: evalution for evs submit.
        evs_setting: params setting for evs evaluation.
        filter_vcs_range: valid vcs range in the validation. Could be a dict
            containing valid vcs range for every class, or a list containing
            valid vcs range for all classes.
        eval_mode: the way used to count the metric, in ["bev_iou", "let_iou"].
        let_iou_param: the parameters used to calculate the let_iou metric.
        time_delta: time delta used to calculate the velo metric.
    """

    select_keys = [
        "ids_list",
        "gt_traj_list",
        "gt_mask_list",
        "valid_classes",
    ]

    txtmap_idx = {
        "fid": 0,
        "tid": 1,
        "cid": 2,
        "H_vcs": 10,
        "W_vcs": 11,
        "L_vcs": 12,
        "X_vcs": 13,
        "Y_vcs": 14,
        "Z_vcs": 15,
        "yaw_vcs": 16,
        "score": 17,
        "Vx_vcs": 18,
        "Vy_vcs": 19,
        "timestamp": 21,
        "age": 22,
        "ego_x": 23,
        "ego_y": 24,
        "ego_yaw": 26,
        "ego_vx": 27,
        "ego_vy": 28,
        "Ax_vcs": 29,
        "Ay_vcs": 30,
    }

    def __init__(
        self,
        eval_category_ids: Sequence[Union[int, str]],
        score_threshold: float,
        iou_threshold: float,
        gt_max_depth: float,
        metrics: Sequence[str],
        classes: Sequence[str],
        vcs_range: Sequence[float],
        e2e_predefined_classes: Sequence[str],
        time_delta: float,
        vis_setting: Optional[dict] = None,
        e2e_eval_tracking: bool = False,
        e2e_eval_velocity: bool = False,
        e2e_eval_trajectory: bool = False,
        e2e_submit_evs: bool = False,
        evs_setting: Optional[dict] = None,
        filter_vcs_range: Union[
            Optional[Sequence[float]], Optional[dict]
        ] = None,
        task_name: Optional[str] = "e2e_dynamic_detection",
        eval_mode: Optional[str] = "bev_iou",
        let_iou_param: Optional[Mapping[str, float]] = None,
        decode_rot_setting: Mapping = None,
        **kwargs,
    ):
        super().__init__(
            eval_category_ids=eval_category_ids,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            gt_max_depth=gt_max_depth,
            metrics=metrics,
            vis_setting=vis_setting,
            eval_mode=eval_mode,
            let_iou_param=let_iou_param,
            **kwargs,
        )

        self.eval_evs_in_parallel = e2e_submit_evs
        self.e2e_eval_tracking = e2e_eval_tracking or e2e_submit_evs
        self.e2e_eval_velocity = e2e_eval_velocity
        self.e2e_eval_trajectory = e2e_eval_trajectory
        self.vcs_range = vcs_range
        self.e2e_predefined_classes = e2e_predefined_classes
        self.delta_time = 0.1
        self.trajectory_frame_num = {}
        self.classes = classes
        self.task_name = task_name
        self.filter_vcs_range = filter_vcs_range
        self.time_delta = time_delta
        self.let_iou_param = let_iou_param
        if self.filter_vcs_range is not None:
            assert isinstance(self.filter_vcs_range, dict) or isinstance(
                self.filter_vcs_range, list
            )
        # tmp dir for process synchronization
        if self.save_path is None:
            self.save_path = "tmp_output/e2e_dynamic"
        self.tmp_dir = os.path.join(self.save_path, "tmp")
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        assert eval_mode in [
            "bev_iou",
            "let_iou",
        ], f"eval mode [{eval_mode}] not supported!"
        self.eval_mode = eval_mode

        if self.e2e_eval_trajectory:
            filter_by_token = True
            self.filter_by_token = filter_by_token
            self.trajpred_eval = MetricTrajpred()

        # Note: open vis function when vis is ready.
        if vis_setting is not None:
            self.save_video = vis_setting.get("save_video", False)
        else:
            self.save_video = False
        if self.save_video:
            assert (
                self.save_vis_dir
            ), "save_vis_dir should be provided if video need to be saved!"
        if self.save_vis_dir:
            rank, _ = get_dist_info()
            if rank == 0:
                if os.path.exists(self.save_vis_dir):
                    logging.warning("vis dir exists, remove it.")
                    shutil.rmtree(self.save_vis_dir)
            self.e2e_vis = ANCBevE2EVisualize(self.save_vis_dir, **vis_setting)
        if self.eval_evs_in_parallel:
            assert evs_setting, (
                "'evs_setting' cannot be None, "
                + "if the results is evaluated on evs."
            )
            self.evs_setting = evs_setting
            self.evs_task_name = evs_setting.get(
                "evs_task_name", "EVS_DEFAULT_TASK"
            )
            # aidi cpu queue is required for evs evaluation,
            # or the evs submit task will failed.
            self.evs_data_cfg_path = evs_setting.get(
                "evs_data_cfg_path",
                "projects/pilot/configs/datasets/e2e/evs_dataset.yaml",
            )
            with open(self.evs_data_cfg_path, "r") as f:
                self.evs_dataset_cfg = yaml.load(f, Loader=yaml.FullLoader)
                self.pack_dirs = {
                    v["pack_dir"] for v in self.evs_dataset_cfg.values()
                }
        self.decode_rot_setting = None
        if decode_rot_setting is not None:
            if "psc_rot" in decode_rot_setting:
                psc_rot_setting = decode_rot_setting.pop("psc_rot")
                assert "N_steps_PSC_rot" in psc_rot_setting
                coef_sin, coef_cos = get_coef_of_psc(
                    psc_rot_setting["N_steps_PSC_rot"],
                    return_tensor=True,
                )
                psc_rot_setting["coef_sin"] = coef_sin
                psc_rot_setting["coef_cos"] = coef_cos
                decode_rot_setting["psc_rot"] = psc_rot_setting
            self.decode_rot_setting = decode_rot_setting

    def _init_states(self):
        super()._init_states()

        self.all_gt_trajpreds = []
        self.mot_gt_dt_targets = {}
        self.cur_pack_name = ""
        self.CUR_FRAME_IDX = -1
        self.all_match_instance = {
            "gt_instances": [],
            "dt_instances": [],
            "unmatch_gt": [],
            "unmatch_dt": [],
        }

    def reset(self) -> None:
        super().reset()
        self.det_root = os.path.join(
            self.save_path, f"Eval_{self.metric_index}", "det"
        )
        os.makedirs(self.det_root, exist_ok=True)
        if self.e2e_eval_tracking:
            self.gt_root = os.path.join(
                self.save_path, f"Eval_{self.metric_index}", "trk/kf_gt"
            )
            self.dt_root = os.path.join(
                self.save_path, f"Eval_{self.metric_index}", "trk/kf_dt"
            )
            self.trk_root = os.path.join(
                self.save_path, f"Eval_{self.metric_index}", "trk"
            )
            os.makedirs(self.gt_root, exist_ok=True)
            os.makedirs(self.dt_root, exist_ok=True)
            os.makedirs(self.trk_root, exist_ok=True)
        if self.e2e_eval_trajectory:
            self.traj_root = os.path.join(
                self.save_path, f"Eval_{self.metric_index}", "traj"
            )
            os.makedirs(self.traj_root, exist_ok=True)
        if self.eval_evs_in_parallel:
            self.evs_dt_root = os.path.join(
                self.save_path, f"Eval_{self.metric_index}", "trk/evs_det"
            )
            os.makedirs(self.evs_dt_root, exist_ok=True)

    def save_det_results(self, output, timestamps):
        super().save_results(output, timestamps)

    def save_trk_results(self):
        rank, world_size = get_dist_info()
        for pack_name, pack_data in self.mot_gt_dt_targets.items():
            with open(
                os.path.join(
                    self.trk_root, f"evaluate_tracking.seqmap_rank{rank}.val"
                ),
                "a+",
            ) as f:
                f.writelines(
                    "{} -1 {} {}\n".format(
                        pack_name, 1, pack_data["frame_num"]
                    )
                )
            with open(
                os.path.join(self.gt_root, "{}.txt".format(pack_name)), "w"
            ) as f:
                f.writelines(pack_data["gt_targets"])
            with open(
                os.path.join(self.dt_root, "{}.txt".format(pack_name)), "w"
            ) as f:
                f.writelines(pack_data["dt_targets"])

            if self.eval_evs_in_parallel:
                with open(
                    os.path.join(self.evs_dt_root, "{}.txt".format(pack_name)),
                    "w",
                ) as f:
                    f.writelines(pack_data["evs_dt_targets"])
                if len(pack_data["evs_dt_targets"]) == 0:
                    self.eval_evs_in_parallel = False
                else:
                    self.evs_txt2csv(pack_name, self.evs_dt_root)

    def instance2strlist(self, instance, ego_info=None):
        targets_per_frame = []
        for idx in range(len(instance["scores"])):
            track_id = int(instance["obj_idxes"][idx])
            if track_id in self.trajectory_frame_num:
                self.trajectory_frame_num[track_id] += 1
            else:
                self.trajectory_frame_num[track_id] = 1
            info_item = " ".join(
                [
                    str(self.CUR_FRAME_IDX),
                    str(int(instance["obj_idxes"][idx])),
                    self.classes[instance["labels"][idx]],
                    "-1 -1 -1",
                    str(
                        instance["boxes"][idx][0]
                        - instance["boxes"][idx][2] / 2.0
                    ),
                    str(
                        instance["boxes"][idx][1]
                        - instance["boxes"][idx][3] / 2.0
                    ),
                    str(
                        instance["boxes"][idx][0]
                        + instance["boxes"][idx][2] / 2.0
                    ),
                    str(
                        instance["boxes"][idx][1]
                        + instance["boxes"][idx][3] / 2.0
                    ),
                    str(instance["heights"][idx]),  # H
                    str(instance["boxes"][idx][3]),  # W
                    str(instance["boxes"][idx][2]),  # L
                    str(instance["boxes"][idx][0]),  # X
                    str(instance["boxes"][idx][1]),  # Y
                    str(instance["bev_loc_z"][idx]),  # Z
                    str(
                        np.arctan2(
                            instance["yaws"][idx][1],
                            instance["yaws"][idx][0],
                        )
                    ),
                    str(instance["scores"][idx]),
                ]
            )
            if ego_info is not None:
                info_item = " ".join(
                    [
                        info_item,
                        str(instance["velocities"][idx][0])
                        if ("velocities" in instance)
                        else "0",  # Vx
                        str(instance["velocities"][idx][1])
                        if ("velocities" in instance)
                        else "0",
                        str(instance["velocities"][idx][2])
                        if ("velocities" in instance)
                        else "0",
                        str(int(ego_info["timestamp"])),
                        str(self.trajectory_frame_num[track_id]),
                        str(ego_info["ego_x"]),
                        str(ego_info["ego_y"]),
                        str(ego_info["ego_z"]),
                        str(ego_info["ego_yaw"]),
                        str(ego_info["ego_vx"]),
                        str(ego_info["ego_vy"]),
                    ]
                )

            targets_per_frame.append(info_item + "\n")
        return targets_per_frame

    def convert_to_save_format(self, batch_output, timestamps):
        save_res = defaultdict(list)
        batch_timestamps = timestamps
        for i, output in enumerate(batch_output):
            front_img_timestamp = batch_timestamps[i]
            key = os.path.join(self.cur_pack_name, str(front_img_timestamp))
            location = np.concatenate(
                (output["boxes"][:, :2], output["bev_loc_z"][:, None]), axis=-1
            )
            for j in range(len(output["scores"])):
                # filter the padded objs in data transform
                if output["scores"][j] > self.save_score_thr:
                    save_res[key].append(
                        # all the key_names are used to adap to adas_eval
                        {
                            "dimensions": np.concatenate(
                                (
                                    output["heights"][:, None],
                                    output["boxes"][:, [3, 2]],
                                ),
                                axis=-1,
                            ).tolist()[j],
                            "class_id": output["labels"][j],
                            "score": output["scores"][j],
                            "yaw": np.arctan2(
                                output["yaws"][..., 1], output["yaws"][..., 0]
                            )[j],
                            "location": location[j],
                            "timestamp": str(front_img_timestamp),
                        }
                    )
            if len(save_res[key]) < 1:
                save_res[key] = []
        return save_res

    def update_fake_data(self, eval_mode):
        for cid in self.eval_category_ids:
            for _, interval in enumerate(self.all_vis_intervals):
                prefix = "category_{}_{}_{}_".format(
                    cid, interval, self.eval_mode
                )
                device = getattr(self, prefix + "seg_tp", None).device
                seg_tp = torch.zeros(
                    len(self.all_depth_intervals),
                    dtype=torch.float64,
                    device=device,
                )
                seg_fn = torch.zeros(
                    len(self.all_depth_intervals),
                    dtype=torch.float64,
                    device=device,
                )
                seg_fp = torch.zeros(
                    len(self.all_depth_intervals),
                    dtype=torch.float64,
                    device=device,
                )
                num_valid_image = 0
                num_total_image = 0
                setattr(
                    self,
                    prefix + "seg_tp",
                    getattr(self, prefix + "seg_tp", None) + seg_tp,
                )
                setattr(
                    self,
                    prefix + "seg_fn",
                    getattr(self, prefix + "seg_fn", None) + seg_fn,
                )
                setattr(
                    self,
                    prefix + "seg_fp",
                    getattr(self, prefix + "seg_fp", None) + seg_fp,
                )
                setattr(
                    self,
                    prefix + "num_valid_image",
                    getattr(self, prefix + "num_valid_image", None)
                    + num_valid_image,
                )
                setattr(
                    self,
                    prefix + "num_total_image",
                    getattr(self, prefix + "num_total_image", None)
                    + num_total_image,
                )

                if eval_mode == "let_iou":
                    seg_al = torch.zeros(
                        len(self.all_depth_intervals),
                        dtype=torch.float64,
                        device=device,
                    )
                    setattr(
                        self,
                        prefix + "seg_al",
                        getattr(self, prefix + "seg_al", None) + seg_al,
                    )
                for metric_name in self.metrics:
                    for dep_interval in self.all_depth_intervals:
                        name = prefix + dep_interval + "_" + metric_name

                        setattr(
                            self,
                            name,
                            getattr(self, name, None)
                            + np.zeros(1, dtype=np.float64)[0],
                        )
                        setattr(
                            self,
                            name + "_num",
                            getattr(self, name + "_num", None) + 0,
                        )

    def update_bev3d(self, batch, output, timestamps, eval_mode):
        (
            _gt_group_by_cid,
            _det_group_by_cid,
            _timestamps,
        ) = collect_data_for_bev3d(
            batch=batch,
            output=output,
            gt_cids=self.gt_cids,
            det_cids=self.det_cids,
            timestamps=timestamps,
        )
        if "all" in self.eval_category_ids:
            (
                _gt_group_by_cid_all,
                _det_group_by_cid_all,
                _,
            ) = collect_data_for_bev3d(
                batch=batch,
                output=output,
                gt_cids=[0],
                det_cids=[0],
                timestamps=timestamps,
                eval_all_category="all" in self.eval_category_ids,
            )
            _gt_group_by_cid["all"] = _gt_group_by_cid_all[0]
            _det_group_by_cid["all"] = _det_group_by_cid_all[0]

        res_aps = defaultdict(list)
        matched_dict = defaultdict(list)
        for cid in self.eval_category_ids:
            gt = {
                "timestamps": _timestamps,
                "annotations": _gt_group_by_cid[cid],
            }
            det = _det_group_by_cid[cid]
            if self.filter_vcs_range is not None:
                if isinstance(self.filter_vcs_range, dict):
                    assert cid in self.filter_vcs_range
                    vcs_range_class = self.filter_vcs_range[cid]
                else:
                    vcs_range_class = self.filter_vcs_range
            else:
                vcs_range_class = None
            res = e2e_dynamic_bbox_eval(
                det,
                gt,
                self.depth_intervals,
                self.score_threshold,
                self.iou_threshold,
                self.gt_max_depth,
                vcs_range_class,
                self.enable_ignore,
                self.all_vis_intervals,
                self.ego_ignore_range,
                self.time_delta,
                self.eval_mode,
                self.let_iou_param,
            )
            matched_dict[cid] = res["counts"]["result_aps"][
                "matched_dict_in_batch"
            ]
            for vidx, interval in enumerate(self.all_vis_intervals):
                prefix = "category_{}_{}_{}_".format(
                    cid, interval, self.eval_mode
                )
                device = getattr(self, prefix + "seg_tp", None).device
                seg_tp = torch.tensor(
                    res["counts"]["gt_matched"][vidx], device=device
                )
                seg_fn = torch.tensor(
                    res["counts"]["gt_missed"][vidx], device=device
                )
                seg_fp = torch.tensor(
                    res["counts"]["redundant_det"], device=device
                )
                num_valid_image = res["counts"]["timestamp_count"]
                num_total_image = res["counts"]["total_timestamp_count"]
                setattr(
                    self,
                    prefix + "seg_tp",
                    getattr(self, prefix + "seg_tp", None) + seg_tp,
                )
                setattr(
                    self,
                    prefix + "seg_fn",
                    getattr(self, prefix + "seg_fn", None) + seg_fn,
                )
                setattr(
                    self,
                    prefix + "seg_fp",
                    getattr(self, prefix + "seg_fp", None) + seg_fp,
                )
                setattr(
                    self,
                    prefix + "num_valid_image",
                    getattr(self, prefix + "num_valid_image", None)
                    + num_valid_image,
                )
                setattr(
                    self,
                    prefix + "num_total_image",
                    getattr(self, prefix + "num_total_image", None)
                    + num_total_image,
                )

                if eval_mode == "let_iou":
                    seg_al = torch.tensor(
                        res["counts"]["gt_al"][vidx], device=device
                    )
                    setattr(
                        self,
                        prefix + "seg_al",
                        getattr(self, prefix + "seg_al", None) + seg_al,
                    )

                for metric_name in self.metrics:
                    for dep_interval, metric in zip(
                        self.all_depth_intervals, res[metric_name][interval]
                    ):
                        name = prefix + dep_interval + "_" + metric_name
                        if "drot_evs" in name:
                            if len(metric) > 0:
                                metric = np.array(metric)
                                metric = np.minimum(metric, 180 - metric)
                                metric = metric.tolist()
                        setattr(
                            self,
                            name,
                            getattr(self, name, None) + np.sum(metric),
                        )
                        setattr(
                            self,
                            name + "_num",
                            getattr(self, name + "_num", None) + len(metric),
                        )
                # ap results
                cid_aps = res["counts"]["result_aps"]
                cid_aps.pop("matched_dict_in_batch")
                res_aps[cid].append(cid_aps)

        self.save_det_results(output, timestamps)

        if self.prcurv_save_path:
            if self.eval_mode not in self.rank_json_save_path:
                self.rank_json_save_path = self.rank_json_save_path.replace(
                    ".json", f"_{self.eval_mode}.json"
                )
            with open(self.rank_json_save_path, "a") as f:
                write_item = json.dumps(res_aps)
                f.write(write_item + "\n")

        return matched_dict

    def update_tracking(
        self,
        batch_gt_instances,
        batch_dt_instances,
        timestamps,
        ego_hisArrs=None,
    ):
        for idx, (gt, dt) in enumerate(
            zip(batch_gt_instances, batch_dt_instances)
        ):
            self.CUR_FRAME_IDX += 1
            self.mot_gt_dt_targets[self.cur_pack_name]["frame_num"] += 1
            self.mot_gt_dt_targets[self.cur_pack_name][
                "gt_targets"
            ] += self.instance2strlist(gt)
            self.mot_gt_dt_targets[self.cur_pack_name][
                "dt_targets"
            ] += self.instance2strlist(dt)
            if self.eval_evs_in_parallel:
                assert (
                    ego_hisArrs is not None
                ), "ego_hisArrs cannot be None when eval_evs_in_parallel is True"  # noqa
                cur_ego_hisArrs = ego_hisArrs[idx]
                tmp_idx = -1
                ego_vx = (
                    cur_ego_hisArrs[tmp_idx][0]
                    - cur_ego_hisArrs[tmp_idx - 1][0]
                ) / self.delta_time
                ego_vy = (
                    cur_ego_hisArrs[tmp_idx][1]
                    - cur_ego_hisArrs[tmp_idx - 1][1]
                ) / self.delta_time
                ego_info = {
                    "timestamp": timestamps[idx],
                    "ego_x": cur_ego_hisArrs[tmp_idx][0],
                    "ego_y": cur_ego_hisArrs[tmp_idx][1],
                    "ego_z": 0,  # no use, placeholder
                    "ego_yaw": cur_ego_hisArrs[tmp_idx][-1],
                    "ego_vx": ego_vx,
                    "ego_vy": ego_vy,
                }

                self.mot_gt_dt_targets[self.cur_pack_name][
                    "evs_dt_targets"
                ] += self.instance2strlist(dt, ego_info)

    def update_trajectory(
        self, trajectory_pred, batch, output, batch_matched_dict=None
    ):
        match_instance = {
            "gt_instances": {},
            "dt_instances": {},
            "unmatch_gt": {},
            "unmatch_dt": {},
        }
        for idx in range(len(batch)):
            cur_gt_trajectory = {}
            _trajectory_pred = trajectory_pred[idx]
            for key, val in _trajectory_pred.items():
                if key in BEVE2Eval.select_keys:
                    if isinstance(val, torch.Tensor):
                        val = [tmp.cpu().numpy() for tmp in val]
                    cur_gt_trajectory[key] = [val]
            cur_gt_trajectory["pack_name"] = self.cur_pack_name
            self.all_gt_trajpreds.append(cur_gt_trajectory)

        for cid in self.det_cids:
            valid_idx = 0
            for idx in range(len(batch)):
                if idx not in match_instance["gt_instances"]:
                    match_instance["gt_instances"][idx] = []
                    match_instance["dt_instances"][idx] = []
                    match_instance["unmatch_gt"][idx] = []
                    match_instance["unmatch_dt"][idx] = []
                if len(batch_matched_dict[cid]) == 0:
                    continue
                # filter out the det and gt with current cid
                gt_instance = dict_select(
                    batch[idx], batch[idx]["labels"] == cid
                )
                dt_instance = dict_select(
                    output[idx], output[idx]["labels"] == cid
                )
                if len(gt_instance["scores"]) == 0:
                    if len(dt_instance["scores"]) != 0:
                        match_instance["unmatch_dt"][idx].append(dt_instance)
                    continue
                if len(dt_instance["scores"]) == 0:
                    if len(gt_instance["scores"]) != 0:
                        match_instance["unmatch_gt"][idx].append(gt_instance)
                    continue
                matched_dict = batch_matched_dict[cid][valid_idx]
                valid_idx += 1
                assert len(gt_instance["labels"]) == len(
                    matched_dict["det_assign"]
                ), "The number of gt should be equal to the len of det_assign"
                det_assigns = matched_dict["det_assign"]
                inds = np.array(list(range(len(det_assigns))))
                mask = det_assigns != -1
                det_assigns, inds = det_assigns[mask], inds[mask]
                if np.sum(mask) == 0:
                    continue
                corr_gts = dict_select(gt_instance, inds)
                match_dts = dict_select(dt_instance, det_assigns)
                match_dts.update(
                    {
                        "gt_traj_regs": corr_gts["gt_traj_regs"].copy(),
                        "gt_traj_masks": corr_gts["gt_traj_masks"].copy(),
                    }
                )
                match_instance["gt_instances"][idx].append(corr_gts)
                match_instance["dt_instances"][idx].append(match_dts)
                rest_inds = np.array(
                    list(
                        set(np.arange(len(gt_instance["obj_idxes"])))
                        - set(inds)
                    )
                )
                if len(rest_inds) > 0:
                    rest_gts = dict_select(gt_instance, rest_inds)
                    match_instance["unmatch_gt"][idx].append(rest_gts)
                rest_det_inds = np.array(
                    list(
                        set(np.arange(len(dt_instance["obj_idxes"])))
                        - set(det_assigns)
                    )
                )
                if len(rest_det_inds) > 0:
                    rest_dts = dict_select(dt_instance, rest_det_inds)
                    match_instance["unmatch_dt"][idx].append(rest_dts)
        for k in self.all_match_instance.keys():
            self.all_match_instance[k].extend(
                [match_instance[k][i] for i in range(len(batch))]
            )

    def update(self, batch, output):
        """Update the step vars.

        Args:
            batch (batch_data, task_name)
            batch_data: Dict, the GT used here.

                .. code-block:: none

                    pil_imgs: List[PIL.Image], bs x clip_length*view_num
                        resize+crop之后的图像数据

                    timestamp: tensor, bs*clip_length, 每一帧对应的时间戳

                    pack_dir: List[str], bs, 每个sample 对应的采样pack路径

                    pack_names: str, 1, 每个sample的pack名

                    motr_targets: List[sample_dict], bs

                        sample_dict结构如下:
                            bev_tracking: List[each_frame]
                                obj_idxes: tensor.long, M, 目标对应的轨迹id,
                                    -1表示不参加评测, M 表示目标的个数

                                labels: tensor.long, M, 目标对应class 标签

                                boxes: tensor, M x 4, normalized xywh
                                    in img coord

                                yaws: tensor, M x 2

                                bev_loc_z: tensor, M

                                heights: tensor, M

                                scores: tensor, M

                            trajectory_pred: List[each_frame]
                                ego_hisArrs: tensor, his x 3, 表示历史的自车信息，
                                    每一行表示vc_x, vcs_y和yaw, his表示历史帧长度

                                ids_list: array, M, 轨迹的track id

                                valid_ids: array, 有效的轨迹的id

                                valid_classes: dict[id: label],
                                    有效估计对应的class label

                                gt_traj_list: dict[id: future_num x 2 (array)]
                                    每条轨迹未来的点位置vcs坐标系

                                gt_mask_list: dict[id: future_num x 2 (array)]
                                    每条轨迹未来的点是否有效

        """
        # NOTE. Only support bs=1.
        batch = batch[0]

        # 端到端任务评测过程中，需要保证一块GPU上至少有一个pack。这里对于GPU数目大于pack数的情况，
        # 补充fake dataset 对pack数进行填补，fake dataset不参与评测。
        if batch["fake_dataset_flag"][0]:
            self.update_fake_data(self.eval_mode)
            return
        if batch["pack_names"][0] != self.cur_pack_name:
            self.cur_pack_name = batch["pack_names"][0]
            self.mot_gt_dt_targets.update(
                {
                    self.cur_pack_name: {
                        "gt_targets": [],
                        "dt_targets": [],
                        "evs_dt_targets": [],
                        "frame_num": 0,
                    }
                }
            )
            if self.save_vis_dir:
                vis_frame_path = os.path.join(
                    self.save_vis_dir, self.cur_pack_name
                )
                self.mot_gt_dt_targets[self.cur_pack_name].update(
                    {"frame_path": vis_frame_path}
                )
            self.CUR_FRAME_IDX = -1

        (
            batch_gt_instances,
            batch_det_instances,
            batch_timestamps,
        ) = collect_data_for_e2e(
            batch,
            output["e2e_dynamic"][0]["bev_stage2_e2e_dynamic_head_predict"],
            self.vcs_range,
            self.gt_max_depth,
            self.score_threshold,
            self.filter_vcs_range,
            self.ego_ignore_range,
            self.decode_rot_setting,
            self.task_name,
        )
        matched_dict = self.update_bev3d(
            batch_gt_instances,
            batch_det_instances,
            batch_timestamps,
            self.eval_mode,
        )
        if self.e2e_eval_tracking:
            odos = (
                batch["odo_info"].cpu().numpy()
                if self.eval_evs_in_parallel
                else None
            )
            self.update_tracking(
                batch_gt_instances,
                batch_det_instances,
                batch_timestamps,
                odos,
            )
        if self.e2e_eval_trajectory:
            trajectory_pred = batch["motr_targets"][0]["trajectory_pred"]
            self.update_trajectory(
                trajectory_pred,
                batch_gt_instances,
                batch_det_instances,
                matched_dict,
            )

        if self.save_vis_dir:
            num_frames_per_clip = len(batch_timestamps)
            views_num = len(batch["origin_imgs"]) // num_frames_per_clip
            imgs_data = {
                "meta_info": batch["meta_info"],
                "pack_dir": batch["pack_dir"][0],
                "origin_imgs": [
                    batch["origin_imgs"][i * views_num : (i + 1) * views_num]
                    for i in range(num_frames_per_clip)
                ],
            }
            self.e2e_vis.save_imgs(
                batch_det_instances,
                batch_gt_instances,
                imgs_data,
                batch_timestamps,
                self.cur_pack_name,
            )

    def compute(self):
        self.metric_index += 1
        # generate videos
        if self.save_video:
            logger.info("\n Saving the result videos ...\n")
            for pack_name, pack_data in self.mot_gt_dt_targets.items():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_name = os.path.join(
                    self.save_vis_dir, pack_name + ".mp4"
                )
                images = [
                    img
                    for img in os.listdir(pack_data["frame_path"])
                    if img.endswith(".jpg")
                ]
                images.sort()
                frame = cv2.imread(
                    os.path.join(pack_data["frame_path"], images[0])
                )
                height, width, _ = frame.shape
                video = cv2.VideoWriter(
                    video_name, fourcc, VIDEO_FPS, (width // 2, height // 2)
                )
                for image in tqdm(images):
                    img = cv2.imread(
                        os.path.join(pack_data["frame_path"], image)
                    )
                    resize_img = cv2.resize(
                        img,
                        (width // 2, height // 2),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    video.write(resize_img)
                cv2.destroyAllWindows()
                video.release()

        # 保存变量
        if self.e2e_eval_tracking:
            self.save_trk_results()
        rank, world_size = get_dist_info()
        if self.e2e_eval_trajectory:
            np.save(
                os.path.join(self.traj_root, f"gt_trajpreds_rank{rank}.npy"),
                self.all_gt_trajpreds,
            )
            np.save(
                os.path.join(self.traj_root, f"matched_dict_rank{rank}.npy"),
                self.all_match_instance,
            )

        recall, precision = -1, -1
        metirc_dict_all = {}
        traj_metric_norm, traj_metric_center = None, None
        if rank != 0:
            finish_file = open(
                os.path.join(self.tmp_dir, "evalflag_{}.txt".format(rank)), "w"
            )
            finish_file.close()
        else:
            for i in range(1, world_size):
                while not os.path.exists(
                    os.path.join(self.tmp_dir, "evalflag_{}.txt".format(i))
                ):
                    time.sleep(10)
                os.remove(
                    os.path.join(self.tmp_dir, "evalflag_{}.txt".format(i))
                )
            time.sleep(10)

            summary_str = f"""
                eval_detection: True
                e2e_eval_tracking: {self.e2e_eval_tracking},
                e2e_eval_trajectory: {self.e2e_eval_trajectory},
                eval_evs_in_parallel: {self.eval_evs_in_parallel},
                save_dir: {self.save_path}\n

            """
            # detection
            get_rank0 = rank == 0
            if self.prcurv_save_path and get_rank0:
                if self.eval_mode == "bev_iou":
                    all_ap_results, ap_depth_results = self.reduce_ap_resluts(
                        self.eval_mode
                    )
                elif self.eval_mode == "let_iou":
                    (
                        all_ap_results,
                        ap_depth_results,
                        max_det_rate_results_let,
                        all_apl_results_let,
                        apl_depth_results_let,
                    ) = self.reduce_ap_resluts(self.eval_mode)

                save_path_name, save_path_suffix = os.path.splitext(
                    self.json_save_path
                )
                summarize_results_path = (
                    save_path_name + f"_{self.eval_mode}" + save_path_suffix
                )
                with open(summarize_results_path, "r") as f:
                    summarize_results = json.load(f)

            metirc_dict_all = {}
            for cid in self.eval_category_ids:
                for interval in self.all_vis_intervals:
                    names, values = [], []
                    prefix = "category_{}_{}_{}_".format(
                        cid, interval, self.eval_mode
                    )
                    num_valid_image = int(
                        getattr(self, prefix + "num_valid_image", None)
                        .cpu()
                        .numpy()
                    )
                    num_total_image = int(
                        getattr(self, prefix + "num_total_image", None)
                        .cpu()
                        .numpy()
                    )
                    # porcess seg precision-recall
                    pr_results = {}
                    seg_recall = []
                    seg_precision = []
                    seg_tp = (
                        getattr(self, prefix + "seg_tp", None).cpu().numpy()
                    )
                    seg_fn = (
                        getattr(self, prefix + "seg_fn", None).cpu().numpy()
                    )
                    seg_fp = (
                        getattr(self, prefix + "seg_fp", None).cpu().numpy()
                    )

                    if self.eval_mode == "let_iou":
                        seg_precision_al = []
                        seg_al = (
                            getattr(self, prefix + "seg_al", None)
                            .cpu()
                            .numpy()
                        )

                    for i in range(len(seg_tp)):
                        tmp_precision = seg_tp[i] / (
                            seg_tp[i] + seg_fp[i] + self.eps
                        )
                        tmp_recall = seg_tp[i] / (
                            seg_tp[i] + seg_fn[i] + self.eps
                        )
                        seg_recall.append(tmp_recall)
                        seg_precision.append(tmp_precision)

                        if self.eval_mode == "let_iou":
                            tmp_precision_al = seg_al[i] / (
                                seg_tp[i] + seg_fp[i] + self.eps
                            )
                            seg_precision_al.append(tmp_precision_al)

                    pr_results["TP"] = seg_tp
                    pr_results["FP"] = seg_fp
                    pr_results["FN"] = seg_fn
                    pr_results["Recall"] = seg_recall
                    pr_results["Precision"] = seg_precision
                    tp = seg_tp.sum()
                    fn = seg_fn.sum()
                    fp = seg_fp.sum()
                    precision = tp / (tp + fp + self.eps)
                    recall = tp / (tp + fn + self.eps)
                    if self.eval_mode == "let_iou":
                        pr_results["AL"] = seg_al
                        pr_results["Precision_al"] = seg_precision_al
                        al = seg_al.sum()
                        precision_al = al / (tp + fp + self.eps)

                    det_rate = summarize_results[str(cid)]["detection_rate"]
                    if len(det_rate) == 0:
                        max_rate_score = max_rate_pre = max_rate_rec = 0.0
                    else:
                        max_det_rate = max(det_rate)
                        max_det_rate_index = det_rate.index(max_det_rate)
                        max_rate_score = summarize_results[str(cid)]["conf"][
                            max_det_rate_index
                        ]
                        max_rate_pre = summarize_results[str(cid)][
                            "precision"
                        ][max_det_rate_index]
                        max_rate_rec = summarize_results[str(cid)]["recall"][
                            max_det_rate_index
                        ]

                    logger_show_metric = deepcopy(self.metrics)

                    if self.verbose:
                        # HARD CODE: Bev3d metric has updated evaluation logic,
                        # supporting bev_iou and let_iou. Need to confirm the
                        # input of eval_mode by zongwei.zhou.
                        for metric_name in self.pr_metric[self.eval_mode]:
                            for dep_interval, metric in zip(
                                self.all_depth_intervals,
                                pr_results[metric_name],
                            ):
                                names += [
                                    prefix + dep_interval + "_" + metric_name
                                ]
                                values += [metric]

                        if self.prcurv_save_path and get_rank0:
                            for ap_metric_name in self.ap_metric:
                                for dep_interval, metric in zip(
                                    self.all_depth_intervals,
                                    ap_depth_results[cid],
                                ):
                                    names += [
                                        prefix
                                        + dep_interval
                                        + "_"
                                        + ap_metric_name
                                    ]
                                    values += [metric]

                    for metric in self.metrics:
                        for dep_interval in self.all_depth_intervals:
                            metric_name = prefix + dep_interval + "_" + metric
                            val = (
                                getattr(self, metric_name, None).cpu().numpy()
                            )
                            num = int(
                                getattr(self, metric_name + "_num", None)
                                .cpu()
                                .numpy()
                            )
                            names += [metric_name]
                            values += [val / (num + self.eps)]

                    summary_str += (
                        "~~~~ %s class %s Summary metrics %s ~~~~\n"
                        % (
                            self.name,
                            str(str(cid)),
                            self.eval_mode,
                        )
                    )
                    summary_str += "Summary:\n"
                    summary_str += "BEV_3D Overview: \n"
                    nl = "\n"
                    metirc_dict = {}
                    metirc_dict[
                        "all_depth_intervals"
                    ] = self.all_depth_intervals
                    # Only evaluation in whole interval supports PR metric.
                    # Otherwise we can only settle for using TP metric.
                    if interval == "(0,1)":
                        summary_str += f" total_images : {num_total_image} \
                            {nl} valid_images : {num_valid_image} \
                            {nl} TP: {tp} {nl} FP: {fp} {nl} FN: {fn} \
                            {nl} Recall: {recall: .5f} \
                            {nl} Precision: {precision: .5f} \
                            {nl} Threshod@MaxDetRate: {max_rate_score: .5f} \
                            {nl} Recall@MaxDetRate: {max_rate_rec: .5f} \
                            {nl} Precision@MaxDetRate: {max_rate_pre: .5f} \
                                    {nl}"
                        if self.eval_mode == "let_iou":
                            summary_str += f" AL: {al: .5f} \
                                    {nl} Precision_al: {precision_al: .5f} \
                                    {nl}"
                            metirc_dict["AL"] = al
                            metirc_dict["Precision_al"] = precision_al
                        if self.prcurv_save_path and get_rank0:
                            summary_str += f" AP: {all_ap_results[cid]: .4f} \
                                            {nl}"
                            if self.eval_mode == "let_iou":
                                summary_str += f" APL: {all_apl_results_let[cid]: .4f} \
                                            {nl}"
                        logger_show_metric += self.pr_metric[self.eval_mode]
                        metirc_dict["total_images"] = num_total_image
                        metirc_dict["valid_images"] = num_valid_image
                        metirc_dict["TP"] = tp
                        metirc_dict["FP"] = fp
                        metirc_dict["FN"] = fn
                        metirc_dict["Recall"] = recall
                        metirc_dict["Precision"] = precision

                        if self.prcurv_save_path and get_rank0:
                            metirc_dict["AP"] = all_ap_results[cid]
                            if self.eval_mode == "let_iou":
                                metirc_dict["APL"] = all_apl_results_let[cid]
                    else:
                        summary_str += f" visibility : {interval} \
                            {nl} total_images : {num_total_image} \
                            {nl} valid_images : {num_valid_image} \
                            {nl} TP: {tp} {nl} FN: {fn} \
                            {nl} Recall: {recall: .5f} \
                            {nl} Threshod@MaxDetRate: {max_rate_score: .5f} \
                            {nl} Recall@MaxDetRate: {max_rate_rec: .5f} \
                            {nl} Precision@MaxDetRate: {max_rate_pre: .5f} \
                            {nl}"
                        logger_show_metric += self.vis_metric
                        metirc_dict["visibility"] = interval
                        metirc_dict["total_images"] = num_total_image
                        metirc_dict["valid_images"] = num_valid_image
                        metirc_dict["TP"] = tp
                        metirc_dict["FN"] = fn
                        metirc_dict["Recall"] = recall
                    metirc_dict["depthwise_metric"] = {}
                    tp_value = seg_tp
                    tp_sum = tp
                    for _metric in logger_show_metric:
                        json_interval_dict = {}
                        summary_str += _metric.ljust(10)
                        value_str = "".ljust(10)
                        all_range_sum = 0
                        for _dep_interval in self.all_depth_intervals:
                            _metric_name = (
                                prefix + _dep_interval + "_" + _metric
                            )
                            if _metric_name in names:
                                name_idx = names.index(_metric_name)
                                summary_str += _dep_interval.ljust(15)
                                format = (
                                    ".3f"
                                    if _metric not in ["TP", "FP", "FN"]
                                    else ".1f"
                                )
                                value_str += (
                                    f"{values[name_idx]: {format}}".ljust(15)
                                )
                                json_interval_dict[_dep_interval] = values[
                                    name_idx
                                ]
                                tp_length = len(tp_value)
                                all_range_sum += (
                                    values[name_idx]
                                    * tp_value[name_idx % tp_length]
                                )
                        range_start = (
                            self.all_depth_intervals[0]
                            .split(",")[0]
                            .split("(")[-1]
                        )
                        range_end = (
                            self.all_depth_intervals[-1]
                            .split(",")[-1]
                            .split(")")[0]
                        )
                        all_range = f"{(int(range_start),int(range_end))}"
                        summary_str += all_range.ljust(15)
                        if _metric not in [
                            "TP",
                            "FP",
                            "FN",
                            "Recall",
                            "Precision",
                            "AP_Dist",
                        ]:
                            json_interval_dict[all_range] = (
                                all_range_sum / tp_sum
                            )
                            value_str += (
                                f"{all_range_sum / tp_sum: {format}}".ljust(15)
                            )
                        else:
                            json_interval_dict[all_range] = metirc_dict[
                                _metric
                            ]
                            value_str += (
                                f"{metirc_dict[_metric]: {format}}".ljust(15)
                            )
                        metirc_dict["depthwise_metric"][
                            _metric
                        ] = json_interval_dict
                        summary_str += "\n"
                        summary_str += value_str
                        summary_str += "\n"

                    if (
                        interval == "(0,1)"
                        and self.prcurv_save_path
                        and get_rank0
                    ):
                        for _metric in self.ap_metric:
                            json_ap_metric_dict = {}
                            summary_str += _metric.ljust(10)
                            value_str = "".ljust(10)
                            for _dep_interval in self.all_depth_intervals:
                                _metric_name = (
                                    prefix + _dep_interval + "_" + _metric
                                )
                                if _metric_name in names:
                                    name_idx = names.index(_metric_name)
                                    summary_str += _dep_interval.ljust(15)
                                    format = (
                                        ".3f"
                                        if _metric not in ["TP", "FP", "FN"]
                                        else ".1f"
                                    )
                                    value_str += (
                                        f"{values[name_idx]: {format}}".ljust(
                                            15
                                        )
                                    )
                                    json_ap_metric_dict[
                                        _dep_interval
                                    ] = values[name_idx]
                            metirc_dict["depthwise_metric"][
                                _metric
                            ] = json_ap_metric_dict
                            summary_str += all_range.ljust(15)
                            AP = metirc_dict["AP"]
                            json_interval_dict[all_range] = AP
                            value_str += f"{AP: {format}}".ljust(15)
                            summary_str += "\n"
                            summary_str += value_str
                            summary_str += "\n"
                    metirc_dict_all[cid] = metirc_dict
            logger.info(summary_str)

            with open(
                os.path.join(self.det_root, "det_results.txt"),
                "w",
            ) as fwr:
                fwr.write(summary_str)

            # trajectory
            if self.e2e_eval_trajectory:
                all_gt_trajpreds, all_match_instances = [], defaultdict(list)
                for rank in range(world_size):
                    gt_traj_path = os.path.join(
                        self.traj_root,
                        f"gt_trajpreds_rank{rank}.npy",
                    )
                    match_traj_path = os.path.join(
                        self.traj_root, f"matched_dict_rank{rank}.npy"
                    )
                    if os.path.exists(gt_traj_path) and os.path.exists(
                        match_traj_path
                    ):
                        all_gt_trajpreds.extend(
                            list(
                                np.load(
                                    gt_traj_path,
                                    allow_pickle=True,
                                )
                            )
                        )
                        rank_match_instances = np.load(
                            match_traj_path,
                            allow_pickle=True,
                        ).item()
                        for k, v in rank_match_instances.items():
                            all_match_instances[k].extend(v)

                self.trajpred_eval.clear_key_dicts()
                trajoutstr_norm, traj_metric_norm = self.trajpred_eval(
                    all_gt_trajpreds, all_match_instances
                )
                self.trajpred_eval.clear_key_dicts()
                trajoutstr_center, traj_metric_center = self.trajpred_eval(
                    all_gt_trajpreds,
                    all_match_instances,
                    FilterByLateralrange=True,
                    lateral_range=[-3, 3],
                )
                trajoutstr = trajoutstr_norm + trajoutstr_center
                logger.info(trajoutstr)

                with open(
                    os.path.join(
                        self.traj_root, "trajpred_val_metric_lateral_range.txt"
                    ),
                    "w",
                ) as fwr:
                    fwr.write(trajoutstr)
            # tracking
            if self.e2e_eval_tracking:
                pack_lists_for_eval = []
                for rank in range(world_size):
                    trk_save_path = os.path.join(
                        self.trk_root,
                        f"evaluate_tracking.seqmap_rank{rank}.val",
                    )
                    if os.path.exists(trk_save_path):
                        with open(
                            os.path.join(
                                self.trk_root,
                                f"evaluate_tracking.seqmap_rank{rank}.val",
                            ),
                            "r",
                        ) as f:
                            pack_lists_for_eval.extend(f.readlines())
                with open(
                    os.path.join(self.trk_root, "evaluate_tracking.seqmap"),
                    "w+",
                ) as f:
                    f.writelines(pack_lists_for_eval)

                if self.eval_evs_in_parallel:
                    os.system(
                        "python3 projects/fsd/tools/e2e_evs_submit.py"
                        + f" --evs_task_name {self.evs_task_name}"
                        + f" --dt_root {self.evs_dt_root}"
                        + f" --evs_data_cfg_path {self.evs_data_cfg_path}"
                    )
                trk_metric_all = e2e_dynamic_bevtracking_eval(
                    gt_dir=self.trk_root,
                    result_dir=self.trk_root,
                    predefined_classes=self.e2e_predefined_classes,
                    eval_mode=self.eval_mode,
                    let_iou_param=self.let_iou_param,
                    iou_threshold=self.iou_threshold,
                )
                for k in trk_metric_all:
                    metirc_dict_all[self.classes.index(k)].update(
                        trk_metric_all[k]
                    )

        return (
            ("e2e_eval_tracking", "e2e_eval_trajectory", "eval_evs"),
            (
                self.e2e_eval_tracking,
                self.e2e_eval_trajectory,
                self.eval_evs_in_parallel,
            ),
            (metirc_dict_all, traj_metric_norm, traj_metric_center),
        )

    def evs_txt2csv(self, pack_name, dt_root):
        pack_date = pack_name[:PACK_NAME_LENGTH]
        pack_path = None
        for pack_dir in list(self.pack_dirs):
            pack_path_ = "{}/{}/{}/wheel/".format(
                pack_dir, pack_date, pack_name[PACK_NAME_LENGTH + 1 :]
            )
            if os.path.exists(pack_path_):
                pack_path = pack_path_
                break

        assert pack_path, (
            "The provided 'pack_dir' in evs_setting may be incorrect, "
            + f"I cannot find the path: {pack_path}"
        )
        with open(os.path.join(dt_root, "{}.txt".format(pack_name)), "r") as f:
            lines = [line.strip("\n").split(" ") for line in f.readlines()]
            lines = sorted(lines, key=lambda x: int(x[21]))

        category2idx = {"car": "1", "pedestrian": "2", "bicycle": "18"}

        second_line = {
            "source": "wchangan",
            "dataset": pack_date,
            "data_type": "obs",
            "data_blocks": [
                {
                    "data_type": "obs_point",
                    "row_offset": 2,
                    "row_number": len(lines),
                    "data_info": {},
                }
            ],
            "start_time": int(lines[0][BEVE2Eval.txtmap_idx["timestamp"]]),
            "end_time": int(lines[-1][BEVE2Eval.txtmap_idx["timestamp"]]),
            "pack_name": pack_name,
        }
        csv_lines = [
            "version,1,exporter,pack2csv,exporter_version,2.0.0\n",
            json.dumps(second_line) + "\n",
            "stamp,sensor_id,id,type,point_pos,x,y,z,roll,pitch,yaw,linear_vx,linear_vy,linear_vz,angular_vx,angular_vy,angular_vz,length,width,height,acceleration_x,acceleration_y,acceleration_z,age,cipv,motion_status,motion_category,measure_status,valid_info,image_left,image_top,image_right,image_bottom,has_cutin_info,objCutInFlag,objDistInLane,cut_lane_id,cut_lane_nearest_dist,cut_lane_farthest_dist,asyn_predict_time_slot,Bus,Small_Medium_Car,Trucks,Special_vehicle,Lorry,Motor,Tiny_car,head,rear\n",  # noqa
        ]
        timestamps_speed = {}
        timestamps_acc = {}
        for line in tqdm(lines):

            timestamp = int(line[BEVE2Eval.txtmap_idx["timestamp"]])

            if timestamp not in timestamps_speed:
                delta_t = 1
                while True:
                    wheel_json = "{}.json".format(timestamp)
                    if os.path.exists(os.path.join(pack_path, wheel_json)):
                        break
                    else:
                        wheel_json = "{}.json".format(timestamp - delta_t)
                        if os.path.exists(os.path.join(pack_path, wheel_json)):
                            break
                        else:
                            wheel_json = "{}.json".format(timestamp + delta_t)
                            if os.path.exists(
                                os.path.join(pack_path, wheel_json)
                            ):
                                break
                    delta_t += 1

                with open(os.path.join(pack_path, wheel_json), "r") as f:
                    json_data = json.load(f)
                    speed = json_data["speed"]
                    ego_ax = json_data["ax"]
                    ego_ay = json_data["ay"]

                timestamps_speed.update({timestamp: speed})
                timestamps_acc.update({timestamp: [ego_ax, ego_ay]})

            line_str = ",".join(
                [
                    line[BEVE2Eval.txtmap_idx["timestamp"]],
                    "0",
                    line[BEVE2Eval.txtmap_idx["tid"]],
                    category2idx[line[BEVE2Eval.txtmap_idx["cid"]]],
                    "0",
                    line[13],
                    line[14],
                    "0",
                    "0,0",
                    line[16],
                    str(float(line[18]) - timestamps_speed[timestamp]),
                    line[19],
                    "0",
                    "0,0,0",
                    line[12],
                    line[11],
                    line[10],
                    # 加速度 x, y, z
                    str(float(line[29]) - timestamps_acc[timestamp][0])
                    if len(line) >= 30
                    else "0",
                    str(float(line[30]) - timestamps_acc[timestamp][1])
                    if len(line) >= 30
                    else "0",
                    "0",
                    line[BEVE2Eval.txtmap_idx["age"]],
                    "0,0,0,0,17344,0,0,0,0,1,0,0,-1,0,0,2000,0,0,0,0,0,0,0,0,0\n",  # noqa
                ]
            )
            csv_lines.append(line_str)
        with open("{}/{}.csv".format(dt_root, pack_name), "w") as f:
            f.writelines(csv_lines)
