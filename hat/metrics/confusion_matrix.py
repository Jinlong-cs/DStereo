# Copyright (c) Horizon Robotics. All rights reserved.

import glob
import logging
import os
import pickle
from collections import defaultdict
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed as dist
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix

from hat.metrics.metric_3dv_utils import rotate_iou_matching
from hat.registry import OBJECT_REGISTRY
from hat.utils.aidi import EvalResult
from hat.utils.apply_func import _as_list
from hat.utils.apply_func import convert_numpy as to_numpy
from hat.utils.distributed import get_dist_info, rank_zero_only  # noqa
from .metric import EvalMetric

__all__ = ["ConfusionMatrix", "ConfusionMatrixBEV3D"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ConfusionMatrix(EvalMetric):
    """Evaluation segmentation results.

    Args:
        seg_class: a list of classes the segmentation dataset includes，
            the order should be the same as the label.
        name: name of this metric instance for display, also used as
            monitor params for Checkpoint.
        ignore_index: the label index that will be ignored in evaluation.

    """

    def __init__(
        self,
        seg_class: List[str],
        name: str = "ConfusionMatrix",
        ignore_index: int = 255,
    ):
        self.num_classes = len(seg_class)
        self.label_ids = range(self.num_classes)
        self.seg_class = seg_class
        super(ConfusionMatrix, self).__init__(name)
        self.ignore_index = ignore_index

    def _init_states(self):

        self.add_state(
            "confusion_matrix",
            default=torch.zeros((self.num_classes, self.num_classes)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "pred_label",
            default=torch.zeros((self.num_classes,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "label",
            default=torch.zeros((self.num_classes,)),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        label: torch.Tensor,
        preds: Union[Sequence[torch.Tensor], torch.Tensor],
    ):
        """
        Update internal buffer with latest predictions.

        Note that the statistics are not available until
        you call self.get() to return the metrics.

        Args:
            preds: model output.
            label: gt.

        """
        # only one pred and one gt used in MeanIOU calculation.
        pred_label = _as_list(preds)[0].detach()

        mask = label != self.ignore_index
        pred_label = pred_label[mask].float()
        label = label[mask].float()

        cls_confusion_matrix = confusion_matrix(
            label.numpy(), pred_label.numpy(), labels=self.label_ids
        )
        area_pred_label = torch.histc(
            pred_label, bins=self.num_classes, max=self.num_classes - 1
        )
        area_label = torch.histc(
            label, bins=self.num_classes, max=self.num_classes - 1
        )

        self.confusion_matrix += cls_confusion_matrix
        self.label += area_label
        self.pred_label += area_pred_label

    def compute(self):
        """Get evaluation metrics."""

        for i in range(self.num_classes):
            self.confusion_matrix[i] = self.confusion_matrix[i] / max(
                self.label[i], 1
            )

        return (self.confusion_matrix, self.label, self.pred_label)


@OBJECT_REGISTRY.register
class ConfusionMatrixBEV3D(EvalMetric):
    """Calculate confusion matrix for BEV3D.

    Args:
        num_classes: class num for calculating confusion matrix.
        score_threshold: threshold for filtering pred results.
        iou_threshold: threshold for iou-based matching strategy.
        eval_vcs_range: valid vcs range for evaluation.
        enable_ignore: if True, pred bbox3d matched with ignored
            gt bbox3d will be ignored.
        ego_ignore_range: ego vcs range.
        name: name for different ConfusionMatrixBEV3D instance.
        id2label: Mapping from id to label name.
        pred_keys: valid keys in pred dict.
        confusion_save_path: confusion infos save path, confusion infos
            contain all frame names for each wrong type, e.g.
            "0to1"(gt is 0 but pred is 1).
        pred_save_path: model outputs save path.
        save_score_thr: threshold for filltering pred
            when saving model outputs.
        gt_id_key: key of gt's id for calculating confusion matrix.
        pred_id_key: key of pred's id for calculating confusion matrix.
        anno_name: key of anno for calculating confusion matrix.
        range_mode: mode if vcs range for calculating confusion matrix.
    """

    def __init__(
        self,
        num_classes: int,
        score_threshold: float,
        iou_threshold: float,
        gt_max_depth: float,
        eval_vcs_range: Optional[Sequence[float]] = None,
        enable_ignore: Optional[bool] = True,
        ego_ignore_range: Optional[Sequence[float]] = None,
        name: str = "ConfusionMatrix",
        id2label: Optional[dict] = None,
        pred_keys: Optional[dict] = None,
        confusion_save_path: Optional[str] = None,
        pred_save_path: Optional[str] = None,
        save_score_thr: float = 0.0,
        gt_id_key: str = "vcs_cls_",
        pred_id_key: str = "bev3d_cls_id",
        anno_name: str = "annos_bev_3d",
        range_mode: str = "wide",
    ):
        self.num_classes = num_classes
        self.label_ids = range(self.num_classes)
        self.pred_keys = pred_keys
        self.score_threshold = score_threshold
        self.eval_vcs_range = eval_vcs_range
        self.ego_ignore_range = ego_ignore_range
        self.gt_max_depth = gt_max_depth
        self.enable_ignore = enable_ignore
        self.iou_threshold = iou_threshold
        if self.pred_keys is None:
            self.pred_keys = [
                "bev3d_ct",
                "bev3d_loc_z",
                "bev3d_dim",
                "bev3d_rot",
                "bev3d_score",
                "bev3d_cls_id",
            ]
        if id2label is None:
            id2label = {}
        assert isinstance(id2label, dict), "id2label should be None or dict"
        self.id2label = id2label
        self.confusion_save_path = confusion_save_path
        self.pred_save_path = pred_save_path
        self.save_score_thr = save_score_thr
        self.gt_id_key = gt_id_key
        self.pred_id_key = pred_id_key
        self.anno_name = anno_name
        self.range_mode = range_mode
        super(ConfusionMatrixBEV3D, self).__init__(name)

    def _init_states(self):

        self.add_state(
            "confusion_matrix",
            default=torch.zeros((self.num_classes, self.num_classes)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "pred_label",
            default=torch.zeros((self.num_classes,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "label",
            default=torch.zeros((self.num_classes,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "iou_match_fn_num",
            default=torch.zeros((self.num_classes,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "iou_match_fp_num",
            default=torch.zeros((self.num_classes,)),
            dist_reduce_fx="sum",
        )

    def reset(self):
        super().reset()
        rank, word_size = get_dist_info()
        if self.confusion_save_path:
            self.rank_confusion_info_path = os.path.join(
                self.confusion_save_path,
                self.name,
                "rank_split",
                f"confusion_info_{rank}.pkl",
            )
            self.confusion_info_save_path = os.path.join(
                self.confusion_save_path,
                self.name,
                f"confusion_info_{rank}.pkl",
            )
            confusion_save_root = os.path.dirname(
                self.rank_confusion_info_path
            )
            if not os.path.exists(confusion_save_root):
                os.makedirs(confusion_save_root, exist_ok=True)

        if self.pred_save_path:
            self.rank_pkl_save_path = os.path.join(
                self.pred_save_path,
                self.name,
                "pred",
                "rank_split",
                f"rank_{rank}.pkl",
            )
            self.pkl_save_path = os.path.join(
                self.pred_save_path,
                self.name,
                "pred",
                f"pred_rank_{rank}.pkl",
            )
            pkl_save_root = os.path.dirname(self.rank_pkl_save_path)
            if not os.path.exists(pkl_save_root):
                os.makedirs(pkl_save_root, exist_ok=True)

    @rank_zero_only
    def reduce_rank_pkl(self):
        """Reduce all rank's pkl to total pkl."""
        if self.pred_save_path:
            pred_res_total = {}
            rank_pkl_save_root = os.path.dirname(self.rank_pkl_save_path)
            pkl_file_list = glob.glob(rank_pkl_save_root + "/*.pkl")
            for pkl_file in pkl_file_list:
                rank_pred_res = self.load_pred_pkl(pkl_file)
                pred_res_total.update(rank_pred_res)
            logger.info("remove all pred rank pkl...")
            os.system(f"rm {rank_pkl_save_root}/*.pkl")
            with open(self.pkl_save_path, "wb") as f:
                logger.info(f"dump pred in to {self.pkl_save_path}")
                pickle.dump(pred_res_total, f)
        if self.confusion_save_path:
            confusion_res_total = defaultdict(list)
            confusion_rank_pkl_root = os.path.dirname(
                self.rank_confusion_info_path
            )
            pkl_file_list = glob.glob(confusion_rank_pkl_root + "/*.pkl")
            for pkl_file in pkl_file_list:
                rank_confusion_info = self.load_confusion_info_pkl(pkl_file)
                for key in rank_confusion_info.keys():
                    confusion_res_total[key] += rank_confusion_info[key]
            logger.info("remove all confusion info rank pkl ...")
            os.system(f"rm {confusion_rank_pkl_root}/*.pkl")
            with open(self.confusion_info_save_path, "wb") as f:
                logger.info(
                    f"dump total confusion info into {self.confusion_info_save_path}"  # noqa
                )
                pickle.dump(confusion_res_total, f)

    def load_pred_pkl(self, pkl_path):
        result = {}
        f = open(pkl_path, "rb")
        while True:
            try:
                data = pickle.load(f)
                result.update(data)
            except EOFError:
                break
        f.close()
        return result

    def load_confusion_info_pkl(self, pkl_path):
        result = defaultdict(list)
        f = open(pkl_path, "rb")
        while True:
            try:
                data = pickle.load(f)
                for key in data.keys():
                    result[key] += data[key]
            except EOFError:
                break
        f.close()
        return result

    def update(
        self,
        batch,
        output,
    ):
        """
        Update internal buffer with latest predictions.

        Note that the statistics are not available until
        you call self.get() to return the metrics.

        Args:
            output: model output.
            batch: gt.

        """
        assert "timestamp" in batch
        # process ground truth
        assert (
            self.anno_name in batch
        ), "Please confirm the annos in batch, \
                    check Bev3dTargetGenerator in auto3dv"
        annotations = batch[self.anno_name]
        timestamps = np.array(batch["timestamp"].cpu())
        timestamps = [str(int(_time * 1000)) for _time in timestamps]

        _timestamps = []
        _det_group = []
        _gt_group = []
        results = {k: output[k] for k in self.pred_keys}
        batch_size, num_objs = output[self.pred_keys[0]].shape[:2]
        for bs in range(batch_size):
            for obj_idx in range(num_objs):
                pred_items = {
                    key: val[bs][obj_idx].cpu().numpy()
                    for key, val in results.items()
                }
                pred_items["timestamp"] = timestamps[bs]
                _det_group.append(pred_items)
            # getting image timastamp
            _timestamps.append(timestamps[bs])
        for bs in range(batch_size):
            valid_gt_idx = (
                (annotations[self.gt_id_key][bs] >= 0).nonzero().squeeze(-1)
            )
            valid_gt = {
                key: out[bs][valid_gt_idx] for key, out in annotations.items()
            }
            num_objs_gt = valid_gt[list(valid_gt.keys())[0]].shape[0]
            for gt_obj_idx in range(num_objs_gt):
                gt_items = {
                    key: val[gt_obj_idx].cpu().numpy()
                    for key, val in valid_gt.items()
                }
                gt_items["timestamp"] = timestamps[bs]
                _gt_group.append(gt_items)

        all_dets, all_gts = self.filter_obj(
            det_res=_det_group,
            gt_res=_gt_group,
            score_threshold=self.score_threshold,
            eval_vcs_range=self.eval_vcs_range,
            ego_ignore_range=self.ego_ignore_range,
            gt_max_depth=self.gt_max_depth,
        )
        wrong_info_dict = defaultdict(list)
        iou_match_fn_batch = np.zeros(self.num_classes)
        iou_match_fp_batch = np.zeros(self.num_classes)
        for bs, timestamp in enumerate(_timestamps):
            det_bbox3d, det_scores, det_locs, det_yaw = [], [], [], []
            gt_bbox3d, gt_locs, gt_yaw, gt_ignore = (
                [],
                [],
                [],
                [],
            )
            gt_labels = []
            det_labels = []
            if len(all_gts[timestamp]) == 0 or len(all_dets[timestamp]) == 0:
                for item in all_gts[timestamp]:
                    if not item["vcs_ignore_"]:
                        iou_match_fn_batch[int(item[self.gt_id_key])] += 1
                for item in all_dets[timestamp]:
                    iou_match_fp_batch[int(item[self.pred_id_key])] += 1
                continue
            for gt in all_gts[timestamp]:
                dim = gt["vcs_dim_"]
                yaw = gt["vcs_rot_z_"]
                loc = gt["vcs_loc_"]
                bbox3d = [loc[0], loc[1], dim[2], dim[1], -yaw]
                gt_bbox3d.append(bbox3d)
                gt_locs.append(gt["vcs_loc_"])
                gt_yaw.append(gt["vcs_rot_z_"])
                gt_labels.append(gt[self.gt_id_key])
                if self.enable_ignore:
                    gt_ignore.append(gt["vcs_ignore_"])

            for det in all_dets[timestamp]:
                dim = det["bev3d_dim"]
                yaw = det["bev3d_rot"]
                loc = det["bev3d_ct"]
                # [x, y, l, w, -yaw], -yaw means change the yaw from \
                # counterclockwise -> clockwise
                bbox3d = [loc[0], loc[1], dim[2], dim[1], -yaw]
                det_bbox3d.append(bbox3d)
                assert det["bev3d_score"] >= 0
                det_scores.append(det["bev3d_score"])
                det_locs.append(det["bev3d_ct"])
                det_yaw.append(det["bev3d_rot"])
                det_labels.append(det[self.pred_id_key])

            det_locs = np.array(det_locs)
            det_bbox3d = np.array(det_bbox3d)
            det_scores = np.array(det_scores)
            gt_locs = np.array(gt_locs)
            gt_bbox3d = np.array(gt_bbox3d)
            gt_labels = np.array(gt_labels)
            det_labels = np.array(det_labels)
            matched_dict, iou_match_fp_index, _ = rotate_iou_matching(
                det_bbox3d,
                det_locs,
                gt_bbox3d,
                gt_locs,
                det_scores,
                self.iou_threshold,
                gt_ignore,
            )
            gt_ignore = np.array(gt_ignore).astype(np.bool_)
            det_assign = matched_dict["det_assign"]
            iou_match_fn_mask = (~gt_ignore) & (det_assign < 0)
            iou_match_fn_label = gt_labels[iou_match_fn_mask]
            iou_match_fp_label = det_labels[iou_match_fp_index]
            for index in range(self.num_classes):
                iou_match_fn_batch[index] += np.sum(
                    iou_match_fn_label == index
                )
                iou_match_fp_batch[index] += np.sum(
                    iou_match_fp_label == index
                )
            matched_gt_labels = gt_labels[det_assign >= 0]
            matched_det_labels = det_labels[det_assign[det_assign >= 0]]
            if len(matched_gt_labels) == 0 or len(matched_det_labels) == 0:
                continue
            cls_confusion_matrix = confusion_matrix(
                matched_gt_labels, matched_det_labels, labels=self.label_ids
            )
            device = getattr(self, "confusion_matrix", None).device
            self.confusion_matrix += torch.tensor(
                cls_confusion_matrix, device=device
            )
            hist_gt = torch.histc(
                torch.tensor(matched_gt_labels, device=device),
                self.num_classes,
                max=self.num_classes - 1,
            )
            hist_det = torch.histc(
                torch.tensor(matched_det_labels, device=device),
                self.num_classes,
                max=self.num_classes - 1,
            )
            self.label += hist_gt
            self.pred_label += hist_det
            wrong_mask = matched_gt_labels != matched_det_labels
            wrong_gts = matched_gt_labels[wrong_mask]
            wrong_dets = matched_det_labels[wrong_mask]
            if len(wrong_gts) == 0:
                continue
            if self.confusion_save_path:
                key = os.path.join(batch["pack_dir"][bs], timestamp)
                for wrong_gt, wrong_det in zip(wrong_gts, wrong_dets):
                    wrong_info_dict[
                        f"{int(wrong_gt)}to{int(wrong_det)}"
                    ].append(key)
        device = getattr(self, "iou_match_fn_num", None).device
        self.iou_match_fn_num += torch.tensor(
            iou_match_fn_batch, device=device
        )
        self.iou_match_fp_num += torch.tensor(
            iou_match_fp_batch, device=device
        )
        if self.confusion_save_path and len(wrong_info_dict):
            with open(self.rank_confusion_info_path, "ab") as f:
                pickle.dump(wrong_info_dict, f)
        if self.pred_save_path:
            pkl_res = self.convert_to_save_format(batch, output)
            with open(self.rank_pkl_save_path, "ab") as f:
                pickle.dump(pkl_res, f)

    def get(self):
        eval_result = self.compute()
        self.reduce_rank_pkl()
        return self.name + "_confusion_matrix_result", eval_result

    def compute(self):
        """Get evaluation metrics."""
        if dist.get_rank() == 0:
            tb = PrettyTable()
            tb.title = f"Confusion Matrix for {self.pred_id_key}(Left column is GT name)"  # noqa
            row_header = ["label"]
            for i in range(self.num_classes):
                row_header.append(
                    str(i)
                    if not self.id2label.get(i, None)
                    else self.id2label[i]
                )
            row_header.append("matched_total")
            row_header.append("classification_Recall")
            row_header.append("iou_match_fn")
            tb.field_names = row_header
            header_length = len(row_header)
            rows = []
            matrix = []
            for i in range(self.num_classes):
                row_info = []
                label_name = (
                    str(i)
                    if not self.id2label.get(i, None)
                    else self.id2label[i]
                )
                row_info.append(f"{label_name}_GT")
                for j in range(self.num_classes):
                    row_info.append(
                        int(self.confusion_matrix[i][j].cpu().numpy().tolist())
                    )
                matrix.append(row_info[1:])
                row_info.append(int(self.label[i].cpu().numpy().tolist()))
                recall = self.confusion_matrix[i][i] / (
                    self.confusion_matrix[i].sum() + 1e-6
                )
                recall = round(recall.cpu().numpy().tolist(), 4)
                row_info.append(recall)
                row_info.append(
                    int(self.iou_match_fn_num[i].cpu().numpy().tolist())
                )
                rows.append(row_info)
            iou_match_fp_row = ["iou_match_fp"]
            matched_total = ["matched_total"]
            precison_row = ["classification_Precision"]
            for i in range(self.num_classes):
                matched_total.append(
                    int(self.pred_label[i].cpu().numpy().tolist())
                )
                precision = self.confusion_matrix[i][i] / (
                    self.confusion_matrix[:, i].sum() + 1e-6
                )
                precision = round(precision.cpu().numpy().tolist(), 4)
                precison_row.append(precision)
                iou_match_fp_row.append(
                    int(self.iou_match_fp_num[i].cpu().numpy().tolist())
                )
            rows.append(matched_total)
            rows.append(precison_row)
            rows.append(iou_match_fp_row)
            for row in rows:
                if len(row) <= header_length:
                    row.extend([""] * (header_length - len(row)))
            tb.add_rows(rows)
            logger.info(f"\n{tb}")
            return EvalResult(
                confusion_matrixes=[
                    {
                        "name": self.range_mode + "_" + self.name,
                        "labels": row_header[1 : self.num_classes + 1],
                        "matrix": matrix,
                    },
                ]
            )

    def filter_obj(
        self,
        det_res,
        gt_res,
        score_threshold,
        eval_vcs_range,
        ego_ignore_range,
        gt_max_depth,
    ):
        all_dets = defaultdict(list)
        all_gts = defaultdict(list)
        for det in det_res:
            if det["bev3d_score"] < score_threshold:
                continue
            if eval_vcs_range is not None:
                if not (
                    eval_vcs_range[0] < det["bev3d_ct"][0] < eval_vcs_range[2]
                    and eval_vcs_range[1]
                    < det["bev3d_ct"][1]
                    < eval_vcs_range[-1]
                ):
                    continue
            if ego_ignore_range is not None:
                if (
                    ego_ignore_range[0]
                    <= det["bev3d_ct"][0]
                    <= ego_ignore_range[2]
                    and ego_ignore_range[1]
                    <= det["bev3d_ct"][1]
                    <= ego_ignore_range[-1]
                ):
                    continue
            all_dets[det["timestamp"]].append(det)

        for gt in gt_res:
            gt_depth = abs(gt["vcs_loc_"][0])  # vcs: abs(x)=depth
            if eval_vcs_range is not None:
                if not (
                    eval_vcs_range[0] < gt["vcs_loc_"][0] < eval_vcs_range[2]
                    and eval_vcs_range[1]
                    < gt["vcs_loc_"][1]
                    < eval_vcs_range[-1]
                ):
                    continue
            else:
                if gt_depth > gt_max_depth:
                    continue
            if ego_ignore_range is not None:
                if (
                    ego_ignore_range[0]
                    <= gt["vcs_loc_"][0]
                    <= ego_ignore_range[2]
                    and ego_ignore_range[1]
                    <= gt["vcs_loc_"][1]
                    <= ego_ignore_range[-1]
                ):
                    continue
            all_gts[gt["timestamp"]].append(gt)

        return all_dets, all_gts

    def convert_to_save_format(self, batch, output):
        save_res = defaultdict(list)
        assert "timestamp" in batch
        batch_timestamps = np.array(batch["timestamp"].cpu())
        batch_timestamps = [
            str(int(_bs_time * 1000)) for _bs_time in batch_timestamps
        ]
        batch_size, num_objs = output["bev3d_ct"].shape[:2]
        # cat bev3d_ct (x,y) with bev3d_loc_z (z) -> vcs_location (x, y, z)
        location = torch.cat(
            (output["bev3d_ct"], output["bev3d_loc_z"].unsqueeze(-1)),
            dim=-1,
        )
        for i in range(batch_size):
            front_img_timestamp = batch_timestamps[i]
            pack_dir = batch["pack_dir"][i]
            # the default timestamp in auto3dv is camera_front
            key = os.path.join(pack_dir, front_img_timestamp)

            for j in range(num_objs):
                # filter the padded objs in data transform
                if output["bev3d_score"][i][j] > self.save_score_thr:
                    # all the key_names are used to adap to adas_eval
                    item = {
                        "dimensions": to_numpy(
                            output["bev3d_dim"][i][j]
                        ).tolist(),
                        "class_id": to_numpy(
                            output["bev3d_cls_id"][i][j], dtype="int16"
                        ),
                        "score": to_numpy(output["bev3d_score"][i][j]),
                        "yaw": to_numpy(output["bev3d_rot"][i][j]),
                        "location": to_numpy(location[i][j]).tolist(),
                        "timestamp": str(front_img_timestamp),
                    }
                    if self.pred_id_key != "bev3d_cls_id":
                        item[self.pred_id_key] = (
                            to_numpy(output[self.pred_id_key][i][j]),
                        )
                    save_res[key].append(item)
            if len(save_res[key]) < 1:
                save_res[key] = []
        return save_res
