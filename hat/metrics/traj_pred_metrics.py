# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from collections import defaultdict
from itertools import cycle
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from hat.metrics.traj_pred_metric_utils import (
    top_k_fpvped_min_ade_calculator,
    top_k_fpvped_min_fde_calculator,
    top_k_fpvped_min_rmse_calculator,
    top_k_min_ade_calculator,
    top_k_min_fde_calculator,
    top_k_miss_rate_calculator,
)
from hat.registry import OBJECT_REGISTRY
from .metric import EvalMetric

__all__ = ["TrajPredMetric", "BehavPredMetric", "DenseTNTtrajMetric"]


@OBJECT_REGISTRY.register
class TrajPredMetric(EvalMetric):
    """Calculate validation metrics for trajectory predition.

    Here, we calculate two metrics: \
    1. Minimum Average Displacement Error over k (minADE_k). \
    2. Minimum Final Displacement Error over k (minFDE_k). \
    3. Miss Rate over k, \

    ADE is the average of pointwise L2 distances between the predicted
    trajectory and ground truth. FDE is the L2 distance between the final
    points of the prediction and ground truth. minADE_k (minFDE_k) takes the
    minimum ADE (FDE) over the k most likely predictions and average over all
    agents. Miss rate is a proportion of misses over all agents, we define the
    the prediction whose maximum pointwise L2 distance with ground truth is
    greater than the miss_rate_dis_thr is a miss. For each agent, we take the
    k most likely predictions and evaluate if any are misses.
    """

    SUPPORTED_HEADS = ["traj_head"]
    REQUIRED_HEADS = ["traj_head"]

    def __init__(
        self,
        head_names: Dict,
        k_values: List,
        name: Optional[str] = None,
        miss_rate_dis_thr: float = 2.0,
        ego_track_id: int = -42,
        sorted_top_k: bool = True,
        agent_classes: List = (0, 1, 2),
    ):
        """Initialize method.

        Args:
            head_names: the dict of used head names.
            k_values: list of top-k values to calculate the metric.
            name: name of this metric instance for display. It should be
                consistent with the key's prefix of prediction outputs.
            miss_rate_dis_thr: Threshold to consider if a prediction
                is a hit or not.
            ego_track_id: the track id of the ego vehicle.
            sorted_top_k: whether the top k trajectories are sorted.
            agent_classes: all agent classes.
        """
        for key in head_names.keys():
            assert (
                key in self.SUPPORTED_HEADS
            ), f"Unsupported head {key} in the traj viz module."
        for key in self.REQUIRED_HEADS:
            assert (
                key in head_names
            ), f"Required head {key} is not defined in traj viz module."
        self.head_names = head_names
        self.k_values = k_values
        self.name = head_names["traj_head"] if name is None else name
        assert len(k_values), "Please input at least k-value."
        self.miss_rate_dis_thr = miss_rate_dis_thr
        self.min_ade_calculator = top_k_min_ade_calculator
        self.min_fde_calculator = top_k_min_fde_calculator
        self.miss_rate_calculator = top_k_miss_rate_calculator
        self.ego_track_id = ego_track_id
        self.sorted_top_k = sorted_top_k
        self.agent_classes = agent_classes
        super(TrajPredMetric, self).__init__(self.name)

    def _init_states(self):
        device = torch.cuda.current_device()
        self.add_state(
            "global_topk_ade",
            default=[],
            dist_reduce_fx="cat",
        )
        self.add_state(
            "global_topk_fde",
            default=[],
            dist_reduce_fx="cat",
        )
        self.add_state(
            "global_topk_miss_flag",
            default=[],
            dist_reduce_fx="cat",
        )
        self.add_state(
            "global_obs_cls",
            default=[],
            dist_reduce_fx="cat",
        )
        self.add_state(
            "global_num_obj",
            default=torch.tensor(0.0, device=device),
            dist_reduce_fx="sum",
        )
        for agent_cls in self.agent_classes:
            key = f"global_num_obj_cls_{str(int(agent_cls))}"
            self.add_state(
                key,
                default=torch.tensor(0.0, device=device),
                dist_reduce_fx="sum",
            )
        # self.global_exp_obs_dict = defaultdict(lambda: defaultdict(int))
        self.customized_init()

    def reset(self):
        """Clear and rest states after each iteration of validation."""
        device = torch.cuda.current_device()
        self.global_topk_ade = []
        self.global_topk_fde = []
        self.global_topk_miss_flag = []
        self.global_obs_cls = []
        self.global_num_obj = torch.tensor(0.0, device=device)
        for agent_cls in self.agent_classes:
            setattr(
                self,
                f"global_num_obj_cls_{str(int(agent_cls))}",
                torch.tensor(0.0, device=device),
            )
        # self.global_exp_obs_dict = defaultdict(lambda: defaultdict(int))
        self.customized_reset()

    def update(self, output):
        """Update internal buffer with the latest prediction results.

        Args:
            output (Dict): A dict of model output which includes the predicted
                and ground-truth trajectories. It should have the following
                keys: \
                1. '{}_pred_trajs': \
                   (FloatTensor, [num_obj, all_samples, traj_len, 2]): \
                   the predicted trajectories. \
                2. '{}_traj_probs': \
                   (FloatTensor, [num_obj, all_samples]): \
                   the normalized likelihood of each sample trajectory. \
                3. '{}_gt_trajs': \
                   (FloatTensor, [num_obj, all_samples, traj_len, 2]): \
                   the ground-truth trajectories. \
                3. '{}_timestep_masks': \
                   (FloatTensor, [num_obj, traj_len]): \
                   the masks to indicate whether a time stamp is valid. \
                4. '{}_track_valid_cls': \
                    (FloatTensor, [num_obj]) \
                    the predicted prob of obj is valid to predict. \
                5. '{}_track_id_gt': \
                    (FloatTensor, [num_obj]) \
                    the ground truth of obj is valid to predict.
        """
        device = torch.cuda.current_device()
        traj_head_name = self.head_names["traj_head"]
        if "predict_trajs" in output:
            pred_traj = output["predict_trajs"]
            # 差分转正常轨迹计算loss
            if "diff_fut_trajs" in output:
                pred_traj = torch.cumsum(pred_traj, dim=2)
            pred_traj = pred_traj.cpu()
        else:
            pred_traj = output[f"{traj_head_name}_pred_trajs"]
        if f"{traj_head_name}_traj_probs" in output:
            prob = output[f"{traj_head_name}_traj_probs"]
        else:
            num_obs, k_max = pred_traj.shape[0:2]
            if k_max == 1:
                prob = torch.ones(num_obs, k_max)
            else:
                tmp_prob = output["real_scores"].cpu()
                if not self.sorted_top_k:
                    # 轨迹是有序的，但概率是无序的，则需要对概率排序
                    prob, _ = torch.sort(
                        tmp_prob[:, :, 0, 0], dim=1, descending=True
                    )
                    prob = prob[:, :k_max]
                else:
                    prob = tmp_prob[:, :, 0, 0]

        if f"{traj_head_name}_gt_trajs" in output:
            gt_traj = output[f"{traj_head_name}_gt_trajs"]
        else:
            gt_traj = output["future_trajectories"]
            gt_traj = gt_traj.cpu()
        if f"{traj_head_name}_timestep_masks" in output:
            mask = output[f"{traj_head_name}_timestep_masks"]
        else:
            mask = output["valid_masks"].cpu()

        pred_traj = pred_traj.numpy()
        mask = None if (mask is None) else mask.numpy().astype(bool)
        prob = prob.numpy()
        gt_traj = gt_traj.numpy()

        obs_cls_key = f"{traj_head_name}_obs_cls"
        if obs_cls_key in output:
            obs_cls = output[obs_cls_key].numpy()
        elif "classes" in output:
            obs_cls = output["classes"].cpu().numpy()
        else:
            obs_cls = np.zeros([len(mask)])
        obj_mask = np.any(mask, axis=-1)
        obs_cls = obs_cls[obj_mask]

        topk_min_ade = self.min_ade_calculator(
            self.k_values, pred_traj, prob, gt_traj, mask
        )
        topk_min_fde = self.min_fde_calculator(
            self.k_values, pred_traj, prob, gt_traj, mask
        )
        topk_miss_flag = self.miss_rate_calculator(
            self.k_values,
            self.miss_rate_dis_thr,
            pred_traj,
            prob,
            gt_traj,
            mask,
        )

        to_update_dict = {
            "global_obs_cls": obs_cls,
            "global_topk_ade": topk_min_ade,
            "global_topk_fde": topk_min_fde,
            "global_topk_miss_flag": topk_miss_flag,
            "global_num_obj": len(topk_min_ade),
        }

        all_obs_classes_key = f"{traj_head_name}_all_obs_classes"
        if all_obs_classes_key in output:
            all_obs_classes = output[all_obs_classes_key]
        elif "agent_classes" in output:
            all_obs_classes = output["agent_classes"]
        else:
            all_obs_classes = []
        num_tmp_cls = defaultdict(int)
        for sample_cls in all_obs_classes:
            tmp_classes = set(sample_cls)
            for tmp_cls in tmp_classes:
                num_tmp_cls[tmp_cls] += np.sum(np.array(sample_cls) == tmp_cls)
        for tmp_cls, value in num_tmp_cls.items():
            key = f"global_num_obj_cls_{str(int(tmp_cls))}"
            to_update_dict[key] = value

        for k, v in to_update_dict.items():
            global_item = getattr(self, k)
            tmp_value = torch.tensor(v, device=device)
            global_item += tmp_value
            setattr(self, k, global_item)

        self.customized_update(output)

    def compute(self, epoch_id=-1):
        device = torch.cuda.current_device()
        num_k = len(self.k_values)
        if type(self.global_topk_ade) is list:
            all_min_ade = (
                torch.cat(self.global_topk_ade, dim=0)
                .cpu()
                .numpy()
                .reshape(-1, num_k)
            )
            all_min_fde = (
                torch.cat(self.global_topk_fde, dim=0)
                .cpu()
                .numpy()
                .reshape(-1, num_k)
            )
            all_miss_flag = (
                torch.cat(self.global_topk_miss_flag, dim=0)
                .cpu()
                .numpy()
                .reshape(-1, num_k)
            )
            all_obs_cls = (
                torch.tensor(self.global_obs_cls, device=device).cpu().numpy()
            )
        else:
            all_min_ade = self.global_topk_ade.cpu().numpy().reshape(-1, num_k)
            all_min_fde = self.global_topk_fde.cpu().numpy().reshape(-1, num_k)
            all_miss_flag = (
                self.global_topk_miss_flag.cpu().numpy().reshape(-1, num_k)
            )
            all_obs_cls = self.global_obs_cls.cpu().numpy()

        metric_dict = {}
        if len(all_obs_cls):
            all_valid_cls = np.unique(all_obs_cls)
            for cls_id in all_valid_cls:
                cls_id = int(cls_id)
                cls_mask = all_obs_cls == cls_id
                if not np.sum(cls_mask):
                    continue
                tmp_min_ade = np.mean(all_min_ade[cls_mask, :], axis=0)
                tmp_min_fde = np.mean(all_min_fde[cls_mask, :], axis=0)
                tmp_miss_flag = all_miss_flag[cls_mask, :]
                tmp_miss_rate = (
                    np.sum(tmp_miss_flag, axis=0) / tmp_miss_flag.shape[0]
                )
                for i, k in enumerate(self.k_values):
                    metric_dict[
                        f"head_{self.name}_min_ade_{k}_cls_{cls_id}"
                    ] = tmp_min_ade[i]
                    metric_dict[
                        f"head_{self.name}_min_fde_{k}_cls_{cls_id}"
                    ] = tmp_min_fde[i]
                    metric_dict[
                        f"head_{self.name}_miss_rate_{k}_cls_{cls_id}"
                    ] = tmp_miss_rate[i]
                metric_dict[f"head_{self.name}_cls_{cls_id}_num"] = np.sum(
                    cls_mask
                )

        all_min_ade = np.mean(all_min_ade, axis=0)
        all_min_fde = np.mean(all_min_fde, axis=0)
        all_miss_rate = np.sum(all_miss_flag, axis=0) / all_miss_flag.shape[0]
        for i, k in enumerate(self.k_values):
            metric_dict[f"head_{self.name}_min_ade_{k}"] = all_min_ade[i]
            metric_dict[f"head_{self.name}_min_fde_{k}"] = all_min_fde[i]
            metric_dict[f"head_{self.name}_miss_rate_{k}"] = all_miss_rate[i]

        metric_dict[
            f"head_{self.name}_all_obstacle_num"
        ] = self.global_num_obj.cpu().numpy()
        for agent_cls in self.agent_classes:
            key = f"global_num_obj_cls_{str(int(agent_cls))}"
            item = getattr(self, key)
            tmp_metric_key = (
                f"head_{self.name}_all_obstacle_cls_{str(int(agent_cls))}"
            )
            metric_dict[tmp_metric_key] = item.cpu().numpy()

        return metric_dict

    def get(self, epoch_id=-1):
        """Calculate metrics with collected elements.

        Args:
            epoch_id (int): the parameter is ineffective to traj pred metrics.

        Returns:
            names (list, [len(self.k_values)]): the names of the metrics.
            values (list, [len(self.k_values)]): the values of the metrics.
        """
        metric_dict = self.compute(epoch_id)
        # metric_dict.update(self.customized_compute(epoch_id))
        values = [metric_dict[name] for name in metric_dict]
        return list(metric_dict.keys()), values

    def customized_init(self):
        pass

    def customized_reset(self):
        pass

    def customized_update(self, output):
        return {}

    def customized_compute(self, epoch_id=-1):
        return {}


@OBJECT_REGISTRY.register
class BehavPredMetric(EvalMetric):
    """Calculate validation metrics for behavior prediction.

    Here, we calculate following metrics for lateral behavior prediction:
        1. latbehav_F-score
        2. keeplane_auc
        3. keeplane_F-score
        4. keeplane_recall
        5. lchange_auc
        6. lchange_F-score
        7. lchange_recall
        8. rchange_auc
        9. rchange_F-score
        10. rchange_recall
    and following metrics for longitude behavior prediction:
        1. lonbehav_acc
        2. keepvelo_recall
        3. speedup_recall
        4. slowdown_recall
    """

    def __init__(
        self,
        name: str = "behav_head",
        F_score_weights: List[float] = None,
        F_score_beta: int = 1,
        save_dir: str = None,
        tag_masks: Dict = None,
    ):
        """Initialize method.

        Args:
            name: Name of this metric instance for display. It should be
                consistent with the key's prefix of prediction outputs.
            F_score_weights: Weights for keeplane, lchange, rchange
                in calculating multi-class F_score.
            F_score_beta: Determines weight of recall in the combined score,
                F_score_beta < 1 lends more weight to precision,
                while F_score_beta > 1 favors recall.
            save_dir: Path to save the metrics and figures.
            tag_masks:

        """

        self.name = name
        self.F_score_beta = F_score_beta
        self.F_score_weights = F_score_weights
        self.tag_masks = tag_masks
        self.lat_class_names = ["keeplane", "lchange", "rchange"]
        self.lon_class_names = ["keepvelo", "speedup", "slowdown"]
        super(BehavPredMetric, self).__init__(self.name)
        self.save_dir = save_dir

    def _init_states(self):
        """Initialize state variables.

        Note that only variables initialized by self.add_state() method will be
            synchronized in distributed environment.
        """
        self.add_state(
            "global_lat_pred",
            default=[],
            dist_reduce_fx="cat",
        )
        self.add_state(
            "global_lat_gts",
            default=[],
            dist_reduce_fx="cat",
        )
        self.add_state(
            "global_lon_pred",
            default=[],
            dist_reduce_fx="cat",
        )
        self.add_state(
            "global_lon_gts",
            default=[],
            dist_reduce_fx="cat",
        )
        self.add_state(
            "global_tags",
            default=[],
            dist_reduce_fx="cat",
        )

    def reset(self):
        """Clear and reset states after each iteration of validation."""
        self.global_lat_pred = []
        self.global_lat_gts = []
        self.global_lon_pred = []
        self.global_lon_gts = []
        self.global_tags = []

    def update(self, output):
        """Update internal buffer with the latest prediction results.

        Args:
            output (Dict): A dict of model output which includes the predicted
                and ground-truth behaviors. It should have the following keys:
                1. '{}_pred_lat_behav' (FloatTensor, [num_obj]):
                    the predicted lateral behavior labels.
                2. '{}_lat_behav_gts' (FloatTensor, [num_obj]):
                    the true lateral behavior labels.
                3. '{}_pred_lon_behav' (FloatTensor, [num_obj]):
                    the predicted longitudinal behavior labels.
                4. '{}_lon_behav_gts' (FloatTensor, [num_obj]):
                    the true longitudinal behavior labels.
                5. "tag_arry" (FloatTensor, [num_obj]):
                    the array saving the sample's tags.
        """

        behav_head_name = self.name
        pred_lat_behav = output[f"{behav_head_name}_pred_lat_behav"]
        lat_behav_gt = output[f"{behav_head_name}_lat_behav_gts"]
        pred_lon_behav = output[f"{behav_head_name}_pred_lon_behav"]
        lon_behav_gt = output[f"{behav_head_name}_lon_behav_gts"]
        if self.tag_masks:
            tag_array = output["tag_array"]
        else:
            tag_array = []
        for local_item, global_item in zip(
            [
                pred_lat_behav,
                lat_behav_gt,
                pred_lon_behav,
                lon_behav_gt,
                tag_array,
            ],
            [
                self.global_lat_pred,
                self.global_lat_gts,
                self.global_lon_pred,
                self.global_lon_gts,
                self.global_tags,
            ],
        ):
            global_item += local_item

    def compute(self, epoch_id=-1):
        """Compute behavior metrics with buffered metric states.

        All states variables registered with `self.add_state` are synchronized
        across devices before the execution of this method.
        """
        if isinstance(self.global_lat_gts, list):
            global_lat_gts = np.array(self.global_lat_gts)
            global_lat_pred = np.array(self.global_lat_pred)
            global_lon_gts = np.array(self.global_lon_gts)
            global_lon_pred = np.array(self.global_lon_pred)

        elif isinstance(self.global_lat_gts, torch.Tensor):
            global_lat_gts = self.global_lat_gts.cpu().numpy()
            global_lat_pred = self.global_lat_pred.cpu().numpy()
            global_lon_gts = self.global_lon_gts.cpu().numpy()
            global_lon_pred = self.global_lon_pred.cpu().numpy()

        if isinstance(self.global_tags, list):
            global_tags = np.array(
                [np.array(tags) for tags in self.global_tags]
            )
        elif isinstance(self.global_tags, torch.Tensor):
            global_tags = self.global_tags.cpu().numpy()

        logging.info(f"Total {len(self.global_lat_gts)} objects")
        metric_dict = {}
        metric_dict.update(
            self.cal_lat_metrics(
                global_lat_gts, global_lat_pred, "global", epoch_id
            )
        )
        metric_dict.update(
            self.cal_lon_metrics(
                global_lon_gts, global_lon_pred, "global", epoch_id
            )
        )

        if self.tag_masks:
            global_tags = global_tags.reshape((-1, 13))
            for prefix in self.tag_masks:
                tag_lists = self.tag_masks[prefix]
                subset_mask = np.array([], dtype=np.int64)
                for tag_list in tag_lists:
                    tag_list = np.array(tag_list)
                    tag_indices = tag_list[:, 0]
                    tag_value = tag_list[:, 1]
                    tmp_mask = np.where(
                        np.all(
                            global_tags[:, tag_indices] == tag_value, axis=1
                        )
                    )[0]
                    subset_mask = np.union1d(subset_mask, tmp_mask)

                if len(subset_mask):
                    metric_dict.update(
                        self.cal_lat_metrics(
                            global_lat_gts[subset_mask],
                            global_lat_pred[subset_mask],
                            prefix,
                            epoch_id,
                        )
                    )
                logging.info(
                    f"Total {len(subset_mask)} objects of {prefix} samples"
                )

        return metric_dict

    def get(self, epoch_id=-1):
        metric_dict = self.compute(epoch_id)
        values = [metric_dict[name] for name in metric_dict]
        return list(metric_dict.keys()), values

    def cal_lat_metrics(self, lat_gts, lat_preds, prefix, epoch_id=-1):
        """Calculate lateral behavior metrics."""
        metric_dict = {}
        labels = np.unique(lat_gts)
        if len(labels) == 1:
            class_name = self.lat_class_names[labels[0]]
            recall = np.sum(lat_preds == labels[0]) / len(lat_gts)
            metric_dict[f"behav_head_{prefix}_{class_name}_recall"] = recall
            return metric_dict

        lat_cls_results = classification_report(
            lat_gts,
            lat_preds,
            labels=[0, 1, 2],
            target_names=self.lat_class_names,
            output_dict=True,
        )
        weighted_F_score = 0
        for i, class_name in enumerate(self.lat_class_names):
            recall = lat_cls_results[class_name]["recall"]
            precision = lat_cls_results[class_name]["precision"]
            f_score = (
                (1 + self.F_score_beta ** 2)
                * precision
                * recall
                / ((self.F_score_beta ** 2 * precision) + recall + 1e-5)
            )
            metric_dict[f"behav_head_{prefix}_{class_name}_F-score"] = f_score
            metric_dict[f"behav_head_{prefix}_{class_name}_recall"] = recall
            weighted_F_score += f_score * self.F_score_weights[i]
        metric_dict[f"behav_head_{prefix}_latbehav_F-score"] = weighted_F_score

        (
            metric_dict[f"behav_head_{prefix}_keeplane_auc"],
            metric_dict[f"behav_head_{prefix}_lchange_auc"],
            metric_dict[f"behav_head_{prefix}_rchange_auc"],
        ) = self.cal_roc_auc(lat_gts, lat_preds, epoch_id)

        lat_confusion_mat = np.array(confusion_matrix(lat_gts, lat_preds))
        plt.matshow(lat_confusion_mat, cmap=plt.cm.Reds)
        for i in range(len(lat_confusion_mat)):
            for j in range(len(lat_confusion_mat)):
                plt.annotate(
                    lat_confusion_mat[j, i],
                    xy=(i, j),
                    horizontalalignment="center",
                    verticalalignment="center",
                )
        plt.ylabel("True")
        plt.xlabel("Predict")
        plt.title("lat_confusion_matrix")
        plt.savefig(
            self.save_dir
            + f"/epoch{epoch_id:0>2d}_{prefix}_lat_confusion_matrix.png"
        )
        return metric_dict

    def cal_lon_metrics(self, lon_gts, lon_preds, prefix, epoch_id=-1):
        """Calculate longitude behavior metrics."""
        metric_dict = {}
        lon_cls_results = classification_report(
            lon_gts,
            lon_preds,
            labels=[0, 1, 2],
            target_names=self.lon_class_names,
            output_dict=True,
        )
        metric_dict[f"behav_head_{prefix}_lonbehav_acc"] = lon_cls_results[
            "accuracy"
        ]
        for class_name in self.lon_class_names:
            metric_dict[
                f"behav_head_{prefix}_{class_name}_recall"
            ] = lon_cls_results[class_name]["recall"]

        lon_confusion_mat = np.array(confusion_matrix(lon_gts, lon_preds))
        plt.matshow(lon_confusion_mat, cmap=plt.cm.Reds)
        for i in range(len(lon_confusion_mat)):
            for j in range(len(lon_confusion_mat)):
                plt.annotate(
                    lon_confusion_mat[j, i],
                    xy=(i, j),
                    horizontalalignment="center",
                    verticalalignment="center",
                )
        plt.ylabel("True")
        plt.xlabel("Predict")
        plt.title("lon_confusion_matrix")
        plt.savefig(
            self.save_dir
            + f"/epoch{epoch_id:0>2d}_{prefix}_lon_confusion_matrix.png"
        )
        return metric_dict

    def cal_roc_auc(self, data_label, data_pred, epoch_id):
        """Calculate roc_auc and save ROC curve figure."""
        y_label = label_binarize(data_label, classes=[0, 1, 2])
        y_pred = label_binarize(data_pred, classes=[0, 1, 2])
        n_classes = y_label.shape[1]
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_label.ravel(), y_pred.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.figure()
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.4f})".format(
                roc_auc["micro"]
            ),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.4f})".format(
                roc_auc["macro"]
            ),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label="ROC curve of class {0} (area = {1:0.4f})".format(
                    i, roc_auc[i]
                ),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("Lat behav ROC-AUC")
        plt.legend(loc="lower right")
        plt.savefig(self.save_dir + f"/epoch{epoch_id:0>2d}_ROC.png")
        return roc_auc[0], roc_auc[1], roc_auc[2]


@OBJECT_REGISTRY.register
class TrajPredFPVPedMetric(EvalMetric):
    """Calculate validation metrics for FPV trajectory predition."""

    def __init__(
        self,
        metrics_type: dict,
        name: str = "fpvped",
        top_k_values: tuple = (1,),
    ):
        """Initialize method.

        Args:
            name: this metrics is for pedestrian.
            top_k_values: tuple of k values to calculate the top-k metric.
            metrics_type: the metrics will be calculated, for example,
                metrics_type = {
                    "05": 3,
                    "10": 5,
                    "15": 8,
                }
                the keys of the dict represent the trajectory length used to
                calculate the metric, "05" is 0.5s, "10" is 1s and "15" is
                1.5s.
        """
        self.metrics_type = metrics_type
        assert len(metrics_type), "Please input at least one metric type."
        super(TrajPredFPVPedMetric, self).__init__(name)
        self.top_k_values = list(top_k_values)
        assert len(top_k_values), "Please input at least one k-value."
        self.rmse_calculator = top_k_fpvped_min_rmse_calculator
        self.ade_calculator = top_k_fpvped_min_ade_calculator
        self.fde_calculator = top_k_fpvped_min_fde_calculator

    def _init_states(self):
        """Init the states."""
        for key in self.metrics_type.keys():
            setattr(self, f"global_rmse{key}", [])
            setattr(self, f"global_ade_center{key}", [])
            setattr(self, f"global_ade_mpb{key}", [])
            setattr(self, f"global_fde_center{key}", [])
            setattr(self, f"global_fde_mpb{key}", [])

    def update(self, output):
        """Update internal buffer with the latest prediction results.

        Args:
            output: A dict of model output which includes the predicted
                and ground-truth trajectories.The following keys of putput:
                will be used in calculating metrics.
                1. 'pred_traj' (Tensor, [batch_size, K, dec_step, dim]):
                   the predicted trajectories in x1x2y1y2 style.
                2. 'pred_cxcympb' (Tensor, [batch_size, K, dec_step, dim]):
                   the predicted trajectories in cxcympb style, mpb: middle
                   point of the bottom edge..
                3. 'target_traj' (Tensor, [batch_size, 1, dec_step, dim]):
                   the ground-truth trajectories in x1x2y1y2 style.
                3. 'target_cxcympb' (Tensor, [batch_size, 1, dec_step, dim]):
                   the ground-truth trajectories in cxcympb style.
                4. 'probabilities' (Tensor, [batch_size, 1, 1, K]):
                    the predicted probs of the K trajectories, this item
                    only exists when the K > 1.
        """
        # [batch_size, K, dec_step, dim]
        pred_traj = output["pred_traj"].cpu().detach().numpy()
        pred_cxcympb = output["pred_cxcympb"].cpu().detach().numpy()
        # [batch_size, 1, dec_step, dim]
        target_traj = output["target_traj"].cpu().detach().numpy()
        target_cxcympb = output["target_cxcympb"].cpu().detach().numpy()
        k_value = pred_traj.shape[1]
        batch_size = pred_traj.shape[0]
        # [batch_size, 1, 1, K]
        if k_value > 1 and "probabilities" in output:
            prob = output["probabilities"]
        else:
            prob = torch.ones((batch_size, 1, 1, k_value))
        # [batch_size, 1, 1, k_value] -> [batch_size, k_value]
        prob = prob.squeeze(1).squeeze(1).cpu().detach().numpy()
        batch_rmse = self.rmse_calculator(
            target_traj, pred_traj, prob, self.top_k_values, self.metrics_type
        )
        batch_ade_center, batch_ade_mpb = self.ade_calculator(
            target_cxcympb,
            pred_cxcympb,
            prob,
            self.top_k_values,
            self.metrics_type,
        )
        batch_fde_center, batch_fde_mpb = self.fde_calculator(
            target_cxcympb,
            pred_cxcympb,
            prob,
            self.top_k_values,
            self.metrics_type,
        )
        for key in self.metrics_type.keys():
            rmse_tmp = getattr(self, f"global_rmse{key}")
            rmse_tmp.append(batch_rmse[key])
            setattr(self, f"global_rmse{key}", rmse_tmp)
            ade_center_tmp = getattr(self, f"global_ade_center{key}")
            ade_center_tmp.append(batch_ade_center[key])
            setattr(self, f"global_ade_center{key}", ade_center_tmp)
            ade_mpb_temp = getattr(self, f"global_ade_mpb{key}")
            ade_mpb_temp.append(batch_ade_mpb[key])
            setattr(self, f"global_ade_mpb{key}", ade_mpb_temp)
            fde_center_tmp = getattr(self, f"global_fde_center{key}")
            fde_center_tmp.append(batch_fde_center[key])
            setattr(self, f"global_fde_center{key}", fde_center_tmp)
            fde_mpb_temp = getattr(self, f"global_fde_mpb{key}")
            fde_mpb_temp.append(batch_fde_mpb[key])
            setattr(self, f"global_fde_mpb{key}", fde_mpb_temp)

    def get(self):
        """Calculate metrics with collected elements.

        Returns:
            names (List, [len(self.k_values)*len(self.metrics_type)]):
                the names of the metrics.
            values (List, [len(self.k_values)*len(self.metrics_type)]):
                the values of the metrics.
        """
        names, values = [], []
        op = "np.concatenate"
        for key in self.metrics_type.keys():
            exec(f"all_rmse{key} = {op}(self.global_rmse{key}, 0)")
            exec(f"all_ade_center{key} = {op}(self.global_ade_center{key}, 0)")
            exec(f"all_ade_mpb{key} = {op}(self.global_ade_mpb{key}, 0)")
            exec(f"all_fde_center{key} = {op}(self.global_fde_center{key}, 0)")
            exec(f"all_fde_mpb{key} = {op}(self.global_fde_mpb{key}, 0)")

        metrics = {}
        for key in self.metrics_type.keys():
            metrics[f"all_rmse{key}"] = [
                eval(f"np.mean(all_rmse{key}, 0)"),
                eval(f"np.percentile(all_rmse{key}, 90, 0)"),
                eval(f"np.percentile(all_rmse{key}, 95, 0)"),
            ]
            metrics[f"all_ade_center{key}"] = [
                eval(f"np.mean(all_ade_center{key},0)"),
                eval(f"np.percentile(all_ade_center{key}, 90, 0)"),
                eval(f"np.percentile(all_ade_center{key}, 95, 0)"),
            ]
            metrics[f"all_ade_mpb{key}"] = [
                eval(f"np.mean(all_ade_mpb{key}, 0)"),
                eval(f"np.percentile(all_ade_mpb{key}, 90, 0)"),
                eval(f"np.percentile(all_ade_mpb{key}, 95, 0)"),
            ]
            metrics[f"all_fde_center{key}"] = [
                eval(f"np.mean(all_fde_center{key}, 0)"),
                eval(f"np.percentile(all_fde_center{key}, 90, 0)"),
                eval(f"np.percentile(all_fde_center{key}, 95, 0)"),
            ]
            metrics[f"all_fde_mpb{key}"] = [
                eval(f"np.mean(all_fde_mpb{key}, 0)"),
                eval(f"np.percentile(all_fde_mpb{key}, 90, 0)"),
                eval(f"np.percentile(all_fde_mpb{key}, 95, 0)"),
            ]

        for key in metrics.keys():
            metrics[key] = [np.round(x, 4) for x in metrics[key]]

        for metric in self.metrics_type:
            for k_idx, k in enumerate(self.top_k_values):
                for stat_idx, stat in enumerate(["avg", "90th", "95th"]):
                    names.append(
                        "{}_min_rmse{}_{}_{}".format(
                            self.name, metric, k, stat
                        )
                    )
                    values.append(
                        metrics["all_rmse{}".format(metric)][stat_idx][k_idx]
                    )
                    names.append(
                        "{}_min_ade_center{}_{}_{}".format(
                            self.name, metric, k, stat
                        )
                    )
                    values.append(
                        metrics["all_ade_center{}".format(metric)][stat_idx][
                            k_idx
                        ]
                    )
                    names.append(
                        "{}_min_ade_mpb{}_{}_{}".format(
                            self.name, metric, k, stat
                        )
                    )
                    values.append(
                        metrics["all_ade_mpb{}".format(metric)][stat_idx][
                            k_idx
                        ]
                    )
                    names.append(
                        "{}_min_fde_center{}_{}_{}".format(
                            self.name, metric, k, stat
                        )
                    )
                    values.append(
                        metrics["all_fde_center{}".format(metric)][stat_idx][
                            k_idx
                        ]
                    )
                    names.append(
                        "{}_min_fde_mpb{}_{}_{}".format(
                            self.name, metric, k, stat
                        )
                    )
                    values.append(
                        metrics["all_fde_mpb{}".format(metric)][stat_idx][
                            k_idx
                        ]
                    )
        return names, values

    def reset(self):
        """Clear and rest states after each iteration of validation."""
        for key in self.metrics_type.keys():
            setattr(self, f"global_rmse{key}", [])
            setattr(self, f"global_ade_center{key}", [])
            setattr(self, f"global_ade_mpb{key}", [])
            setattr(self, f"global_fde_center{key}", [])
            setattr(self, f"global_fde_mpb{key}", [])


@OBJECT_REGISTRY.register
class DenseTNTtrajMetric(TrajPredMetric):
    """Calculate validation metrics for DenseTNT.

    Here, we calculate six metrics: \
    1. Minimum Average Displacement Error over k (minADE_k). \
    2. Minimum Final Displacement Error over k (minFDE_k). \
    3. Miss Rate over k, \
    4. min_goal_fde: DenseTNT goal选择中，通过score选择的topk goal中 \
       的min_fde，描述的是goal选择的效果。 \
    5. minimal_goal_fde: 在DenseTNT二阶段（set_predictor）时生效，描述 \
       的是set predictor生成的goal点距离gt的min_ade，描述的是goal生成的 \
       效果，是min_goal_fde的理论上限。 \
    6. min_label_fde: 所有采样goal点中，与gt最近的goal点的min_fde，可以 \
       认为是一阶段min_goal_fde的优化上限。 \

    ADE is the average of pointwise L2 distances between the predicted
    trajectory and ground truth. FDE is the L2 distance between the final
    points of the prediction and ground truth. minADE_k (minFDE_k) takes the
    minimum ADE (FDE) over the k most likely predictions and average over all
    agents. Miss rate is a proportion of misses over all agents, we define the
    the prediction whose maximum pointwise L2 distance with ground truth is
    greater than the miss_rate_dis_thr is a miss. For each agent, we take the
    k most likely predictions and evaluate if any are misses.
    """

    def __init__(
        self,
        head_names: Dict,
        k_values: List,
        name: Optional[str] = None,
        miss_rate_dis_thr: float = 2.0,
        ego_track_id: int = -42,
        goal_coords_scale: float = 1,
        sorted_top_k: bool = True,
        agent_classes: List = (0, 1, 2),
    ):
        """Initialize method.

        Args:
            head_names: the dict of used head names.
            k_values: list of top-k values to calculate the metric.
            name: name of this metric instance for display. It should be
                consistent with the key's prefix of prediction outputs.
            miss_rate_dis_thr: Threshold to consider if a prediction
                is a hit or not.
            ego_track_id: the track id of the ego vehicle.
            goal_coords_scale: the scale rate of the input goal coorinates.
            sorted_top_k: whether the top k trajectories are sorted.
            agent_classes: all agent classes.
        """
        self.goal_coords_scale = goal_coords_scale
        kwargs = {
            "head_names": head_names,
            "k_values": k_values,
            "name": name,
            "miss_rate_dis_thr": miss_rate_dis_thr,
            "ego_track_id": ego_track_id,
            "sorted_top_k": sorted_top_k,
            "agent_classes": agent_classes,
        }
        super(DenseTNTtrajMetric, self).__init__(**kwargs)

    def customized_init(self):
        """Customize metirc initialization."""
        self.add_state(
            "global_topk_goal_fde",
            default=[],
            dist_reduce_fx="cat",
        )
        self.add_state(
            "global_minimal_goal_fde",
            default=[],
            dist_reduce_fx="cat",
        )
        self.add_state(
            "global_min_label_fde",
            default=[],
            dist_reduce_fx="cat",
        )
        self.add_state(
            "goal_valid_mask",
            default=[],
            dist_reduce_fx="cat",
        )

    def customized_reset(self):
        """Customize metirc initialization."""
        self.global_topk_goal_fde = []
        self.global_minimal_goal_fde = []
        self.global_min_label_fde = []
        self.goal_valid_mask = []

    def customized_update(self, output):
        """Customize metirc updater."""
        device = torch.cuda.current_device()
        if "predict_goal_scores" not in output:
            return
        traj_head_name = self.head_names["traj_head"]
        if f"{traj_head_name}_timestep_masks" in output:
            mask = output[f"{traj_head_name}_timestep_masks"]
        else:
            mask = output["valid_masks"].cpu()
        mask = None if (mask is None) else mask.numpy().astype(bool)
        mask_1 = np.any(mask, axis=-1)
        mask_2 = mask[:, -1]

        obj_mask = mask_2[:, None]

        # 预测或选择的最佳点与真值点位置差的统计
        if "set_pred_scores" in output and "set_pred_goals" in output:
            goal_scores = output["set_pred_scores"].cpu().numpy()
            goal_coords = output["set_pred_goals"].cpu().numpy()
        else:
            goal_scores = output["predict_goal_scores"].cpu().numpy()
            goal_coords = output["goal_coords"].cpu().numpy()
        end_points = output["end_points"].cpu().numpy()

        goal_coords = goal_coords.transpose([0, 3, 2, 1])
        end_points = end_points.transpose([0, 3, 2, 1])
        goal_scores = goal_scores[:, 0, 0, :]
        topk_goal_min_fde = (
            top_k_min_fde_calculator(
                self.k_values, goal_coords, goal_scores, end_points, obj_mask
            )
            / self.goal_coords_scale
        )

        # 预测的点集中距离真值点位置最近的点的距离差的统计
        label_diff = goal_coords - end_points
        label_diff = np.sqrt(np.sum(label_diff ** 2, axis=-1))
        label_scores = -label_diff[:, :, 0]

        topk_goal_minimal_fde = (
            top_k_min_fde_calculator(
                self.k_values, goal_coords, label_scores, end_points, obj_mask
            )
            / self.goal_coords_scale
        )

        # 采样点或伪标签选择的最佳点与真值点位置差的统计
        if "pseudo_labels" in output:
            # [num_obs, 2, 1, k]
            label_coords = output["pseudo_labels"].cpu().numpy()
            label_coords = label_coords.transpose([0, 3, 2, 1])
        else:
            label_coords = goal_coords
        label_diff = label_coords - end_points
        label_diff = np.sqrt(np.sum(label_diff ** 2, axis=-1))
        label_scores = -label_diff[:, :, 0]

        topk_min_label_fde = (
            top_k_min_fde_calculator(
                self.k_values, label_coords, label_scores, end_points, obj_mask
            )
            / self.goal_coords_scale
        )

        to_update_dict = {
            "goal_valid_mask": mask_2[mask_1],
            "global_topk_goal_fde": topk_goal_min_fde,
            "global_minimal_goal_fde": topk_goal_minimal_fde,
            "global_min_label_fde": topk_min_label_fde,
        }

        for k, v in to_update_dict.items():
            global_item = getattr(self, k)
            tmp_value = torch.tensor(v, device=device)
            global_item += tmp_value
            setattr(self, k, global_item)

    def compute(self, epoch_id=-1):
        device = torch.cuda.current_device()
        num_k = len(self.k_values)
        if type(self.global_topk_ade) is list:
            all_goal_fde = (
                torch.cat(self.global_topk_goal_fde, dim=0)
                .cpu()
                .numpy()
                .reshape(-1, num_k)
            )
            all_minimal_goal_fde = (
                torch.cat(self.global_minimal_goal_fde, dim=0)
                .cpu()
                .numpy()
                .reshape(-1, num_k)
            )
            all_label_fde = (
                torch.cat(self.global_min_label_fde, dim=0)
                .cpu()
                .numpy()
                .reshape(-1, num_k)
            )
            all_goal_valid_mask = (
                torch.tensor(self.goal_valid_mask, device=device).cpu().numpy()
            )
            all_obs_cls = (
                torch.tensor(self.global_obs_cls, device=device).cpu().numpy()
            )
        else:
            all_goal_fde = (
                self.global_topk_goal_fde.cpu().numpy().reshape(-1, num_k)
            )
            all_minimal_goal_fde = (
                self.global_minimal_goal_fde.cpu().numpy().reshape(-1, num_k)
            )
            all_label_fde = (
                self.global_min_label_fde.cpu().numpy().reshape(-1, num_k)
            )
            all_goal_valid_mask = self.goal_valid_mask.cpu().numpy()
            all_obs_cls = self.global_obs_cls.cpu().numpy()

        metric_dict = {}
        if len(all_obs_cls):
            all_valid_cls = np.unique(all_obs_cls)
            for cls_id in all_valid_cls:
                cls_id = int(cls_id)
                cls_mask = (all_obs_cls == cls_id)[all_goal_valid_mask]
                if not np.sum(cls_mask):
                    continue
                tmp_min_fde = np.mean(all_goal_fde[cls_mask, :], axis=0)
                tmp_minimal_fde = np.mean(
                    all_minimal_goal_fde[cls_mask, :], axis=0
                )
                tmp_min_label = np.mean(all_label_fde[cls_mask, :], axis=0)
                for i, k in enumerate(self.k_values):
                    metric_dict[
                        f"head_{self.name}_min_goal_fde_{k}_cls_{cls_id}"
                    ] = np.mean(tmp_min_fde[i])
                    metric_dict[
                        f"head_{self.name}_minimal_goal_fde_{k}_cls_{cls_id}"
                    ] = np.mean(tmp_minimal_fde[i])
                    metric_dict[
                        f"head_{self.name}_min_label_fde_{k}_cls_{cls_id}"
                    ] = np.mean(tmp_min_label[i])

        all_goal_fde = np.mean(all_goal_fde, axis=0)
        for i, k in enumerate(self.k_values):
            metric_dict[f"head_{self.name}_min_goal_fde_{k}"] = np.mean(
                all_goal_fde[i]
            )

        all_minimal_goal_fde = np.mean(all_minimal_goal_fde, axis=0)
        for i, k in enumerate(self.k_values):
            metric_dict[f"head_{self.name}_minimal_goal_fde_{k}"] = np.mean(
                all_minimal_goal_fde[i]
            )

        all_label_fde = np.mean(all_label_fde, axis=0)
        for i, k in enumerate(self.k_values):
            metric_dict[f"head_{self.name}_min_label_fde_{k}"] = np.mean(
                all_label_fde[i]
            )

        metric_dict.update(TrajPredMetric.compute(self, epoch_id))

        return metric_dict
