# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import os
import pickle
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from tqdm import tqdm

from hat.callbacks.metric_updater import MetricUpdater
from hat.core.traj_pred_typing import SeqCenter, SeqIndex, TrajGroupIndex
from hat.core.traj_pred_viz_module import ImageRendererWrapper
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from .callbacks import CallbackMixin

__all__ = [
    "TrajPredViz",
    "TrajPredMetricUpdater",
    "BehavPredViz",
    "BehavPredMetricUpdater",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class TrajPredViz(CallbackMixin):
    """The visualization callback for the trajectory prediction task."""

    SUPPORTED_HEADS = ["traj_head", "behav_head"]
    REQUIRED_HEADS = ["traj_head"]

    def __init__(
        self,
        head_names: Dict,
        height: int,
        width: int,
        resolution: float,
        map_origin: List,
        video_save_path: str,
        video_name: Optional[str] = None,
        video_fps: int = 1,
        viz_on_every_epoch: bool = False,
        max_epoch: int = 0,
        ego_track_id: int = -42,
        num_traj_to_viz: int = 5,
        num_workers: int = 1,
        filter_traj_prob_thr: float = 0,
        output_reorganize: Callable = None,
        del_img_dir: bool = True,
        no_pred: bool = False,
        use_behav_head: bool = False,
        freq_ratio: int = 1,
        first_fut_frame_idx: int = 4,
        fut_len: int = 12,
    ):
        """Initialize method.

        Args:
            head_names: the dict of used head names.
            height: height of context image.
            width: width of context image.
            resolution: resolution of context image.
            map_origin: the coordinate of the map orgin [m] in the
                map coordinates.
            video_save_path: the path to save the video.
            video_name: the name of the video.
            video_fps: the fps of the video.
            viz_on_every_epoch: whether to visualize on the end of each
                epoch. Default to False, i.e., the callback will only
                visualize on the end of the step.
            max_epoch: the maximum epoch of this step.
            ego_track_id: the track id of the ego vehicle.
            num_traj_to_viz: the max number of the multi-modal prediction
                results to visualze.
            num_workers: the number of workers.
            filter_traj_prob_thr: the threshold to filter the predicted
                trajectories that have too small probabilities.
            output_reorganize: a callable object used to reorganize model
                outputs to satisfy the interface of the collect fuction.
                For example, the type of outputs in graph model is
                'namedtuple', which need to be converted to 'dict'.
            del_img_dir: whether to delete image directory after
                generating video.
            no_pred: whether to visualize ground-truth trajectories without
                predicted trajectories.
            use_behav_head: whether to use video visualization for
                behavior prediction.
            first_fut_frame_idx: the first future frame index.
            fut_len: the len of future frames.
        """
        for key in head_names.keys():
            assert (
                key in self.SUPPORTED_HEADS
            ), f"Unsupported head {key} in the viz module."
        for key in self.REQUIRED_HEADS:
            assert (
                key in head_names
            ), f"Required head {key} is not defined in viz module."
        self.head_names = head_names
        self.video_save_path = video_save_path
        self.ego_track_id = ego_track_id
        self.num_traj_to_viz = num_traj_to_viz
        self.filter_traj_prob_thr = filter_traj_prob_thr
        self.video_name = video_name
        self.viz_on_every_epoch = viz_on_every_epoch
        self.max_epoch = max_epoch
        self.output_reorganize = output_reorganize
        self.no_pred = no_pred
        self.use_behav_head = use_behav_head
        self.first_fut_frame_idx = first_fut_frame_idx
        self.fut_len = fut_len

        image_offset = np.array(map_origin) / resolution
        self.video_renderer = ImageRendererWrapper(
            height,
            width,
            resolution,
            num_workers,
            image_offset=image_offset,
            reverse=True,
            ego_track_id=ego_track_id,
            prepend_last_groundtruth=True,
            save_dataframe=False,
            fps=video_fps,
            del_img_dir=del_img_dir,
            use_behav_head=self.use_behav_head,
            freq_ratio=freq_ratio,
            first_fut_frame_idx=self.first_fut_frame_idx,
            fut_len=self.fut_len,
        )

        self.split_pred_trajectories = []
        self.split_seq_df = []
        self.split_seq_center = []
        self.split_dataset_index = []
        self.split_seq_index = []
        self.split_image = []
        self.split_track_ids = []
        self.split_seq_mask = []
        self.split_seq_prob = []
        self.split_seq_anchor_idx = []
        self.split_track_status = []
        self.split_track_yaw = []
        self.split_agent_classes = []
        if self.use_behav_head:
            self.split_pred_lat_behavs = []
            self.split_pred_lon_behavs = []
        self.split_exp_important_obs = []

    def collect(self, batch: Dict, output: Dict):
        """Collect elements for visualization during each validation steps.

        Args:
            batch: the batch data from the dataloader.
            output: the model output.
        """
        traj_head_name = self.head_names["traj_head"]

        # TODO(zifan.li):densetnt的轨迹输出变量名称，后续最好与主线统一
        traj_name = f"{traj_head_name}_trajectories"
        if traj_name in output:
            # shape=(num_obs,k,1,traj_len,2)
            trajectories = output[traj_name]
        else:
            # shape=(num_obs,k,traj_len,2)
            trajectories = output["predict_trajs"]
            # 差分轨迹转正常轨迹
            if "diff_fut_trajs" in output:
                trajectories = torch.cumsum(trajectories, dim=2)

        # TODO(zifan.li): densetnt 的 prob_name 与主线统一会更好
        prob_name = f"{traj_head_name}_log_anchors_probs"
        if prob_name in output:
            # shape=(num_obs,k)
            anchor_probs = output[prob_name]
        else:  # densetnt不输出anchor,因此轨迹概率用predict_topk_scores表示
            anchor_probs = output["real_scores"][:, :, 0, 0]

        if self.use_behav_head:
            behav_head_name = self.head_names["behav_head"]
            pred_lat_behavs = output[behav_head_name + "_pred_lat_behav"]
            pred_lon_behavs = output[behav_head_name + "_pred_lon_behav"]

        batch_track_ids = batch["valid_track_ids"]
        batch_dataset_index = batch["dataset_index"]
        batch_rendered_frames = output["rendered_frames"]
        batch_classification = batch["valid_class"]
        batch_seq_df = batch["seq_df"]
        batch_seq_index = []
        for seq_idx in batch["seq_index"]:
            if isinstance(seq_idx, SeqIndex):
                seq_idx[0] = TrajGroupIndex(*seq_idx[0])
                seq_idx = SeqIndex(*seq_idx)
            elif not isinstance(seq_idx, List):
                raise ValueError("Unsupported type of seq_idx.")
            batch_seq_index.append(seq_idx)
        batch_seq_center = [
            SeqCenter(*list(np.array(i.cpu()))) for i in batch["seq_center"]
        ]
        if "track_stat_dict" in batch:
            batch_track_stat_dict = batch["track_stat_dict"]
        else:
            batch_track_stat_dict = None
        if "track_yaw_dict" in batch:
            batch_track_yaw_dict = batch["track_yaw_dict"]
        else:
            batch_track_yaw_dict = None

        # The input pred trajs has shape [n_obj, ...], we have to resplit this
        # tensor to align with the input batches in order to conduct later
        # visualization.
        seq_s_indices = []
        seq_e_indices = []
        seq_batch_indices = []
        batch_idx = 0
        index = 0
        for track_ids in batch_track_ids:
            seq_s_indices.append(index)
            index += len(track_ids)
            seq_e_indices.append(index)
            seq_batch_indices.append(batch_idx)
            batch_idx += 1

        # type checking and conversion
        if not isinstance(batch_rendered_frames, np.ndarray):
            if type(batch_rendered_frames) is not torch.Tensor:
                batch_rendered_frames = batch_rendered_frames.as_subclass(
                    torch.Tensor
                )
            batch_rendered_frames = (
                batch_rendered_frames.cpu().detach().numpy()
            )
            batch_rendered_frames = batch_rendered_frames.transpose(
                [0, 2, 3, 1]
            )
        if not isinstance(anchor_probs, np.ndarray):
            anchor_probs = anchor_probs.cpu().numpy()
        if not isinstance(trajectories, np.ndarray):
            trajectories = trajectories.cpu().numpy()

        if len(trajectories.shape) != 5:
            if len(trajectories.shape) == 4:
                trajectories = np.expand_dims(trajectories, axis=2)
            else:
                raise ValueError(
                    "len of trajectories's shape should be either 4 or 5."
                )

        if self.use_behav_head:
            if not isinstance(pred_lat_behavs, np.ndarray):
                pred_lat_behavs = pred_lat_behavs.cpu().numpy()
            if not isinstance(pred_lon_behavs, np.ndarray):
                pred_lon_behavs = pred_lon_behavs.cpu().numpy()

        # Iterate over seqs to pack elements for one seq.
        for batch_idx, s_index, e_index in zip(
            seq_batch_indices, seq_s_indices, seq_e_indices
        ):
            seq_pred_trajs = []
            seq_mask = []
            seq_prob = []
            seq_anc_idx = []
            seq_track_id = []
            seq_classification = []
            seq_pred_lat_behavs = []
            seq_pred_lon_behavs = []
            for j in range(s_index, e_index):
                pred_traj = trajectories[j, :, :, :, :]
                pred_prob = anchor_probs[j, :]
                seq_track_id_tmp = batch_track_ids[batch_idx][j - s_index]
                if self.use_behav_head:
                    pred_lat_behav = pred_lat_behavs[j]
                    pred_lon_behav = pred_lon_behavs[j]
                seq_track_id.append(seq_track_id_tmp)
                seq_classification.append(
                    batch_classification[batch_idx][j - s_index]
                )
                index_indices = np.arange(0, len(pred_prob), 1)
                top_indices = (-pred_prob).argsort()
                top_indices = top_indices[: self.num_traj_to_viz]
                top_prob = np.exp(pred_prob[top_indices])
                index_indices = index_indices[top_indices]
                prob_mask = top_prob > self.filter_traj_prob_thr
                if not prob_mask[0]:
                    prob_mask[0] = True
                if self.no_pred:
                    prob_mask[:] = False
                seq_pred_trajs.append(pred_traj[top_indices, :, :, :])
                if self.use_behav_head:
                    seq_pred_lat_behavs.append(pred_lat_behav)
                    seq_pred_lon_behavs.append(pred_lon_behav)
                seq_mask.append(prob_mask)
                seq_prob.append(top_prob)
                seq_anc_idx.append(index_indices)
            seq_top_pred_trajs = np.stack(seq_pred_trajs, axis=0)
            seq_top_pred_mask = np.stack(seq_mask, axis=0)
            seq_top_pred_prob = np.stack(seq_prob, axis=0)
            seq_top_anchor_idx = np.stack(seq_anc_idx, axis=0)
            seq_pred_lat_behavs = np.array(seq_pred_lat_behavs)
            seq_pred_lon_behavs = np.array(seq_pred_lon_behavs)
            roadnet_image = (
                batch_rendered_frames[batch_idx, :, :, :3] * 128 + 128
            ).astype(np.uint8)

            self.split_pred_trajectories.append(seq_top_pred_trajs)
            self.split_seq_df.append(batch_seq_df[batch_idx])
            self.split_seq_center.append(batch_seq_center[batch_idx])
            self.split_seq_index.append(batch_seq_index[batch_idx])
            self.split_dataset_index.append(batch_dataset_index[batch_idx])
            self.split_image.append(roadnet_image)
            self.split_track_ids.append(seq_track_id)
            self.split_seq_mask.append(seq_top_pred_mask)
            self.split_seq_prob.append(seq_top_pred_prob)
            self.split_seq_anchor_idx.append(seq_top_anchor_idx)
            self.split_agent_classes.append(seq_classification)
            if self.use_behav_head:
                self.split_pred_lat_behavs.append(seq_pred_lat_behavs)
                self.split_pred_lon_behavs.append(seq_pred_lon_behavs)
            if batch_track_stat_dict is not None:
                self.split_track_status.append(
                    batch_track_stat_dict[batch_idx]
                )
            if batch_track_yaw_dict is not None:
                self.split_track_yaw.append(batch_track_yaw_dict[batch_idx])

        important_obs_key = f"{traj_head_name}_expand_important_obs"
        if important_obs_key in output:
            important_obs = output[important_obs_key]
            self.split_exp_important_obs += important_obs

    def clear(self):
        """Clear and rest states after each iteration of validation."""
        self.split_pred_trajectories = []
        self.split_seq_df = []
        self.split_seq_center = []
        self.split_dataset_index = []
        self.split_seq_index = []
        self.split_image = []
        self.split_track_ids = []
        self.split_seq_mask = []
        self.split_seq_prob = []
        self.split_seq_anchor_idx = []
        self.split_track_status = []
        self.split_track_yaw = []
        self.split_agent_classes = []
        if self.use_behav_head:
            self.split_pred_lat_behavs = []
            self.split_pred_lon_behavs = []
        self.split_exp_important_obs = []

    def render(self):
        """Render the videos for visualization."""
        split_pred_lat_behavs = None
        split_pred_lon_behavs = None
        if self.use_behav_head:
            split_pred_lat_behavs = self.split_pred_lat_behavs
            split_pred_lon_behavs = self.split_pred_lon_behavs
        self.video_renderer(
            gt_df=self.split_seq_df,
            seq_center=self.split_seq_center,
            pred_trajectories=self.split_pred_trajectories,
            track_ids=self.split_track_ids,
            agent_classes=self.split_agent_classes,
            track_status=self.split_track_status,
            image=self.split_image,
            dataset_index=self.split_dataset_index,
            seq_index=self.split_seq_index,
            seq_mask=self.split_seq_mask,
            seq_prob=self.split_seq_prob,
            anc_indices=self.split_seq_anchor_idx,
            video_dir=self.video_save_path,
            filename_info=self.video_name,
            track_yaw=self.split_track_yaw,
            pred_lat_behavs=split_pred_lat_behavs,
            pred_lon_behavs=split_pred_lon_behavs,
            exp_important_obs=self.split_exp_important_obs,
        )

    def on_epoch_begin(self, train_metrics, **kwargs):
        self.clear()

    def on_epoch_end(self, epoch_id, train_metrics, **kwargs):
        if not self.viz_on_every_epoch:
            if epoch_id == self.max_epoch:
                self.render()
        else:
            self.render()
        self.clear()

    def on_batch_end(self, batch, model_outs, **kwargs):
        if self.output_reorganize is not None:
            model_outs = self.output_reorganize(model_outs)
        model_out = {}
        for k, v in model_outs.items():
            if type(v) is torch.Tensor:
                v = v.cpu()
            model_out[k] = v
        batch_data = {}
        for k, v in batch.items():
            if type(v) is torch.Tensor:
                v = v.cpu()
            batch_data[k] = v
        self.collect(batch_data, model_out)


@OBJECT_REGISTRY.register
class TrajPredMetricUpdater(MetricUpdater):
    """Callback to output metric logs for traj pred task.

    The reset and update functions are not change, please refer
    to the base class.
    """

    def __init__(
        self,
        metric_update_func: Callable,
        metrics: Optional[Sequence] = None,
        filter_condition: Optional[Callable] = None,
        step_log_freq: Optional[int] = 1,
        reset_metrics_by: Optional[str] = "epoch",
        epoch_log_freq: Optional[int] = 1,
        log_prefix: Optional[str] = "",
        display_head_names: list = None,
        traj_display_k_values: List = (1, 5),
        display_cls_dict: Optional[Dict] = None,
        save_metric_path: str = None,
        support_behav_metric: bool = False,
        traj_metrics_keys: Optional[List] = None,
    ):
        """Initialize method.

        Args:
            the common parameters refer to the base class.
            display_head_name: the name of the head to display metrics.
            traj_display_k_values: the values of top-k metrics to display.
            display_cls_dict: the keys are the obstacle type. The values
                are the corresponding type id. The type id is the value
                after remapping transfrom. Usually, it should be:
                {"vehicle": 0, "pedestrain": 1, "cyclist": 2}.
            save_metric_path: the path to save metric file.
        """
        super(TrajPredMetricUpdater, self).__init__(
            metric_update_func,
            metrics,
            filter_condition,
            step_log_freq,
            reset_metrics_by,
            epoch_log_freq,
            log_prefix,
        )
        self.display_head_names = display_head_names
        self.traj_display_k_values = traj_display_k_values
        self.display_cls_dict = display_cls_dict
        if traj_metrics_keys is not None:
            self.traj_metrics_keys = traj_metrics_keys
        else:
            self.traj_metrics_keys = ["min_ade", "min_fde", "miss_rate"]

        self.behav_metrics_keys = []
        self.support_behav_metric = support_behav_metric
        if self.support_behav_metric:
            self.behav_lat_metrics_keys = [
                "latbehav_F-score",
                "keeplane_auc",
                "keeplane_F-score",
                "keeplane_recall",
                "lchange_auc",
                "lchange_F-score",
                "lchange_recall",
                "rchange_auc",
                "rchange_F-score",
                "rchange_recall",
            ]
            self.behav_lon_metrics_keys = [
                "lonbehav_acc",
                "keepvelo_recall",
                "speedup_recall",
                "slowdown_recall",
            ]
        self.save_metric_path = save_metric_path
        if save_metric_path is not None:
            os.makedirs(save_metric_path, exist_ok=True)

    def on_epoch_end(self, epoch_id, train_metrics, **kwargs):
        metrics = train_metrics if self.metrics is None else self.metrics
        if (epoch_id + 1) % self.epoch_log_freq == 0:
            prefix = "Epoch[%d] %s: " % (epoch_id, self.log_prefix)
            self._log(metrics, prefix, epoch_id, True)

    def _log(self, train_metrics, prefix="", epoch_id=-1, save_txt=False):
        """Write logs."""
        log_list = []
        logger.info(prefix)
        log_list.append(prefix)

        for metric in train_metrics:
            name, value = metric.get(epoch_id)
            metric_dict = {
                k: v for k, v in zip(_as_list(name), _as_list(value))
            }
            if metric.name == "traj_head":
                log_list += self._get_one_group_log(metric_dict, metric.name)
                for cls_name, type_id in self.display_cls_dict.items():
                    type_id = int(type_id)
                    suffix = f"cls_{type_id}"
                    log_list += self._get_one_group_log(
                        metric_dict, metric.name, suffix, cls_name, type_id
                    )
            if metric.name == "behav_head" and self.support_behav_metric:
                log_list += self._get_behav_group_log(metric_dict, metric.name)
        spilit_array = "-" * 50
        logger.info(spilit_array)
        log_list.append(spilit_array)

        if save_txt and os.path.exists(self.save_metric_path):
            with open(
                os.path.join(self.save_metric_path, "metric.txt"), "w"
            ) as f:
                for line in log_list:
                    f.writelines(f"{line}\n")

    def _get_one_group_log(
        self,
        metric_dict,
        head_name,
        suffix=None,
        group_name=None,
        type_id=None,
    ):
        """Write the log for one group of metrics."""
        log_list = []
        spilit_array = "-" * 50
        if group_name is None:
            group_name = "all"
        col_name = "       " + "  ".join(self.traj_metrics_keys)
        logger.info(spilit_array)
        logger.info(f"{group_name}:")
        logger.info(col_name)
        log_list += [spilit_array, f"{group_name}:", col_name]
        for topk in self.traj_display_k_values:
            tmp_info = f"top {topk}: "
            for metric in self.traj_metrics_keys:
                key = f"head_{head_name}_{metric}_{topk}"
                if suffix is not None:
                    key = f"{key}_{suffix}"
                if key in metric_dict:
                    tmp_info += "%.3f     " % (metric_dict[key])
                else:
                    tmp_info += "no  "
            logger.info(tmp_info)
            log_list.append(tmp_info)
        if type_id is not None:
            # Key obstacles
            key = f"head_{head_name}_cls_{type_id}_num"
            key2 = f"head_{head_name}_all_obstacle_cls_{type_id}"
            if key in metric_dict:
                total_num = metric_dict[key]
            else:
                total_num = 0
            if key2 in metric_dict:
                total_raw_num = metric_dict[key2]
            else:
                total_raw_num = 0
            count_info = "total: "
            count_info += "%d / %d " % (total_num, total_raw_num)
            logger.info(count_info)
            log_list.append(count_info)

            # key = f"head_{head_name}_exp_important_cnt"
            # exp_obs_info = "expand important obstacle: "
            # exp_obs_info += str(dict(metric_dict[key][type_id]))
            # logger.info(exp_obs_info)
            # log_list.append(exp_obs_info)

        return log_list

    def _get_behav_group_log(self, metric_dict, head_name, group_name=None):
        log_list = []
        spilit_array = "-" * 50
        if group_name is None:
            group_name = "all"
        lat_col_name = "  ".join(self.behav_lat_metrics_keys)
        lon_col_name = "  ".join(self.behav_lon_metrics_keys)
        logger.info(spilit_array)
        logger.info(f"{group_name}:")
        log_list += [spilit_array, f"{group_name}:"]

        for col_name, metrics in zip(
            [lat_col_name, lon_col_name],
            [self.behav_lat_metrics_keys, self.behav_lon_metrics_keys],
        ):
            logger.info(col_name)
            log_list.append(col_name)
            tmp_info = "     "
            for metric in metrics:
                behav_key = metric.split("_")[0]
                behav_key = behav_key.replace("*", "")
                metric_key = metric.split("_")[1]
                key = f"head_{head_name}_{behav_key}_mean_{metric_key}"
                if key in metric_dict:
                    tmp_info += "%.3f           " % (metric_dict[key])
                else:
                    tmp_info += "no            "
            logger.info(tmp_info)
            log_list.append(tmp_info)
        return log_list


@OBJECT_REGISTRY.register
class BehavPredViz(CallbackMixin):
    """The visualization callback for the behavior prediction task."""

    def __init__(
        self,
        save_path: str,
        dump_file_name: str,
        vis_track_ids: List[int] = None,
        video_fps: int = 10,
        is_visualize: bool = False,
        with_gt: bool = False,
    ):
        """Initialize method.

        Args:
            save_path: the path to save the visualization results.
            vis_track_ids: the track ids to visualize.
            dump_file_name: the name of the dump file.
            video_fps: the fps of the video.
            is_visualize: whether to visualize. Default to False.
            with_gt: whether to visualize ground-truth behaviors.
        """
        super().__init__()
        self.save_path = save_path
        self.vis_track_ids = vis_track_ids
        self.dump_file_name = dump_file_name
        self.video_fps = video_fps
        self.is_visualize = is_visualize
        self.with_gt = with_gt

        self.all_sample_dict = {}
        self.track_id_dict = {}

    def collect(self, batch_data: Dict, model_out: Dict):
        """Collect elements for visualization during each validation steps.

        Args:
            batch_data: the batch data from the dataloader.
            model_out: the model output.
        """
        token_list = []
        for scene_token in model_out["scene_token"]:
            plate = scene_token.split("_")[0]
            date_token = (
                plate
                + "-"
                + scene_token.split("_")[1]
                + "_"
                + scene_token.split("_")[2]
            )
            token_list.append(date_token)

        lcf_timestamp_list = (
            batch_data["lcf_timestamp"].cpu().numpy().astype(np.float64)
        )
        batch_track_id = (
            batch_data["track_id"].cpu().numpy().astype(np.float64)
        )

        feats_keys = [
            "struct_road_feats",
            "struct_road_masks",
            "struct_traj_feats",
            "struct_traj_masks",
            "behav_head_pred_lat_behav",
            "behav_head_lat_behav_probs",
        ]
        if self.with_gt:
            feats_keys.append("lat_behaviors")

        batch_size = len(model_out["track_id"])
        for batch_idx in range(batch_size):
            track_id = batch_track_id[batch_idx]
            lcf_ts = int(lcf_timestamp_list[batch_idx])
            sample_token = token_list[batch_idx] + "_" + str(int(track_id))
            if sample_token not in self.all_sample_dict.keys():
                self.all_sample_dict[sample_token] = {}
            sample = {}
            for key in feats_keys:
                sample[key] = model_out[key][batch_idx].detach().cpu().numpy()
            if track_id in self.track_id_dict:
                self.track_id_dict[track_id].append(lcf_ts)
            else:
                self.track_id_dict[track_id] = [lcf_ts]
            self.all_sample_dict[sample_token][lcf_ts] = sample

    def clear(self):
        """Clear and rest states after each iteration of validation."""
        pass

    def on_epoch_begin(self, train_metrics, **kwargs):
        self.clear()

    def on_epoch_end(self, epoch_id, train_metrics, **kwargs):
        self.render()
        with open(
            os.path.join(self.save_path, self.dump_file_name), "wb"
        ) as file:
            pickle.dump(self.all_sample_dict, file)
        with open(
            os.path.join(self.save_path, "traci_dict.pkl"), "wb"
        ) as file:
            pickle.dump(self.track_id_dict, file)

    def on_batch_end(self, batch, model_outs, **kwargs):
        model_out = {}
        for k, v in model_outs.items():
            if type(v) is torch.Tensor:
                v = v.cpu()
            model_out[k] = v
        batch_data = {}
        for k, v in batch.items():
            if type(v) is torch.Tensor:
                v = v.cpu()
            batch_data[k] = v
        self.collect(batch_data, model_out)

    def render(self):
        """Render the videos for visualization."""
        for sample_token in tqdm(self.all_sample_dict.keys()):
            # sample_token like: H3165-20220421-123209_224_4354
            plate = sample_token[:5]
            date = sample_token.split("-")[1]
            date_token = (
                date
                + "-"
                + sample_token.split("-")[2][:6]
                + "_"
                + sample_token.split("-")[2][7:10]
            )
            track_id = sample_token.split("_")[-1]
            if (
                self.vis_track_ids is not None
                and int(track_id) not in self.vis_track_ids
            ):
                continue
            sample_seq = self.all_sample_dict[sample_token]
            sorted_timestamp = sorted(sample_seq)
            tmp_img_path = os.path.join(
                self.save_path, f"{plate}_{date_token}_{track_id}"
            )
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(tmp_img_path, exist_ok=True)
            video = cv2.VideoWriter(
                f"{self.save_path}/{sample_token}.mp4",
                cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                self.video_fps,
                (1200, 800),
            )
            for lcf_ts in sorted_timestamp:
                sample = sample_seq[lcf_ts]
                sample["lcf_ts"] = lcf_ts
                sample["track_id"] = track_id
                fig = self.vis_one_frame(sample)
                img_path = os.path.join(tmp_img_path, f"{lcf_ts}.png")
                fig.savefig(img_path)
                video.write(cv2.imread(img_path))
                plt.close()
            video.release()

    def vis_one_frame(self, sample):
        traj_feats = sample["struct_traj_feats"].transpose(1, 2, 0) / 0.04
        road_feats = sample["struct_road_feats"].transpose(1, 2, 0) / 0.04
        track_id = sample["track_id"]
        frame_ts = sample["lcf_ts"]

        obs_width = 2
        obs_length = 4

        num_obs = int(np.sum(sample["struct_traj_masks"]))
        obs_history_trajs = np.zeros((num_obs - 1, 16, 2))
        history_traj = np.zeros((16, 2))
        history_traj[:-1, :] = traj_feats[0, :, :2]
        history_traj = history_traj[3::4, :]
        obs_history_trajs[:, :-1, :] = traj_feats[1:num_obs, :, :2]
        obs_history_trajs[:, -1, :] = traj_feats[1:num_obs, -1, 2:4]

        road_segs = road_feats[:, :, :2]
        road_segs = road_segs[sample["struct_road_masks"] == 1]

        fig, ax = plt.subplots(figsize=(15, 10), dpi=80)
        plt.xlim(-20, 40)
        plt.ylim(-20, 20)

        cmap = [
            (1, 0, 0, j / len(history_traj)) for j in range(len(history_traj))
        ]
        ax.scatter(
            history_traj[:, 0], history_traj[:, 1], c=cmap, s=10, marker=None
        )
        for traj in history_traj:
            rect = Rectangle(
                (traj[0] - (obs_length / 2), traj[1] - (obs_width / 2)),
                obs_length,
                obs_width,
                linewidth=1,
                edgecolor="g",
                facecolor="none",
            )
            ax.add_patch(rect)

        for obs_history_traj in obs_history_trajs:
            traj = obs_history_traj[
                np.logical_or(
                    obs_history_traj[:, 0] != 0, obs_history_traj[:, 1] != 0
                )
            ]
            if traj.shape[0] == 0:
                continue
            rect = Rectangle(
                (traj[-1, 0] - 2, traj[-1, 1] - 1.75),
                4,
                2,
                linewidth=1,
                edgecolor="b",
                facecolor="blue",
                alpha=1,
            )
            ax.add_patch(rect)

        for road_seg in road_segs:
            plt.plot(road_seg[:, 0], road_seg[:, 1], "r--", alpha=1, lw=3)

        behav_pred_probs = sample["behav_head_lat_behav_probs"].squeeze()
        behav_pred_probs = np.exp(behav_pred_probs) / sum(
            np.exp(behav_pred_probs)
        )
        if self.with_gt:
            behav_gt = sample["lat_behaviors"]
            if np.argmax(behav_pred_probs) == behav_gt:
                title_color = "black"
            else:
                title_color = "red"
            fig.suptitle(
                f"track_id: {track_id}  timestamp: {frame_ts}\
                \ngt: {behav_gt} behav_probs: [{behav_pred_probs[0]:.2}, {behav_pred_probs[1]:.2}, {behav_pred_probs[2]:.2}]",  # noqa: E501,
                fontsize=20,
                color=title_color,
            )
        else:
            fig.suptitle(
                f"track_id: {track_id}    timestamp: {frame_ts}\
                \nbehav_probs: [{behav_pred_probs[0]:.2}, {behav_pred_probs[1]:.2}, {behav_pred_probs[2]:.2}]",  # noqa: E501,
                fontsize=20,
                color="black",
            )
            fig.suptitle(
                f"         track_id: {track_id}  timestamp: {frame_ts}\
                \nbehav_probs: [{behav_pred_probs[0]:.2}, {behav_pred_probs[1]:.2},{behav_pred_probs[2]:.2}]",  # noqa: E501,
                fontsize=20,
                color=title_color,
            )
        return fig


@OBJECT_REGISTRY.register
class BehavPredMetricUpdater(MetricUpdater):
    """Callback to output metric logs for behav pred task.

    The reset and update functions are not change, please refer
    to the base class.
    """

    def __init__(
        self,
        metric_update_func: Callable,
        metrics: Optional[Sequence] = None,
        filter_condition: Optional[Callable] = None,
        step_log_freq: Optional[int] = 1,
        reset_metrics_by: Optional[str] = "epoch",
        epoch_log_freq: Optional[int] = 1,
        log_prefix: Optional[str] = "",
        save_metric_path: str = None,
    ):
        super(BehavPredMetricUpdater, self).__init__(
            metric_update_func,
            metrics,
            filter_condition,
            step_log_freq,
            reset_metrics_by,
            epoch_log_freq,
            log_prefix,
        )
        self.save_metric_path = save_metric_path
        if save_metric_path is not None:
            os.makedirs(save_metric_path, exist_ok=True)
        self.behav_lat_metrics_keys = [
            "latbehav_F-score",
            "keeplane_auc",
            "keeplane_F-score",
            "keeplane_recall",
            "lchange_auc",
            "lchange_F-score",
            "lchange_recall",
            "rchange_auc",
            "rchange_F-score",
            "rchange_recall",
        ]
        self.behav_lon_metrics_keys = [
            "lonbehav_acc",
            "keepvelo_recall",
            "speedup_recall",
            "slowdown_recall",
        ]

    def on_epoch_end(self, epoch_id, train_metrics, **kwargs):
        metrics = train_metrics if self.metrics is None else self.metrics
        if (epoch_id + 1) % self.epoch_log_freq == 0:
            prefix = "Epoch[%d] %s: " % (epoch_id, self.log_prefix)
            self._log(metrics, prefix, epoch_id, True)

    def _log(self, metrics, prefix="", epoch_id=-1, save_txt=False):
        log_list = []
        logger.info(prefix)
        log_list.append(prefix)

        for metric in metrics:
            name, value = metric.get(epoch_id)
            metric_dict = {
                k: v for k, v in zip(_as_list(name), _as_list(value))
            }
            if metric.name == "behav_head":
                log_list += self._get_behav_group_log(metric_dict)
        spilit_array = "-" * 50
        logger.info(spilit_array)
        log_list.append(spilit_array)

        if save_txt and os.path.exists(self.save_metric_path):
            with open(
                os.path.join(self.save_metric_path, "metric.txt"), "w"
            ) as f:
                for line in log_list:
                    f.writelines(f"{line}\n")

    def _get_behav_group_log(self, metric_dict):
        log_list = []
        spilit_array = "-" * 50
        prefix_list = []
        for metric_name in metric_dict:
            prefix_list.append(metric_name.split("_")[2])

        for prefix in set(prefix_list):
            logger.info(spilit_array)
            logger.info(f"{prefix}:")
            log_list += [spilit_array, f"{prefix}:"]
            tmp_info = ""
            for metric_key in self.behav_lat_metrics_keys:
                key = f"behav_head_{prefix}_{metric_key}"
                if key in metric_dict:
                    tmp_info += f"{metric_key}: "
                    tmp_info += "%.3f    " % (metric_dict[key])
            if tmp_info:
                logger.info(tmp_info)
                log_list.append(tmp_info)

            tmp_info = ""
            for metric_key in self.behav_lon_metrics_keys:
                key = f"behav_head_{prefix}_{metric_key}"
                if key in metric_dict:
                    tmp_info += f"{metric_key}: "
                    tmp_info += "%.3f    " % (metric_dict[key])
            if tmp_info:
                logger.info(tmp_info)
                log_list.append(tmp_info)
        return log_list
