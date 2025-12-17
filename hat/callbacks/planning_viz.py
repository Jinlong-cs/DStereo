# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import os
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np
import torch

from hat.callbacks.metric_updater import MetricUpdater
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.distributed import rank_zero_only
from .callbacks import CallbackMixin

__all__ = ["PlanningMetricUpdater", "PlanningViz"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class PlanningViz(CallbackMixin):
    """The visualization callback for the imitation learning task.

    Args:
        height: height of context image.
        width: width of context image.
        resolution: resolution of context image.
        bev_ego_center: ego location in bev coordinates.
        video_save_path: the path to save the video.
        video_name: the name of the video.
        video_fps: the fps of the video.
        viz_on_every_epoch: whether to visualize after each epoch.
        viz_epoch: the specific epoch of visualize.
        del_img_dir: whether to delete image directory after
            generating video.

    """

    def __init__(
        self,
        height: int,
        width: int,
        resolution: float,
        bev_ego_center: List[int],
        video_save_path: str,
        video_name: Optional[str] = None,
        video_fps: int = 1,
        viz_on_every_epoch: bool = False,
        viz_epoch: int = 0,
        del_img_dir: bool = False,
    ):

        self.video_save_path = video_save_path
        self.viz_on_every_epoch = viz_on_every_epoch
        self.viz_epoch = viz_epoch
        self.img_size = (height, width)
        self.resolution = resolution
        self.bev_ego_center = bev_ego_center
        self.del_img = del_img_dir
        self.viz = False

        self.color_table = {
            "agents_obs": [(50, 0, 0), (100, 0, 0), (150, 0, 0), (250, 0, 0)],
            "agents_ego": [(0, 50, 0), (0, 100, 0), (0, 150, 0), (0, 250, 0)],
            "svf_goal": (250, 250, 0),
            "traj_gt": (0, 200, 0),
        }

        video_name = video_name + ".avi"
        self.video_name = os.path.join(self.video_save_path, video_name)
        self.image_path = os.path.join(video_save_path, "image")

        self.video_renderer = cv2.VideoWriter(
            self.video_name,
            cv2.VideoWriter_fourcc(*"XVID"),
            video_fps,
            self.img_size,
        )

    def render(self, batch, output):
        """Collect elements for visualization during each validation steps.

        Args:
            batch (Dict): the batch data from the dataloader.
            output (Dict): the model output.
        """

        batch_rendered_map = batch["rendered_map"]
        batch_rendered_obs = batch["rendered_obs"]
        batch_rendered_ego = batch["plan_rendered_ego"]
        batch_goal = batch["plan_svf_goal"]
        batch_traj = batch["plan_fut_state"]
        batch_timestamp = batch["lcf_timestamp"]
        batch_svf = output["reward_head_svf"]

        # type checking and conversion
        if not isinstance(batch_rendered_map, np.ndarray):
            batch_rendered_map = batch_rendered_map.cpu().detach().numpy()
            batch_rendered_map = batch_rendered_map.transpose([0, 2, 3, 1])
        if not isinstance(batch_rendered_obs, np.ndarray):
            batch_rendered_obs = batch_rendered_obs.cpu().detach().numpy()
            batch_rendered_obs = batch_rendered_obs.transpose([0, 2, 3, 1])
        if not isinstance(batch_rendered_ego, np.ndarray):
            batch_rendered_ego = batch_rendered_ego.cpu().detach().numpy()
            batch_rendered_ego = batch_rendered_ego.transpose([0, 2, 3, 1])
        if not isinstance(batch_goal, np.ndarray):
            batch_goal = torch.nn.functional.upsample_nearest(
                batch_goal, self.img_size
            )
            batch_goal = batch_goal.cpu().detach().numpy()
            batch_goal = batch_goal.transpose([0, 2, 3, 1])
        if not isinstance(batch_svf, np.ndarray):
            batch_svf = torch.nn.functional.upsample_nearest(
                batch_svf, self.img_size
            )
            batch_svf = batch_svf.cpu().detach().numpy()
            batch_svf = batch_svf.transpose([0, 2, 3, 1])
        if not isinstance(batch_traj, np.ndarray):
            batch_traj = batch_traj.cpu().detach().numpy()
            batch_traj = batch_traj[:, :, :2]
            batch_img_traj = (
                -batch_traj / self.resolution + self.bev_ego_center
            )

        batch_rendered_frames = batch_rendered_map * 128 + 128
        batch_rendered_frames = batch_rendered_frames.astype(np.uint8).copy()

        # render obstacle images
        mask_obs = batch_rendered_obs > 0
        for i in range(batch_rendered_obs.shape[-1]):
            batch_rendered_frames[mask_obs[:, :, :, i]] = self.color_table[
                "agents_obs"
            ][i]

        # render obstacle images
        mask_ego = batch_rendered_ego > 0
        for i in range(batch_rendered_ego.shape[-1]):
            batch_rendered_frames[mask_ego[:, :, :, i]] = self.color_table[
                "agents_ego"
            ][i]

        # render svf path images
        for ind, svf in enumerate(batch_svf):
            svf = (svf / svf.max() * 250).astype(np.uint8)
            svf = cv2.applyColorMap(svf, cv2.COLORMAP_PINK)
            batch_rendered_frames[ind] = cv2.add(
                svf, batch_rendered_frames[ind]
            )

        # render navigation images
        mask_goal = batch_goal > 0
        batch_rendered_frames[mask_goal[:, :, :, 0]] = self.color_table[
            "svf_goal"
        ]

        # render experts trajectory images
        batch_rendered_frames = self._render_traj(
            batch_rendered_frames, batch_img_traj
        )

        # img file save
        for ind, img in enumerate(batch_rendered_frames):
            file_name = str(batch_timestamp[ind]) + ".jpg"
            file_path = os.path.join(self.image_path, file_name)
            cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    @rank_zero_only
    def video_render(self):
        """Render the videos for visualization."""

        file_list = os.listdir(self.image_path)
        file_list.sort()

        for file in file_list:
            path = os.path.join(self.image_path, file)
            img = cv2.imread(path)
            self.video_renderer.write(img)

        self.video_renderer.release()
        if self.del_img:
            for file in file_list:
                os.remove(os.path.join(self.image_path, file))
            os.removedirs(self.image_path)

    def on_epoch_begin(self, epoch_id, train_metrics, **kwargs):
        self.viz = self.viz_on_every_epoch or epoch_id == self.viz_epoch

        if self.viz:
            if not os.path.exists(self.image_path):
                os.mkdir(self.image_path)

    def on_epoch_end(self, epoch_id, train_metrics, **kwargs):
        if self.viz:
            self.video_render()

    def on_batch_end(self, epoch_id, batch, model_outs, **kwargs):
        batch_out = {}
        for k, v in model_outs.items():
            if type(v) is torch.Tensor:
                v = v.cpu()
            batch_out[k] = v
        batch_in = {}
        for k, v in batch.items():
            if type(v) is torch.Tensor:
                v = v.cpu()
            batch_in[k] = v
        if self.viz:
            self.render(batch_in, batch_out)

    def _render_traj(self, batch_img, batch_traj):
        """Render trajectory on bev map."""

        rendered_img = batch_img.astype(np.uint8).copy()

        for n in range(batch_img.shape[0]):
            img = rendered_img[n]
            traj = batch_traj[n]

            p0 = traj[0]
            cv2.circle(
                img,
                (int(p0[1]), int(p0[0])),
                3,
                self.color_table["traj_gt"],
                -1,
            )
            for i in range(traj.shape[0] - 1):
                p1 = traj[i]
                p2 = traj[i + 1]
                cv2.line(
                    img,
                    (int(p1[1]), int(p1[0])),
                    (int(p2[1]), int(p2[0])),
                    self.color_table["traj_gt"],
                    2,
                )
                cv2.circle(
                    img,
                    (int(p2[1]), int(p2[0])),
                    3,
                    self.color_table["traj_gt"],
                    -1,
                )

        return rendered_img


@OBJECT_REGISTRY.register
class PlanningMetricUpdater(MetricUpdater):
    """Callback to output metric logs for imitation planning task.

    Args:
        metric_update_func: Function with `metrics`, `batch` and `model_outs`
            as inputs, filter out labels, predictions then update corresponding
            metric.
        metrics: Metric configs or metric instances, for multi-task.
        filter_condition: Function to filter current task metric inputs
            on batch end, including `model_outs` and `batch`. Useful in
            multitask training.
        step_log_freq: Logging every `step_log_freq` steps. If < 1, disable
            step log output.
        reset_metrics_by: When are metrics reset during training, can be
            either one of 'step', 'log' and 'epoch'.
        epoch_log_freq: Logging every `epoch_log_freq` epochs. This argument
            works only when reset_metric_by == 'epoch'.
        log_prefix: Logging info prefix.
        save_metric_path: Metric results save path in txt format.
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
        super(PlanningMetricUpdater, self).__init__(
            metric_update_func,
            metrics,
            filter_condition,
            step_log_freq,
            reset_metrics_by,
            epoch_log_freq,
            log_prefix,
        )

        self.save_metric_path = save_metric_path
        self.metric_file_path = os.path.join(
            self.save_metric_path, "metric.txt"
        )
        if save_metric_path is not None:
            os.makedirs(save_metric_path, exist_ok=True)
            with open(self.metric_file_path, "a") as f:
                f.writelines("Imitation Planning Metrics\n")

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
            name, value = metric.get()
            log_list += self._get_one_group_log(
                _as_list(name), _as_list(value)
            )

        spilit_array = "-" * 50
        logger.info(spilit_array)
        log_list.append(spilit_array)

        rank_zero = torch.distributed.get_rank() == 0
        if save_txt and os.path.exists(self.save_metric_path) and rank_zero:
            with open(self.metric_file_path, "a") as f:
                for line in log_list:
                    f.writelines(f"{line}\n")

    def _get_one_group_log(
        self,
        metric_keys,
        metric_vals,
        group_name=None,
    ):
        """Write the log for one group of metrics."""
        log_list = []
        spilit_array = "-" * 50
        if group_name is None:
            group_name = "all"
        col_name = "          " + "  ".join(metric_keys)
        logger.info(spilit_array)
        logger.info(f"{group_name}:")
        logger.info(col_name)
        log_list += [spilit_array, f"{group_name}:", col_name]

        tmp_info = "results : "
        for value in metric_vals:
            tmp_info += "%.5f   " % (value)

        logger.info(tmp_info)
        log_list.append(tmp_info)
        return log_list


def gen_color_table(label_colors):
    all_seg_labels = list(label_colors.keys())
    color_table = {}
    rand_table_before = (np.random.randint(-128, 127, size=128) / 128).tolist()
    rand_table_after = (
        np.random.randint(-128, 127, size=128 - len(all_seg_labels)) / 128
    ).tolist()

    color_table["r_val"] = tuple(
        rand_table_before
        + [(label_colors[label][0] - 128) / 128 for label in all_seg_labels]
        + rand_table_after
    )
    color_table["g_val"] = tuple(
        rand_table_before
        + [(label_colors[label][1] - 128) / 128 for label in all_seg_labels]
        + rand_table_after
    )
    color_table["b_val"] = tuple(
        rand_table_before
        + [(label_colors[label][2] - 128) / 128 for label in all_seg_labels]
        + rand_table_after
    )
    return color_table


def gen_roadmap_color_table():
    # Get color table.
    label_colors = {
        "others": [0, 0, 0],
        "roadedge": [0, 0, 255],
        "roadarrow": [47, 79, 79],
        "solid_lanes": [200, 200, 200],
        "stopline": [192, 0, 64],
        "crosswalk": [255, 127, 80],
        "sections": [0, 0, 0],
        "junctions": [0, 0, 0],
        "virtuallanes": [50, 50, 50],
        "zones": [0, 0, 0],
        "parking_slots": [34, 139, 34],
        "dashed_lines": [255, 255, 0],  # 青色
        "static_objects": [240, 32, 160],  # 紫色
        "traffic_light_unknown": [205, 250, 255],  # 淡黄
        "traffic_light_off": [56, 94, 15],  # 绿土
        "traffic_light_green": [0, 255, 0],
        "traffic_light_yellow": [255, 255, 0],
        "traffic_light_red": [0, 0, 100],
    }
    return gen_color_table(label_colors)
