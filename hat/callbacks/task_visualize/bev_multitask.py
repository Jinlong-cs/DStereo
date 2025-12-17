# Copyright (c) Horizon Robotics. All rights reserved.
import json
import logging
import os
from typing import Dict, Sequence, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import get_dist_info
from hat.visualize.detection_bev.visualize import (
    draw_multi_camera_imgs,
    get_virtual_params,
)
from .visualize import BaseVisualize

__all__ = ["BEVMultitaskVisualize"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class BEVMultitaskVisualize(BaseVisualize):
    """
    BEVVisualize callback is used for visualize results on bev-detection tasks.

    Args:
        out_keys: Visualize tasks.
        bird_eye_size: Size of bev_map.
        vcs_range: Range of vcs valid range.
        vis_image_layout: multi view picture arrange.
        vis_configs: Visualize configs for multi-task.
        ori_img_keys: Key of ori img in batch.
        output_dir: Output dir for saving results.
        save_viz_imgs: Whether to save viz imgs.
    """

    def __init__(
        self,
        out_keys: Union[str, Sequence[str]],
        bird_eye_size: Sequence[int],
        vcs_range: Sequence[Union[float, int]],
        vis_image_layout: Dict[str, Sequence[int]],
        vis_configs: Dict[str, Dict],
        output_dir: str = "./tmp_viz_imgs/",
        ori_img_keys: Sequence[str] = ["img_ori"],  # noqa: B006
        save_viz_imgs: bool = False,
    ):
        super().__init__(output_dir=output_dir)
        self.out_keys = out_keys
        self.bird_eye_size = bird_eye_size
        self.vcs_range = vcs_range
        self.vis_image_layout = vis_image_layout
        self.vis_configs = vis_configs
        self.output_dir = output_dir
        self.save_viz_imgs = save_viz_imgs
        self.ori_img_keys = ori_img_keys

        rank, _ = get_dist_info()
        if rank == 0 and self.save_viz_imgs:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

    def visualize(self, global_id, batch, result):
        meta_info = (
            batch[0]["meta"] if isinstance(batch, tuple) else batch["meta"]
        )
        num_batch = len(meta_info)
        if num_batch == 0:
            return
        if isinstance(meta_info[0], str):
            meta_list = [json.loads(meta) for meta in meta_info]
        else:
            meta_list = meta_info

        model_outs = [result[key] for key in self.out_keys]

        if isinstance(batch, tuple):
            batch = batch[0]

        # for each sample of batch
        for batch_idx, model_out in enumerate(zip(*model_outs)):
            sample_key = meta_list[batch_idx]["timestamp"]
            img_multi_view = []
            for key in self.ori_img_keys:
                ori_imgs = batch[key].cpu().numpy()
                _n_views = ori_imgs.shape[0] // num_batch
                img_list = list(
                    ori_imgs[_n_views * batch_idx : _n_views * (batch_idx + 1)]
                )
                img_multi_view.extend(img_list)

            if not self.save_viz_imgs:
                continue

            meta_multi_view = meta_list[batch_idx]["view_anno"]
            assert len(img_multi_view) == len(meta_multi_view)

            # draw bird eye view map
            bird_eye_size = self.bird_eye_size
            vcs_range = self.vcs_range
            bev_map = np.zeros(
                (bird_eye_size[0], bird_eye_size[1], 3), dtype=np.uint8
            )
            cu, cv, vf = get_virtual_params(bird_eye_size, vcs_range)
            cv2.circle(
                bev_map,
                (int(cu + 0.5), int(cv + 0.5)),
                3,
                (0, 255, 0),
                thickness=2,
            )
            for r in range(10, int(abs(vcs_range[0])) + 5, 10):
                r = int(vf * r)
                cv2.circle(bev_map, (cu, cv), r, (0, 255, 0), 1)
            # for each task
            img_list = list(img_multi_view)
            for k, res in zip(self.out_keys, model_out):
                img_list, bev_map = draw_multi_camera_imgs(
                    img_list,
                    bev_map,
                    [
                        meta["meta"]["calib"]
                        for meta in meta_multi_view.values()
                    ],
                    res,
                    vis_configs=self.vis_configs[k],
                    bird_eye_size=bird_eye_size,
                    vcs_range=vcs_range,
                )

            # reorder and concat
            row = (
                max([layout[0] for layout in self.vis_image_layout.values()])
                + 1
            )
            col = (
                max([layout[1] for layout in self.vis_image_layout.values()])
                + 1
            )
            fig = plt.figure(figsize=(col * 18, row * 12))
            # vis bird eye view map
            layout = self.vis_image_layout["bird_eye_view"]
            ax = fig.add_subplot(row, col, layout[0] * col + layout[1] + 1)
            ax.imshow(bev_map)
            ax.axis("off")
            ax.set_title("bird eye map", fontsize=18)
            # vis each camera image
            for idx, camera in enumerate(meta_multi_view.keys()):
                if camera in self.vis_image_layout:
                    vis_img = img_list[idx]
                    layout = self.vis_image_layout[camera]
                    ax = fig.add_subplot(
                        row, col, layout[0] * col + layout[1] + 1
                    )
                    ax.imshow(vis_img)
                    ax.axis("off")
                    if (
                        "image_key"
                        in meta_list[batch_idx]["view_anno"][camera]["meta"]
                    ):
                        plt_key = meta_list[batch_idx]["view_anno"][camera][
                            "meta"
                        ]["image_key"]
                    else:
                        plt_key = meta_list[batch_idx]["timestamp"]
                    ax.set_title(
                        f"{camera}:{plt_key}",  # noqa
                        fontsize=18,
                    )
            plt.tight_layout()
            plt.savefig(
                f"{os.path.join(self.output_dir, str(sample_key))}.jpg"
            )
            plt.close()
        return global_id, None, result

    def __repr__(self):
        return "BEVMultitaskVisualize"
