# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os
from typing import Dict, Sequence, Union

import cv2

from hat.core.data_struct.img_structures import ImgBase
from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import get_dist_info
from hat.visualize.vis_img_struct import vis_img_struct
from .visualize import BaseVisualize

__all__ = ["DetMultitaskVisualize"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class DetMultitaskVisualize(BaseVisualize):
    """
    DetVisualize callback is used for visualize results on 2d-detection tasks.

    Args:
        out_keys: Visualize tasks.
        vis_configs: Visualize configs for multi-task.
        output_dir: Output dir for saving results.
        save_viz_imgs: Whether to save viz imgs.
        overwrite: Whether to overwrite existed viz imgs.
    """

    def __init__(
        self,
        out_keys: Union[str, Sequence[str]],
        vis_configs: Dict[str, Dict],
        output_dir: str = "./tmp_viz_imgs/",
        save_viz_imgs: bool = False,
        overwrite: bool = False,
    ):
        super().__init__()
        self.out_keys = out_keys
        self.vis_configs = vis_configs
        self.output_dir = output_dir
        self.save_viz_imgs = save_viz_imgs
        self.overwrite = overwrite

        rank, _ = get_dist_info()
        if rank == 0 and self.save_viz_imgs:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

    def visualize(self, global_id, batch, result):

        model_outs = [result[key] for key in self.out_keys]

        if isinstance(batch, tuple):
            batch = batch[0]

        for i, model_out in enumerate(zip(*model_outs)):
            calib = batch.get("calib", None)
            if calib is not None:
                calib = calib[i]
            distCoeffs = batch.get("distCoeffs", None)
            if distCoeffs is not None:
                distCoeffs = distCoeffs[i]
            img_struct = ImgBase(
                img_id=batch["img_id"][i].item(),
                img=batch["img"][i],
                ori_img=batch["ori_img"][i],
                layout=batch["layout"][i],
                color_space=batch["color_space"][i],
                img_width=batch["img_width"][i],
                img_height=batch["img_width"][i],
                calib=calib,
                distCoeffs=distCoeffs,
            )

            vis_img = None
            for k, res in zip(self.out_keys, model_out):
                vis_configs = self.vis_configs.get(k, None)
                if self.save_viz_imgs and vis_configs is not None:
                    res = res.to("cpu")
                    vis_img = vis_img_struct(
                        img_struct, res, vis_configs, vis_image=vis_img
                    )

            if self.save_viz_imgs:
                write_path = os.path.join(
                    self.output_dir, f"{img_struct.img_id}.png"
                )

                if os.path.exists(write_path) and self.overwrite:
                    break

                cv2.imwrite(write_path, vis_img)

        return global_id, None, result

    def __repr__(self):
        return "DetMultitaskVisualize"
