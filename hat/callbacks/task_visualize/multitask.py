# Copyright (c) Horizon Robotics. All rights reserved.
import os
from typing import Any, Callable, Dict, Optional, Sequence, Union

from easydict import EasyDict
from PIL import Image

from hat.callbacks.task_visualize.visualize import BaseVisualize
from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import get_dist_info

__all__ = ["MultitaskVisualize"]


@OBJECT_REGISTRY.register
class MultitaskVisualize(BaseVisualize):
    def __init__(
        self,
        task_vis_func: Union[Callable, Dict[str, Callable]],
        match_task_output: Optional[Callable] = None,
        max_save_num: int = -1,
        output_dir: str = "./tmp_viz_imgs/",
    ):
        """Visualize callback for cloud model multitask model.

        Used for visualizing the results from multiple tasks.

        Args:
            task_vis_func: Functions for visualizing multitask results.
            match_task_output: Function to match current task outputs.
            max_save_num: Maximum image number for visualization.
            output_dir: Output dir for saving results.
        """
        super().__init__(
            output_dir=output_dir, match_task_output=match_task_output
        )
        self.task_vis_func = task_vis_func
        self.max_save_num = max_save_num
        self.output_dir = output_dir
        self.rank, self.world_size = get_dist_info()
        if self.rank == 0:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

    def on_batch_end(
        self,
        global_step_id: int,
        batch: Sequence[Dict[str, Any]],
        model_outs: Dict,
        **kwargs,
    ):
        if isinstance(batch, tuple):
            batch = batch[0]

        batch_size = len(batch["img"])
        step_id = kwargs.get("step_id", None)

        if (
            step_id
            and self.max_save_num > 0
            and (step_id * batch_size * self.world_size >= self.max_save_num)
        ):
            return

        if self.match_task_output:
            (
                batch_output,
                match_output,
            ) = self.match_task_output(batch, model_outs, **kwargs)
        else:
            batch_output, match_output = batch, model_outs

        self.visualize(global_step_id, batch_output, match_output)

    def visualize(
        self,
        global_step_id: int,
        batch: Sequence[Dict[str, Any]],
        results: Sequence[Dict[str, Any]],
    ):
        for batch_i, result in zip(batch, results):
            render_img = batch_i.img
            for task_name, pred in result.items():
                if isinstance(self.task_vis_func, dict):
                    vis_func = self.task_vis_func[task_name]
                else:
                    vis_func = self.task_vis_func
                render_img = vis_func(render_img, pred)

            write_path = os.path.join(
                self.output_dir, f"{batch_i.image_key}.jpg"
            )
            Image.fromarray(render_img).save(
                write_path, format="JPEG", quality=80
            )

    def __repr__(self):
        return "MultitaskVisualize"

    @staticmethod
    def default_match_func(
        batch: Dict[str, Any],
        model_outs: Sequence[Any],
        task_names: Sequence[str],
        verbose_name: bool = True,
        meta_keys: Optional[tuple] = (
            "img_height",
            "img_width",
            "orig_img",
            "img_id",
            "img_name",
            "color_space",
        ),
        **kwargs,
    ):
        def _get_img_metas(batch_in, meta_keys_in):
            if isinstance(batch_in, tuple):
                batch_in = batch_in[0]
            batch_size = len(batch_in["img"])
            ret = [
                EasyDict({k: batch_in[k][ind] for k in meta_keys_in})
                for ind in range(batch_size)
            ]

            return ret

        img_metas = _get_img_metas(batch, meta_keys)
        preds = [
            {k: v[ind] for k, v in model_outs.items() if k in task_names}
            for ind in range(len(img_metas))
        ]
        if len(preds[0]) == 0:
            return [], []

        epoch_id = kwargs.get("epoch_id", 0)
        task_name = "_".join(task_names)

        object_rets = []
        for img_meta in img_metas:
            img_id = int(img_meta.img_id)
            img_name = os.path.splitext(img_meta.img_name)[0]
            image_key = (
                f"epoch{epoch_id}_{task_name}_imgId{img_id}_{img_name}"
                if verbose_name
                else img_name
            )

            img_rgb = img_meta.orig_img
            if img_meta.color_space == "bgr":
                img_rgb = img_rgb[:, :, ::-1].copy()

            object_ret = EasyDict(
                image_key=image_key,
                img=img_rgb,
                img_width=int(img_meta.img_width),
                img_height=int(img_meta.img_height),
                layout="hwc",
                color_space="rgb",
            )
            object_rets.append(object_ret)
        return object_rets, preds
