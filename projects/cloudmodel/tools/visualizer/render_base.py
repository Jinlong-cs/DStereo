import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import matplotlib.patches as mpatches
import numpy
import numpy as np
from dataclasses_json import DataClassJsonMixin
from hatbc.workflow.operator import Operator

from projects.cloudmodel.tools.visualizer.base.structure import ColorMode
from projects.cloudmodel.tools.visualizer.base.visualizer_base import (
    VisualizerBase,
)
from projects.cloudmodel.tools.visualizer.class_metas.utils import (
    ClassMetaBase,
)

logger = logging.getLogger(__name__)

__all__ = ["RenderBase"]


@dataclass
class ImageClassification(DataClassJsonMixin):
    """
    Data class for image classification configuration.

    Attributes:
        image_classification_on (Optional[bool]): Flag indicating if image classification render is enabled.
            Default is True.
    """

    image_classification_on: Optional[bool] = True


@dataclass
class InstanceConfigs(DataClassJsonMixin):
    """
    Data class for instance configuration.

    Attributes:
        object_detection_on (Optional[bool]): Flag indicating if object detection is enabled.
            Default is True.
        keypoint_instance_on (Optional[bool]): Flag indicating if keypoint instance is enabled.
            Default is True.
        lane_instanceseg_on (Optional[bool]): Flag indicating if lane instance segmentation is enabled.
            Default is True.
        vis_class_name (Optional[Union[str, List]]): Visualization class name. Can be a string or a list.
            Default is "all", indicating render all valid classes.
        instance_threshold (Optional[Union[float, Dict]]): Instance threshold value. Can be a float or a dictionary.
            If float is given, all class share the same threshold. If dict is given, the key-value of the dict should
            be classname and the respect threshold
    """

    object_detection_on: Optional[bool] = True
    keypoint_instance_on: Optional[bool] = True
    lane_instanceseg_on: Optional[bool] = True
    vis_class_name: Optional[Union[str, List]] = "all"
    instance_threshold: Optional[Union[float, Dict]] = 0.05


@dataclass
class SemanticSegConfigs(DataClassJsonMixin):
    """
    Data class for semantic segmentation configuration.

    Attributes:
        parsing_on (Optional[bool]): Flag indicating if parsing is enabled.
            Default is True.
        lane_parsing_on (Optional[bool]): Flag indicating if lane parsing is enabled.
            Default is True.
    """

    parsing_on: Optional[bool] = True
    lane_parsing_on: Optional[bool] = True


@dataclass
class Real3DConfigs(DataClassJsonMixin):
    """
    Data class for Real3D configuration.

    Attributes:
        real3d_on (Optional[bool]): Flag indicating if Real3D is enabled.
            Default is False.
        real3d_bev_on (Optional[bool]): Flag indicating if Real3D bev image is enabled.
            Default is False.
        bev_overlay_on_image (Optional[bool]): Flag indicating if bev image overlay on the image.
            Default is True.
        real3d_score_threshold (Optional[float]): Score threshold for Real3D.
            Default is True.
        real3d_depth_threshold (Optional[float]): Depth threshold for Real3D.
            Default is True.
        real3d_truncation_thresh (Optional[float]): Truncation threshold for Real3D.
            Default is True.
    """

    real3d_on: Optional[bool] = False
    real3d_bev_on: Optional[bool] = False
    bev_overlay_on_image: Optional[bool] = True
    real3d_score_threshold: Optional[float] = True
    real3d_depth_threshold: Optional[float] = True
    real3d_truncation_thresh: Optional[float] = True


class RenderBase(Operator):
    def __init__(
        self,
        image_scale: Optional[float] = 1.0,
        color_mode: Optional[int] = ColorMode.CLASS_FIX,
        label_legend: Optional[bool] = False,
        classification_config: Optional[ImageClassification] = None,
        instance_config: Optional[InstanceConfigs] = None,
        semanticseg_config: Optional[SemanticSegConfigs] = None,
        real3d_config: Optional[Real3DConfigs] = None,
    ):
        super(RenderBase, self).__init__()
        self.drawer = None
        self.image_scale = image_scale
        self.color_mode = color_mode
        self.label_legend = label_legend
        self.classification_config = (
            classification_config or ImageClassification()
        )
        self.instance_config = instance_config or InstanceConfigs()
        self.semanticseg_config = semanticseg_config or SemanticSegConfigs()
        self.real3d_config = real3d_config or Real3DConfigs()
        assert (self.real3d_config.real3d_on is False) and (
            self.real3d_config.real3d_bev_on is False
        ), "Real3D result do not supported yet"

    def forward(
        self,
        source_data: Any,
        source_types: str,
        source_image: Optional[numpy.ndarray] = None,
        **kwargs,
    ):
        raise NotImplementedError

    def render_source_generator(
        self, image_data: Any, image_shape: List, **kwargs
    ):
        raise NotImplementedError

    def _init_visualizer(
        self,
        img_rgb: Optional[numpy.ndarray],
        metadata: Optional[ClassMetaBase] = None,
        scale: Optional[float] = 1.0,
        instance_mode: Optional[int] = ColorMode.CLASS_FIX,
        label_legend: Optional[bool] = True,
    ):
        assert self.drawer is None
        if self.real3d_config.real3d_bev_on:
            x_ratio = 0.15 if self.real3d_config.bev_overlay_on_image else 0.3
            bev_size = int(img_rgb.shape[1] * x_ratio)
            bev_img = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
        else:
            bev_img = None
        self.drawer = VisualizerBase(
            img_rgb,
            bev_img,
            metadata,
            scale,
            instance_mode,
            label_legend,
            bev_overlay_on_image=self.real3d_config.bev_overlay_on_image,
        )

    def generate_legend(
        self,
        task_name: str,
        loc: str,
        fontsize: float,
        ncol: Optional[int] = 1,
    ):
        handles = self.block_legend(task_name)
        if len(handles) == 0:
            return
        self.drawer.output.fig.legend(
            handles=handles,
            loc=loc,
            title=task_name,
            fontsize=fontsize,
            frameon=True,
            framealpha=0.5,
            ncol=ncol,
            handlelength=2,
            handleheight=2,
        )

    def block_legend(self, task_name: str):
        patches = [
            mpatches.Patch(color=i[1], label=f"{i[0]}")
            for i in sorted(self.drawer.cls_info[task_name])
        ]
        return patches

    def render_parsing(
        self, label_mask: np.ndarray, color_map: Dict, task_name
    ):

        if label_mask is not None:
            for label_name, label_color in color_map.items():
                self.drawer.cls_info[task_name].add(
                    (label_name, tuple([i / 255.0 for i in label_color]))
                )
            self.drawer.draw_sem_label(
                label_mask,
                color_map,
                area_threshold=0,
                alpha=0.35,
            )

    def render_instance(
        self,
        predictions,
    ):
        self.drawer.draw_instance_predictions(predictions)

    def render_real3d_instance(
        self,
        instances_real3d,
    ):
        self.drawer.draw_bbox3d(instances_real3d)
        if self.real3d_config.real3d_bev_on:
            self.drawer.draw_bev(instances_real3d)

    def render_image_class(self, classes):
        self.drawer.draw_image_classification(classes)
