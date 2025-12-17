import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from easydict import EasyDict
from hatbc.message.frame import CameraFrame
from hatbc.message.objects import DenseMap, Instance
from hatbc.utils.utils import _as_list

from projects.cloudmodel.tools.visualizer.base.structure import (
    ClassName,
    ColorMode,
    KeyPoint,
)
from projects.cloudmodel.tools.visualizer.base.visualizer_base import (
    VisualizerBase,
)
from projects.cloudmodel.tools.visualizer.class_metas.utils import (
    ClassMetaBase,
)
from projects.cloudmodel.tools.visualizer.render_base import (
    ImageClassification,
    InstanceConfigs,
    Real3DConfigs,
    RenderBase,
    SemanticSegConfigs,
)

logger = logging.getLogger(__name__)

__all__ = ["RenderCameraFrame"]


class RenderCameraFrame(RenderBase):
    """
        This is the constructor method for the  RenderCameraFrame.
        RenderCameraFrame is used to render CameraFrames in a specific configuration.

    Args:
        image_scale : A float value in [0, 1] representing
            the result image scale relative to original image.
        color_mode: An integer value representing the color mode.
        label_legend: An instance of the  ImageClassification representing
            the configuration for image classification.
        classification_config: An instance of the InstanceConfigs
            representing the configuration for instance segmentation.
        instance_config: An instance of the  InstanceConfigs
            representing the configuration for instance segmentation.
        semanticseg_config: An instance of the  SemanticSegConfigs
            representing the configuration for semantic segmentation.
        real3d_config: An instance of the  Real3DConfigs
            representing the configuration for 3D rendering.
    """

    _LEGEND_FONT_SCALE = 3e-3

    def __init__(
        self,
        image_scale: Optional[float] = 1.0,
        color_mode: Optional[int] = ColorMode.CLASS_FIX,
        label_legend: Optional[bool] = True,
        classification_config: Optional[ImageClassification] = None,
        instance_config: Optional[InstanceConfigs] = None,
        semanticseg_config: Optional[SemanticSegConfigs] = None,
        real3d_config: Optional[Real3DConfigs] = None,
    ):
        super(RenderCameraFrame, self).__init__(
            image_scale=image_scale,
            color_mode=color_mode,
            label_legend=label_legend,
            classification_config=classification_config,
            instance_config=instance_config,
            semanticseg_config=semanticseg_config,
            real3d_config=real3d_config,
        )
        self.drawer = None

    def forward(
        self,
        frame_data: CameraFrame,
        source_image: Optional[np.ndarray] = None,
        metadata: Optional[ClassMetaBase] = None,
        **kwargs,
    ):
        """
        Render the CameraFrame in a specific configuration.

        Args:
            frame_data: The CameraFrame object to be rendered.
            source_image: Optional numpy array representing the source image.
                If not provided, it will be extracted from the frame_data.
            metadata: Optional ClassMetaBase object representing the metadata.
                If not provided, a new instance of ClassMetaBase will be created.

        Returns:
            The rendered result.
        """
        if source_image is None:
            source_image = self.image_from_source_data(frame_data)
        else:
            assert isinstance(
                source_image, np.ndarray
            ), "source_image must be a numpy array"
        if metadata is None:
            metadata = ClassMetaBase()
        self.drawer: Optional[VisualizerBase] = None
        self._init_visualizer(
            img_rgb=source_image,
            metadata=metadata,
            scale=self.image_scale,
            instance_mode=self.color_mode,
            label_legend=self.label_legend,
        )
        render_tasks = self.render_source_generator(
            frame_data, source_image.shape
        )
        render_res = self.apply_render(render_tasks)

        if self.label_legend:
            _, img_w, _ = source_image.shape
            fontsize = self._LEGEND_FONT_SCALE * img_w * self.image_scale
            self.generate_legend("thing_class", "upper left", fontsize)
            self.generate_legend("lane_parsing", "upper right", fontsize)
            self.generate_legend("parsing", "upper center", fontsize, ncol=20)

        return render_res

    def render_source_generator(
        self,
        frame_data: CameraFrame,
        image_shape: List,
        **kwargs,
    ):
        """
        Generate render tasks based on the given frame data and image shape.

        Args:
            frame_data: The CameraFrame object containing perception data.
            image_shape: The shape of the source image.
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary containing the render tasks.
        """
        results = defaultdict(list)

        if self.instance_config.object_detection_on:
            render_boxes = (
                list()
            )  # list of (x1 y1 x2 y2) or (x1 y1 x2 y2 angle)
            render_scores = list()  # list of float
            # list of dict, each dict contain 2 keys:
            # (`class_name`=str, `attrs`=dict(str=str))
            render_classes = list()
            if isinstance(self.instance_config.instance_threshold, float):
                class_wise_threshold = {
                    k: self.instance_config.instance_threshold
                    for k in self.drawer.metadata.thing_classes
                }
            elif isinstance(self.instance_config.instance_threshold, dict):
                class_wise_threshold = self.instance_config.instance_threshold
            else:
                raise TypeError("instance_threshold should a dict or float")

            def _gen_bbox_instance(instance: Instance):
                bbox2d = instance.bbox2ds[0]
                class_name = bbox2d.topic
                if class_name not in self.drawer.metadata.thing_classes:
                    return
                if (
                    self.instance_config.vis_class_name != "all"
                    and class_name
                    not in _as_list(self.instance_config.vis_class_name)
                ):
                    return

                score = -1 if bbox2d.score is None else bbox2d.score
                # thresholding
                if score < class_wise_threshold.get(class_name, -math.inf):
                    return
                render_boxes.append(
                    [bbox2d.x1, bbox2d.y1, bbox2d.x2, bbox2d.y2]
                )
                render_scores.append(score)
                render_classes.append(
                    ClassName(
                        classname=class_name,
                        label_position=self.drawer.metadata.class_name2position.get(
                            class_name, "left"
                        ),
                        attrs={i.topic: i.value for i in instance.attributes},
                    )
                )
                for sub_instance in instance.instances:
                    _gen_bbox_instance(sub_instance)

            # object detection results
            for perception in frame_data.perceptions:
                if isinstance(perception, Instance):
                    if len(perception.bbox2ds) == 0:
                        continue
                    else:
                        assert len(perception.bbox2ds) == 1
                    # we do not render bbox with polygons
                    if len(perception.polygon2ds) == 0:
                        _gen_bbox_instance(perception)
            if (
                len(render_boxes) > 0
                or len(render_scores) > 0
                or len(render_classes) > 0
            ):
                instances = EasyDict(
                    boxes=np.array(render_boxes)
                    if len(render_boxes) > 0
                    else None,
                    scores=render_scores if len(render_scores) > 0 else None,
                    classes=render_classes
                    if len(render_classes) > 0
                    else None,
                    keypoints=None,
                    masks=None,
                )
                results["instances"].append(instances)

        # polygon instance
        if self.instance_config.keypoint_instance_on:
            render_keypoints = list()
            render_scores = list()
            render_classes = list()
            for perception in frame_data.perceptions:
                if (
                    isinstance(perception, Instance)
                    and len(perception.polygon2ds) > 0
                ):
                    if (
                        perception.topic
                        in self.drawer.metadata.keypoint_task_to_names
                    ):
                        for polygon in perception.polygon2ds:
                            render_keypoints.append(
                                KeyPoint(
                                    point=[
                                        i + [1.0]
                                        for i in np.array(
                                            polygon.polygon
                                        ).tolist()
                                    ],
                                    point_type=polygon.topic,
                                )
                            )
                            render_classes.append(
                                ClassName(
                                    classname=polygon.topic,
                                    label_position=self.drawer.metadata.class_name2position.get(
                                        polygon.topic, "left"
                                    ),
                                )
                            )
                        # instence level scores stored in bbox
                        if len(perception.bbox2ds) == len(
                            perception.polygon2ds
                        ):
                            for bbox in perception.bbox2ds:
                                render_scores.append(float(bbox.score))

            instances = EasyDict(
                boxes=None,
                scores=render_scores if len(render_scores) > 0 else None,
                classes=render_classes if len(render_classes) > 0 else None,
                keypoints=render_keypoints
                if len(render_keypoints) > 0
                else None,
                masks=None,
            )
            results["keypoints"].append(instances)

        # image level classification
        if self.classification_config.image_classification_on:
            render_classes = list()
            for perception in frame_data.perceptions:
                if isinstance(perception, Instance):
                    if (
                        len(perception.attributes) > 0
                        and len(perception.bbox2ds) == 0
                        and len(perception.mask2ds) == 0
                        and len(perception.keypoint2ds) == 0
                        and len(perception.polygon2ds) == 0
                    ):
                        render_classes.append(
                            ClassName(
                                classname=perception.topic,
                                label_position="left",
                                attrs={
                                    i.topic: f"{i.value}:{i.score:.3f}"
                                    for i in perception.attributes
                                },
                            )
                        )
            # draw classification class as instance class
            if len(render_classes) > 0:
                h, w, c = image_shape
                num_topic = len(render_classes)
                rois = [
                    [
                        10,
                        int(h * (0.5 + i * 0.1)),
                        10 + int(w * 0.1),
                        int(h * (0.5 + (i + 1) * 0.1)),
                    ]
                    for i in range(num_topic)
                ]
                classes = EasyDict(
                    roi_pos=np.array(rois),
                    classes=render_classes
                    if len(render_classes) > 0
                    else None,
                )
                results["image_classification"].append(classes)
        # lane instance segmentation
        if self.instance_config.lane_instanceseg_on:
            render_scores = list()
            render_classes = list()
            render_masks = list()
            for perception in frame_data.perceptions:
                if isinstance(perception, Instance):
                    # TODO hardcode here, we assueme a instance with empty
                    #  bbox2ds as lane instance segmentation results
                    if (
                        len(perception.bbox2ds) > 0
                        or len(perception.mask2ds) == 0
                    ):
                        continue
                    assert len(perception.mask2ds) == 1
                    mask = perception.mask2ds[0].data.as_numpy()
                    render_masks.append(mask)
                    render_classes.append(
                        ClassName(
                            label_position="center",
                            attrs={
                                i.topic: f"{i.value}:{i.score:.2f}"
                                for i in perception.attributes
                            },
                        )
                    )
            if len(render_classes) > 0 or len(render_masks) > 0:
                instances = EasyDict(
                    boxes=None,
                    scores=None,
                    classes=render_classes
                    if len(render_classes) > 0
                    else None,
                    keypoints=None,
                    pred_masks=render_masks if len(render_masks) > 0 else None,
                )
                results["lane_instanceseg"].append(instances)

        if self.semanticseg_config.parsing_on:
            for perception in frame_data.perceptions:
                if isinstance(perception, DenseMap):
                    task_name = perception.topic
                    masks = perception.mask2ds
                    assert len(masks) == 1
                    mask = masks[0].data.as_numpy()
                    if (
                        task_name in self.drawer.metadata.parsing_colormap
                        and self.semanticseg_config.parsing_on
                    ):
                        results["parsing"].append(
                            dict(
                                task="parsing",
                                colormap=self.drawer.metadata.parsing_colormap[
                                    task_name
                                ],
                                mask=mask,
                            )
                        )
                    elif (
                        task_name in self.drawer.metadata.lane_parsing_colormap
                        and self.semanticseg_config.lane_parsing_on
                    ):
                        results["lane_parsing"].append(
                            dict(
                                task="lane_parsing",
                                colormap=self.drawer.metadata.lane_parsing_colormap[
                                    task_name
                                ],
                                mask=mask,
                            )
                        )
        return results

    def apply_render(self, render_tasks: Dict):
        """
        Applies the given render tasks to the image.

        Args:
            render_tasks: A dictionary containing the render tasks and their data.

        Returns:
            The rendered image as a numpy array.
        """
        for render_task_name, datas in render_tasks.items():
            if render_task_name in ["parsing", "lane_parsing"]:
                for data in datas:
                    self.render_parsing(
                        data["mask"],
                        color_map=data["colormap"],
                        task_name=render_task_name,
                    )
            elif render_task_name in [
                "instances",
                "lane_instanceseg",
                "keypoints",
            ]:
                for data in datas:
                    self.render_instance(data)
            elif render_task_name in [
                "image_classification",
            ]:
                for data in datas:
                    self.render_image_class(data)
        return self.drawer.output

    @staticmethod
    def image_from_source_data(source_data: CameraFrame):
        if source_data.image.data is None:
            source_data.image.read()
        image_rgb = source_data.image.as_numpy()
        return image_rgb
