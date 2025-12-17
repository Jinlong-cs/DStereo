from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from projects.cloudmodel.tools.visualizer.base.element_drawer import (
    ElementDrawer,
)
from projects.cloudmodel.tools.visualizer.base.structure import (
    ClassName,
    ColorMode,
    GenericMask,
    KeyPoint,
)
from projects.cloudmodel.tools.visualizer.class_metas.utils import (
    ClassMetaBase,
)

__all__ = ["VisImage", "VisualizerBase"]


class VisImage:
    def __init__(
        self,
        img: np.ndarray,
        img_bev: Optional[np.ndarray] = None,
        scale: Optional[float] = 1.0,
        bev_overlay_on_image: Optional[bool] = True,
    ):
        self.img = img
        self.img_bev = img_bev
        self.bev_overlay_on_image = bev_overlay_on_image
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(
        self,
        img: np.ndarray,
    ):
        def _set_fig():
            self.dpi = fig.get_dpi()
            # add a small 1e-2 to avoid precision lost due to matplotlib's
            # truncation (https://github.com/matplotlib/matplotlib/issues/15363)  # noqa
            fig.set_size_inches(
                (fig_w * self.scale + 1e-2) / self.dpi,
                (fig_h * self.scale + 1e-2) / self.dpi,
            )
            self.canvas = FigureCanvasAgg(fig)

        fig = mplfigure.Figure(frameon=False)
        if self.img_bev is None:
            fig_w = self.width
            fig_h = self.height
            _set_fig()
            ax_img = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax_img.axis("off")
            # Need to imshow this first so that other patches can be drawn on top  # noqa
            ax_img.imshow(
                img,
                extent=(0, self.width, self.height, 0),
                interpolation="nearest",
            )
            ax_bev = None
        else:
            fig_w = (
                self.width
                if self.bev_overlay_on_image
                else self.width + self.img_bev.shape[1]
            )
            fig_h = self.height
            _set_fig()
            ax_img = fig.subplots(
                1,
                1,
                gridspec_kw=dict(
                    left=0,
                    bottom=0,
                    right=self.width / fig_w,
                    top=self.height / fig_h,
                ),
            )
            ax_img.axis("off")
            ax_img.imshow(
                img,
                extent=(0, self.width, self.height, 0),
                interpolation="nearest",
            )
            r = 1
            b = 0.6
            ax_bev = fig.subplots(
                1,
                1,
                gridspec_kw=dict(
                    left=r - self.img_bev.shape[1] / fig_w,
                    bottom=b,
                    right=r,
                    top=b + self.img_bev.shape[0] / fig_h,
                ),
            )
            if self.bev_overlay_on_image:
                bev_x1 = fig_w - self.img_bev.shape[1]
                bev_x2 = fig_w
                bev_y2 = fig_h // 2
                bev_y1 = fig_h // 2 - self.img_bev.shape[0]
                self.img_bev = (
                    self.img_bev * 0.8
                    + img[bev_y1:bev_y2, bev_x1:bev_x2, :] * 0.2
                ).astype(np.uint8)
            ax_bev.axis("off")
            ax_bev.imshow(
                self.img_bev,
                extent=(0, self.img_bev.shape[1], self.img_bev.shape[0], 0),
                interpolation="nearest",
            )
            ax_bev.yaxis.grid(True)
            ax_bev.yaxis.grid(True)
        self.fig = fig
        self.ax = ax_img
        self.ax_bev = ax_bev

    def save(self, filepath: str):
        self.fig.savefig(filepath)

    def get_image(self):
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


class VisualizerBase:
    def __init__(
        self,
        img_rgb: np.ndarray,
        image_bev: Optional[np.ndarray] = None,
        metadata: Optional[ClassMetaBase] = None,
        scale: Optional[float] = 1.0,
        instance_mode: Optional[ColorMode] = ColorMode.CLASS_FIX,
        label_legend: Optional[bool] = True,
        bev_overlay_on_image: Optional[bool] = True,
    ):
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.image_bev = (
            np.asarray(image_bev).clip(0, 255).astype(np.uint8)
            if image_bev is not None
            else None
        )
        self.metadata = metadata
        assert isinstance(self.metadata, ClassMetaBase)
        self.output = VisImage(
            self.img,
            self.image_bev,
            scale=scale,
            bev_overlay_on_image=bev_overlay_on_image,
        )

        # too small texts are useless, therefore clamp to 9
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )
        self._instance_mode = instance_mode
        self.label_legend = label_legend
        self.cls_info = defaultdict(set)
        self.elem_drawer = ElementDrawer(self.output, scale, metadata)

    def instance2color(self, classes: List[ClassName], colormap):
        has_class_name = all([i.classname is not None for i in classes])
        if (
            self._instance_mode == ColorMode.SEGMENTATION
            and colormap
            and has_class_name
        ):
            colors = [
                self._jitter(tuple(x for x in colormap[c.classname]))
                for c in classes
            ]
            alpha = 0.8
        elif (
            self._instance_mode == ColorMode.CLASS_FIX
            and colormap
            and has_class_name
        ):
            colors = []
            for c in classes:
                colors.append(list(colormap[c.classname]))
            alpha = 0.5
        else:
            colors = None
            alpha = 0.5
        if self._instance_mode == ColorMode.IMAGE_BW:
            alpha = 0.3

        return colors, alpha

    def draw_instance_predictions(self, predictions: Dict):
        boxes = predictions.get("boxes", None)
        scores = predictions.get("scores", None)
        classes = predictions.get("classes", None)
        keypoints = predictions.get("keypoints", None)
        labels, labels_pos = self._create_text_labels(classes, scores)

        if predictions.get("pred_masks", None):
            masks = np.asarray(predictions["pred_masks"])
            masks = [
                GenericMask(x, self.output.height, self.output.width)
                for x in masks
            ]
        else:
            masks = None

        if classes is not None:
            colors, alpha = self.instance2color(
                classes, colormap=self.metadata.thing_colors
            )
        else:
            colors = None
            alpha = 0.4

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(
                (predictions["pred_masks"].any(dim=0) > 0).numpy()
                if predictions.get("pred_masks", None)
                else None
            )

        if colors is not None:
            assert len(colors) == len(classes)
            for cls, col in zip(classes, colors):
                self.cls_info["thing_class"].add((cls.classname, tuple(col)))

        self.elem_drawer.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            labels_pos=labels_pos,
            assigned_colors=colors,
            alpha=alpha,
        )

        # draw keypoints
        if keypoints is not None:
            for idx, keypoints_groups in enumerate(keypoints):
                if isinstance(keypoints_groups, KeyPoint):
                    keypoints_per_instance = keypoints_groups.point
                    keypoints_task_name = keypoints_groups.point_type
                else:
                    keypoints_per_instance = keypoints
                    keypoints_task_name = None
                keypoints_per_instance = self._convert_keypoints(
                    keypoints_per_instance
                )

                self.elem_drawer.draw_and_connect_keypoints(
                    keypoints_per_instance,
                    keypoints_task_name,
                    instance_label=labels[idx] if labels is not None else None,
                    instance_color=colors[idx] if colors is not None else None,
                )
        return self.output

    def draw_bbox3d(self, instances_real3d):
        for instance_3d in instances_real3d:
            cls = instance_3d["class_name"]
            color = instance_3d["color"]
            colors, alpha = self.instance2color(
                [cls], colormap={cls.classname: color}
            )
            self.cls_info["thing_class"].add((cls.classname, tuple(colors[0])))
            labels, poses = self._create_text_labels(
                class_names=[cls],
                scores=[instance_3d["score"]],
            )
            if instance_3d["render_mode"] == "box2d":
                bbox2d = instance_3d["bbox2d"]
                self.elem_drawer.overlay_instances(
                    masks=None,
                    boxes=np.array([bbox2d]),
                    labels=labels,
                    labels_pos=poses,
                    assigned_colors=colors,
                    alpha=alpha,
                )
            elif instance_3d["render_mode"] == "box3d":
                self.elem_drawer.overlay_real3d_instance(
                    corners_pts_2d_list=[instance_3d["corners_pts_2d"]],
                    labels=labels,
                    labels_pos=poses,
                    assigned_colors=colors,
                    face_idx=instance_3d["face_idx"],
                    show_arrow=instance_3d["show_arrow"],
                )
            else:
                raise ValueError("[3D Render]render_mode do not support.")

    def draw_bev(self, instances_bev):
        for instance_bev in instances_bev:
            cls = instance_bev["class_name"]
            color = instance_bev["color"]
            colors, alpha = self.instance2color(
                [cls], colormap={cls.classname: color}
            )
            labels, _ = self._create_text_labels(
                class_names=[cls],
                scores=[instance_bev["score"]],
            )
            self.elem_drawer.overlay_instance_on_bev(
                rect_bev_2ds=[instance_bev["rect_bev_2d"]],
                labels=labels,
                labels_pos=None,
                assigned_colors=colors,
                alpha=alpha,
            )

    def draw_sem_label(
        self,
        label_img: np.ndarray,
        color_map: Dict[str, List],
        area_threshold: Optional[int] = None,
        alpha: Optional[float] = 0.8,
    ):
        labels, areas = np.unique(label_img, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        label_colors = list(color_map.values())
        label_names = list(color_map.keys())

        for label in filter(lambda l: l < len(color_map), labels):
            try:
                mask_color = [x / 255 for x in label_colors[label]]
            except (AttributeError, IndexError):
                mask_color = None

            binary_mask = (label_img == label).astype(np.uint8)
            text = (
                label_names[label]
                if not self.label_legend
                and label_names[label] not in ["background"]
                else None
            )
            self.elem_drawer.draw_binary_mask(
                binary_mask,
                color=mask_color,
                edge_color=None,
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        return self.output

    def draw_image_classification(self, class_preds: Dict):
        roi_pos = class_preds["roi_pos"]
        classes = class_preds["classes"]

        for roi, class_i in zip(roi_pos, classes):
            labels, _ = self._create_text_labels([class_i], None)
            label = labels[0]
            color = self.metadata.image_cls_colors[class_i.classname]
            x0, y0, x1, y1 = roi
            text_pos = (x0, y0)
            horiz_align = "left"
            lighter_color = self.elem_drawer._change_color_brightness(
                color, brightness_factor=0.1
            )
            # for small objects, draw text at the side to avoid occlusion
            vertical_align = "baseline"

            height_ratio = (y1 - y0) / np.sqrt(
                self.output.height * self.output.width
            )
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 0.3, 1.0, 1.6)
                * 0.8
                * self._default_font_size
            )
            self.elem_drawer.draw_text(
                label,
                text_pos,
                color=lighter_color,
                horizontal_alignment=horiz_align,
                font_size=font_size,
                verticalalignment=vertical_align,
            )

    def _jitter(self, color: Union[List, Tuple]):
        color = mplc.to_rgb(color)
        vec = np.random.rand(3)
        # better to do it in another color space
        vec = vec / np.linalg.norm(vec) * 0.5
        res = np.clip(vec + color, 0, 1)
        return tuple(res)

    def _create_grayscale_image(self, mask: Optional[np.ndarray] = None):
        img_bw = self.img.astype("f4").mean(axis=2)
        img_bw = np.stack([img_bw] * 3, axis=2)
        if mask is not None:
            img_bw[mask] = self.img[mask]
        return img_bw

    @staticmethod
    def _convert_keypoints(keypoints: Union[np.ndarray, list]):
        if isinstance(keypoints, np.ndarray):
            assert keypoints.shape[-1] in [3]
            return keypoints
        elif isinstance(keypoints, list):
            return np.array(keypoints)
        else:
            raise TypeError(f"Unsupported keypoint types: {type(keypoints)}")

    def _create_text_labels(
        self,
        class_names: List[ClassName],
        scores: List[float],
        attr_map: Optional[Dict] = None,
    ):
        labels = None
        if attr_map is None:
            attr_map = self.metadata.task_name_map
        # parse class labels
        label_position = None
        class_labels = None
        if class_names is not None:
            # do not draw class name when using label_legend
            if not self.label_legend:
                class_labels = [str(i.classname) for i in class_names]
            label_position = [str(i.label_position) for i in class_names]
        # parse attributes
        attrs = None
        if class_names is not None:
            attrs = [i.attrs for i in class_names]
            attrs = [
                [
                    f"{attr_map[k]}:{v}" if k in attr_map else f"{k}:{v}"
                    for k, v in attr.items()
                ]
                for attr in attrs
            ]
            attrs = ["\n".join(i) if len(i) > 0 else None for i in attrs]
        # generate texts
        if scores is not None and class_labels is not None:
            labels = [f"{c}|{s:.2f}" for (c, s) in zip(class_labels, scores)]
        elif scores is None and class_labels is not None:
            labels = [f"{c}" for c in class_labels]
        elif scores is not None and class_labels is None:
            labels = [f"{s:.3f}" for s in scores]
        if attrs:
            if labels is None:
                labels = attrs
            else:
                labels = [
                    f"{ll}" if a is None else f"{ll}\n{a}"
                    for ll, a in zip(labels, attrs)
                ]

        return labels, label_position

    def get_output(self):
        return self.output
