import colorsys
import math
from typing import Union

import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import numpy as np
import pycocotools.mask as mask_util

from projects.cloudmodel.tools.visualizer.base.structure import GenericMask
from projects.cloudmodel.tools.visualizer.class_metas.utils import (
    ClassMetaBase,
    random_color,
)


class ElementDrawer:
    _RED = (1.0, 0, 0)
    _LARGE_MASK_AREA_THRESH = 120000
    _KEYPOINT_THRESHOLD = 0.05

    def __init__(self, fig, scale: float, class_meta: ClassMetaBase):
        self.output = fig
        self.class_meta = class_meta
        # too small texts are useless, therefore clamp to 9
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )

    # basic component
    def draw_rotated_box_with_label(
        self,
        rotated_box,
        alpha=0.5,
        edge_color="g",
        line_style="-",
        label=None,
        linewidth=None,
    ):
        cnt_x, cnt_y, w, h, angle = rotated_box
        # use thinner lines when the box is small
        linewidth = 1.0 if linewidth is None else linewidth

        theta = angle * math.pi / 180.0
        c = math.cos(theta)
        s = math.sin(theta)
        rect = [
            (-w / 2, h / 2),
            (-w / 2, -h / 2),
            (w / 2, -h / 2),
            (w / 2, h / 2),
        ]
        # x: left->right ; y: top->down
        rotated_rect = [
            (s * yy + c * xx + cnt_x, c * yy - s * xx + cnt_y)
            for (xx, yy) in rect
        ]
        for k in range(4):
            j = (k + 1) % 4
            self.draw_line(
                [rotated_rect[k][0], rotated_rect[j][0]],
                [rotated_rect[k][1], rotated_rect[j][1]],
                color=edge_color,
                linestyle="--" if k == 1 else line_style,
                linewidth=linewidth,
            )

        if label is not None:
            text_pos = rotated_rect[1]  # topleft corner

            height_ratio = h / np.sqrt(self.output.height * self.output.width)
            label_color = self._change_color_brightness(
                edge_color, brightness_factor=0.1
            )
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 0.3, 1.0, 1.6)
                * 0.4
                * self._default_font_size
            )
            self.draw_text(
                label,
                text_pos,
                color=label_color,
                font_size=font_size,
                rotation=angle,
            )

        return self.output

    def draw_binary_mask(
        self,
        binary_mask,
        color=None,
        *,
        edge_color=None,
        text=None,
        alpha=0.5,
        area_threshold=0,
    ):
        if color is None:
            color = random_color(rgb=True, maximum=1)
        color = mplc.to_rgb(color)

        has_valid_segment = False
        binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
        mask = GenericMask(binary_mask, self.output.height, self.output.width)
        shape2d = (binary_mask.shape[0], binary_mask.shape[1])

        if not mask.has_holes:
            # draw polygons for regular masks
            for segment in mask.polygons:
                area = mask_util.area(
                    mask_util.frPyObjects([segment], shape2d[0], shape2d[1])
                )
                if area < (area_threshold or 0):
                    continue
                has_valid_segment = True
                segment = segment.reshape(-1, 2)
                self.draw_polygon(
                    segment, color=color, edge_color=edge_color, alpha=alpha
                )
        else:
            # TODO: Use Path/PathPatch to draw vector graphics:
            # https://stackoverflow.com/questions/8919719/how-to-plot-a-complex-polygon
            rgba = np.zeros(shape2d + (4,), dtype="float32")
            rgba[:, :, :3] = color
            rgba[:, :, 3] = (mask.mask == 1).astype("float32") * alpha
            has_valid_segment = True
            self.output.ax.imshow(
                rgba, extent=(0, self.output.width, self.output.height, 0)
            )

        if text is not None and has_valid_segment:
            lighter_color = self._change_color_brightness(
                color, brightness_factor=0.1
            )
            (
                _num_cc,
                cc_labels,
                stats,
                centroids,
            ) = cv2.connectedComponentsWithStats(binary_mask, 8)
            largest_component_id = np.argmax(stats[1:, -1]) + 1

            # draw text on the largest component,
            # as well as other very large components.
            for cid in range(1, _num_cc):
                if (
                    cid == largest_component_id
                    or stats[cid, -1] > self._LARGE_MASK_AREA_THRESH
                ):
                    # median is more stable than centroid
                    # center = centroids[largest_component_id]
                    center = np.median((cc_labels == cid).nonzero(), axis=1)[
                        ::-1
                    ]
                    self.draw_text(text, center, color=lighter_color)
        return self.output

    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        labels_pos=None,
        masks=None,
        assigned_colors=None,
        alpha=0.5,
    ):
        num_instances = 0
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if assigned_colors is None:
            assigned_colors = [
                random_color(rgb=True, maximum=1) for _ in range(num_instances)
            ]
        if num_instances == 0:
            return self.output
        if labels is not None:
            assert len(labels) == num_instances
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = (
                [labels[k] for k in sorted_idxs]
                if labels is not None
                else None
            )
            labels_pos = (
                [labels_pos[k] for k in sorted_idxs]
                if labels_pos is not None
                else None
            )
            masks = (
                [masks[idx] for idx in sorted_idxs]
                if masks is not None
                else None
            )
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(
                        segment.reshape(-1, 2), color, alpha=alpha
                    )

            if labels is not None:
                # mask first maybe better
                if masks is not None:
                    # skip small mask without polygon
                    if len(masks[i].polygons) == 0:
                        continue
                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median)
                    # when box is not drawn median is less sensitive
                    # to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                elif boxes is not None:
                    posi = labels_pos[i] if labels_pos is not None else None
                    x0, y0, x1, y1 = boxes[i]
                    text_pos, horiz_align = self._instance_label_posi_wrt_box(
                        box=[x0, y0, x1, y1], labels_pos=posi
                    )
                else:
                    continue
                lighter_color = self._change_color_brightness(
                    color, brightness_factor=0.1
                )
                # for small objects, draw text at the side to avoid occlusion
                vertical_align = "baseline"

                height_ratio = (y1 - y0) / np.sqrt(
                    self.output.height * self.output.width
                )
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 0.3, 1.0, 1.6)
                    * 0.45
                    * self._default_font_size
                )
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                    verticalalignment=vertical_align,
                )

        return self.output

    def overlay_rotated_instances(
        self, boxes=None, labels=None, assigned_colors=None
    ):
        num_instances = len(boxes)

        if assigned_colors is None:
            assigned_colors = [
                random_color(rgb=True, maximum=1) for _ in range(num_instances)
            ]
        if num_instances == 0:
            return self.output

        # Display in largest to smallest order to reduce occlusion.
        if boxes is not None:
            areas = boxes[:, 2] * boxes[:, 3]

        sorted_idxs = np.argsort(-areas).tolist()
        # Re-order overlapped instances in descending order.
        boxes = boxes[sorted_idxs]
        labels = (
            [labels[k] for k in sorted_idxs] if labels is not None else None
        )
        colors = [assigned_colors[idx] for idx in sorted_idxs]

        for i in range(num_instances):
            self.draw_rotated_box_with_label(
                boxes[i],
                edge_color=colors[i],
                label=labels[i] if labels is not None else None,
            )

        return self.output

    def overlay_real3d_instance(
        self,
        *,
        corners_pts_2d_list=None,
        labels=None,
        labels_pos=None,
        assigned_colors=None,
        face_idx=((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
        show_arrow=False,
    ):
        if labels is not None:
            assert len(labels) == len(corners_pts_2d_list)
        if labels_pos is not None:
            assert len(labels_pos) == len(corners_pts_2d_list)
        if assigned_colors is None:
            assigned_colors = [
                random_color(rgb=True, maximum=1)
                for _ in range(corners_pts_2d_list)
            ]
        draw_lines = list()
        draw_corners = list()
        draw_labels = list()
        draw_arrows = list()
        for idx, corners_pts_2d in enumerate(corners_pts_2d_list):
            # points
            draw_corners.extend(
                [
                    dict(pt=list(i), color=assigned_colors[idx])
                    for i in corners_pts_2d
                ]
            )
            # lines
            for idx_f in range(3, -1, -1):
                face = face_idx[idx_f]
                for j in range(4):
                    draw_lines.append(
                        dict(
                            x_data=[
                                corners_pts_2d[face[j], 0],
                                corners_pts_2d[face[(j + 1) % 4], 0],
                            ],
                            y_data=[
                                corners_pts_2d[face[j], 1],
                                corners_pts_2d[face[(j + 1) % 4], 1],
                            ],
                            color=assigned_colors[idx],
                        )
                    )
                if show_arrow:
                    p1 = (
                        corners_pts_2d[0, :]
                        + corners_pts_2d[1, :]
                        + corners_pts_2d[2, :]
                        + corners_pts_2d[3, :]
                    ) / 4.0
                    p2 = (corners_pts_2d[0, :] + corners_pts_2d[1, :]) / 2
                    p3 = p2 + (p2 - p1) * 0.5
                    draw_arrows.append(
                        dict(
                            x_data=[p1[0], p3[0]],
                            y_data=[p1[1], p3[1]],
                            color=assigned_colors[idx],
                        )
                    )
                else:
                    # draw "X" for front face
                    if face_idx == 0:
                        draw_lines.extend(
                            [
                                dict(
                                    x_data=[
                                        corners_pts_2d[face[0], 0],
                                        corners_pts_2d[face[2], 0],
                                    ],
                                    y_data=[
                                        corners_pts_2d[face[0], 1],
                                        corners_pts_2d[face[2], 1],
                                    ],
                                    color=assigned_colors[idx],
                                ),
                                dict(
                                    x_data=[
                                        corners_pts_2d[[1], 0],
                                        corners_pts_2d[face[3], 0],
                                    ],
                                    y_data=[
                                        corners_pts_2d[face[1], 1],
                                        corners_pts_2d[face[3], 1],
                                    ],
                                    color=assigned_colors[idx],
                                ),
                            ]
                        )
            # labels
            if labels is not None:
                draw_labels.append(
                    dict(
                        label=labels[idx],
                        labels_pos=labels_pos[idx]
                        if labels_pos is not None
                        else None,
                        color=self._change_color_brightness(
                            assigned_colors[idx], brightness_factor=0.1
                        ),
                        box=[
                            corners_pts_2d[:, 0].min(),
                            corners_pts_2d[:, 1].min(),
                            corners_pts_2d[:, 0].max(),
                            corners_pts_2d[:, 1].max(),
                        ],
                    )
                )

        # draw lines
        for line in draw_lines:
            self.draw_line(
                x_data=line["x_data"],
                y_data=line["y_data"],
                color=line["color"],
                linewidth=1.0,
            )
        # draw errors
        for arrow in draw_arrows:
            self.draw_arrow(
                x_data=arrow["x_data"],
                y_data=arrow["y_data"],
                color=arrow["color"],
                linewidth=15.0,
            )
        # draw corner points
        for corner in draw_corners:
            self.draw_circle(
                circle_coord=corner["pt"], color=corner["color"], radius=2
            )
        # draw labels
        for label in draw_labels:
            text_pos, horiz_align = self._instance_label_posi_wrt_box(
                box=label["box"], labels_pos=label["labels_pos"]
            )
            vertical_align = "baseline"
            # y2 - y1
            height_ratio = (label["box"][3] - label["box"][1]) / np.sqrt(
                self.output.height * self.output.width
            )
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 0.3, 1.0, 1.6)
                * 0.4
                * self._default_font_size
            )
            self.draw_text(
                label["label"],
                text_pos,
                color=label["color"],
                horizontal_alignment=horiz_align,
                font_size=font_size,
                verticalalignment=vertical_align,
            )

    def overlay_instance_on_bev(
        self,
        rect_bev_2ds,
        labels,
        labels_pos,
        assigned_colors,
        alpha=0.5,
    ):
        if self.output.ax_bev is None:
            return
        if labels is not None:
            assert len(labels) == len(rect_bev_2ds)
        if labels_pos is not None:
            assert len(labels_pos) == len(rect_bev_2ds)
        if assigned_colors is None:
            assigned_colors = [
                random_color(rgb=True, maximum=1) for _ in range(rect_bev_2ds)
            ]
        for idx, rect_bev_2d in enumerate(rect_bev_2ds):
            # poly lines
            self.output.ax_bev.add_patch(
                mpl.patches.Polygon(
                    rect_bev_2d,
                    alpha=alpha,
                    color=assigned_colors[idx],
                )
            )
            p1 = np.mean(rect_bev_2d, axis=0)
            p2 = (rect_bev_2d[0] + rect_bev_2d[1]) / 2
            p3 = p2 + (p2 - p1) / 2
            p1, p3 = p1.astype(np.int32), p3.astype(np.int32)
            self.output.ax_bev.add_patch(
                mpl.patches.Arrow(
                    x=p1[0],
                    dx=p3[0] - p1[0],
                    y=p1[1],
                    dy=p3[1] - p1[1],
                    color=assigned_colors[idx],
                    width=5 * self.output.scale,
                    linestyle="-",
                )
            )

    def draw_and_connect_keypoints(
        self,
        keypoints,
        keypoints_task_name,
        instance_label=None,
        instance_color=None,
    ):
        visible = {}
        keypoint_names = self.class_meta.keypoint_names.get(
            keypoints_task_name, {}
        ).get("name", None)
        pt_color = self.class_meta.keypoint_names.get(
            keypoints_task_name, {}
        ).get("color", None)
        for idx, keypoint in enumerate(keypoints):
            # draw keypoint
            x, y, prob = keypoint
            c = pt_color[idx] if pt_color is not None else self._RED
            if prob > self._KEYPOINT_THRESHOLD:
                self.draw_circle((x, y), color=c)
                if keypoint_names:
                    keypoint_name = keypoint_names[idx]
                    visible[keypoint_name] = (x, y)

        if self.class_meta.keypoint_connection_rules.get(
            keypoints_task_name, None
        ):
            for kp0, kp1, color in self.class_meta.keypoint_connection_rules[
                keypoints_task_name
            ]:
                if kp0 in visible and kp1 in visible:
                    x0, y0 = visible[kp0]
                    x1, y1 = visible[kp1]
                    color = tuple(x for x in color)
                    self.draw_line(
                        [x0, x1], [y0, y1], color=color, linewidth=1.0
                    )
        if instance_label is not None:
            horiz_align = "center"
            center = (np.mean(keypoints[:, 0]), np.mean(keypoints[:, 1]))
            lighter_color = self._change_color_brightness(
                instance_color, brightness_factor=0.1
            )
            y0 = np.min(keypoints[:, 1])
            y1 = np.max(keypoints[:, 1])
            vertical_align = "baseline"
            height_ratio = (y1 - y0) / np.sqrt(
                self.output.height * self.output.width
            )
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 0.3, 1.0, 1.6)
                * 0.30
                * self._default_font_size
            )
            self.draw_text(
                instance_label,
                center,
                color=lighter_color,
                horizontal_alignment=horiz_align,
                font_size=font_size,
                verticalalignment=vertical_align,
            )

        return self.output

    # base elements

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color: Union[str, tuple] = "g",
        horizontal_alignment="center",
        rotation=0,
        verticalalignment="baseline",
    ):
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={
                "facecolor": "black",
                "alpha": 0.5,
                "pad": 0.4,
                "edgecolor": "none",
            },
            verticalalignment=verticalalignment,
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

    def draw_box(
        self,
        box_coord,
        alpha=0.8,
        edge_color="g",
        line_style="-",
        linewidth=None,
    ):
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = (
            max(self._default_font_size // 14, 1)
            if linewidth is None
            else linewidth
        )

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

    def draw_circle(self, circle_coord, color, radius=3):
        self.output.ax.add_patch(
            mpl.patches.Circle(
                circle_coord, radius=radius, fill=True, color=color
            )
        )
        return self.output

    def draw_line(self, x_data, y_data, color, linestyle="-", linewidth=None):
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        self.output.ax.add_line(
            mpl.lines.Line2D(
                x_data,
                y_data,
                linewidth=linewidth * self.output.scale,
                color=color,
                linestyle=linestyle,
            )
        )

        return self.output

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = self._change_color_brightness(
                    color, brightness_factor=-0.7
                )
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=0.5,
        )
        self.output.ax.add_patch(polygon)
        return self.output

    def draw_arrow(self, x_data, y_data, color, linestyle="-", linewidth=None):

        self.output.ax.add_patch(
            mpl.patches.Arrow(
                x=x_data[0],
                dx=x_data[1] - x_data[0],
                y=y_data[0],
                dy=y_data[1] - y_data[0],
                color=color,
                width=linewidth * self.output.scale,
                linestyle=linestyle,
            )
        )

    def _convert_masks(self, masks_or_polygons):
        ret = []
        for x in masks_or_polygons:
            if isinstance(x, GenericMask):
                ret.append(x)
            else:
                ret.append(
                    GenericMask(x, self.output.height, self.output.width)
                )
        return ret

    @staticmethod
    def _convert_boxes(boxes):
        if isinstance(boxes, np.ndarray):
            assert boxes.shape[-1] in [4, 5]
            return boxes

    @staticmethod
    def _change_color_brightness(color, brightness_factor):
        assert -1.0 <= brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (
            brightness_factor * polygon_color[1]
        )
        modified_lightness = (
            0.0 if modified_lightness < 0.0 else modified_lightness
        )
        modified_lightness = (
            1.0 if modified_lightness > 1.0 else modified_lightness
        )
        modified_color = colorsys.hls_to_rgb(
            polygon_color[0], modified_lightness, polygon_color[2]
        )
        return modified_color

    @staticmethod
    def _instance_label_posi_wrt_box(
        box,
        labels_pos=None,
    ):
        x0, y0, x1, y1 = box[:]
        if labels_pos is None:
            text_pos = (
                x0,
                y0 - 4,
            )  # if drawing boxes, put text on the box corner.
            horiz_align = "right"
        elif labels_pos.lower() in ["left", "l"]:
            text_pos = (
                x0,
                y0 - 4,
            )  # if drawing boxes, put text on the box corner.
            horiz_align = "right"
        elif labels_pos.lower() in ["right", "r"]:
            text_pos = (
                x1,
                y0 - 4,
            )  # if drawing boxes, put text on the box corner.
            horiz_align = "left"
        else:
            raise ValueError(f"invalid position labels:{labels_pos}")
        return text_pos, horiz_align
