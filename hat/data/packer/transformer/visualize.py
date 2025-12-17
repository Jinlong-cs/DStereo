import json
import os
import os.path as osp
import random
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = [
    "VizDenseBoxDetAnno",
    "VizRoiDenseBoxDetAnno",
    "plot_densebox_image_record_bbox",
    "VizFlankCornersMatrixGluon",
]  # noqa


def _draw_bbox(image, bbox, color=(235, 135, 206), thickness=2):
    image = cv2.rectangle(
        image,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])),
        color=color,
        thickness=thickness,
    )

    return image


def _draw_square(image, points, color=(255, 0, 0), thickness=2):
    for i in range(len(points) - 1):
        image = _draw_line(image, points[i], points[i + 1], color, thickness)
    image = _draw_line(
        image, points[len(points) - 1], points[0], color, thickness
    )

    return image


def _draw_line(image, pt1, pt2, color=(255, 0, 0), thickness=2):
    pt1 = (int(pt1[0]), int(pt1[1]))
    pt2 = (int(pt2[0]), int(pt2[1]))
    return cv2.line(image, pt1, pt2, color=color, thickness=thickness)


@OBJECT_REGISTRY.register
class VizFlankDetAnno(object):
    """
    Visualize Vehicle Flank detection annotation.

    Args:
        task_name:
            What kind of Flank task, possible values are
            {'flanks'}
        save_flag :
            Whether save rendered image, by default False
        save_path:, optional
            Where to save image, by default None
    """

    def __init__(
        self,
        task_name: str = "flanks",
        save_flag: bool = False,
        save_path: str = None,
    ):
        self.task_name = task_name
        self.save_flag = save_flag
        self.save_path = save_path

        self._mkdir()

    def _mkdir(self):
        if self.save_flag and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def __call__(self, img, anno, **kwargs):
        image_url = anno["image"]
        bbox_lst = anno["bboxes"]
        flank_lst = anno[self.task_name]
        ignore_lst = anno["ignore_regions"]
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # draw positive and negative
        for bbox, flank in zip(bbox_lst, flank_lst):
            x1, y1, x2, y2 = bbox["data"]
            flank_points = flank["data"]  # noqa
            flank_cls = flank["class_id"]
            if flank_cls == 0:
                # negative
                image = _draw_bbox(image, [x1, y1, x2, y2], color=(0, 255, 0))
            else:
                # positive
                image = _draw_bbox(image, [x1, y1, x2, y2], color=(0, 0, 255))
                pt0, pt1, pt2, pt3 = flank_points
                image = _draw_square(
                    image, [pt0, pt1, pt3, pt2], color=(255, 0, 0)
                )

        # draw ignore
        for ignore in ignore_lst:
            x1, y1 = ignore["left_top"]
            x2, y2 = ignore["right_bottom"]
            image = _draw_bbox(image, [x1, y1, x2, y2], color=(192, 192, 192))

        # save image
        if self.save_flag:
            image_name = os.path.basename(image_url)
            cv2.imwrite(os.path.join(self.save_path, image_name), image)

        return image


@OBJECT_REGISTRY.register
class VizDenseBoxDetAnno(object):
    def __init__(
        self,
        viz_class_id,
        class_name,
        save_flag=False,
        save_path=None,
        lt_point_id=0,
        rb_point_id=2,
    ):  # noqa
        self.save_flag = save_flag
        self.save_path = save_path
        self.viz_class_id = viz_class_id
        self.class_name = class_name
        self.lt_point_id = lt_point_id
        self.rb_point_id = rb_point_id
        self._mkdir()

    def _mkdir(self):
        if self.save_flag and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def __call__(self, img, anno, **kwargs):
        image_name = anno["img_url"].split("/")[-1]
        plot_densebox_image_record_bbox(
            img,
            anno,
            lt_point_id=self.lt_point_id,
            rb_point_id=self.rb_point_id,
            viz_class_id=self.viz_class_id,
            class_names=self.class_name,
            viz_normal=True,
            viz_hard=True,
            viz_ignore=True,
        )
        # save image
        if self.save_flag:
            plt.savefig(os.path.join(self.save_path, image_name))


@OBJECT_REGISTRY.register
class VizRoiDenseBoxDetAnno(object):
    """
    Visualize roi densebox detection annotation.

    Args:
        viz_class_id:
            Visualized class ids.
        class_name:
            Class names.
        save_flag:
            Whether save rendered image, by default False
        save_path:
            Where to save image, by default None
        lt_point_id:, optional
            Point id of left top, by default 0
        rb_point_id:, optional
            Point id of right bottom, by default 2
        roi_lt_point_id:, optional
            Roi point id of left top, by default 10
        roi_rb_point_id:, optional
            Roi point id of right bottom, by default 12
    """

    def __init__(
        self,
        viz_class_id: Union[List[int], Tuple[int]],
        class_name: Union[List[str], Tuple[str]],
        save_flag: bool = False,
        save_path: str = None,
        lt_point_id: int = 0,
        rb_point_id: int = 2,
        roi_lt_point_id: int = 10,
        roi_rb_point_id: int = 12,
    ):  # noqa
        self.save_flag = save_flag
        self.save_path = save_path
        self.viz_class_id = viz_class_id
        self.class_name = class_name
        self.lt_point_id = lt_point_id
        self.rb_point_id = rb_point_id
        self.roi_lt_point_id = roi_lt_point_id
        self.roi_rb_point_id = roi_rb_point_id
        self._mkdir()

    def _mkdir(self):
        if self.save_flag and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def __call__(self, img, anno, **kwargs):
        image_name = anno["img_url"].split("/")[-1]
        ax = plot_densebox_image_record_bbox(
            img,
            anno,
            lt_point_id=self.lt_point_id,
            rb_point_id=self.rb_point_id,
            viz_class_id=self.viz_class_id,
            class_names=self.class_name,
            viz_normal=True,
            viz_hard=True,
            viz_ignore=True,
        )
        ax = plot_densebox_image_record_bbox(
            img,
            anno,
            lt_point_id=self.roi_lt_point_id,
            rb_point_id=self.roi_rb_point_id,
            viz_class_id=self.viz_class_id,
            class_names=self.class_name,
            viz_normal=True,
            viz_hard=True,
            viz_ignore=True,
            ax=ax,
        )
        # save image
        if self.save_flag:
            plt.savefig(os.path.join(self.save_path, image_name))


def plot_densebox_image_record_bbox(
    img: np.ndarray,
    image_record: "ImageRecord",  # noqa
    lt_point_id: int,
    rb_point_id: int,
    viz_class_id: Union[List[int], Tuple[int]] = None,
    class_names: Union[List[int], Tuple[int]] = None,
    viz_normal: bool = True,
    colors: Dict = None,
    viz_hard: bool = False,
    hard_color: Union[List[int], Tuple[int]] = (0, 255, 255),
    viz_ignore: bool = False,
    ignore_color: Union[List[int], Tuple[int]] = (255, 255, 255),
    scale_x: int = 1.0,
    scale_y: int = 1.0,
    ax: Union[np.ndarray, "mxnet.nd.NDArray"] = None,  # noqa
):
    """Plot bbox for densebox ImageRecord.

    Args:
        img:
            Image with shape (H, W, C)
        image_record:
            densebox annotation.
        lt_point_id:
            Left top point index.
        rb_point_id:
            Right bottom point index.
        viz_class_id:
            Visualized class id.

            .. note::

                Start from 1.
        class_names:
            Class names.
        viz_normal:
            Whether to visualize normal instance.
        colors: dict of list/tuple of int
            Colors for normal instance. Looks like

            .. code-block:: none

                {
                    0: (0, 0, 255),
                    1: (0, 255, 0),
                    ...
                }
        viz_hard:
            Whether to visualize hard instance.
        hard_color:
            Colors for hard instance.
        viz_ignore:
            Whether to visualize ignore regions.
        ignore_color: list/tuple of int
            Colors for ignore regions.
        scale_x:
            Rescale factor in width.
        scale_y:
            Rescale factor in height.
        ax:
            You can reuse previous axes if provided.

    Returns:
        The ploted axes.
    """

    viz_class_id = _as_list(viz_class_id) if viz_class_id is not None else None
    if viz_class_id:
        for i in viz_class_id:
            assert i > 0, "class id should begin from 1"
    class_names = _as_list(class_names) if class_names is not None else None
    if isinstance(viz_class_id, (list, tuple)) and isinstance(
        class_names, (list, tuple)
    ):
        assert len(viz_class_id) == len(class_names), "%d vs. %d" % (
            len(viz_class_id),
            len(class_names),
        )

    if isinstance(image_record, str):
        image_record = json.loads(image_record)
    elif isinstance(image_record, dict):
        image_record = image_record

    normal_bboxes = []
    normal_class_ids = []
    hard_bboxes = []
    ignore_bboxes = []
    for _, instance in enumerate(image_record["instances"]):
        class_id = instance["class_id"][0]
        if viz_class_id is not None and class_id not in viz_class_id:
            continue
        is_hard = instance["is_hard"][0] == 1
        if not is_hard and not viz_normal:
            continue
        if is_hard and not viz_hard:
            continue
        # get point id
        if is_hard:
            hard_bboxes.append(
                [
                    instance["points_data"][lt_point_id][0],
                    instance["points_data"][lt_point_id][1],
                    instance["points_data"][rb_point_id][0],
                    instance["points_data"][rb_point_id][1],
                ]
            )
        else:
            normal_bboxes.append(
                [
                    instance["points_data"][lt_point_id][0],
                    instance["points_data"][lt_point_id][1],
                    instance["points_data"][rb_point_id][0],
                    instance["points_data"][rb_point_id][1],
                ]
            )
            normal_class_ids.append(class_id - 1)  # make class id begin from 0
    if viz_ignore:
        for ignore_region in image_record["ignore_regions"]:
            if "left_top" in ignore_region:
                ignore_bboxes.append(
                    [
                        ignore_region["left_top"][0],
                        ignore_region["left_top"][1],
                        ignore_region["right_bottom"][0],
                        ignore_region["right_bottom"][1],
                    ]
                )
            else:
                left_top = ignore_region["contour"][0]
                right_bottom = ignore_region["contour"][1]
                ignore_bboxes.append(
                    [
                        left_top[0],
                        left_top[1],
                        right_bottom[0],
                        right_bottom[1],
                    ]
                )
    normal_bboxes = np.array(normal_bboxes)
    normal_class_ids = np.array(normal_class_ids)
    hard_bboxes = np.array(hard_bboxes)
    ignore_bboxes = np.array(ignore_bboxes)

    # rescale image and bbox
    if scale_x != 1.0 or scale_y != 1.0:
        img = cv2.resize(img, (0, 0), fx=scale_x, fy=scale_y)
        normal_bboxes = _rescale_bbox(normal_bboxes, scale_x, scale_y)
        hard_bboxes = _rescale_bbox(hard_bboxes, scale_x, scale_y)
        ignore_bboxes = _rescale_bbox(ignore_bboxes, scale_x, scale_y)

    if viz_normal:
        colors = _normalize_color(colors)
        ax = gcv_plot_bbox(
            img,
            normal_bboxes,
            labels=normal_class_ids,
            class_names=class_names,
            colors=colors,
            ax=ax,
        )
    if viz_hard:
        hard_color = _normalize_color(hard_color)
        ax = gcv_plot_bbox(
            img,
            hard_bboxes,
            colors={0: hard_color},
            labels=np.zeros((hard_bboxes.shape[0])),
            ax=ax,
            class_names=["hard"],
        )
    if viz_ignore:
        ignore_color = _normalize_color(ignore_color)
        ax = gcv_plot_bbox(
            img,
            ignore_bboxes,
            colors={0: ignore_color},
            labels=np.zeros((ignore_bboxes.shape[0])),
            ax=ax,
            class_names=["ignore"],
        )
    return ax


def _rescale_bbox(bbox, scale_x, scale_y):
    """Rescale bbox."""
    if bbox.size > 0:
        bbox[:, 0] *= scale_x
        bbox[:, 1] *= scale_y
        bbox[:, 2] *= scale_x
        bbox[:, 3] *= scale_y
    return bbox


def _normalize_color(colors):
    """Normalize rgb color to within [0.0, 1.0]."""
    if colors is None:
        return colors
    if isinstance(colors, dict):
        for key in colors:
            colors[key] = _normalize_color(colors[key])
    elif isinstance(colors, (list, tuple)):
        colors = [i / 255.0 for i in colors]
    else:
        raise NotImplementedError(
            "unsupported type %s for color" % str(type(colors))
        )
    return colors


def gcv_plot_bbox(
    img: Union[np.ndarray, "mxnet.nd.NDArray"],  # noqa
    bboxes: Union[np.ndarray, "mxnet.nd.NDArray"],  # noqa
    scores: Union[np.ndarray, "mxnet.nd.NDArray"] = None,  # noqa
    labels: Union[np.ndarray, "mxnet.nd.NDArray"] = None,  # noqa
    thresh: float = 0.5,
    class_names: List[str] = None,
    colors: Dict = None,
    ax: Union[np.ndarray, "mxnet.nd.NDArray"] = None,  # noqa
    reverse_rgb: bool = False,
    absolute_coordinates: bool = True,
):
    """Visualize bounding boxes.

    Args:
        img:
            Image with shape `H, W, 3`.
        bboxes:
            Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
        scores:
            Confidence scores of the provided `bboxes` with shape `N`.
        labels:
            Class labels of the provided `bboxes` with shape `N`.
        thresh:
            Display threshold if `scores` is provided.
            Scores with less than `thresh` will be ignored in display,
            this is visually more elegant if you have
            a large number of bounding boxes with very small scores.
        class_names:
            Description of parameter `class_names`.
        colors:
            You can provide desired colors as
            {0: (255, 0, 0), 1:(0, 255, 0), ...},
            otherwise random colors will be substituted.
        ax:
            You can reuse previous axes if provided.
        reverse_rgb:
            Reverse RGB<->BGR orders if `True`.
        absolute_coordinates:
            If `True`, absolute coordinates will be considered,
            otherwise coordinates are interpreted as in range(0, 1).

    Returns:
        matplotlib axes
            The ploted axes.

    """
    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError(
            "The length of labels and bboxes mismatch, {} vs {}".format(
                len(labels), len(bboxes)
            )
        )
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError(
            "The length of scores and bboxes mismatch, {} vs {}".format(
                len(scores), len(bboxes)
            )
        )

    ax = plot_image(img, ax=ax, reverse_rgb=reverse_rgb)

    if len(bboxes) < 1:
        return ax

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height

    # use random colors if None is provided
    if colors is None:
        colors = {}
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        if cls_id not in colors:
            if class_names is not None:
                colors[cls_id] = plt.get_cmap("hsv")(cls_id / len(class_names))
            else:
                colors[cls_id] = (
                    random.random(),
                    random.random(),
                    random.random(),
                )
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=colors[cls_id],
            linewidth=3.5,
        )
        ax.add_patch(rect)
        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ""
        score = "{:.3f}".format(scores.flat[i]) if scores is not None else ""
        if class_name or score:
            ax.text(
                xmin,
                ymin - 2,
                "{:s} {:s}".format(class_name, score),
                bbox={
                    "facecolor": colors[cls_id],
                    "alpha": 0.5,
                },
                fontsize=12,
                color="white",
            )
    return ax


def plot_image(
    img,
    ax: Union[np.ndarray, "mxnet.nd.NDArray"] = None,  # noqa
    reverse_rgb: bool = False,
):
    """Visualize image.

    Args:
        img:
            Image with shape `H, W, 3`.
        ax:
            You can reuse previous axes if provided.
        reverse_rgb:
            Reverse RGB<->BGR orders if `True`.

    Returns:
        matplotlib axes
            The ploted axes.
    Examples:
        from matplotlib import pyplot as plt
        ax = plot_image(img)
        plt.show()
    """
    if ax is None:
        # create new axes
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    img = img.copy()
    if reverse_rgb:
        img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
    ax.imshow(img.astype(np.uint8))
    return ax


@OBJECT_REGISTRY.register
class VizKpsDetAnno(object):
    """
    Visualize key point detection annotation.

    Args:
        num_kps:
            The number of key points
        task_name:
            What kind of key point task, possible values are
            {'cyclist_2_kps', 'vehicle_2_kps', 'vehicle_12_kps',
            'vehicle_8_kps','vehicle_4_kps'}
        save_flag :
            Whether save rendered image, by default False
        save_path:, optional
            Where to save image, by default None
        use_cam_params :
            Whether visualize with camera parameter.
    """

    def __init__(self, num_kps, task_name, save_flag=False, save_path=None):
        self.num_kps = num_kps
        self.task_name = task_name
        self.save_flag = save_flag
        self.save_path = save_path

        self._mkdir()

    def _mkdir(self):
        if self.save_flag and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def __call__(self, img, anno, **kwargs):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gt_contents = anno
        image_abs_path = gt_contents["image"]
        image_name = image_abs_path.strip().split("/")[-1]
        cv2.imwrite(os.path.join(self.save_path, "orig" + image_name), img)
        # check object number
        num_obj = len(gt_contents["boxes"])
        if num_obj <= 0:
            return

        # traverse each object instance
        for i in range(int(num_obj)):
            gt_obj_cur = gt_contents["boxes"][i]
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = map(float, gt_obj_cur)
            keypoints = gt_contents["keypoints"]

            wheel_back = keypoints[i][0:2]
            wheel_front = keypoints[i][3:5]

            img = visualize_onraw_img_2_kps(
                image=img,
                bbox=[bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax],
                lms_wheel_back=wheel_back,
                lms_wheel_front=wheel_front,
                ind=i,
            )

        # save image
        if self.save_flag:
            cv2.imwrite(os.path.join(self.save_path, image_name), img)
        return img


def visualize_onraw_img_2_kps(
    image, bbox, lms_wheel_back, lms_wheel_front, ind, norm_error=0.0
):
    """Draw 2 anno kps on the raw image."""
    thick = 2
    xmin, ymin, xmax, ymax = bbox

    # draw wheel line
    cv2.line(
        image,
        (int(lms_wheel_back[0]), int(lms_wheel_back[1])),
        (int(lms_wheel_front[0]), int(lms_wheel_front[1])),
        (0, 255, 0),
        thick + 1,
    )

    # red
    cv2.circle(
        image, (int(lms_wheel_back[0]), int(lms_wheel_back[1])), 3, (0, 0, 255)
    )

    # blue
    cv2.circle(
        image,
        (int(lms_wheel_front[0]), int(lms_wheel_front[1])),
        3,
        (255, 0, 0),
    )

    # bbox
    cv2.rectangle(
        image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 140, 255), 2
    )

    # draw annotation
    cv2.putText(
        image,
        "id:" + str(ind),
        (int(xmin), int(ymin) - 20),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 140, 255),
    )
    cv2.putText(
        image,
        "lme:" + "%.2f" % norm_error,
        (int(xmin), int(ymin)),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 140, 255),
    )
    cv2.putText(
        image,
        str(0),
        (
            max(int(lms_wheel_back[0]) - 20, 0),
            max(int(lms_wheel_back[1]) - 20, 0),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3,
    )
    cv2.putText(
        image,
        str(1),
        (
            max(int(lms_wheel_front[0]) - 20, 0),
            max(int(lms_wheel_front[1]) - 20, 0),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3,
    )

    return image


def draw_points(
    image,
    points,
    color: Optional[Tuple[int, int, int]] = (0, 0, 255),
    radius: Optional[int] = 2,
    thickness: Optional[int] = 2,
):
    if len(points) == 0:
        return image

    points = np.asarray(points)
    point_dimension = points.shape[-1]
    points = points.reshape(-1, point_dimension)
    for point in points:
        x, y = map(int, point[0:2])
        image = cv2.circle(
            image,
            (x, y),
            color=color,
            radius=radius,
            thickness=thickness,
        )

    return image


def draw_lines(
    image,
    lines,
    color: Optional[Tuple[int, int, int]] = (0, 255, 255),
    thickness: Optional[int] = 2,
):
    if len(lines) == 0:
        return image

    lines = np.asarray(lines)
    point_dimension = lines.shape[-1]
    lines = lines.reshape(-1, 2, point_dimension)
    for start, end in lines:
        sx, sy = map(int, start[0:2])
        ex, ey = map(int, end[0:2])
        image = cv2.line(
            image, (sx, sy), (ex, ey), color=color, thickness=thickness
        )

    return image


@OBJECT_REGISTRY.register
class VizFlankCornersMatrixGluon(object):
    """
    Visualize key point detection annotation.

    Args:
        save_flag :
            Whether save rendered image, by default False
        save_path:, optional
            Where to save image, by default None
    """

    def __init__(
        self,
        save_flag: Optional[bool] = False,
        save_path: Optional[str] = None,
    ):
        self.save_flag = save_flag
        self.save_path = save_path

        self._mkdir()

    def _mkdir(self):
        if self.save_flag and not osp.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def __call__(self, img, anno, **kwargs):
        image_name = osp.basename(anno["image"])
        bboxes = anno.get("bboxes", [])
        keypoints = anno.get("keypoints", [])
        flank_classes = anno.get("flank_classes", [])
        interpolation_points = anno.get("interpolation_points", [])
        indices = anno.get("interpolation_indices", [])
        assert len(bboxes) == len(keypoints) == len(flank_classes)

        for bbox, kps, cls, inps in zip(
            bboxes, keypoints, flank_classes, interpolation_points
        ):
            img = self.render(img, bbox, kps, cls, inps, indices)

        # save image
        if self.save_flag:
            cv2.imwrite(os.path.join(self.save_path, image_name), img)

        return img

    def render(self, img, bbox, kps, cls, inps, indices):
        _red = (0, 0, 255)
        _green = (0, 255, 0)
        _sky_blue = (235, 206, 135)
        _yellow = (0, 255, 255)

        if cls == 1:
            # positive
            img = _draw_bbox(
                img,
                bbox,
                color=_sky_blue,
                thickness=2,
            )
            img = draw_points(
                img,
                [kps],
                color=_red,
                thickness=8,
            )
            img = draw_lines(
                img,
                [kps],
                color=_red,
                thickness=2,
            )

            for idx in range(len(indices)):
                for points in inps[idx]:
                    img = draw_points(
                        img,
                        points,
                        color=_green,
                        thickness=4,
                    )
                lines = [
                    [inps[idx][i], inps[idx][i + 1]]
                    for i in range(len(inps[idx]) - 1)
                ]
                img = draw_lines(img, lines, color=_green, thickness=2)
        else:
            # negative: only bboxes exist
            img = _draw_bbox(
                img,
                bbox,
                color=_yellow,
                thickness=2,
            )

        return img
