import copy
import logging
import math
import pprint
import warnings
from itertools import zip_longest
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from easydict import EasyDict
from hatbc.utils import _as_list

from hat.core.box_utils import bbox_overlaps
from hat.data.datasets.densebox_dataset import (
    DenseboxDataset,
    build_dataset,
    get_idx_path,
)
from hat.data.datasets.legacy_densebox import HAT_LEGACYDENSEBOX_AVAILABLE
from hat.data.datasets.utils import img_to_rgb
from hat.registry import OBJECT_REGISTRY
from .affine import _pad_array, get_interp_method
from .bbox import clip_bbox, remap_bbox_label_by_area
from .detection import ToTensor

try:
    import albumentations
except ImportError:
    albumentations = None

import cv2

logger = logging.getLogger(__name__)

__all__ = [
    "TrafficSignDetectionLableTs",
    "DenseboxDataset2PETrafficSign",
    "DecodeDenseBoxDatasetToDetFormatWithImageInfo",
    "TrafficSignROICropResizeTransform",
    "TrafficSignImageAugmentation",
    "ClassIdRemap",
    "ClassificationLabelTs",
]


@OBJECT_REGISTRY.register
class DenseboxDataset2PETrafficSign(DenseboxDataset):
    """Dataset for densebox record data in auto, extend for 2pe.

    Args:
        data_path: Path of data relative to buket path.
        anno_path: Path of annotation.
        transforms: List of transform.
        to_rgb: Convert bgr(cv2 imread) to rgb.
        task_type: Consist of 'detection', 'segmentation'
        class_id: the rec's class id, 1base
        category: the used category, 0base
        rec_idx_file_path: index file related to data_path. Used only when
            there is already index file somewhere.
        disable_default_densebox_log: Disable default print output from
            `LegacyDenseBoxImageRecordDataset`. Default is True.
        ignore_hard: Ignore hard instances if `hard` tag in annotation.
            Default is False.
        use_ignore: Whether to use ignore regions in annotation.
            Default is False.
        with_img_buf: Whether return img buf.
            Default is False.
        return_orig_img: Whether to return an extra original img which can
            be used for visualization. Default is False.
        remove_det_duplicate: Whether to filter out duplicate gt bboxes in one
            image. Sometimes there are duplicate gt bboxes in one image to
            emphasize certain training examples, but which may cause inaccurate
            performance evaluation. Default is False.
        version: The packaging format, the default is densebox v1.
    """

    def __init__(
        self,
        data_path: str,
        anno_path: str,
        roi_lt_id: int = 10,
        roi_rb_id: int = 12,
        gt_lt_id: int = 0,
        gt_rb_id: int = 2,
        transforms: Optional[List] = None,
        to_rgb: Optional[bool] = False,
        task_type: Optional[str] = "detection",
        task_name: Optional[str] = None,
        class_id: Union[int, List] = -1,
        category: Union[int, Dict] = -1,
        rec_idx_file_path: Optional[str] = None,
        disable_default_densebox_log: Optional[bool] = True,
        ignore_hard: Optional[bool] = False,
        use_ignore: Optional[bool] = False,
        maximum_instances_per_image: Optional[int] = 100,
        with_img_buf: Optional[bool] = False,
        read_only: Optional[bool] = False,
        return_orig_img: Optional[bool] = False,
        remove_det_duplicate: Optional[bool] = False,
        rand_sampling_bbox: Optional[bool] = True,
        version: str = "v1",
        image_processing_backend: str = "opencv",
    ):
        assert (
            HAT_LEGACYDENSEBOX_AVAILABLE
        ), "horizon_plugin_pytorch >= 1.0.0 is required."

        self.data_path = data_path
        self.anno_path = anno_path
        self.transforms = transforms
        self.to_rgb = to_rgb
        self.task_type = task_type
        self.task_name = task_name
        self.class_id = class_id
        self.category = category
        self.maximum_instances_per_image = maximum_instances_per_image
        self.rec_idx_file_path = rec_idx_file_path
        self.disable_default_densebox_log = disable_default_densebox_log
        self.ignore_hard = ignore_hard
        self.use_ignore = use_ignore
        self.with_img_buf = with_img_buf
        self.return_orig_img = return_orig_img
        self.remove_det_duplicate = remove_det_duplicate
        self.rand_sampling_bbox = rand_sampling_bbox
        self.roi_lt_id = roi_lt_id
        self.roi_rb_id = roi_rb_id
        self.gt_lt_id = gt_lt_id
        self.gt_rb_id = gt_rb_id
        self.version = version

        if self.rec_idx_file_path is None:
            self.rec_idx_file_path = get_idx_path(self.data_path)

        if self.task_type == "detection" or self.task_type == "classification":
            self.kwargs = {}
        else:
            raise Exception(
                "error task_type, your task_type[{}],"
                " we need classification or detection".format(self.task_type)
            )

        assert image_processing_backend in [
            "opencv",
            "imageio",
            "PIL",
            "turbojpeg",
        ]
        self.image_processing_backend = image_processing_backend
        self.kwargs["with_img_buf"] = self.with_img_buf
        self.kwargs["image_processing_backend"] = image_processing_backend
        self.dataset = build_dataset(
            self.data_path,
            self.anno_path,
            self.rec_idx_file_path,
            self.disable_default_densebox_log,
            read_only,
            **self.kwargs,
        )
        logging.info(f"dataset path: {self.data_path}")
        logging.info(f"dataset length: {len(self.dataset)}")

    def __getitem__(self, index: int) -> List:
        if self.with_img_buf:
            image, img_buf, anno = self.dataset[index]
        else:
            image, anno = self.dataset[index]
        data = {}

        color_space = "bgr"
        if self.image_processing_backend in ["imageio", "PIL", "turbojpeg"]:
            color_space = "rgb"
            self.to_rgb = False

        if self.to_rgb:
            image = img_to_rgb(
                image
            )  # change channel order faster than cv2.cvtColor
            color_space = "rgb"

        anno = anno.to_dict()
        data["img_name"] = anno["img_url"].split("/")[-1]
        data["img_height"] = anno["img_h"]
        data["img_width"] = anno["img_w"]
        data["img_id"] = np.expand_dims(anno["idx"], 0)
        data["img"] = image
        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image.shape

        if self.with_img_buf:
            data["img_buf"] = img_buf
        if self.return_orig_img:
            data["orig_img"] = image.copy()
        results = []
        dummy_data = data.copy()

        if self.rand_sampling_bbox:
            bbox_idx_list = np.random.choice(
                len(anno["instances"]),
                min(len(anno["instances"]), self.maximum_instances_per_image),
                replace=False,
            )
        else:
            bbox_idx_list = np.arange(len(anno["instances"]))

        for instance_idx in bbox_idx_list:
            data = dummy_data.copy()
            if self.task_type == "detection":
                gt_bboxes = []
                gt_classes = []
                points_data = anno["instances"][instance_idx]["points_data"]
                x1_crop, y1_crop = points_data[self.roi_lt_id]
                x2_crop, y2_crop = points_data[self.roi_rb_id]
                crop_roi = [x1_crop, y1_crop, x2_crop, y2_crop]

                for ins in anno["instances"]:
                    is_hard = ins["is_hard"][0]
                    points_data = ins["points_data"]
                    class_id = int(ins["class_id"][0])
                    bbox = []
                    bbox.extend(points_data[self.gt_lt_id])
                    bbox.extend(points_data[self.gt_rb_id])
                    gt_bboxes.append(bbox)
                    if class_id in self.class_id:
                        if is_hard and self.ignore_hard:
                            gt_classes.append(-1)
                        else:
                            gt_classes.append(self.category[class_id])
                    else:
                        gt_classes.append(-1)

                data["crop_roi"] = np.array(crop_roi, dtype=np.float32)
                data["gt_bboxes"] = np.array(gt_bboxes, dtype=np.float32)
                data["gt_classes"] = np.array(gt_classes, dtype=np.int64)

                ig_bboxes = []
                if self.use_ignore:
                    for ig_region in anno["ignore_regions"]:
                        left_top = ig_region["contour"][0]
                        right_bottom = ig_region["contour"][1]
                        class_id = int(ig_region["class_id"][0])
                        if not (
                            left_top == [0, 0]
                            and right_bottom == [anno["img_w"], anno["img_h"]]
                        ):  # must be first n in anno["ignore_regions"]
                            ig_bbox = left_top + right_bottom + [class_id]
                            ig_bboxes.append(ig_bbox)
                        else:
                            break
                data["ig_bboxes"] = (
                    np.array(ig_bboxes, dtype=np.float32)
                    if len(ig_bboxes) > 0
                    else np.zeros((0, 5), dtype=np.float32)
                )
                # Remove duplicate gt boxes in one image
                if self.remove_det_duplicate:
                    data["gt_bboxes"], unique_indices = np.unique(
                        data["gt_bboxes"], return_index=True, axis=0
                    )
                    data["gt_classes"] = data["gt_classes"][unique_indices]

            elif self.task_type == "classification":
                gt_classes = []
                ins = anno["instances"][instance_idx]
                points_data = ins["points_data"]
                x1_crop, y1_crop = points_data[self.gt_lt_id]
                x2_crop, y2_crop = points_data[self.gt_rb_id]
                crop_roi = [x1_crop, y1_crop, x2_crop, y2_crop]

                is_hard = ins["is_hard"][0]
                if self.version == "v1":
                    class_id = int(ins["class_id"][0])
                elif self.version == "v2":
                    ins_attr = ins.get("attribute", [])
                    class_id = int(ins_attr[-1]) + 1
                else:
                    raise NotImplementedError
                if class_id in self.class_id:
                    if is_hard and self.ignore_hard:
                        gt_classes.append(-1)
                    else:
                        gt_classes.append(class_id - 1)
                else:
                    gt_classes.append(-1)
                data["crop_roi"] = np.array(crop_roi, dtype=np.float32)
                data["gt_classes"] = np.array(gt_classes, dtype=np.int64)

            if self.transforms is not None:
                results.append(self.transforms(data))
        return results


@OBJECT_REGISTRY.register
class TrafficSignDetectionLableTs(object):
    """Label transform for traffic sign assistant model.

    Args:
        pad_det_data: Whether pad gt bboxes.
        max_gt_boxes_num: Max gt bboxes number.
        max_ig_regions_num: Max ignore regions number.
        regroup_gt_bboxes: Whther regroup gt bboxes
        gt_boxes_key: Gt bboxes key, default gt_bboxes
        ig_regions_key: Ignore regions key, default ig_bboxes.
        gt_classes_key: Gt classes key.
        keys_mapping: key dict to rename,
            in dict(old key:new key) format.
    """

    def __init__(
        self,
        pad_det_data: bool = True,
        max_gt_boxes_num: int = 100,
        max_ig_regions_num: int = 100,
        regroup_gt_bboxes: bool = True,
        gt_boxes_key: str = "gt_bboxes",
        ig_regions_key: str = "ig_bboxes",
        gt_classes_key: str = "gt_classes",
        keys_mapping: Optional[Dict] = None,
    ):
        self.pad_det_data = pad_det_data
        self.max_gt_boxes_num = max_gt_boxes_num
        self.max_ig_regions_num = max_ig_regions_num
        self.regroup_gt_bboxes = regroup_gt_bboxes
        self.gt_boxes_key = gt_boxes_key
        self.ig_regions_key = ig_regions_key
        self.gt_classes_key = gt_classes_key
        self.keys_mapping = keys_mapping

    def __call__(self, data):

        if self.regroup_gt_bboxes:
            assert len(data[self.gt_boxes_key]) == len(
                data[self.gt_classes_key]
            ), f"{len(data[self.gt_boxes_key])} vs {len(data[self.gt_classes_key])}"  # noqa

            if len(data[self.gt_boxes_key]) < 1:
                data[self.gt_boxes_key] = np.array([[0, 0, 0, 0, 0]])
            else:
                gt_classes = data[self.gt_classes_key][:, np.newaxis]
                data[self.gt_boxes_key] = np.concatenate(
                    (data[self.gt_boxes_key], gt_classes),
                    axis=1,
                )

        if self.pad_det_data:
            # pading gt boxes
            pad_shape = list(data[self.gt_boxes_key].shape)
            pad_shape[0] = self.max_gt_boxes_num
            data["gt_boxes_num"] = (
                np.array(data[self.gt_boxes_key].shape[0])
                .reshape((1,))
                .astype(np.float32)
            )
            data[self.gt_boxes_key] = _pad_array(
                data[self.gt_boxes_key], pad_shape, self.gt_boxes_key
            ).astype(np.float32)

            # padding ignore regions
            ig_regions = data.get(self.ig_regions_key, None)
            if len(ig_regions) < 1:
                data["ig_regions"] = np.array([[0, 0, 0, 0, 0]])
                data["ig_regions_num"] = 0

            pad_shape = list(data[self.ig_regions_key].shape)
            pad_shape[0] = self.max_ig_regions_num
            data["ig_regions_num"] = (
                np.array(data[self.ig_regions_key].shape[0])
                .reshape((1,))
                .astype(np.float32)
            )
            data[self.ig_regions_key] = _pad_array(
                data[self.ig_regions_key], pad_shape, self.ig_regions_key
            ).astype(np.float32)

        if self.keys_mapping is not None:
            for lkey, rkey in self.keys_mapping.items():
                if lkey in data:
                    data[rkey] = data.pop(lkey)

        data["im_hw"] = np.array(data["img"].shape[:2])

        return data


@OBJECT_REGISTRY.register
class DecodeDenseBoxDatasetToDetFormatWithImageInfo(object):
    """A transformer that transform the densebox record data to be det format.

    Args:
        selected_class_ids: Selected class ids, classes that are not in this list will be filter out.
        lt_point_id: Point id of left top.
        rb_point_id: Point id of right bottom.
        min_edge_size: Filter out bbox whose edge is less than this value, by default 0.1.
        norm_point_id1: The first point ID of the two points that determine the scale of the object.
        norm_point_id2: The second point ID of the two points that determine the scale of the object.
        show_image_info: Whether show image info.
        class_id_key: The key of class id.
    """  # noqa

    def __init__(
        self,
        selected_class_ids: Union[List[int], Tuple[int]],
        lt_point_id: int,
        rb_point_id: int,
        min_edge_size: float = 0.1,
        norm_point_id1: int = None,
        norm_point_id2: int = None,
        show_image_info: bool = False,
        class_id_key: str = "class_id",
    ):
        if selected_class_ids is not None:
            selected_class_ids = _as_list(selected_class_ids)
            # class id should begin from 1
            self.classidmap = dict(
                zip(selected_class_ids, range(1, len(selected_class_ids) + 1))
            )
        else:
            self.classidmap = None
        self.lt_point_id = lt_point_id
        self.rb_point_id = rb_point_id
        self.min_edge_size = min_edge_size
        self.norm_point_id1 = norm_point_id1
        self.norm_point_id2 = norm_point_id2
        self.show_image_info = show_image_info
        self.class_id_key = class_id_key
        assert self.class_id_key in ["class_id", "attribute"]

    def _get_bbox_wh(self, bbox):
        assert len(bbox) >= 4
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])

    def __call__(self, data):

        img, anno = data["img"], data["anno"]

        def _is_selected_class_id(class_id):
            if self.classidmap is None:
                return True
            return class_id in self.classidmap.keys()

        def _remap_class_id(class_id):
            if self.classidmap is None:
                return class_id
            return self.classidmap[class_id]

        gt_boxes = []
        norm_points = []
        for inst_i in anno.instances:
            if isinstance(inst_i, dict):
                inst_i = EasyDict(inst_i)
            lt = inst_i.points_data[self.lt_point_id]
            rb = inst_i.points_data[self.rb_point_id]
            class_id = getattr(inst_i, self.class_id_key)[0]
            if _is_selected_class_id(class_id):
                class_id = _remap_class_id(class_id)
            else:
                continue
            hard_flag = inst_i.is_hard[0]
            if hard_flag in [1, True, "1", "True"]:
                class_id *= -1
            gt_boxes_i = [lt[0], lt[1], rb[0], rb[1], class_id]
            gt_boxes_i_wh = self._get_bbox_wh(gt_boxes_i)
            if (
                gt_boxes_i_wh[0] < self.min_edge_size
                or gt_boxes_i_wh[1] < self.min_edge_size
            ):
                msg = (
                    "Ignore gt_boxes %s since its min edge size is invalid..."
                    % gt_boxes_i
                )  # noqa
                warnings.warn(msg)
                continue
            if self.norm_point_id1 and self.norm_point_id2:
                norm_point1 = inst_i.points_data[self.norm_point_id1]
                norm_point2 = inst_i.points_data[self.norm_point_id2]
                norm_points.append((norm_point1, norm_point2))
            gt_boxes.append(gt_boxes_i)

        gt_boxes = (
            np.array(gt_boxes, dtype=np.float32)
            if len(gt_boxes) > 0
            else np.zeros((0, 5), dtype=np.float32)
        )

        ig_regions = []
        for inst_i in anno.ignore_regions:
            if isinstance(inst_i, dict):
                inst_i = EasyDict(inst_i)
            lt = inst_i.contour[0]
            rb = inst_i.contour[1]
            class_id = getattr(inst_i, self.class_id_key)[0]
            if _is_selected_class_id(class_id):
                class_id = _remap_class_id(class_id)
            else:
                continue
            ig_regions_i = [lt[0], lt[1], rb[0], rb[1], class_id]
            ig_regions_i_wh = self._get_bbox_wh(ig_regions_i)
            if ig_regions_i_wh[0] <= 0 or ig_regions_i_wh[1] <= 0:
                msg = "Ignore invalid ig_regions %s" % ig_regions_i
                warnings.warn(msg)
                continue
            ig_regions.append(ig_regions_i)

        ig_regions = (
            np.array(ig_regions, dtype=np.float32)
            if len(ig_regions) > 0
            else np.zeros((0, 5), dtype=np.float32)
        )

        image_info = anno.to_dict()

        results = {
            "img": img,
            "gt_boxes": gt_boxes,
            "ig_regions": ig_regions,
            "norm_points": norm_points,
            "color_space": data["color_space"],
        }

        if self.show_image_info:
            # imag_info = json.dumps(image_info)
            results["image_info"] = image_info

        return results


def _generate_random_roi(
    img_wh: Union[List[int], Tuple[int]],
    gt_boxes: List,
    min_side: int = 16,
    max_side: int = 128,
    max_try_number: int = 10,
    random_roi_class_id: int = -1,
):
    """Generate random roi.

    Args:
        img_wh: Image width and height.
        gt_boxes: List of ground truth boxes.
        min_side: The minimum side gap.
        max_side: The maximum side gap.
        max_try_number: Maximum number of attempts to find ROI.
        random_roi_class_id: The class id of random roi.
    """

    valid_bbox = [0, img_wh[1] - 30, 30, img_wh[1]]
    for _ in range(max_try_number):
        x = max(np.random.uniform(1, img_wh[0] - max_side - 1), 0)
        y = max(np.random.uniform(1, img_wh[1] - max_side - 1), 0)
        w = min(np.random.uniform(min_side, max_side), img_wh[0] - x)
        h = min(np.random.uniform(min_side, max_side), img_wh[1] - y)
        box = [x, y, x + w, y + h]

        if len(gt_boxes) == 0:
            valid_bbox = box
            break
        else:
            iou_matrix = bbox_overlaps(np.array([box]), gt_boxes[:, 0:4])
            if iou_matrix.sum(axis=1)[0] == 0:
                valid_bbox = box
                break
    valid_bbox.append(random_roi_class_id)
    valid_bbox = np.array(valid_bbox)
    valid_bbox[:2] = np.maximum(valid_bbox[:2], 0)
    valid_bbox[2:4] = np.minimum(valid_bbox[2:4], np.array(img_wh))
    return valid_bbox


@OBJECT_REGISTRY.register
class TrafficSignROICropResizeTransform(object):
    """A transformer that crops roi and resize.

    Args:
        target_wh: Target width and height.
        crop_method: Crop method.
        padding: Padding distance afer croping.
        max_jitter_ratio: Max jitter ratio when random croping roi.
        random_jitter_bilateral: Whether to perform bilateral random jitter.
        keep_aspect_ratio: Whether keep aspect ratio.
        inter_method: Interpolation methods used when resize.
        affine_transform: Whether to perform affine transformation.
        filter_valid_roi: Whether to filter valid roi.
        min_valid_area: Minimum valid area.
        min_edge_size: Minimum edge size.
        random_roi_class_id: Random roi class id.
        show_origin_image: Whether show origin image.
        show_origin_bbox: Whether show origin bbox.
        show_debug_info: Whether show debug info.
    """

    def __init__(
        self,
        target_wh: Tuple = (64, 64),
        crop_method: str = "CropAndPad",
        padding: Tuple = (0, 0, 0, 0),
        max_jitter_ratio: Tuple = (0.1, 0.1),
        random_jitter_bilateral: bool = True,
        keep_aspect_ratio: bool = False,
        inter_method: int = 10,
        affine_transform: bool = False,
        filter_valid_roi: bool = True,
        min_valid_area: int = 16,
        min_edge_size: int = 8,
        random_roi_class_id: int = -1,
        show_origin_image: bool = False,
        show_origin_bbox: bool = False,
        show_debug_info: bool = False,
    ):
        self.target_wh = target_wh
        self.keep_aspect_ratio = keep_aspect_ratio
        self.inter_method = inter_method
        self.affine_transform = affine_transform
        self.filter_valid_roi = filter_valid_roi
        self.min_valid_area = min_valid_area
        self.min_edge_size = min_edge_size
        self.random_roi_class_id = random_roi_class_id

        # for debug
        self.show_origin_image = show_origin_image
        self.show_origin_bbox = show_origin_bbox
        self.show_debug_info = show_debug_info

        assert crop_method in [
            "CropAndPad",
            "RandomCropNearBBox",
            "RandomCropNearBBoxV2",
        ]
        self.crop_method = crop_method
        self.padding = padding
        self.crop_func_factory = {
            "CropAndPad": self.crop_and_pad,
            "RandomCropNearBBox": self.random_crop_near_bbox,
            "RandomCropNearBBoxV2": self.random_crop_near_bbox_v2,
        }
        self.crop_func = self.crop_func_factory[crop_method]

        if padding[0] <= 1:
            # 如果小于等于1，则认为是百分比
            assert all([it <= 1 and it >= 0 for it in padding])
            padding_left = padding[0] * target_wh[0]
            padding_right = padding[2] * target_wh[0]
            padding_top = padding[1] * target_wh[1]
            padding_bottom = padding[3] * target_wh[1]
            self.padding = [
                padding_left,
                padding_top,
                padding_right,
                padding_bottom,
            ]
        else:
            # 如果大于1，则认为是像素宽度
            self.padding = [int(it) for it in padding]

        self.random_jitter_bilateral = random_jitter_bilateral
        if len(max_jitter_ratio) == 1:
            self.max_jitter_ratio = _as_list(max_jitter_ratio) * 4
        elif len(max_jitter_ratio) == 2:
            self.max_jitter_ratio = max_jitter_ratio * 2
        elif len(max_jitter_ratio) == 4:
            self.max_jitter_ratio = max_jitter_ratio
        else:
            raise ValueError(
                f"Length of 'max_jitter_ratio' must be one of [1, 2, 4],"
                f"but you supply {len(max_jitter_ratio)}"
            )

    def crop_and_pad(self, bboxes, img_wh):
        roi_bboxes = np.array(bboxes.copy())
        padding = np.array(self.padding)

        roi_bboxes[:, :2] = np.maximum(roi_bboxes[:, :2] - padding[:2], 0)
        roi_bboxes[:, 2:4] = np.minimum(
            roi_bboxes[:, 2:4] + padding[2:], np.array(img_wh)
        )

        return roi_bboxes, bboxes

    def random_crop_near_bbox(self, bboxes, img_wh):
        roi_bboxes = np.array(bboxes.copy())

        h_max_shift = np.round(
            (roi_bboxes[:, 3] - roi_bboxes[:, 1]) * self.max_jitter_ratio[1]
        )
        w_max_shift = np.round(
            (roi_bboxes[:, 2] - roi_bboxes[:, 0]) * self.max_jitter_ratio[0]
        )

        h_max_shift = np.clip(h_max_shift, 1, None)
        w_max_shift = np.clip(w_max_shift, 1, None)

        roi_bboxes[:, 0] -= np.random.randint(-w_max_shift, w_max_shift)
        roi_bboxes[:, 2] += np.random.randint(-w_max_shift, w_max_shift)

        roi_bboxes[:, 1] -= np.random.randint(-h_max_shift, h_max_shift)
        roi_bboxes[:, 3] += np.random.randint(-h_max_shift, h_max_shift)

        roi_bboxes[:, :2] = np.maximum(roi_bboxes[:, :2], 0)
        roi_bboxes[:, 2:4] = np.minimum(roi_bboxes[:, 2:4], np.array(img_wh))

        roi_bboxes_ws = roi_bboxes[:, 2:3] - roi_bboxes[:, 0:1]
        roi_bboxes_hs = roi_bboxes[:, 3:4] - roi_bboxes[:, 1:2]

        roi_bboxes[:, [0, 2]] = np.where(
            roi_bboxes_ws <= 0, bboxes[:, [0, 2]], roi_bboxes[:, [0, 2]]
        )
        roi_bboxes[:, [1, 3]] = np.where(
            roi_bboxes_hs <= 0, bboxes[:, [1, 3]], roi_bboxes[:, [1, 3]]
        )

        if self.show_debug_info and (
            (roi_bboxes_ws <= 0).astype("int").sum() > 0
            or (roi_bboxes_hs <= 0).astype("int").sum() > 0
        ):
            warnings.warn("error! width or height less than 0!")

        return roi_bboxes, bboxes

    def random_crop_near_bbox_v2(self, bboxes, img_wh):
        roi_bboxes = np.array(bboxes.copy())

        left_max_shift = (
            roi_bboxes[:, 2] - roi_bboxes[:, 0]
        ) * self.max_jitter_ratio[0]
        top_max_shift = (
            roi_bboxes[:, 3] - roi_bboxes[:, 1]
        ) * self.max_jitter_ratio[1]
        right_max_shift = (
            roi_bboxes[:, 2] - roi_bboxes[:, 0]
        ) * self.max_jitter_ratio[2]
        bottom_max_shift = (
            roi_bboxes[:, 3] - roi_bboxes[:, 1]
        ) * self.max_jitter_ratio[3]

        left_min_shift = np.minimum(0, left_max_shift)
        left_max_shift = np.maximum(0, left_max_shift)
        top_min_shift = np.minimum(0, top_max_shift)
        top_max_shift = np.maximum(0, top_max_shift)
        right_min_shift = np.minimum(0, right_max_shift)
        right_max_shift = np.maximum(0, right_max_shift)
        bottom_min_shift = np.minimum(0, bottom_max_shift)
        bottom_max_shift = np.maximum(0, bottom_max_shift)

        if self.random_jitter_bilateral:
            left_shift = np.random.uniform(-left_max_shift, left_max_shift)
            top_shift = np.random.uniform(-top_max_shift, top_max_shift)
            right_shift = np.random.uniform(-right_max_shift, right_max_shift)
            bottom_shift = np.random.uniform(
                -bottom_max_shift, bottom_max_shift
            )
        else:
            left_shift = np.random.uniform(left_min_shift, left_max_shift)
            top_shift = np.random.uniform(top_min_shift, top_max_shift)
            right_shift = np.random.uniform(right_min_shift, right_max_shift)
            bottom_shift = np.random.uniform(
                bottom_min_shift, bottom_max_shift
            )

        if self.show_debug_info:
            pprint.pprint(
                dict(  # noqa
                    left_min_shift=left_min_shift,
                    left_max_shift=left_max_shift,
                    top_min_shift=top_min_shift,
                    top_max_shift=top_max_shift,
                    right_min_shift=right_min_shift,
                    right_max_shift=right_max_shift,
                    bottom_min_shift=bottom_min_shift,
                    bottom_max_shift=bottom_max_shift,
                    left_shift=left_shift,
                    top_shift=top_shift,
                    right_shift=right_shift,
                    bottom_shift=bottom_shift,
                )
            )

        roi_bboxes[:, 0] -= left_shift
        roi_bboxes[:, 1] -= top_shift
        roi_bboxes[:, 2] += right_shift
        roi_bboxes[:, 3] += bottom_shift

        roi_bboxes[:, :2] = np.maximum(roi_bboxes[:, :2], 0)
        roi_bboxes[:, 2:4] = np.minimum(roi_bboxes[:, 2:4], np.array(img_wh))

        roi_bboxes_ws = roi_bboxes[:, 2:3] - roi_bboxes[:, 0:1]
        roi_bboxes_hs = roi_bboxes[:, 3:4] - roi_bboxes[:, 1:2]

        if self.show_debug_info and (
            (roi_bboxes_ws <= 0).astype("int").sum() > 0
            or (roi_bboxes_hs <= 0).astype("int").sum() > 0
        ):
            warnings.warn("error! width or height less than 0!")

        roi_bboxes[:, [0, 2]] = np.where(
            roi_bboxes_ws <= 0, bboxes[:, [0, 2]], roi_bboxes[:, [0, 2]]
        )
        roi_bboxes[:, [1, 3]] = np.where(
            roi_bboxes_hs <= 0, bboxes[:, [1, 3]], roi_bboxes[:, [1, 3]]
        )

        return roi_bboxes, bboxes

    def filter_roi_bboxes(self, roi_bboxes, origin_roi_bboxes, img_wh):
        def filter_bbox(
            bbox, img_roi, allow_outside_center=True, min_edge_size=0.1
        ):
            assert bbox.ndim == 2
            assert len(img_roi) == 4
            roi = np.array(img_roi)
            if allow_outside_center:
                mask = np.ones(bbox.shape[0], dtype=bool)
            else:
                centers = (bbox[:, :2] + bbox[:, 2:4]) / 2
                mask = np.logical_and(
                    roi[:2] <= centers, centers < roi[2:]
                ).all(axis=1)
            mask = np.logical_and(
                mask, (bbox[:, :2] + min_edge_size < bbox[:, 2:4]).all(axis=1)
            )

            bbox[np.logical_not(mask), 4] *= -1
            # bbox = bbox[mask]
            return bbox, mask

        if self.filter_valid_roi:
            img_roi = (0, 0, img_wh[0], img_wh[1])
            clip_roi_bboxes = clip_bbox(roi_bboxes, img_roi)
            clip_roi_bboxes = remap_bbox_label_by_area(
                clip_roi_bboxes, self.min_valid_area
            )
            clip_roi_bboxes, mask = filter_bbox(
                clip_roi_bboxes,
                img_roi,
                allow_outside_center=True,
                min_edge_size=self.min_edge_size,
            )
        else:
            clip_roi_bboxes = roi_bboxes

        return clip_roi_bboxes, origin_roi_bboxes

    def resize_img(self, img):
        assert img is not None
        h, w = img.shape[:2]
        inter_method = get_interp_method(
            self.inter_method,
            (h, w, self.target_wh[1], self.target_wh[0]),
        )

        resize_img = cv2.resize(
            img,
            self.target_wh,
            interpolation=inter_method,
        )

        return resize_img

    def __call__(self, data):
        img = data["img"]
        gt_boxes = data["gt_boxes"]
        data["ig_regions"]
        data["norm_points"]
        image_info = data.get("image_info", None)

        img_height, img_width = img.shape[:2]

        crop_images = []
        origin_crop_images = []
        roi_boxes, origin_roi_boxes = self.crop_func(
            gt_boxes, (img_width, img_height)
        )

        origin_crop_roi_boxes = roi_boxes.copy()

        roi_boxes, origin_roi_boxes = self.filter_roi_bboxes(
            roi_boxes, origin_roi_boxes, (img_width, img_height)
        )

        if len(roi_boxes) == 0:
            random_roi = _generate_random_roi(
                (img_width, img_height),
                gt_boxes,
                random_roi_class_id=self.random_roi_class_id,
            )
            roi_boxes = [random_roi]
            origin_roi_boxes = [random_roi]

        for roi_box, origin_roi_box in zip_longest(
            roi_boxes, origin_roi_boxes
        ):
            left, top = (math.floor(roi_box[0]), math.floor(roi_box[1]))
            right, bottom = (math.ceil(roi_box[2]), math.ceil(roi_box[3]))
            crop_img = img[top:bottom, left:right, :]
            resize_crop_img = self.resize_img(crop_img)
            crop_images.append(resize_crop_img)

            if self.show_origin_image:
                left, top = (
                    math.floor(origin_roi_box[0]),
                    math.floor(origin_roi_box[1]),
                )
                right, bottom = (
                    math.ceil(origin_roi_box[2]),
                    math.ceil(origin_roi_box[3]),
                )
                origin_crop_img = img[top:bottom, left:right, :]
                origin_resize_crop_img = self.resize_img(origin_crop_img)
                origin_crop_images.append(origin_resize_crop_img)

        results = {
            "imgs": crop_images,
            # "gt_bboxes": gt_boxes,
            "roi_bboxes": roi_boxes,
            # "ig_bboxes": ig_regions,
            "img_height": img_height,
            "img_width": img_width,
            "color_space": data["color_space"],
        }

        if image_info is not None:
            results["image_info"] = image_info

        if self.show_origin_bbox:
            results["origin_roi_bboxes"] = origin_roi_boxes
            results["origin_crop_roi_boxes"] = origin_crop_roi_boxes

        if self.show_origin_image:
            results["origin_imgs"] = origin_crop_images

        return results


def build_albumentations_transform(transform_cfg):
    """Build albumentations transform.

    Args:
        transform_cfg: Cfg for building albumentations transform.
    """

    if transform_cfg is None:
        return None

    assert isinstance(transform_cfg, dict)
    transform_cfg = copy.deepcopy(transform_cfg)
    assert (
        albumentations is not None
    ), "Please install albumentations using pip!"
    base_type = transform_cfg.pop("albumentations_type")
    if base_type in ["Compose", "SomeOf", "OneOf", "Sequential", "OneOrOther"]:
        assert "transforms" in transform_cfg
        transforms = []
        for cfg in transform_cfg["transforms"]:
            transforms.append(build_albumentations_transform(cfg))
        transform_cfg["transforms"] = transforms
        transform = getattr(albumentations, base_type)(**transform_cfg)
    else:
        transform = getattr(albumentations, base_type)(**transform_cfg)

    return transform


@OBJECT_REGISTRY.register
class TrafficSignImageAugmentation(object):
    """Image Augmentation for traffic sign.

    Args:
        base_transformer: Cfg for building base transformer.
        color_transformer: Cfg for building color transformer.
        flip_transformer: Cfg for building flip transformer.
        classname2idxs: Mapping from class name to index.
        flip_trans_all: Whether flip transform for all classes.
        flip_label_mapping: Mapping for fliping label.
        color_trans_all: Whether color transform for all classes.
        color_trans_types: Class type for color transform.
    """

    def __init__(
        self,
        base_transformer: dict = None,
        color_transformer: dict = None,
        flip_transformer: dict = None,
        classname2idxs: dict = None,
        flip_trans_all: bool = False,
        flip_label_mapping: Union[list, dict] = None,
        color_trans_all: bool = False,
        color_trans_types: list = None,
    ):
        self.base_transformer = build_albumentations_transform(
            base_transformer
        )
        self.color_transformer = build_albumentations_transform(
            color_transformer
        )
        self.flip_transformer = build_albumentations_transform(
            flip_transformer
        )

        self.classname2idxs = classname2idxs
        if classname2idxs is not None:
            self.classidx2names = {
                idx: label for label, idx in classname2idxs.items()
            }
        else:
            self.classidx2names = None

        if isinstance(flip_label_mapping, list):
            self.flip_label_mapping = {val: val for val in flip_label_mapping}
        elif (
            isinstance(flip_label_mapping, dict) or flip_label_mapping is None
        ):
            self.flip_label_mapping = flip_label_mapping
        else:
            raise TypeError(
                "type of 'flip_label_mapping' must be list or dict"
            )

        self.flip_trans_all = flip_trans_all
        if flip_label_mapping is not None:
            assert (
                classname2idxs is not None
            ), "Apply flip transform at some types, needs to parse 'classname2idxs'"  # noqa
            for key, value in self.flip_label_mapping.items():
                if key not in classname2idxs:
                    raise ValueError(
                        f"'flip_label_mapping''s key and value must in 'classname2idxs', "  # noqa
                        f"but key '{key}' not in 'classname2idxs'"
                    )
                if value not in classname2idxs:
                    raise ValueError(
                        f"'flip_label_mapping''s key and value must in 'classname2idxs', "  # noqa
                        f"but value '{value}' not in 'classname2idxs'"
                    )

        self.color_trans_all = color_trans_all
        if color_trans_types is not None:
            assert (
                classname2idxs is not None
            ), "Apply color transform at some types, needs to parse 'channel_labels'"  # noqa
            self.color_trans_types = color_trans_types
        else:
            self.color_trans_types = None

    def transform(self, img, roi_box):
        class_id = roi_box[-1]

        if class_id < 0:
            class_type = None
        else:
            class_type = (
                self.classidx2names[class_id]
                if self.classidx2names is not None
                else None
            )

        ts_img = img

        # apply color transform
        if self.color_transformer is not None and (
            self.color_trans_all
            or (
                self.color_trans_types is not None
                and class_type in self.color_trans_types
            )
        ):
            ts_img = self.color_transformer(image=ts_img)["image"]

        # apply base transform
        if self.base_transformer is not None:
            ts_img = self.base_transformer(image=ts_img)["image"]

        # apply flip transform
        if self.flip_transformer is not None and (
            self.flip_trans_all
            or (
                self.flip_label_mapping is not None
                and class_type in self.flip_label_mapping
            )
        ):
            ts_img_ori = ts_img.copy()
            ts_img = self.flip_transformer(image=ts_img)["image"]
            flip = not (ts_img == ts_img_ori).all()

            if flip and self.flip_label_mapping is not None:
                convert_class_type = self.flip_label_mapping.get(
                    class_type, class_type
                )
                convert_class_id = self.classname2idxs[convert_class_type]
                roi_box[-1] = convert_class_id

        return ts_img, roi_box

    def __call__(self, data):
        imgs = data["imgs"]
        roi_bboxes = data["roi_bboxes"]

        ts_imgs = []
        ts_roi_bboxes = []
        for img, roi_box in zip(imgs, roi_bboxes.copy()):
            ts_img, ts_roi_box = self.transform(img, roi_box)
            ts_imgs.append(ts_img)
            ts_roi_bboxes.append(ts_roi_box)

        data["imgs"] = ts_imgs
        data["roi_bboxes"] = ts_roi_bboxes

        return data


@OBJECT_REGISTRY.register
class ClassIdRemap(object):
    """Remap class id.

    Args:
        selected_class_ids: List of selected class ids used for remap.
        ignore_label: Label of ignore classe.
    """

    def __init__(
        self,
        selected_class_ids: List,
        ignore_label: int = -1,
    ):
        self.classidmap = dict(
            zip(selected_class_ids, range(len(selected_class_ids)))
        )
        self.ignore_label = ignore_label

    def _remap_class_id(self, class_id):
        old_class_id = int(class_id)
        if old_class_id == self.ignore_label:
            return float(old_class_id)
        new_class_id = (
            self.classidmap[old_class_id]
            if old_class_id > 0
            else self.ignore_label
        )
        return float(new_class_id)

    def __call__(self, data):

        gt_bboxes = data.get("gt_bboxes", [])
        ig_bboxes = data.get("ig_bboxes", [])
        roi_boxes = data.get("roi_bboxes", [])

        for gt_boxes_per_img in gt_bboxes:
            for gt_box in gt_boxes_per_img:
                gt_box[-1] = self._remap_class_id(gt_box[-1])
        for ig_boxes_per_img in ig_bboxes:
            for ig_box in ig_boxes_per_img:
                ig_box[-1] = self._remap_class_id(ig_box[-1])
        for roi_box in roi_boxes:
            roi_box[-1] = self._remap_class_id(roi_box[-1])

        return data


@OBJECT_REGISTRY.register
class ClassificationLabelTs(object):
    """A transformer that transform classification labels.

    Args:
        to_tensor: Whether transform to tensor.
        to_yuv: Whether transfor to yuv.
        task_type: Task Type.
        ignore_label: Label of ignore class.
        show_image_info: Whether show image info.
        show_origin_image: Whether show origin image.
        show_origin_class_id: Whether show origin class id.
        show_origin_bbox: Whether show origin bbox.
    """

    def __init__(
        self,
        to_tensor: bool = True,
        to_yuv: bool = False,
        task_type: str = "detection",
        ignore_label: int = -1,
        show_image_info: bool = False,
        show_origin_image: bool = False,
        show_origin_class_id: bool = False,
        show_origin_bbox: bool = False,
    ):
        self._to_tensor = ToTensor(to_yuv=to_yuv) if to_tensor else None
        self.task_type = task_type
        self.ignore_label = ignore_label
        self.show_image_info = show_image_info
        self.show_origin_image = show_origin_image
        self.show_origin_class_id = show_origin_class_id
        self.show_origin_bbox = show_origin_bbox

    def __call__(self, data):
        imgs = data["imgs"]
        origin_imgs = data.get("origin_imgs", [])
        gt_bboxes = data.get("gt_bboxes", [])
        roi_bboxes = data.get("roi_bboxes", [])
        origin_roi_bboxes = data.get("origin_roi_bboxes", [])
        origin_crop_roi_boxes = data.get("origin_crop_roi_boxes", [])
        ig_bboxes = data.get("ig_bboxes", [])
        image_info = data.get("image_info", None)
        img_height = data["img_height"]
        img_width = data["img_width"]
        color_space = data["color_space"]

        results = []
        for (
            img,
            origin_img,
            gt_boxes,
            roi_box,
            origin_roi_box,
            origin_crop_roi_box,
            ig_boxes,
        ) in zip_longest(
            imgs,
            origin_imgs,
            gt_bboxes,
            roi_bboxes,
            origin_roi_bboxes,
            origin_crop_roi_boxes,
            ig_bboxes,
        ):
            if self.task_type == "detection":
                assert len(gt_bboxes) > 0
                gt_boxes = gt_boxes[:, :-1]
                ig_boxes = ig_boxes[:, :-1]
                gt_classes = gt_boxes[:, -1].astype(np.int64)
            elif self.task_type == "classification":
                assert len(roi_bboxes) > 0
                gt_classes = np.array(roi_box[-1], dtype=np.int64)
                origin_gt_classes = gt_classes
                if origin_roi_box is not None:
                    origin_gt_classes = np.array(
                        origin_roi_box[-1], dtype=np.int64
                    )
            item = {
                "img": img,
                "layout": "hwc",
                "gt_classes": gt_classes,
                "img_height": img_height,
                "img_width": img_width,
                "img_shape": img.shape,
                "color_space": color_space,
            }

            if self.task_type == "detection":
                item["gt_bboxes"] = gt_boxes
                item["ig_bboxes"] = ig_boxes
            elif self.task_type == "classification":
                if self.show_origin_class_id:
                    item["origin_gt_classes"] = origin_gt_classes
                if self.show_origin_image:
                    item["origin_img"] = origin_img
                if self.show_origin_bbox and origin_roi_box is not None:
                    item["origin_roi_box"] = origin_roi_box
                    item["origin_crop_roi_box"] = origin_crop_roi_box
                    item["roi_box"] = roi_box

            if self._to_tensor is not None:
                item = self._to_tensor(item)

            if self.show_image_info and image_info is not None:
                item["image_info"] = image_info

            results.append(item)

        return results
