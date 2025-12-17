import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch.utils.data as data

try:
    from auto_matrix.data.dataset import DenseBoxAnnoDataset, KPSAnnoDataset
except ImportError:
    DenseBoxAnnoDataset = None
    KPSAnnoDataset = None

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

__all__ = [
    "DatasetFromAnno",
    "RCNNDetDatasetFromAnno",
    "Auto2dFromRawJson",
]


@OBJECT_REGISTRY.register
class DatasetFromAnno(data.Dataset):
    """Dataset which gets img data from the annotation file.

    This dataset can used for getting raw image buffer.

    Args:
        anno_path: The path where the pb_rec is stored.
        class_id_map: map each class's rec class id to used category.
        transforms: List of transform.
        task_type: Optional values include
            [`detection`, `rcnn_kps`, `rcnn_classification`].
        buf_only: Save image buffer instead of decoded image.
        to_rgb: Whether to convert to `rgb` color_space.
        return_orig_img: Whether to return an extra original img,
            orig_img can usually be used on visualization.
        ignore_hard: Ignore hard instances if `hard` tag in annotation.
            Default is False.
        input_hw: the shape of input image will be controled.
        use_ignore: Whether to use ignore regions in anno set.
        pad_topk: whether to pad 0 to labels if num_gt is smaller than
            this value.
    """

    @require_packages("auto_matrix")
    def __init__(
        self,
        anno_path: str,
        class_id_map: Optional[Dict] = None,
        transforms: List = None,
        task_type: str = "detection",
        buf_only: bool = False,
        to_rgb: bool = False,
        return_orig_img: bool = False,
        ignore_hard: bool = False,
        input_hw: Tuple[int, int] = (1152, 1408),
        use_ignore: bool = False,
        pad_topk: bool = -1,
    ):
        super().__init__()
        self.anno_path = anno_path
        self.transforms = transforms
        self.task_type = task_type
        self.buf_only = buf_only
        self.to_rgb = to_rgb
        self.return_orig_img = return_orig_img
        self.bucket_root = "/horizon-bucket"
        if not os.path.exists(self.bucket_root):
            self.bucket_root = "/bucket/input"
            assert os.path.exists(
                self.bucket_root
            ), "Please check environment."
        self.ignore_hard = ignore_hard
        self.input_hw = input_hw
        self.use_ignore = use_ignore
        self.class_id_map = class_id_map if class_id_map else {}
        self.pad_topk = pad_topk
        if task_type == "rcnn_kps":
            self.dataset_class = KPSAnnoDataset
        else:
            self.dataset_class = DenseBoxAnnoDataset

        self._get_img_info()

    def _get_img_info(self):
        imgs_info = self.dataset_class(self.anno_path, self.anno_path + ".idx")
        data = []
        for info in imgs_info:
            img_info = info[1]
            if not isinstance(img_info, dict):
                img_info = img_info.to_dict()
            img_h = img_info.get("img_h", img_info.get("height", -1))
            img_w = img_info.get("img_w", img_info.get("width", -1))
            if img_h != self.input_hw[0] or img_w != self.input_hw[1]:
                continue
            data.append(img_info)
        self.imgs_info = data

    def __len__(self):
        return len(self.imgs_info)

    def __getitem__(self, idx):
        data = {}
        img_info = self.imgs_info[idx]
        img_url = img_info.get("img_url", img_info.get("image", None))
        assert img_url is not None
        img_path = self.bucket_root + img_url.split("/horizon-bucket")[1]
        image = cv2.imread(img_path)
        color_space = "bgr"
        if self.to_rgb:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color_space = "rgb"
        data["img_name"] = img_path.split("/")[-1]
        with open(img_path, "rb") as rf:
            data["img_buf"] = rf.read()
        if not self.buf_only:
            data["img"] = image
        if self.return_orig_img:
            data["ori_image"] = image
        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image.shape

        data["img_id"] = img_info.get("idx", -1)
        data["img_height"] = img_info.get(
            "img_h", img_info.get("height", image.shape[0])
        )
        data["img_width"] = img_info.get(
            "img_w", img_info.get("width", image.shape[1])
        )
        data["im_hw"] = (data["img_height"], data["img_width"])

        if self.task_type in [
            "detection",
            "rcnn_classfication",
            "rcnn_detection",
        ]:
            data = self._get_detection_label_data(data, img_info)
        elif self.task_type == "rcnn_kps":
            data = self._get_rcnn_kps_label_data(data, img_info)

        data["task_type"] = self.task_type
        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def _get_rcnn_kps_label_data(self, data, img_info):
        img_boxes = img_info["boxes"]
        img_keypoints = img_info["keypoints"]
        gt_boxes = []
        for box, keypoints in zip(img_boxes, img_keypoints):
            gt_boxes.append(box + keypoints)
        data["gt_boxes"] = np.array(gt_boxes)
        num_gt = len(gt_boxes)
        data["gt_boxes_num"] = np.array(num_gt)
        if self.pad_topk > num_gt:
            pad_num = self.pad_topk - num_gt
            data["gt_boxes"] = self._pad_gt(pad_num, data["gt_boxes"])
        return data

    def _get_detection_label_data(self, data, img_info):
        gt_bboxes = []
        gt_classes = []
        for ins in img_info["instances"]:
            is_hard = ins["is_hard"][0]
            if is_hard and self.ignore_hard:
                continue
            points_data = ins["points_data"]
            class_id = int(ins["class_id"][0])
            bbox = []
            bbox.extend(points_data[0])
            bbox.extend(points_data[2])
            bbox.append(class_id)

            if class_id in self.class_id_map:
                gt_class = self.class_id_map[class_id]
                if is_hard and self.ignore_hard:
                    gt_class = -1 * (gt_class + 1)
                gt_classes.append(gt_class)
            else:
                continue
            gt_bboxes.append(bbox)
        num_gt = len(gt_bboxes)
        data["gt_bboxes"] = np.array(gt_bboxes)
        data["gt_classes"] = np.array(gt_classes, dtype=np.int64)
        data["gt_labels"] = [l for _, l in self.class_id_map.items()]

        ig_bboxes = []
        if self.use_ignore:
            for ig_region in img_info["ignore_regions"]:
                left_top = ig_region["contour"][0]
                right_bottom = ig_region["contour"][1]
                if not (
                    left_top == [0, 0]
                    and right_bottom == [data["img_width"], data["img_height"]]
                ):  # must be first n in anno["ignore_regions"]
                    ig_bbox = left_top + right_bottom
                    ig_bboxes.append(ig_bbox)
                else:
                    break
        data["ig_bboxes"] = np.array(ig_bboxes)
        if self.pad_topk > num_gt:
            pad_num = self.pad_topk - num_gt
            data["gt_bboxes"] = self._pad_gt(pad_num, data["gt_bboxes"])
            data["gt_classes"] = self._pad_gt(pad_num, data["gt_classes"])
            data["ig_bboxes"] = self._pad_gt(pad_num, data["ig_bboxes"])
        return data

    def _pad_gt(self, pad_num, label_array):
        return np.concatenate(
            [
                label_array,
                np.zeros([pad_num, label_array.shape[1]], label_array.dtype),
            ]
        )


@OBJECT_REGISTRY.register
class RCNNDetDatasetFromAnno(DatasetFromAnno):
    """Dataset that extends DatasetFromAnno.

    This dataset is used in rcnn detection task.

    Args:
        anno_path: The path where the pb_rec is stored.
        class_id_map: map each class's rec class id to used category.
        transforms: List of transform.
        buf_only: Save image buffer instead of decoded image.
        to_rgb: Whether to convert to `rgb` color_space.
        return_orig_img: Whether to return an extra original img,
            orig_img can usually be used on visualization.
        ignore_hard: Ignore hard instances if `hard` tag in annotation.
            Default is False.
        input_hw: the shape of input image will be controled.
        use_ignore: Whether to use ignore regions in anno set.
        pad_topk: whether to pad 0 to labels if num_gt is smaller than
            this value.
        lt_point_id: left top point idx of sub box in annotation.
        rb_point_id: right bottom point idx of sub box in annotation.
        parent_lt_point_id: left top point idx of parent box in annotation.
        parent_rb_point_id: right bottom point idx of parent box in annotation.
    """

    def __init__(
        self,
        anno_path: str,
        class_id_map: Optional[Dict] = None,
        transforms: List = None,
        buf_only: bool = False,
        to_rgb: bool = False,
        return_orig_img: bool = False,
        ignore_hard: bool = False,
        input_hw: Tuple[int, int] = (1152, 1408),
        use_ignore: bool = False,
        pad_topk: bool = -1,
        lt_point_id: int = 0,
        rb_point_id: int = 2,
        parent_lt_point_id: int = 10,
        parent_rb_point_id: int = 12,
    ):
        super().__init__(
            anno_path=anno_path,
            transforms=transforms,
            buf_only=buf_only,
            to_rgb=to_rgb,
            return_orig_img=return_orig_img,
            ignore_hard=ignore_hard,
            input_hw=input_hw,
            task_type="rcnn_detection",
            use_ignore=use_ignore,
            pad_topk=pad_topk,
            class_id_map=class_id_map,
        )
        self.lt_point_id = lt_point_id
        self.rb_point_id = rb_point_id
        self.parent_lt_point_id = parent_lt_point_id
        self.parent_rb_point_id = parent_rb_point_id

    def _get_detection_label_data(self, data, img_info):
        def _get_gt_boxes(lt_point_id, rb_point_id):
            gt_boxes = []
            for ins in img_info["instances"]:
                is_hard = ins["is_hard"][0]
                if is_hard and self.ignore_hard:
                    continue
                points_data = ins["points_data"]
                lt = points_data[lt_point_id]
                rb = points_data[rb_point_id]
                gt_boxes_i = [lt[0], lt[1], rb[0], rb[1]]
                gt_boxes.append(gt_boxes_i)
            gt_boxes = np.array(gt_boxes, dtype=np.float32)

            return gt_boxes

        gt_boxes = _get_gt_boxes(self.lt_point_id, self.rb_point_id)
        parent_gt_boxes = _get_gt_boxes(
            self.parent_lt_point_id,
            self.parent_rb_point_id,
        )
        assert len(gt_boxes) == len(parent_gt_boxes)
        data["gt_boxes_num"] = np.array(len(gt_boxes))
        data["parent_gt_boxes_num"] = np.array(len(parent_gt_boxes))
        data["gt_boxes"] = np.zeros((100, 4))
        data["parent_gt_boxes"] = np.zeros((100, 4))
        if len(gt_boxes) > 0:
            data["gt_boxes"][: len(gt_boxes), :] = gt_boxes
            data["parent_gt_boxes"][
                : len(parent_gt_boxes), :
            ] = parent_gt_boxes
        return data


@OBJECT_REGISTRY.register
class Auto2dFromRawJson(data.Dataset):
    """Dataset which gets img data from the data_path.

    This dataset is basicly used in inferring aidi-eval datasets.

    Args:
        anno_path: The path where the pb_rec is stored.
        transforms: List of transform.
        task_type: Optional values include
            [`detection`, `rcnn_detection`, `rcnn_kps`,
            `rcnn_classification`].
        to_rgb: Whether to convert to `rgb` color_space.
        return_orig_img: Whether to return an extra original img,
            orig_img can usually be used on visualization.
        ignore_hard: Ignore hard instances if `hard` tag in annotation.
            Default is False.
        input_hw: the shape of input image will be controled.
    """

    def __init__(
        self,
        anno_path: str,
        img_root: Optional[str] = None,
        transforms: Optional[List] = None,
        task_type: Optional[str] = "detection",
        buf_only: Optional[bool] = False,
        to_rgb: Optional[bool] = False,
        return_orig_img: Optional[bool] = False,
        input_hw: Optional[Tuple[int]] = (1152, 1408),
    ):
        self.anno_path = anno_path
        self.transforms = transforms
        self.task_type = task_type
        self.buf_only = buf_only
        self.to_rgb = to_rgb
        self.return_orig_img = return_orig_img
        self.input_hw = input_hw
        if img_root:
            self.img_root = img_root
        else:
            self.img_root = anno_path.split(".json")[0]

        self._get_img_info()

    def _get_img_info(self):
        data = []
        with open(self.anno_path, "r", encoding="utf-8") as r:
            for line in r.readlines():
                img_info = json.loads(line.strip())
                if (
                    img_info["height"] != self.input_hw[0]
                    or img_info["width"] != self.input_hw[1]
                ):
                    continue
                data.append(img_info)
        self.imgs_info = data

    def __len__(self):
        return len(self.imgs_info)

    def __getitem__(self, idx):
        data = {}
        img_info = self.imgs_info[idx]
        img_key = img_info["image_key"]
        img_path = os.path.join(self.img_root, img_key)
        image = cv2.imread(img_path)
        color_space = "bgr"
        if self.to_rgb:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color_space = "rgb"
        data["img_name"] = img_key
        if self.buf_only:
            with open(img_path, "rb") as rf:
                data["img_buf"] = rf.read()
        else:
            data["img"] = image
        if self.return_orig_img:
            data["ori_image"] = image
        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image.shape
        data["img_id"] = img_info["video_index"]
        data["img_height"] = img_info["height"]
        data["img_width"] = img_info["width"]
        data["im_hw"] = (img_info["height"], img_info["width"])
        # eval on aidi, fake gt
        if self.task_type == "detection":
            data["gt_bboxes"] = np.zeros((1, 4))
            data["gt_classes"] = np.zeros((1), dtype=np.int64)
            data["gt_labels"] = np.zeros((1), dtype=np.int64)
        elif self.task_type == "rcnn_detection":
            data["gt_boxes_num"] = np.zeros((1), dtype=np.int64)
            data["parent_gt_boxes_num"] = np.zeros((1), dtype=np.int64)
            data["gt_boxes"] = np.zeros((1, 4))
            data["parent_gt_boxes"] = np.zeros((1, 4))
        elif self.task_type == "rcnn_kps":
            data["gt_boxes_num"] = np.zeros((1), dtype=np.int64)
            data["gt_boxes"] = np.zeros((1, 10))
        elif self.task_type == "rcnn_classification":
            data["gt_boxes_num"] = np.zeros((1), dtype=np.int64)
            data["gt_boxes"] = np.zeros((1, 5))

        if self.transforms is not None:
            data = self.transforms(data)

        return data
