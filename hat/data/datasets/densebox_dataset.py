# Copyright (c) Horizon Robotics. All rights reserved.
"""Dataset for densebox mx-record data, used in auto."""
import contextlib
import getpass
import logging
import os
import tempfile
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch.utils.data as data

from hat.data.datasets.legacy_densebox import (
    HAT_LEGACYDENSEBOX_AVAILABLE,
    LegacyDenseBoxImageRecordDataset,
)
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "DenseboxDataset",
    "MulticlassDenseboxDataset",
    "DenseboxDataset2PE",
    "VehicleSideDenseboxDataset",
]


@OBJECT_REGISTRY.register
class DenseboxDataset(data.Dataset):
    """Dataset for densebox record data in auto, such as adas-mini.

    Args:
        data_path : Path of data relative to buket path.
        anno_path : Path of annotation.
        transforms : List of transform.
        to_rgb: Convert bgr(cv2 imread) to rgb.
        task_type : Consist of 'detection', 'segmentation'
        class_id : the rec's class id, 1base
        category : the used category, 0base
        rec_idx_file_path: index file related to data_path. Used only when
            there is already index file somewhere.
        disable_default_densebox_log: Disable default print output from
            `LegacyDenseBoxImageRecordDataset`. Default is True.
        ignore_hard : Ignore hard instances if `hard` tag in annotation.
            Default is False.
        use_ignore : Whether to use ignore regions in annotation.
            Default is False.
        with_img_buf: Whether return img buf.
            Default is False.
        return_orig_img: Whether to return an extra original img which can
            be used for visualization. Default is False.
        return_orig_gt_seg: Whether to return an extra original segmentation
            ground-truth image which can be used for visualization.
            Default is False.
        remove_det_duplicate: Whether to filter out duplicate gt bboxes in one
            image. Sometimes there are duplicate gt bboxes in one image to
            emphasize certain training examples, but which may cause inaccurate
            performance evaluation. Default is False.
        abandon_other_category: there may be some bboxes that do not belong to
            current class. If False, set these bboxes as ignore (foreground
            with no loss), else, set as background. Default is False.
        extend_ignore_region_into_gtbox: Whether to extend boxes in ignore
            regions into ground truth bbox with class -1. Default is False.
    """

    def __init__(
        self,
        data_path: str,
        anno_path: str,
        transforms: Optional[List] = None,
        to_rgb: Optional[bool] = False,
        task_type: Optional[str] = "detection",
        class_id: Optional[int] = -1,
        category: Optional[int] = -1,
        rec_idx_file_path: Optional[str] = None,
        disable_default_densebox_log: Optional[bool] = True,
        ignore_hard: Optional[bool] = False,
        use_ignore: Optional[bool] = False,
        with_img_buf: Optional[bool] = False,
        return_orig_img: Optional[bool] = False,
        return_orig_gt_seg: Optional[bool] = False,
        remove_det_duplicate: Optional[bool] = False,
        abandon_other_category: Optional[bool] = False,
        extend_ignore_region_into_gtbox: Optional[bool] = False,
    ):
        assert (
            HAT_LEGACYDENSEBOX_AVAILABLE
        ), "horizon_plugin_pytorch >= 1.0.0 is required."

        self.data_path = data_path
        self.anno_path = anno_path
        self.transforms = transforms
        self.to_rgb = to_rgb
        self.task_type = task_type
        self.class_id = class_id
        self.category = category
        self.rec_idx_file_path = rec_idx_file_path
        self.disable_default_densebox_log = disable_default_densebox_log
        self.ignore_hard = ignore_hard
        self.use_ignore = use_ignore
        self.with_img_buf = with_img_buf
        self.return_orig_img = return_orig_img
        self.return_orig_gt_seg = return_orig_gt_seg
        self.remove_det_duplicate = remove_det_duplicate
        self.abandon_other_category = abandon_other_category
        self.extend_ignore_region_into_gtbox = extend_ignore_region_into_gtbox

        if self.rec_idx_file_path is None:
            self.rec_idx_file_path = get_idx_path(self.data_path)

        if self.task_type == "detection" or self.task_type == "classification":
            self.kwargs = {}
        elif self.task_type == "segmentation":
            self.kwargs = {"with_seg_label": True, "seg_label_dtype": np.int8}
        else:
            raise Exception(
                "error task_type, your task_type[{}],"
                " we need segmentation or detection".format(self.task_type)
            )
        self.kwargs["with_img_buf"] = self.with_img_buf
        self.dataset = build_dataset(
            self.data_path,
            self.anno_path,
            self.rec_idx_file_path,
            self.disable_default_densebox_log,
            **self.kwargs,
        )
        logging.info(f"dataset path: {self.data_path}, {self.anno_path}")
        logging.info(f"dataset length: {len(self.dataset)}")

    def __getitem__(self, index: int) -> Dict:
        if self.with_img_buf:
            image, img_buf, anno = self.dataset[index]
        else:
            image, anno = self.dataset[index]
        data = {}
        color_space = "bgr"
        if self.to_rgb:
            # cv2.cvtColor may be slow.
            # See http://wiki.hobot.cc/pages/viewpage.action?pageId=186775106 for more details.     # noqa
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color_space = "rgb"
        if self.task_type == "detection" or self.task_type == "classification":
            anno = anno
        elif self.task_type == "segmentation":
            seg_label = anno[1]
            seg_label = seg_label.astype(np.uint8)
            anno = anno[0]
        anno = anno.to_dict()
        data["img_name"] = anno["img_url"].split("/")[-1]
        data["data_path"] = self.data_path
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

        if self.task_type == "detection" or self.task_type == "classification":
            gt_bboxes = []
            gt_classes = []
            gt_attributes = []
            for ins in anno["instances"]:
                is_hard = ins["is_hard"][0]
                points_data = ins["points_data"]
                class_id = int(ins["class_id"][0])
                bbox = []
                bbox.extend(points_data[0])
                bbox.extend(points_data[2])
                if self.task_type == "classification":
                    if "attribute" in ins:
                        attribute = ins["attribute"]
                        gt_attributes.append(attribute)
                    gt_bboxes.append(bbox)
                    gt_classes.append(-1)
                else:
                    if class_id == self.class_id:
                        gt_bboxes.append(bbox)
                        if is_hard and self.ignore_hard:
                            gt_classes.append(-1)
                        else:
                            gt_classes.append(self.category)
                    elif not self.abandon_other_category:
                        gt_bboxes.append(bbox)
                        gt_classes.append(-1)
            if self.extend_ignore_region_into_gtbox:
                for ignore in anno.get("ignore_regions", []):
                    class_id = int(ignore["class_id"][0])
                    contour = ignore["contour"]
                    if class_id != self.class_id:
                        continue
                    bbox = []
                    bbox.extend(contour[0])
                    bbox.extend(contour[1])
                    gt_bboxes.append(bbox)
                    gt_classes.append(-1)
            data["gt_bboxes"] = np.array(gt_bboxes)
            data["gt_classes"] = np.array(gt_classes, dtype=np.int64)
            if self.task_type == "classification":
                data["attribute_label"] = np.array(
                    gt_attributes, dtype=np.float32
                )

            ig_bboxes = []
            if self.use_ignore:
                for ig_region in anno["ignore_regions"]:
                    left_top = ig_region["contour"][0]
                    right_bottom = ig_region["contour"][1]
                    if not (
                        left_top == [0, 0]
                        and right_bottom == [anno["img_w"], anno["img_h"]]
                    ):  # must be first n in anno["ignore_regions"]
                        ig_bbox = left_top + right_bottom
                        ig_bboxes.append(ig_bbox)
                    else:
                        break
            data["ig_bboxes"] = ig_bboxes
            # Remove duplicate gt boxes in one image
            if self.remove_det_duplicate:
                data["gt_bboxes"], unique_indices = np.unique(
                    data["gt_bboxes"], return_index=True, axis=0
                )
                data["gt_classes"] = data["gt_classes"][unique_indices]

        elif self.task_type == "segmentation":
            data["gt_seg"] = seg_label
            if self.return_orig_gt_seg:
                data["orig_gt_seg"] = seg_label.copy()

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return "DenseboxDataset"

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("dataset")
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        disable_default_densebox_log = self.__dict__[
            "disable_default_densebox_log"
        ]
        rec_idx_file_path = self.__dict__["rec_idx_file_path"]
        kwargs = self.__dict__["kwargs"]
        data_path = self.__dict__["data_path"]
        anno_path = self.__dict__["anno_path"]

        self.__dict__["dataset"] = build_dataset(
            data_path,
            anno_path,
            rec_idx_file_path,
            disable_default_densebox_log,
            **kwargs,
        )


@OBJECT_REGISTRY.register
class DenseboxDataset2PE(DenseboxDataset):
    """Dataset for densebox record data in auto, extend for 2pe.

    Args:
        data_path : Path of data relative to buket path.
        anno_path : Path of annotation.
        transforms : List of transform.
        to_rgb: Convert bgr(cv2 imread) to rgb.
        task_type : Consist of 'detection', 'segmentation'
        class_id : the rec's class id, 1base
        category : the used category, 0base
        rec_idx_file_path: index file related to data_path. Used only when
            there is already index file somewhere.
        disable_default_densebox_log: Disable default print output from
            `LegacyDenseBoxImageRecordDataset`. Default is True.
        ignore_hard : Ignore hard instances if `hard` tag in annotation.
            Default is False.
        use_ignore : Whether to use ignore regions in annotation.
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
        version: str = "v1",
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
        self.kwargs["with_img_buf"] = self.with_img_buf
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
        if self.to_rgb:
            # cv2.cvtColor may be slow.
            # See http://wiki.hobot.cc/pages/viewpage.action?pageId=186775106 for more details.     # noqa
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        for instance_idx in np.random.choice(
            len(anno["instances"]),
            min(len(anno["instances"]), self.maximum_instances_per_image),
            replace=False,
        ):
            data = dummy_data.copy()
            if self.task_type == "detection":
                gt_bboxes = []
                gt_classes = []
                points_data = anno["instances"][instance_idx]["points_data"]
                x1_crop, y1_crop = points_data[10]
                x2_crop, y2_crop = points_data[12]
                crop_roi = [x1_crop, y1_crop, x2_crop, y2_crop]

                for ins in anno["instances"]:
                    is_hard = ins["is_hard"][0]
                    points_data = ins["points_data"]
                    class_id = int(ins["class_id"][0])
                    bbox = []
                    bbox.extend(points_data[0])
                    bbox.extend(points_data[2])
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
                        if not (
                            left_top == [0, 0]
                            and right_bottom == [anno["img_w"], anno["img_h"]]
                        ):  # must be first n in anno["ignore_regions"]
                            ig_bbox = left_top + right_bottom
                            ig_bboxes.append(ig_bbox)
                        else:
                            break
                data["ig_bboxes"] = ig_bboxes
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
                x1_crop, y1_crop = points_data[0]
                x2_crop, y2_crop = points_data[2]
                crop_roi = [x1_crop, y1_crop, x2_crop, y2_crop]

                is_hard = ins["is_hard"][0]
                if self.version == "v1":
                    class_id = int(ins["class_id"][0])
                elif self.version == "v2":
                    ins_attr = ins.get("attribute", [])
                    # 打包的时候，这里的attribute减过1了，为了和class_id处理保持一致，
                    # 这里先＋1
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


def build_dataset(
    data_path: str,
    anno_path: str,
    rec_idx_file_path: Optional[str] = None,
    disable_default_densebox_log: Optional[bool] = True,
    read_only: Optional[bool] = False,
    **kwargs,
):
    """
    Build dataset.

    Args:
        disable_default_densebox_log: Disable default print output from
            `LegacyDenseBoxImageRecordDataset`. Default is True.
        data_path : Path of data relative to buket path.
        anno_path : Path of annotation.
        rec_idx_file_path: index file related to data_path. Used only when
            there is already index file somewhere.
        kwargs : Kwargs for build dataset.
    """
    try:
        if disable_default_densebox_log:
            temp_fid = tempfile.NamedTemporaryFile("w")
            with contextlib.redirect_stdout(temp_fid):
                dataset = LegacyDenseBoxImageRecordDataset(
                    rec_path=data_path,
                    anno_path=anno_path,
                    rec_idx_file_path=rec_idx_file_path,
                    # we do to_rgb below
                    to_rgb=False,
                    read_only=read_only,
                    **kwargs,
                )
        else:
            dataset = LegacyDenseBoxImageRecordDataset(
                rec_path=data_path,
                anno_path=anno_path,
                rec_idx_file_path=rec_idx_file_path,
                # we do to_rgb below
                to_rgb=False,
                read_only=read_only,
                **kwargs,
            )
        return dataset
    except TypeError as e:
        logging.error("Please update auto_matrix >= 0.4.6b202104231524")
        raise e


@OBJECT_REGISTRY.register
class MulticlassDenseboxDataset(DenseboxDataset):
    """Subclass from DenseboxDataset, extend for multiclass labeled dataset.

    Args:
        data_path : Path of data relative to buket path.
        anno_path : Path of annotation.
        class_id_map : map each class's rec class id to used category.
        transforms : List of transform.
        to_rgb: Convert bgr(cv2 imread) to rgb.
        task_type : Consist of 'detection', 'segmentation'
        rec_idx_file_path: index file related to data_path. Used only when
            there is already index file somewhere.
        disable_default_densebox_log: Disable default print output from
            `LegacyDenseBoxImageRecordDataset`. Default is True.
        ignore_hard : Ignore hard instances if `hard` tag in annotation.
            Default is False.
        use_ignore : Whether to use ignore regions in annotation.
            Default is False.
    """

    def __init__(
        self,
        data_path: str,
        anno_path: str,
        class_id_map: Dict[int, int],
        transforms: Optional[List] = None,
        to_rgb: Optional[bool] = False,
        task_type: Optional[str] = "detection",
        rec_idx_file_path: Optional[str] = None,
        disable_default_densebox_log: Optional[bool] = True,
        ignore_hard: Optional[bool] = False,
        use_ignore: Optional[bool] = False,
    ):
        super().__init__(
            data_path,
            anno_path,
            transforms,
            to_rgb,
            task_type,
            rec_idx_file_path=rec_idx_file_path,
            disable_default_densebox_log=disable_default_densebox_log,
            ignore_hard=ignore_hard,
            use_ignore=use_ignore,
        )
        self.class_id_map = class_id_map

    def __getitem__(self, index: int) -> Dict:
        image, anno = self.dataset[index]
        color_space = "bgr"
        if self.to_rgb:
            # cv2.cvtColor may be slow.
            # See http://wiki.hobot.cc/pages/viewpage.action?pageId=186775106 for more details.  # noqa
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color_space = "rgb"
        if self.task_type == "detection":
            anno = anno
        elif self.task_type == "segmentation":
            seg_label = anno[1]
            seg_label = seg_label.astype(np.uint8)
            anno = anno[0]
        anno = anno.to_dict()
        data = {}
        data["img_name"] = anno["img_url"].split("/")[-1]
        data["img_height"] = anno["img_h"]
        data["img_width"] = anno["img_w"]
        data["img_id"] = np.expand_dims(anno["idx"], 0)
        data["img"] = image
        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image.shape
        if self.task_type == "detection":
            gt_bboxes = []
            gt_classes = []
            gt_labels = [l for _, l in self.class_id_map.items()]
            data["gt_labels"] = np.array(gt_labels, dtype=np.int64)
            for ins in anno["instances"]:
                is_hard = ins["is_hard"][0]
                points_data = ins["points_data"]
                class_id = int(ins["class_id"][0])
                bbox = []
                bbox.extend(points_data[0])
                bbox.extend(points_data[2])
                if class_id in self.class_id_map:
                    gt_class = self.class_id_map[class_id]
                    if is_hard and self.ignore_hard:
                        gt_class = -1 * (gt_class + 1)
                    gt_classes.append(gt_class)
                else:
                    continue
                gt_bboxes.append(bbox)
            data["gt_bboxes"] = np.array(gt_bboxes)
            data["gt_classes"] = np.array(gt_classes, dtype=np.int64)

            if self.use_ignore:
                ig_bboxes = []
                for ig_region in anno["ignore_regions"]:
                    left_top = ig_region["contour"][0]
                    right_bottom = ig_region["contour"][1]
                    if not (
                        left_top == [0, 0]
                        and right_bottom == [anno["img_w"], anno["img_h"]]
                    ):  # must be first n in anno["ignore_regions"]
                        ig_bbox = left_top + right_bottom
                        ig_bboxes.append(ig_bbox)
                    else:
                        break
                data["ig_bboxes"] = ig_bboxes

        elif self.task_type == "segmentation":
            data["gt_seg"] = seg_label

        if self.transforms is not None:
            data = self.transforms(data)
        return data


def get_idx_path(rec_path: str, root: Optional[str] = None) -> str:
    """
    Get index file path based on rec file and root.

    If root is not provided, idx file will be saved in /cluster_home or /tmp.
    """
    existed_idx_file = rec_path + ".idx"
    if os.path.exists(existed_idx_file):
        return existed_idx_file
    idx_file_name = rec_path.replace("/", "_") + ".idx"
    username = getpass.getuser()
    if root:
        idx_path = os.path.join(root, idx_file_name)
    else:
        if os.path.exists("/cluster_home"):
            root = "/cluster_home/idx_files_genereated_by_hat"
        else:
            root = f"/tmp/idx_files_genereated_by_hat_{username}"
        if not os.path.exists(root):
            os.makedirs(root)
        idx_path = os.path.join(root, idx_file_name)
    return idx_path


@OBJECT_REGISTRY.register
class VehicleSideDenseboxDataset(DenseboxDataset):
    """Dataset for densebox record data in auto, such as adas-mini.

    Args:
        data_path : Path of data relative to buket path.
        anno_path : Path of annotation.
        transforms : List of transform.
        to_rgb: Convert bgr(cv2 imread) to rgb.
        task_type : Consist of 'detection', 'segmentation'
        class_id : the rec's class id, 1base
        category : the used category, 0base
        rec_idx_file_path: index file related to data_path. Used only when
            there is already index file somewhere.
        disable_default_densebox_log: Disable default print output from
            `LegacyDenseBoxImageRecordDataset`. Default is True.
        ignore_hard : Ignore hard instances if `hard` tag in annotation.
            Default is False.
    """

    def __getitem__(self, index: int) -> Dict:
        image, anno = self.dataset[index]
        color_space = "bgr"
        if self.to_rgb:
            # cv2.cvtColor may be slow.
            # See http://wiki.hobot.cc/pages/viewpage.action?pageId=186775106 for more details.     # noqa
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color_space = "rgb"
        if self.task_type == "detection":
            anno = anno
        elif self.task_type == "segmentation":
            seg_label = anno[1]
            seg_label = seg_label.astype(np.uint8)
            anno = anno[0]
        anno = anno.to_dict()
        data = {}
        data["img_name"] = anno["img_url"].split("/")[-1]
        data["img_height"] = anno["img_h"]
        data["img_width"] = anno["img_w"]
        data["img_id"] = np.expand_dims(anno["idx"], 0)
        data["img"] = image
        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image.shape
        if self.return_orig_img:
            data["orig_img"] = image.copy()

        if self.task_type == "detection":
            gt_bboxes = []
            gt_classes = []
            gt_tanalphas = []
            for ins in anno["instances"]:
                is_hard = ins["is_hard"][0]
                points_data = ins["points_data"]
                assert len(points_data) == 10
                class_id = int(ins["class_id"][0])
                bbox = []
                bbox.extend(points_data[0])
                bbox.extend(
                    [
                        points_data[2][0],
                        (points_data[2][1] + points_data[3][1]) / 2.0,
                    ]
                )
                # tanalpha = points_data[-1]
                tanalpha = (points_data[2][1] - points_data[3][1]) / (
                    points_data[2][0] - points_data[3][0]
                )
                gt_bboxes.append(bbox)
                gt_tanalphas.append(tanalpha)
                if class_id == self.class_id:
                    if is_hard and self.ignore_hard:
                        gt_classes.append(-1)
                    else:
                        gt_classes.append(self.category)
                else:
                    gt_classes.append(-1)
            if self.extend_ignore_region_into_gtbox:
                for ignore in anno.get("ignore_regions", []):
                    class_id = int(ignore["class_id"][0])
                    contour = ignore["contour"]
                    if class_id != self.class_id:
                        continue
                    bbox = []
                    bbox.extend(contour[0])
                    bbox.extend(contour[1])
                    gt_bboxes.append(bbox)
                    gt_classes.append(-1)
                    gt_tanalphas.append(0.0)
            # data["gt_bboxes"].shape = (n, 4)
            data["gt_bboxes"] = np.array(gt_bboxes)
            # data["gt_tanalphas"].shape = (n,)
            data["gt_tanalphas"] = np.array(gt_tanalphas)
            # data["gt_classes"].shape = (n,)
            data["gt_classes"] = np.array(gt_classes, dtype=np.int64)
        elif self.task_type == "segmentation":
            data["gt_seg"] = seg_label

        if self.transforms is not None:
            data = self.transforms(data)
        return data
