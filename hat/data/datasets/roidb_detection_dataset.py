# Copyright (c) Horizon Robotics. All rights reserved.
"""Dataset for densebox mx-record data, used in auto."""
import logging
import os
import pickle
from copy import deepcopy
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch.utils.data as data

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type import PackTypeMapper
from hat.utils.pack_type.mxrecord import unpack_img

__all__ = ["RoidbDetectionDataset", "RoidbDataset"]


@OBJECT_REGISTRY.register
class RoidbDataset(object):
    """A dataset that can read roidb (pickle) and the relation image record \
        (mxnet record) dataset.

    Args:
        roidb_path: Gt roidb path.
        rec_path: Image record path.
        rec_idx_file_path: Image record index file path,
            default to None;
            if None use: the value will be rec_path.replace('.rec', '.idx')
        rec_lst_file_path: Image record list file path,
            default to None;
            if None use: the value will be rec_path.replace('.rec', '.lst')
        cv_format: Image format option for ``cv2.imdecode``;
            Default to ``cv2.IMREAD_COLOR``.
        data_desc: Data description, used in evaluation process (datasets
            will be filtered and gathered by data_desc)
            Default to ``None``.
    """

    def __init__(
        self,
        roidb_path: str,
        rec_path: str,
        rec_idx_file_path: Optional[str] = None,
        rec_lst_file_path: Optional[str] = None,
        cv_format: Optional[int] = cv2.IMREAD_COLOR,
        data_desc: Optional[str] = None,
    ):
        self.roidb_path = roidb_path
        self.pack_type = PackTypeMapper["mxrecord"]
        self.rec_path = rec_path
        self.rec_idx_file_path = (
            rec_idx_file_path
            if rec_idx_file_path is not None
            else rec_path.replace(".rec", ".idx")
        )
        if not os.path.exists(self.rec_idx_file_path):
            raise FileNotFoundError(
                "rec idx file (%s) not found! " % self.rec_idx_file_path
            )
        self.rec_lst_file_path = (
            rec_lst_file_path
            if rec_lst_file_path is not None
            else rec_path.replace(".rec", ".lst")
        )
        if not os.path.exists(self.rec_lst_file_path):
            raise FileNotFoundError(
                "rec lst file (%s) not found! " % self.rec_lst_file_path
            )
        self.cv_format = cv_format
        self.data_desc = data_desc
        self._init_pack()

    def _init_pack(self):
        self.anno_dataset = self._get_anno_dataset(self.roidb_path)
        self.pack_file = self.pack_type(
            uri=self.rec_path, idx_path=self.rec_idx_file_path, writable=False
        )
        self.pack_file.open()
        self.img_lst = self._read_lst(self.rec_lst_file_path)
        self.add_img_ind_in_rec_for_roidb()
        self._set_group_flag()

    def _get_anno_dataset(self, anno_path):
        with open(anno_path, "rb") as fn:
            anno = pickle.load(fn, encoding="latin1")
            return self._rename_keys(anno)

    def _rename_keys(self, anno):
        renamed_keys = {
            "image": "image_name",
            "height": "image_height",
            "width": "image_width",
            "gt_classes": "classes",
        }
        unused_keys = {
            "degree",
            "flipped",
            "rotation",
            "upper_body",
            "seg_label",
            "seg_label_name",
            "gt_masks",
            "masks",
            "id",
            "reid",
            "attrs",
        }
        for gt_roi in anno:
            for old_name in renamed_keys:
                if old_name in gt_roi:
                    new_name = renamed_keys[old_name]
                    gt_roi[new_name] = gt_roi.pop(old_name)
            for old_name in unused_keys:
                if old_name in gt_roi:
                    del gt_roi[old_name]
        return anno

    def _read_lst(self, lst_path):
        img_lst = {}
        with open(lst_path, "r", encoding="utf-8") as fin:
            for line in iter(fin.readline, ""):
                try:
                    line = line.decode("utf-8")
                except AttributeError:
                    pass
                line = line.strip().split("\t")
                image_index = line[0]
                image_name = line[-1]
                assert image_name not in img_lst
                img_lst[image_name] = int(image_index)
        return img_lst

    def add_img_ind_in_rec_for_roidb(
        self,
    ):
        img_lst = self.img_lst.keys()
        self.img_idx_2_anno_idx = {}
        for anno_idx, _anno in enumerate(self.anno_dataset):
            image_name = _anno["image_name"]
            if image_name not in img_lst:
                image_name = os.path.basename(image_name)
                assert image_name in img_lst, "{} {}".format(
                    image_name, _anno["image_name"]
                )
            img_idx = self.img_lst[image_name]
            if "image_index" in _anno:
                assert _anno["image_index"] == img_idx
            else:
                _anno["image_index"] = img_idx

            # In negtive dataset, may have repeat negative annotation in
            # one image. This means that may have repeat anno_idx
            # match to one same img_idx. This wouldn't appear in
            # tracking dataset.
            self.img_idx_2_anno_idx[img_idx] = anno_idx

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 (w/h > 1) will be
        set as group 1, otherwise set as group 0 (h/w >= 1,
        usually is open source dataset).
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            anno = self.anno_dataset[i]
            if anno["image_width"] / anno["image_height"] > 1:
                self.flag[i] = 1

    def __getstate__(self):
        state = self.__dict__
        state["pack_file"] = None
        state["anno_dataset"] = None
        state["img_lst"] = None
        state["flag"] = None
        state["img_idx_2_anno_idx"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._init_pack()

    def __getitem__(self, idx):
        anno = self.anno_dataset[idx]
        if self.data_desc is not None:
            anno["data_desc"] = self.data_desc
        raw_record = self.pack_file.read(anno["image_index"])
        _, img = unpack_img(raw_record, self.cv_format)
        return img, anno

    def __len__(self):
        return len(self.anno_dataset)


@OBJECT_REGISTRY.register
class RoidbDetectionDataset(data.Dataset):
    """Dataset for detection roidb & record data.

    Args:
        data_path: Path of data relative to bucket path.
        anno_path: Path of annotation.
        selected_class_ids: List of selected class ids,
            classes that are not in this list will be filter out.
        num_point_per_person: Number points per person used in data,
            Default is 15.
        ldmk_pairs: List of ldmk pairs.
        ignore_spec_cls_ids: List of class ids to be ignored if there
            is no corresponding objects existing in image. Used in case
            of no class gt label in image. (add a whole image region
            ignored bbox).
            Default to None.
        transforms: List of transform,
            default to None.
        to_rgb: If convert bgr (cv2 imread) to rgb,
            default to False.
        min_size: min size for bbox width and height,
            size smaller than this value will be filter out, default to 2.
        keep_ori_img: If keep original rgb image to data, used to
            visualize the predictor result.
            Default to False.
        data_desc: Data description, used in evaluation process (datasets
            will be filtered and gathered by data_desc);
            Default to ``None``.
    """

    def __init__(
        self,
        data_path: str,
        anno_path: str,
        selected_class_ids: List[int],
        num_point_per_person: int = 15,
        ldmk_pairs: Optional[List] = None,
        ignore_spec_cls_ids: Optional[List[int]] = None,
        transforms: Optional[List] = None,
        to_rgb: Optional[bool] = False,
        min_size: Optional[int] = 2,
        keep_ori_img: Optional[bool] = False,
        data_desc: Optional[str] = None,
    ):
        self.data_path = data_path
        self.anno_path = anno_path
        self.num_point_per_person = num_point_per_person
        self.ldmk_pairs = ldmk_pairs
        self.data_desc = data_desc
        self.keep_ori_img = keep_ori_img
        self.transforms = transforms
        self.to_rgb = to_rgb
        self.min_size = min_size

        self.valid_selected_class_ids = set(
            {id for id in selected_class_ids if id > 0}
        )
        # class id should begin from 1
        self.class_id_map = dict(
            zip(
                self.valid_selected_class_ids,
                range(1, len(self.valid_selected_class_ids) + 1),
            )
        )
        self.ignore_spec_cls_ids = ignore_spec_cls_ids
        self._init_dataset()

    def _init_dataset(self):
        self.dataset = RoidbDataset(
            roidb_path=self.anno_path,
            rec_path=self.data_path,
            data_desc=self.data_desc,
        )
        if (self.ignore_spec_cls_ids is not None) and (
            len(self.ignore_spec_cls_ids) > 0
        ):
            self._ignore_specific_classes(self.ignore_spec_cls_ids)

        self._set_group_flag()

        logging.debug(f"dataset path: {self.data_path}, {self.anno_path}")
        logging.debug(f"dataset length: {len(self.dataset)}")
        logging.debug(
            f"dataset h/w >=1 fraction: {(self.flag==0).sum() / (len(self.dataset) + 1e-6)}\n"  # noqa: E501
        )

    def _ignore_specific_classes(self, ignored_classes):
        """Set specific classes ignored in the whole image region."""
        assert isinstance(ignored_classes, (list, tuple)) and isinstance(
            ignored_classes[0], int
        )
        for _anno in self.dataset.anno_dataset:
            classes = _anno["classes"]
            for ignored_class in ignored_classes:
                # if ignore_class does not exist in classes, add it
                if (ignored_class not in classes) and (
                    -ignored_class not in classes
                ):
                    ignore_box = np.array(
                        [
                            [
                                0,
                                0,
                                _anno["image_width"] - 1,
                                _anno["image_height"] - 1,
                            ]
                        ],
                        dtype=_anno["boxes"].dtype,
                    )
                    ignore_cls = np.array(
                        [-ignored_class], dtype=_anno["classes"].dtype
                    )
                    _anno["boxes"] = np.vstack([_anno["boxes"], ignore_box])
                    _anno["classes"] = np.hstack(
                        [_anno["classes"], ignore_cls]
                    )

    def _clip_box(self, boxes, width, height):
        boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], width - 1), 0)
        boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], height - 1), 0)
        boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], width - 1), 0)
        boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], height - 1), 0)
        return boxes

    def __getstate__(self):
        state = self.__dict__
        state["dataset"] = None
        state["flag"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._init_dataset()

    def __getitem__(self, index: int) -> Dict:
        data = {}

        image, anno = self.dataset[index]
        color_space = "bgr"
        if self.keep_ori_img:
            data["ori_img"] = deepcopy(image)
        if self.to_rgb:
            # cv2.cvtColor may be slow.
            # See http://wiki.hobot.cc/pages/viewpage.action?pageId=186775106 for more details.     # noqa
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color_space = "rgb"

        # cls_name_id_map = {
        #     1: "person",
        #     2: "head",
        #     3: "face",
        #     4: "hand",
        #     5: "cat",
        #     6: "dog",
        #     7: "bottle",
        #     8: "phone",
        # }
        person_cls_id = 1
        num_person = (
            (anno["classes"] == person_cls_id)
            | (anno["classes"] == -person_cls_id)
        ).sum()
        if "keypoints" not in anno:
            # per point: (x, y, status)
            kps = np.zeros(
                (num_person, self.num_point_per_person * 3), dtype=np.float32
            )
            # 'invisible': 0, 'occluded': 1, 'full_visible': 2, 'ignore': 3
            kps[:, 2::3] = 3
        else:
            kps = anno["keypoints"]
            if (
                (kps.shape[0] == 0)
                or (kps.shape[1] != self.num_point_per_person * 3)
                or (kps.shape[0] != num_person)
            ):
                kps = np.zeros(
                    (num_person, self.num_point_per_person * 3),
                    dtype=np.float32,
                )
                # 'invisible': 0, 'occluded': 1, 'full_visible': 2, 'ignore': 3
                kps[:, 2::3] = 3

        kps_idx = -1

        if self.data_desc is not None:
            data["data_desc"] = anno["data_desc"]
        data["img_name"] = anno["image_name"]
        data["img_height"] = anno["image_height"]
        data["img_width"] = anno["image_width"]
        img_id = anno.get("image_index", 0)
        data["img_id"] = np.expand_dims(img_id, 0)
        data["img"] = image
        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image.shape

        gt_bboxes = []
        gt_classes = []
        gt_ldmks = []
        _boxes = self._clip_box(
            anno["boxes"], data["img_width"], data["img_height"]
        )
        _classes = anno["classes"]
        for bbox, class_id in zip(_boxes, _classes):
            if (class_id == person_cls_id) or (class_id == -person_cls_id):
                kps_idx += 1

            # filter [0, 0, 1, 1] ignored region and other invalid boxes
            if (bbox[2] - bbox[0] < self.min_size) or (
                bbox[3] - bbox[1] < self.min_size
            ):
                continue

            if class_id in self.valid_selected_class_ids:
                gt_bboxes.append(bbox)
                gt_classes.append(self.class_id_map[class_id])
                if (class_id == person_cls_id) or (class_id == -person_cls_id):
                    gt_ldmks.append(kps[kps_idx])
            elif -class_id in self.valid_selected_class_ids:
                gt_bboxes.append(bbox)
                gt_classes.append(-self.class_id_map[-class_id])
                if (class_id == person_cls_id) or (class_id == -person_cls_id):
                    gt_ldmks.append(kps[kps_idx])

        data["gt_bboxes"] = np.array(gt_bboxes).reshape((-1, 4))
        data["gt_classes"] = np.array(gt_classes, dtype=np.int64).reshape(
            (-1,)
        )
        data["gt_ldmk"] = np.array(gt_ldmks, dtype=np.float32).reshape(
            (-1, self.num_point_per_person, 3)
        )
        data["ldmk_pairs"] = self.ldmk_pairs
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def _set_group_flag(self):
        assert hasattr(self.dataset, "flag"), "dataset must has group flag"
        self.flag = self.dataset.flag

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return "RoidbDetectionDataset"
