# Copyright (c) Horizon Robotics. All rights reserved.
"""Dataset for densebox mx-record data, used in auto."""
import contextlib
import json
import logging
import os
import random
import tempfile
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch.utils.data as data

from hat.core.anno_ts_utils import ImageRois, get_img_idx_to_img_rois_map
from hat.core.compose_transform import Compose
from hat.data.datasets.densebox_dataset import get_idx_path
from hat.data.datasets.legacy_densebox import (
    HAT_LEGACYDENSEBOX_AVAILABLE,
    LegacyDenseBoxImageRecordDataset,
)
from hat.data.transforms.affine import ImageAffineTransform
from hat.data.transforms.detection import _transform_bboxes, pad_detection_data
from hat.data.transforms.roi_crop import (
    BBoxTransformParm,
    CropParm,
    ImgCropParm,
    RoiTransformer,
)
from hat.registry import OBJECT_REGISTRY, build_from_registry
from hat.utils.pack_type.mxrecord import MXRecord, MXRecordIO, unpack
from hat.utils.pack_type.recordio_pb2 import RecordUnit


@OBJECT_REGISTRY.register
class LegacyDenseBoxWithRoilistImageRecordDataset(
    LegacyDenseBoxImageRecordDataset
):
    """A dataset that can read densebox image record.

    Args:
        roi_list_path: Roi list file path
        **kwargs: See :py:class:'LegacyDenseBoxImageRecordDataset'
    """

    def __init__(
        self,
        rec_path: str,
        anno_path: str,
        roi_list_path: str = None,
        read_only: bool = False,
        with_img_buf: bool = False,
        with_seg_label: bool = False,
        to_rgb: bool = True,
        seg_label_dtype: type = np.uint8,
        rec_idx_file_path: str = None,
    ):
        super(LegacyDenseBoxWithRoilistImageRecordDataset, self).__init__(
            rec_path,
            anno_path,
            read_only,
            with_img_buf,
            with_seg_label,
            to_rgb,
            rec_idx_file_path,
            seg_label_dtype,
        )

        self._roi_list_dataset = None
        if roi_list_path is not None:
            self._roi_list_dataset = self._get_roi_list_dataset(
                roi_list_path, anno_path
            )
            assert len(self._roi_list_dataset) == len(self._anno_dataset)

    def __getitem__(self, idx):
        result = {}
        if self.with_img_buf:
            image, img_buf, anno = super().__getitem__(idx)
            result["img_bug"] = img_buf
        else:
            image, anno = super().__getitem__(idx)
        result["img"] = image
        result["anno"] = anno
        if self._roi_list_dataset is not None:
            if isinstance(self._roi_list_dataset, dict):
                roi_list_labels = self._roi_list_dataset[idx]
            else:
                roi_list_labels = self._roi_list_dataset.read(idx)
                roi_list_labels = self._decoder_roilist(roi_list_labels)
            result["roi_list"] = roi_list_labels
        return result

    def _get_roi_list_dataset(self, roi_list_path, anno_path):
        ext = os.path.splitext(roi_list_path)[1]
        anno_ext = os.path.splitext(anno_path)[1]

        if ext in [".json"]:
            if anno_ext is not None and anno_ext in [".rec", ".pb_rec"]:
                anno_path = anno_path.replace(".anno.pb_rec", ".json")
            img_idx2roilist_map = get_img_idx_to_img_rois_map(
                roi_list_path, anno_path
            )  # noqa
            idx2roilist_map = {}
            for idx in range(len(self._rec_dataset)):
                raw_record = self._rec_dataset.read(idx)
                header, _ = unpack(raw_record)
                idx2roilist_map[idx] = img_idx2roilist_map[header.id]

            return idx2roilist_map

        elif ext in [".rec", ".pb_rec"] and anno_ext in [".rec", ".pb_rec"]:
            roi_list_idx_path = roi_list_path + ".idx"
            if not os.path.exists(roi_list_idx_path):
                recode_tmp = MXRecordIO(self.rec_path, "r")
                recode_tmp.create_idx_file(roi_list_idx_path)
                recode_tmp.close()
            return MXRecord(
                uri=roi_list_path, idx_path=roi_list_idx_path, writable=False
            )

        else:
            raise ValueError("Invalid roi_list_path %s" % roi_list_path)

    def _decoder_roilist(self, record):
        if self.read_only:
            return record
        _, s = unpack(record)
        rec_data = RecordUnit()
        rec_data.ParseFromString(s)
        rec_data = rec_data.body
        assert len(rec_data.data) in [2]
        assert len(rec_data.extra) == 0
        image_rois = ImageRois(
            json_str=json.loads(bytes.decode(rec_data.data[1].value)),
            force_utf8=False,
        )
        return image_rois


@OBJECT_REGISTRY.register
class DenseboxWithRoilistDataset(data.Dataset):
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
    """

    def __init__(
        self,
        data_path: str,
        anno_path: str,
        roi_list_path: Optional[str] = None,
        transforms: Optional[List] = None,
        to_rgb: Optional[bool] = False,
        task_type: Optional[str] = "detection",
        class_id: Optional[int] = -1,
        category: Optional[int] = -1,
        rec_idx_file_path: Optional[str] = None,
        disable_default_densebox_log: Optional[bool] = True,
        with_img_buf: Optional[bool] = False,
        return_orig_img: Optional[bool] = False,
        return_orig_gt_seg: Optional[bool] = False,
        remove_det_duplicate: Optional[bool] = False,
    ):
        assert (
            HAT_LEGACYDENSEBOX_AVAILABLE
        ), "horizon_plugin_pytorch >= 1.0.0 is required."

        self.data_path = data_path
        self.anno_path = anno_path
        self.roi_list_path = roi_list_path
        self.transforms = transforms
        self.to_rgb = to_rgb
        self.task_type = task_type
        self.class_id = class_id
        self.category = category

        self.rec_idx_file_path = rec_idx_file_path
        self.disable_default_densebox_log = disable_default_densebox_log
        self.with_img_buf = with_img_buf
        self.return_orig_img = return_orig_img
        self.return_orig_gt_seg = return_orig_gt_seg
        self.remove_det_duplicate = remove_det_duplicate

        if self.rec_idx_file_path is None:
            self.rec_idx_file_path = get_idx_path(self.data_path)

        if self.task_type == "detection":
            self.kwargs = {}
        elif self.task_type == "segmentation":
            self.kwargs = {"with_seg_label": True, "seg_label_dtype": np.int8}
        else:
            raise Exception(
                "error task_type, your task_type[{}],"
                " we need segmentation or detection".format(self.task_type)
            )
        self.kwargs["with_img_buf"] = self.with_img_buf
        self.dataset = build_roilist_dataset(
            self.data_path,
            self.anno_path,
            self.roi_list_path,
            self.rec_idx_file_path,
            self.disable_default_densebox_log,
            **self.kwargs,
        )
        logging.info(
            f"dataset path: {self.data_path}, {self.anno_path}, {self.roi_list_path}"  # noqa
        )
        logging.info(f"dataset length: {len(self.dataset)}")

    def __getitem__(self, index: int) -> Dict:
        result = self.dataset[index]
        image, anno = result["img"], result["anno"]
        if self.with_img_buf:
            img_buf = result["img_buf"]
        roi_list = result.get("roi_list", None)
        anno = anno.to_dict()
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
        if self.task_type == "detection":
            anno = anno
        elif self.task_type == "segmentation":
            seg_label = anno[1]
            seg_label = seg_label.astype(np.uint8)
            anno = anno[0]
            data["gt_seg"] = seg_label
            if self.return_orig_gt_seg:
                data["orig_gt_seg"] = seg_label.copy()
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
        data["anno"] = anno
        data["roi_list"] = roi_list

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return "DenseboxWithRoilistDataset"

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
        roi_list_path = self.__dict__["roi_list_path"]

        self.__dict__["dataset"] = build_roilist_dataset(
            data_path,
            anno_path,
            roi_list_path,
            rec_idx_file_path,
            disable_default_densebox_log,
            **kwargs,
        )


def build_roilist_dataset(
    data_path: str,
    anno_path: str,
    roi_list_path: Optional[str] = None,
    rec_idx_file_path: Optional[str] = None,
    disable_default_densebox_log: Optional[bool] = True,
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
                dataset = LegacyDenseBoxWithRoilistImageRecordDataset(
                    rec_path=data_path,
                    anno_path=anno_path,
                    roi_list_path=roi_list_path,
                    rec_idx_file_path=rec_idx_file_path,
                    # we do to_rgb below
                    to_rgb=False,
                    **kwargs,
                )
        else:
            dataset = LegacyDenseBoxWithRoilistImageRecordDataset(
                rec_path=data_path,
                anno_path=anno_path,
                roi_list_path=roi_list_path,
                rec_idx_file_path=rec_idx_file_path,
                # we do to_rgb below
                to_rgb=False,
                **kwargs,
            )
        return dataset
    except TypeError as e:
        logging.error("Please update auto_matrix >= 0.4.6b202104231524")
        raise e


@OBJECT_REGISTRY.register
class RoiTransformCroper(object):
    """
    transformer base on rois for object detection.

    Parameters
    ----------
    resize_wh : list/tuple of 2 int, optional
        Resize input image to target size, by default None
    **kwargs :
        Please see :py:class:`AffineMatFromROIBoxGenerator` and
        :py:class:`ImageAffineTransform`
    """

    def __init__(
        self,
        min_sample_num,
        max_sample_num,
        roi_crop_parm: CropParm or dict,
        img_crop_parm: ImgCropParm or dict,
        bbox_ts_parm: BBoxTransformParm or dict,
        from_roi_ratio: float = 1,
        hard_sample_select_ratio: float = 0,
        transforms: list or dict = None,
    ):
        self.min_sample_num = min_sample_num
        self.max_sample_num = max_sample_num
        if isinstance(img_crop_parm, dict):
            img_crop_parm = ImgCropParm.from_dict(img_crop_parm)
        if isinstance(roi_crop_parm, dict):
            roi_crop_parm = CropParm.from_dict(roi_crop_parm)
        if isinstance(bbox_ts_parm, dict):
            bbox_ts_parm = BBoxTransformParm.from_dict(bbox_ts_parm)
        self._img_ts = ImageAffineTransform(
            dst_wh=img_crop_parm.target_wh,
            inter_method=img_crop_parm.inter_method,
            border_value=0,
            use_pyramid=img_crop_parm.use_pyramid,
            pyramid_min_step=img_crop_parm.pyramid_min_step,
            pyramid_max_step=img_crop_parm.pyramid_max_step,
            pixel_center_aligned=img_crop_parm.pixel_center_aligned,
        )
        self._bbox_ts_kwargs = bbox_ts_parm.to_dict()
        self._roi_crop_parm = roi_crop_parm
        self.label_type = self._bbox_ts_kwargs.pop("label_type")
        assert self.label_type in ["detection", "classification"]

        assert self.label_type in ["detection"]
        self.from_roi_ratio = from_roi_ratio
        self.hard_sample_select_ratio = hard_sample_select_ratio
        if transforms is not None:
            transforms = build_from_registry(transforms)
            if isinstance(transforms, (list, tuple)):
                transforms = Compose(transforms)  # noqa
        self.transforms = transforms

    def __call__(self, data):
        img, gt_boxes = data["img"], data["gt_boxes"]
        ig_regions, roi_lists = data["ig_regions"], data["roi_list"]
        src_h, src_w, _ = img.shape
        self._roi_crop_parm.input_wh = [src_w, src_h]

        has_roi_list = False
        if roi_lists is not None and len(roi_lists.rois) > 0:
            has_roi_list = True
        from_roi = False
        if random.random() < self.from_roi_ratio:
            from_roi = True

        roi_idxs = []
        use_roi_list_crop_img = False
        if has_roi_list and from_roi:
            rois_num = len(roi_lists.rois)
            if rois_num < self.min_sample_num:
                num = np.int64(self.min_sample_num / rois_num) + 1
                roi_idxs = np.arange(num) % rois_num
            elif rois_num <= self.max_sample_num:
                roi_idxs = np.arange(rois_num)
            else:
                roi_idxs = np.arange(rois_num)
                random.shuffle(roi_idxs)
                roi_idxs = roi_idxs[: self.max_sample_num]
            use_roi_list_crop_img = True
        elif len(gt_boxes) > 0:
            rois_num = len(gt_boxes)
            if rois_num < self.min_sample_num:
                num = np.int64(self.min_sample_num / rois_num) + 1
                roi_idxs = np.arange(num) % rois_num
            elif rois_num < self.max_sample_num:
                roi_idxs = np.arange(rois_num)
            else:
                roi_idxs = np.arange(rois_num)
                random.shuffle(roi_idxs)
                roi_idxs = roi_idxs[: self.max_sample_num]
        gt_boxes_tmp = gt_boxes.copy()
        results = []
        for roi_idx in roi_idxs:
            affine_aug_param = None
            if use_roi_list_crop_img:
                roi = roi_lists.rois[roi_idx]
                assert roi.pt1 != [-1, -1]
                affine_aug_param = (
                    self.get_roi_transform_mat_wrt_instance_from_roi_list(
                        self._roi_crop_parm, roi.pt1, roi.pt2, roi.center
                    )
                )
            else:
                roi = gt_boxes_tmp[roi_idx]
                affine_aug_param = self.get_roi_transform_mat_wrt_instance(
                    self._roi_crop_parm, roi[:4]
                )

            ts_img, ts_gt_boxes, ts_ig_regions = self.affine_img_and_bboxes(
                img, gt_boxes, ig_regions, affine_aug_param
            )

            # ################test chenhao
            if os.getenv("TEST_CHENHAO", "0") == "1":
                import copy
                import uuid

                import cv2

                colors = [(0, 0, 255), (0, 255, 0)]

                vis_path = "/jfs-public/users/lele.liu/works/tmp/vis"
                save_name = os.path.join(
                    vis_path, f"ID_{uuid.uuid4().hex[:10]}.jpg"
                )
                draw_img = copy.deepcopy(ts_img[:, :, ::-1])
                for box in ts_gt_boxes:
                    x1, y1, x2, y2, class_id = box[:]
                    color = (0, 0, 0)
                    if np.abs(class_id) == 1:
                        color = colors[0]
                    elif np.abs(class_id) == 2:
                        color = colors[1]

                    cv2.rectangle(
                        draw_img,
                        pt1=(int(x1), int(y1)),
                        pt2=(int(x2), int(y2)),
                        color=color,
                        thickness=3 if class_id > 0 else 1,
                    )
                for box in ts_ig_regions:
                    x1, y1, x2, y2, _ = box[:]
                    cv2.rectangle(
                        draw_img,
                        pt1=(int(x1), int(y1)),
                        pt2=(int(x2), int(y2)),
                        color=(255, 255, 0),
                    )
                cv2.imwrite(save_name, draw_img)
                print(f"write finished:{save_name}")
            result = pad_detection_data(
                ts_img,
                ts_gt_boxes,
                ts_ig_regions,
            )
            if self.transforms is not None:
                result = self.transforms(result)
            result["img"] = result["img"].transpose(2, 0, 1)
            results.append(result)
        return results

    @staticmethod
    def calculate_scale_from_points(crop_parm, pt1, pt2):
        def get_scale():
            bs = np.random.uniform(
                crop_parm.min_crop_scale, crop_parm.max_crop_scale
            )
            inst_norm_dist = np.sqrt(
                (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + 1e-7
            )
            bs *= crop_parm.norm_len / inst_norm_dist
            bs = max(crop_parm.img_min_scale, min(bs, crop_parm.img_max_scale))
            assert 0.01 <= bs <= 100
            return bs

        base_scale = get_scale()
        return base_scale

    def get_roi_transform_mat_wrt_instance_from_roi_list(
        self,
        crop_parm: CropParm,
        pt1,
        pt2,
        center,
    ):
        base_scale = self.calculate_scale_from_points(crop_parm, pt1, pt2)
        roi = np.array([center[0], center[1], center[0], center[1]])
        center, scale = RoiTransformer.calculate_ori_roi_center(
            crop_parm, base_scale, roi
        )
        affine_aug_parm = RoiTransformer.generate_affine_aug_parm(
            crop_parm, center, scale
        )
        return affine_aug_parm

    def affine_img_and_bboxes(
        self, img, gt_boxes, ig_regions, affine_aug_param
    ):
        ts_img = self._img_ts(img, affine_aug_param.mat)
        ts_img_wh = ts_img.shape[:2][::-1]

        if self.label_type == "detection":
            boxes = gt_boxes
            ig_region = ig_regions
        else:
            raise ValueError(f"Unknown label_type:{self.label_type}")
        ts_gt_boxes, ts_ig_regions = _transform_bboxes(
            boxes,
            ig_region,
            (0, 0, ts_img_wh[0], ts_img_wh[1]),
            affine_aug_param,
            **self._bbox_ts_kwargs,
        )
        return ts_img, ts_gt_boxes, ts_ig_regions

    def get_roi_transform_mat_wrt_instance(
        self, crop_parm: CropParm, roi: np.ndarray
    ):
        base_scale = RoiTransformer.calculate_scale(crop_parm, roi)

        center, scale = RoiTransformer.calculate_ori_roi_center(
            crop_parm, base_scale, roi
        )
        affine_aug_parm = RoiTransformer.generate_affine_aug_parm(
            crop_parm, center, scale
        )
        return affine_aug_parm
