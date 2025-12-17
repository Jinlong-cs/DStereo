import logging
import pickle
import warnings
import zlib

import cv2
import numpy as np
from easydict import EasyDict
from torch.utils.data import Dataset

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.bucket import url_to_local_path
from hat.utils.pack_type.lmdb import Lmdb

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class DetSeg2DAnnoDataset(Dataset):
    def __init__(
        self,
        idx_path,
        img_path,
        anno_path,
        decode_img=True,
        transforms=None,
        to_rgb=True,
        to_buf=False,
    ):
        if transforms is not None:
            self.transforms = _as_list(transforms)
        else:
            self.transforms = transforms
        self.idx_path = url_to_local_path(idx_path)
        self._idx_lmdb = Lmdb(
            uri=self.idx_path,
            writable=False,
            map_size=10485760,
            meminit=True,
            map_async=False,
            sync=True,
        )

        idx_lmdb_len = len(self._idx_lmdb)

        self.img_path = url_to_local_path(img_path)

        self._img_lmdb = Lmdb(
            uri=self.img_path,
            writable=False,
            map_size=10485760,
            meminit=True,
            map_async=False,
            sync=True,
        )

        img_lmdb_len = len(self._img_lmdb)

        self.anno_path = url_to_local_path(anno_path)

        self._anno_lmdb = Lmdb(
            uri=self.anno_path,
            writable=False,
            map_size=10485760,
            meminit=True,
            map_async=False,
            sync=True,
        )

        anno_lmdb_len = len(self._anno_lmdb)

        if not (idx_lmdb_len == img_lmdb_len == anno_lmdb_len):
            logger.warning(
                f"{self.idx_path} with different length, idx:{idx_lmdb_len},img:{img_lmdb_len}:anno:{anno_lmdb_len}"  # noqa
            )
        logger.info(
            f"{anno_path}, img nums:{img_lmdb_len}, anno nums:{anno_lmdb_len}"  # noqa
        )
        self.len = idx_lmdb_len
        self.decode_img = decode_img

        self.to_rgb = to_rgb
        self.to_buf = to_buf

    def __getitem__(self, index):
        key = self._idx_lmdb.get(str(index).encode("ascii"))
        raw_img = self._img_lmdb.get(key)
        raw_anno = self._anno_lmdb.get(key)
        color_space = "bgr"
        if self.decode_img:
            img = cv2.imdecode(
                np.frombuffer(raw_img, dtype=np.uint8), flags=cv2.IMREAD_COLOR
            )
            if self.to_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                color_space = "rgb"
        else:
            img = raw_img
        if self.to_buf:
            img_buf = cv2.imencode(".jpg", img)[1].tobytes()
        anno_s = raw_anno
        anno = pickle.loads(anno_s)
        if "seg_label_img_format_bytes" in anno:
            uncompressed_label = zlib.decompress(
                anno["seg_label_img_format_bytes"]
            )
            w = int.from_bytes(uncompressed_label[4:8], "little")
            h = int.from_bytes(uncompressed_label[8:12], "little")
            anno = (
                np.frombuffer(uncompressed_label[12:], dtype=np.float32)
                .reshape(w, h)
                .astype(np.int8)
            )

        data = {
            "img": img,
            "anno": anno,
            "color_space": color_space,
            "layout": "hwc",
        }
        if self.to_buf:
            data["img_buf"] = img_buf
        if self.transforms is not None:
            for transform in self.transforms:
                data = transform(data)
        return data

    def __len__(self):
        return self.len


@OBJECT_REGISTRY.register
class DetSeg2DAnnoDatasetToDetFormat(object):
    """
    A transforemr.

    that transform the :py:class:`DenseBoxDataset` to be detection
    format: image, gt_boxes, ignore_regions

    Parameters
    ----------
    selected_class_ids : list/tuple of int
        Selected class ids, classes that are not in this list will be filter out
    lt_point_id : int
        Point id of left top
    rb_point_id : int
        Point id of right bottom
    min_edge_size : float, optional
        Filter out bbox whose edge is less than this value,  by default 0.1
    """  # noqa

    def __init__(
        self, selected_class_ids, lt_point_id, rb_point_id, min_edge_size=0.1
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

    def _get_bbox_wh(self, bbox):
        assert len(bbox) >= 4
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])

    def __call__(self, data):
        assert "img" in data.keys()
        assert "anno" in data.keys()

        anno = data["anno"]

        def _is_selected_class_id(class_id):
            if self.classidmap is None:
                return True
            return class_id in self.classidmap.keys()

        def _remap_class_id(class_id):
            if self.classidmap is None:
                return class_id
            return self.classidmap[class_id]

        gt_boxes = []
        for inst_i in anno["instances"]:
            if isinstance(inst_i, dict):
                inst_i = EasyDict(inst_i)
            lt = inst_i.points_data[self.lt_point_id]
            rb = inst_i.points_data[self.rb_point_id]
            class_id = inst_i.class_id[0]
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
            gt_boxes.append(gt_boxes_i)

        gt_boxes = (
            np.array(gt_boxes, dtype=np.float32)
            if len(gt_boxes) > 0
            else np.zeros((0, 5), dtype=np.float32)
        )

        ig_regions = []
        for inst_i in anno["ignore_regions"]:
            if isinstance(inst_i, dict):
                inst_i = EasyDict(inst_i)

            if "left_top" in inst_i:
                lt = inst_i["left_top"]
                rb = inst_i["right_bottom"]
            else:
                lt = inst_i["contour"][0]
                rb = inst_i["contour"][1]
            class_id = inst_i.class_id[0]
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

        data["gt_boxes"] = gt_boxes
        data["ig_regions"] = ig_regions
        return data


@OBJECT_REGISTRY.register
class DetSeg2DAnnoDatasetToROIFormat(object):
    """Transform dataset to desired format.

    Parameters
    ----------
    parent_lt_point_id and parent_rb_point_id: used for sub box mode.
    parent_id: used for shared rpn whether sub box is used or not.
        which is remapped parent class id, rather than origin parent classid.
    """

    def __init__(
        self,
        selected_class_ids=None,
        lt_point_id=0,
        rb_point_id=2,
        min_edge_size=0.1,
        parent_lt_point_id=None,
        parent_rb_point_id=None,
        use_parent=False,
        parent_id=1,
    ):
        type_error_msg = "select_class_ids should be " "list/tuple"
        assert isinstance(selected_class_ids, (list, tuple)), type_error_msg
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
        self.parent_lt_point_id = parent_lt_point_id
        self.parent_rb_point_id = parent_rb_point_id
        self.use_parent = use_parent
        self.parent_id = parent_id
        self.min_edge_size = min_edge_size

    def _get_bbox_wh(self, bbox):
        assert len(bbox) >= 4
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])

    def __call__(self, data):
        img = data["img"]
        anno = data["anno"]

        def _is_selected_class_id(class_id):
            if self.classidmap is None:
                return True
            return class_id in self.classidmap.keys()

        def _remap_class_id(class_id):
            if self.classidmap is None:
                return class_id
            return self.classidmap[class_id]

        def _get_gt_boxes(
            lt_point_id, rb_point_id, is_parent=False, parent_id=0
        ):
            gt_boxes = []
            for inst_i in anno["instances"]:
                if isinstance(inst_i, dict):
                    inst_i = EasyDict(inst_i)
                lt = inst_i.points_data[lt_point_id]
                rb = inst_i.points_data[rb_point_id]
                class_id = inst_i.class_id[0]
                if _is_selected_class_id(class_id):
                    if is_parent:
                        class_id = parent_id
                    else:
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
                        "Ignore gt_boxes %s, min edge size is invalid..."
                        % gt_boxes_i
                    )  # noqa
                    warnings.warn(msg)
                    continue
                gt_boxes.append(gt_boxes_i)

            gt_boxes = (
                np.array(gt_boxes, dtype=np.float32)
                if len(gt_boxes) > 0
                else np.zeros((0, 5), dtype=np.float32)
            )
            return gt_boxes

        def _get_ig_regions(is_parent=False, parent_id=0):
            ig_regions = []
            for inst_i in anno["ignore_regions"]:
                if isinstance(inst_i, dict):
                    inst_i = EasyDict(inst_i)
                if "left_top" in inst_i:
                    lt = inst_i["left_top"]
                    rb = inst_i["right_bottom"]
                else:
                    lt = inst_i["contour"][0]
                    rb = inst_i["contour"][1]
                class_id = inst_i.class_id[0]
                if _is_selected_class_id(class_id):
                    if is_parent:
                        class_id = parent_id
                    else:
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
            return ig_regions

        gt_boxes = _get_gt_boxes(self.lt_point_id, self.rb_point_id)
        ig_regions = _get_ig_regions(is_parent=False)
        if self.use_parent:
            if (
                self.parent_lt_point_id is not None
                and self.parent_rb_point_id is not None
            ):
                # use parent point id for sub box mode
                parent_gt_boxes = _get_gt_boxes(
                    self.parent_lt_point_id,
                    self.parent_rb_point_id,
                    is_parent=True,
                    parent_id=self.parent_id,
                )
            else:
                parent_gt_boxes = _get_gt_boxes(
                    self.lt_point_id,
                    self.rb_point_id,
                    is_parent=True,
                    parent_id=self.parent_id,
                )
            parent_ig_regions = _get_ig_regions(
                is_parent=True, parent_id=self.parent_id
            )
            return {
                "img": img,
                "gt_boxes": gt_boxes,
                "ig_regions": ig_regions,
                "parent_gt_boxes": parent_gt_boxes,
                "parent_ig_regions": parent_ig_regions,
            }

        return {
            "img": img,
            "gt_boxes": gt_boxes,
            "ig_regions": ig_regions,
        }


@OBJECT_REGISTRY.register
class DetSeg2DAnnoDatasetToRoiMultiDetectionFormat(object):
    def __init__(
        self,
        selected_class_ids=None,
        lt_point_id=0,
        rb_point_id=2,
        min_edge_size=0.1,
        parent_lt_point_id=None,
        parent_rb_point_id=None,
        use_parent=False,
        parent_id=1,
    ):
        type_error_msg = "select_class_ids should be " "list/tuple"
        assert isinstance(selected_class_ids, (list, tuple)), type_error_msg
        if selected_class_ids is not None:
            selected_class_ids = _as_list(selected_class_ids)
            # class id should begin from 1
            self.classidmap = dict(
                zip(selected_class_ids, range(1, len(selected_class_ids) + 1))
            )
        else:
            self.classidmap = None
        self.subbox_lt_point_id = lt_point_id
        self.subbox_rb_point_id = rb_point_id
        self.parent_lt_point_id = parent_lt_point_id
        self.parent_rb_point_id = parent_rb_point_id
        self.use_parent = use_parent
        self.parent_id = parent_id
        self.min_edge_size = min_edge_size
        self.subbox_max_num = 10

    def __call__(self, data):
        img = data["img"]
        anno = data["anno"]

        def _is_selected_class_id(class_id):
            if self.classidmap is None:
                return True
            return class_id in self.classidmap.keys()

        def _remap_class_id(class_id):
            if self.classidmap is None:
                return class_id
            return self.classidmap[class_id]

        def _get_ig_regions(is_parent=False, parent_id=0):
            ig_regions = []
            for inst_i in anno["ignore_regions"]:
                if isinstance(inst_i, dict):
                    inst_i = EasyDict(inst_i)
                if "left_top" in inst_i:
                    lt = inst_i["left_top"]
                    rb = inst_i["right_bottom"]
                else:
                    lt = inst_i["contour"][0]
                    rb = inst_i["contour"][1]
                class_id = inst_i.class_id[0]
                if _is_selected_class_id(class_id):
                    if is_parent:
                        class_id = parent_id
                    else:
                        class_id = _remap_class_id(class_id)
                else:
                    continue
                ig_regions_i = [lt[0], lt[1], rb[0], rb[1], class_id]
                ig_regions_i_wh = _get_bbox_wh(ig_regions_i)
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
            return ig_regions

        def _get_bbox_wh(bbox):
            assert len(bbox) >= 4
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])

        def _compute_iou(bbox1, bbox2):
            left_column_max = max(bbox1[0], bbox2[0])
            right_column_min = min(bbox1[2], bbox2[2])
            up_row_max = max(bbox1[1], bbox2[1])
            down_row_min = min(bbox1[3], bbox2[3])
            if (
                left_column_max >= right_column_min
                or down_row_min <= up_row_max
            ):  # noqa
                return 0
            else:
                S1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                S2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                S_cross = (down_row_min - up_row_max) * (
                    right_column_min - left_column_max
                )
                return S_cross / (S1 + S2 - S_cross)

        def _check_box_invalid(bbox, wh, min_edge_size):
            # 检查box是否有效
            # 打包时无效的subbox会设置为[-10000, -10000, -10000, -10000]，需要过滤
            if (
                wh[0] < min_edge_size
                or wh[1] < min_edge_size
                or bbox == [-10000, -10000, -10000, -10000]
            ):
                return True
            else:
                return False

        def _get_box_pairs(
            parent_lt_point_id,
            parent_rb_point_id,
            subbox_lt_point_id,
            subbox_rb_point_id,
            min_edge_size,
            parent_id,
        ):
            gt_boxes_pairs = []
            for inst_i in anno["instances"]:
                if isinstance(inst_i, dict):
                    inst_i = EasyDict(inst_i)

                parent_lt = inst_i.points_data[parent_lt_point_id]
                parent_rb = inst_i.points_data[parent_rb_point_id]

                subbox_lt = inst_i.points_data[subbox_lt_point_id]
                subbox_rb = inst_i.points_data[subbox_rb_point_id]

                class_id = inst_i.class_id[0]
                if _is_selected_class_id(class_id):
                    parent_class_id = parent_id
                    subbox_class_id = _remap_class_id(class_id)
                else:
                    continue

                hard_flag = inst_i.is_hard[0]
                if hard_flag in [1, True, "1", "True"]:
                    parent_class_id *= -1
                    subbox_class_id *= -1

                parent_boxes_i = [
                    parent_lt[0],
                    parent_lt[1],
                    parent_rb[0],
                    parent_rb[1],
                    parent_class_id,
                ]
                parent_boxes_i_wh = _get_bbox_wh(parent_boxes_i)
                subbox_boxes_i = [
                    subbox_lt[0],
                    subbox_lt[1],
                    subbox_rb[0],
                    subbox_rb[1],
                    subbox_class_id,
                ]
                subbox_boxes_i_wh = _get_bbox_wh(subbox_boxes_i)

                bbox_is_invalid = (
                    _check_box_invalid(
                        parent_boxes_i, parent_boxes_i_wh, min_edge_size
                    )
                    or _check_box_invalid(
                        subbox_boxes_i, subbox_boxes_i_wh, min_edge_size
                    )
                    or _compute_iou(parent_boxes_i, subbox_boxes_i) <= 0
                )
                if bbox_is_invalid:
                    continue

                gt_boxes_pairs.append([parent_boxes_i, subbox_boxes_i])
            return gt_boxes_pairs

        def _split_box(bbox_list):
            parent_bbox_list = []
            subbox_list = []
            for bbox_pairs_i in bbox_list:
                parent_box_i, subbox_box_i = bbox_pairs_i
                parent_bbox_list.append(parent_box_i)
                subbox_list.append(subbox_box_i)

            assert len(parent_bbox_list) == len(subbox_list)
            parent_bbox_list = (
                np.array(parent_bbox_list, dtype=np.float32)
                if len(parent_bbox_list) > 0
                else np.zeros((0, 5), dtype=np.float32)
            )
            subbox_list = (
                np.array(subbox_list, dtype=np.float32)
                if len(subbox_list) > 0
                else np.zeros((0, 5), dtype=np.float32)
            )

            return parent_bbox_list, subbox_list

        gt_boxes_pairs = _get_box_pairs(
            self.parent_lt_point_id,
            self.parent_rb_point_id,
            self.subbox_lt_point_id,
            self.subbox_rb_point_id,
            self.min_edge_size,
            self.parent_id,
        )
        ig_regions = _get_ig_regions(is_parent=False)
        parent_ig_regions = _get_ig_regions(
            is_parent=True, parent_id=self.parent_id
        )
        parent_bbox_list, subbox_list = _split_box(gt_boxes_pairs)

        return {
            "img": img,
            "gt_boxes": subbox_list,
            "ig_regions": ig_regions,
            "parent_gt_boxes": parent_bbox_list,
            "parent_ig_regions": parent_ig_regions,
        }


@OBJECT_REGISTRY.register
class DetSeg2DAnnoDatasetToSegFormat(object):
    """parse atrributes of the img."""

    def __call__(self, data):
        anno = data.pop("anno")
        data["img_name"] = "placeholder"
        data["gt_seg"] = anno.astype(np.uint8)
        data["img_height"] = anno.shape[0]
        data["img_width"] = anno.shape[1]
        data["img_id"] = 0

        return data
