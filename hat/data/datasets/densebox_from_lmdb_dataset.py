import logging
import pickle
import zlib
from typing import Dict, List, Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

from hat.registry import OBJECT_REGISTRY
from hat.utils.bucket import url_to_local_path
from hat.utils.pack_type.lmdb import Lmdb

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class DenseboxFromLMDBDataset(Dataset):
    """Read the LDMB data and transform it to the DenseBox format.

    Args:
        idx_path: the path to the lmdb index file.
        img_path: the path to the lmdb image file.
        anno_path: the path to the lmdb annotation file.
        decode_img: Whether to decode the image byte data. Default is True.
        task_type : Can be "detection" or "segmentation".
            Default is "detection".
        class_id : the rec's class id, 1 based. Default is -1.
        category : the used category, 0 based. Default is -1.
        to_rgb: Whether to convert the "bgr" order to the "rgb" order.
            Default is True.
        ignore_hard : Ignore hard instances if `hard` tag in annotation.
            Default is False.
        use_ignore : Whether to use ignore regions in annotation.
            Default is False.
        transform_ignore_bboxes: Whether to transform ignore bboxes like
            gt bboxes so that the ignored bboxes can have correct sizes
            in case the image is resized or cropped. Default is False.
        return_orig_img: Whether to return an extra original img which can
            be used for visualization. Default is False.
        return_orig_gt_seg: Whether to return an extra original segmentation
            ground-truth image which can be used for visualization.
            Default is False.
        remove_det_duplicate: Whether to filter out duplicate gt bboxes in one
            image. Sometimes there are duplicate gt bboxes in one image to
            emphasize certain training examples, but which may cause inaccurate
            performance evaluation. Default is False.
        abandon_other_category: There may be some bboxes that do not belong to
            current class. If False, set these bboxes as ignore (foreground
            with no loss), else, set as background. Default is False.
        transforms: A list of image and gt transforms, like Resize.
            Default is None.
    """

    def __init__(
        self,
        idx_path,
        img_path,
        anno_path,
        decode_img: Optional[bool] = True,
        task_type: Optional[str] = "detection",
        class_id: Optional[int] = -1,
        category: Optional[int] = -1,
        to_rgb: Optional[bool] = True,
        ignore_hard: Optional[bool] = False,
        use_ignore: Optional[bool] = False,
        transform_ignore_bboxes: Optional[bool] = False,
        return_orig_img: Optional[bool] = False,
        return_orig_gt_seg: Optional[bool] = False,
        remove_det_duplicate: Optional[bool] = False,
        abandon_other_category: Optional[bool] = False,
        transforms: Optional[List] = None,
    ):

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
                f"{self.idx_path} has different item lengthes, "
                f"idx:{idx_lmdb_len}, img:{img_lmdb_len}, anno:{anno_lmdb_len}"
            )

        self.len = idx_lmdb_len
        self.decode_img = decode_img

        self.task_type = task_type
        self.class_id = class_id
        self.category = category
        self.to_rgb = to_rgb
        self.ignore_hard = ignore_hard
        self.use_ignore = use_ignore
        self.transform_ignore_bboxes = transform_ignore_bboxes
        self.return_orig_img = return_orig_img
        self.return_orig_gt_seg = return_orig_gt_seg
        self.remove_det_duplicate = remove_det_duplicate
        self.abandon_other_category = abandon_other_category
        self.transforms = transforms

        if self.task_type == "detection":
            self.kwargs = {}
        elif self.task_type == "segmentation":
            self.kwargs = {"with_seg_label": True, "seg_label_dtype": np.int8}
        else:
            raise Exception(
                f"Wrong task_type, your task_type is [{self.task_type}], "
                f"but we require segmentation or detection."
            )

        logging.info(f"dataset path: {self.img_path}, {self.anno_path}")
        logging.info(f"dataset length: {self.len}")

    def __getitem__(self, index: int) -> Dict:
        key = self._idx_lmdb.get(str(index).encode("ascii"))
        raw_img = self._img_lmdb.get(key)
        raw_anno = self._anno_lmdb.get(key)

        if self.decode_img:
            image = cv2.imdecode(
                np.frombuffer(raw_img, dtype=np.uint8), flags=cv2.IMREAD_COLOR
            )
        else:
            image = raw_img
        color_space = "bgr"
        if self.to_rgb:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color_space = "rgb"

        anno_s = raw_anno
        anno = pickle.loads(anno_s)
        if (
            "seg_label_img_format_bytes" in anno
            and self.task_type == "segmentation"
        ):
            seg_anno = anno.pop("seg_label_img_format_bytes")
            uncompressed_label = zlib.decompress(seg_anno)
            w = int.from_bytes(uncompressed_label[4:8], "little")
            h = int.from_bytes(uncompressed_label[8:12], "little")
            seg_gt = (
                np.frombuffer(uncompressed_label[12:], dtype=np.float32)
                .reshape(w, h)
                .astype(np.int8)
            )
            anno = (anno, seg_gt)

        data = {}
        if self.task_type == "detection":
            anno = anno
        elif self.task_type == "segmentation":
            assert len(anno) == 2, "there should be two elements in anno"
            seg_label = anno[1]
            seg_label = seg_label.astype(np.uint8)
            anno = anno[0]
        data["img_name"] = anno["img_url"].split("/")[-1]
        data["data_path"] = self.img_path
        data["img_height"] = anno["img_h"]
        data["img_width"] = anno["img_w"]
        data["img_id"] = np.expand_dims(anno.get("idx", 0), 0)
        data["img"] = image
        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image.shape
        if self.return_orig_img:
            data["orig_img"] = image.copy()

        if self.task_type == "detection":
            gt_bboxes = []
            gt_classes = []
            for ins in anno["instances"]:
                is_hard = ins["is_hard"][0]
                points_data = ins["points_data"]
                class_id = int(ins["class_id"][0])
                bbox = []
                bbox.extend(points_data[0])
                bbox.extend(points_data[2])
                if class_id == self.class_id:
                    gt_bboxes.append(bbox)
                    if is_hard and self.ignore_hard:
                        gt_classes.append(-1)
                    else:
                        gt_classes.append(self.category)
                elif not self.abandon_other_category:
                    gt_bboxes.append(bbox)
                    gt_classes.append(-1)
            data["gt_bboxes"] = np.array(gt_bboxes)
            data["gt_classes"] = np.array(gt_classes, dtype=np.int64)

            ig_bboxes = []
            if self.use_ignore:
                for ig_region in anno["ignore_regions"]:
                    if "left_top" in ig_region:
                        left_top = ig_region["left_top"]
                        right_bottom = ig_region["right_bottom"]
                    else:
                        left_top = ig_region["contour"][0]
                        right_bottom = ig_region["contour"][1]
                    if not (
                        list(left_top) == [0, 0]
                        and list(right_bottom)
                        == [anno["img_w"], anno["img_h"]]
                    ):
                        ig_bbox = left_top + right_bottom
                        ig_bboxes.append(list(ig_bbox))
                    else:
                        continue

            # If you would like to transform ignore bboxes like gt bboxes, you
            # need to add them to the gt bbox list and assign them a "-1" label
            # so that those transforms like Resize can treat them as gt bboxes,
            # otherwise they will not be transformed because many transforms
            # do not process "ig_bboxes" and just leave them alone.
            if ig_bboxes and self.transform_ignore_bboxes:
                gt_bboxes.extend(ig_bboxes)
                data["gt_bboxes"] = np.array(gt_bboxes)
                gt_classes.extend([-1] * len(ig_bboxes))
                data["gt_classes"] = np.array(gt_classes, dtype=np.int64)
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
        return self.len

    def __repr__(self):
        return "DenseboxFromLMDBDataset"
