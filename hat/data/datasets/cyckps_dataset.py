import json
import os
from typing import List, Optional

import cv2
import numpy as np
import torch.utils.data as data

from hat.data.datasets.utils import img_to_rgb
from hat.evaluation.wheel_kps.parse_annos import loadGtContentsfromJson
from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.mxrecord import MXRecord, unpack
from hat.utils.pack_type.recordio_pb2 import RecordUnit

__all__ = ["CycKpsRecRecSingleDataset", "CycKpsDataset"]


def uppack_rec_unit(record, decode_type, to_rgb=False):
    _, s = unpack(record)
    rec_data = RecordUnit()
    rec_data.ParseFromString(s)
    rec_data = rec_data.body
    assert len(rec_data.data) in [
        2,
    ]
    assert len(rec_data.extra) == 0
    if decode_type in [
        "image",
    ]:
        image_buf = rec_data.data[0].value
        image = cv2.imdecode(
            np.frombuffer(image_buf, dtype=np.uint8), flags=cv2.IMREAD_COLOR
        )
        if to_rgb:
            image = img_to_rgb(image)
        return image
    elif decode_type in [
        "kps",
    ]:
        img_rec_idx = int(bytes.decode(rec_data.data[0].value))
        label_buf = rec_data.data[1].value
        kps_label = json.loads(label_buf)
        return img_rec_idx, kps_label
    else:
        raise KeyError(f"do not support {decode_type} decode_type")


@OBJECT_REGISTRY.register
class CycKpsRecRecSingleDataset(data.Dataset):
    """
    CycKpsRecRecSingleDataset Dataset.

    Args:
        img_path: image path.
        anno_path: label path.
        to_rgb: Whether output image in rgb color.
        transforms: transforms of data before using. Defaults to None.
    """

    def __init__(
        self,
        img_path: str,
        anno_path: str,
        to_rgb: bool,
        transforms: Optional[List] = None,
    ):
        self.to_rgb = to_rgb
        self.transforms = transforms
        self.rec_dataset = MXRecord(
            uri=img_path, idx_path=img_path + ".idx", writable=False
        )
        self.anno_dataset = MXRecord(
            uri=anno_path, idx_path=anno_path + ".idx", writable=False
        )

    def __repr__(self):
        return "CycKpsRecRecSingleDataset"

    def __getitem__(self, idx):
        anno_record = self.anno_dataset.read(idx)
        img_rec_idx, label = uppack_rec_unit(anno_record, "kps", self.to_rgb)
        img_record = self.rec_dataset.read(img_rec_idx)
        img = uppack_rec_unit(img_record, "image", self.to_rgb)

        data = {
            "img": img,
            "layout": "hwc",
            "color_space": "rgb" if self.to_rgb else "bgr",
            "img_path": label["image"],
            "boxes": np.array(label["boxes"], np.float32),
            "gt_classes": np.array(label["gt_classes"], np.int32),
            "keypoints": np.array(label["keypoints"], np.float32),
        }
        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.anno_dataset)


@OBJECT_REGISTRY.register
class CycKpsDataset(data.ConcatDataset):
    """
    Read image, cyclist wheel kps from rec file list.

    Args:
        image_rec_list: List of path to image rec file.
        anno_rec_list: List of path to anno rec file.
        transforms: Transforms of data augmentation.
        to_rgb: Whether output image in rgb color, by default False.
    """

    def __init__(
        self,
        image_rec_list: List[str],
        anno_rec_list: List[str],
        transforms: Optional[List] = None,
        to_rgb: bool = False,
    ):
        self.image_rec_list = image_rec_list
        self.anno_rec_list = anno_rec_list
        self.transforms = transforms
        self.to_rgb = to_rgb

        self._init_pack()
        super(CycKpsDataset, self).__init__(self.dataset_list)

    def __repr__(self):
        return "CycKpsDataset"

    def _init_pack(self):
        assert len(self.image_rec_list) == len(self.anno_rec_list)
        self.dataset_list = []
        for image_path, anno_path in zip(
            self.image_rec_list, self.anno_rec_list
        ):
            self.dataset_list.append(
                CycKpsRecRecSingleDataset(
                    img_path=image_path,
                    anno_path=anno_path,
                    to_rgb=self.to_rgb,
                    transforms=self.transforms,
                )
            )


@OBJECT_REGISTRY.register
class EvalCycKpsSingleDataset(data.Dataset):
    """
    CycKpsRecRecSingleDataset Dataset.

    Args:
        data_path_prefix: image and gt path prefix.
        to_rgb: Whether output image in rgb color.
        transforms: transforms of data before using. Defaults to None.
    """

    def __init__(
        self,
        data_path_prefix: str,
        to_rgb: bool,
        transforms: Optional[List] = None,
    ):
        self.to_rgb = to_rgb
        self.transforms = transforms
        self.gtfile = data_path_prefix + ".json"
        assert os.path.exists(self.gtfile)
        self.img_prefix = data_path_prefix
        all_dict_info = loadGtContentsfromJson(self.gtfile, self.img_prefix)
        self.image_annos = self.flatten_img_info(all_dict_info)

    def flatten_img_info(self, all_dict_info):
        all_annos = []
        for imgname, anno in all_dict_info.items():
            imgpath = os.path.join(self.img_prefix, imgname)
            for roi in anno.values():
                tmp_item = {
                    "img_path": imgpath,
                    "eval_kps_two_occ": np.array(
                        roi["eval_kps_two_occ"], np.float32
                    ),
                    "fail_offset": np.array(roi["fail_offset"], np.float32),
                    "bbox": np.array(roi["roi_bbox"], np.float32),
                    "wheel_kps": np.array(
                        [roi["wheel_kps_0"], roi["wheel_kps_1"]], np.float32
                    ),
                }
                all_annos.append(tmp_item)
        return all_annos

    def __repr__(self):
        return "EvalCycKpsSingleDataset"

    def __getitem__(self, idx):
        data = self.image_annos[idx]
        image_path = data["img_path"]
        image = cv2.imread(image_path)
        if self.to_rgb:
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)

        data["img"] = image
        data["layout"] = "hwc"
        data["color_space"] = "rgb" if self.to_rgb else "bgr"

        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.image_annos)
