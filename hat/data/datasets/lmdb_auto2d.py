# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os
from typing import Dict, List, Optional

import cv2
import msgpack
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader

from hat.data.datasets.read_raw import read_raw
from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.lmdb import Lmdb
from hat.utils.package_helper import require_packages

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

__all__ = ["AutoDetPacker", "AutoSegPacker", "Auto2D2lmdb", "Auto2dFromLMDB"]

logger = logging.getLogger(__name__)


class AutoDetPacker(data.Dataset):
    """Packer used to create map-style dataset for the detection task.

    Your dataset should be organized according to the following directory:

    └── directory
        ├── annotations
        │   ├── train.json
        │   └── val.json
        ├── train
        └── val

    Args:
        directory: Path for dataset.
        split_name: Split name of data, such as train, val and so on.
    """

    @require_packages("pycocotools")
    def __init__(self, directory: str, split_name: str):
        self.directory = directory
        self.split_name = split_name
        self.coco = COCO(
            os.path.join(directory, "annotations", self.split_name + ".json")
        )
        self.image_ids = self.coco.getImgIds()
        self.load_classes()

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])
        self.labels = {}  # {0: 'person', 1: 'bicycle', ..., 79: 'toothbrush'}
        self.coco_labels = {}  # {0: 1, 1: 2, ..., 79: 90}
        self.coco_labels_inverse = {}  # {1: 0, 2: 1, ..., 90: 79}
        for c in categories:
            self.coco_labels[len(self.labels)] = c["id"]
            self.coco_labels_inverse[c["id"]] = len(self.labels)
            self.labels[len(self.labels)] = c["name"]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_info, img = self.load_image(idx)
        annotations = self.load_annotations(idx, image_info)
        results = {}
        results.update(image_info)
        results.update(annotations)
        results["img"] = img
        gt_img = self.load_target_image(image_info)
        if gt_img is not None:
            results["gt_img"] = gt_img
        return results

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        if image_info.get("bit_nums_upper") is not None:
            if image_info["bit_nums_upper"] == 8:
                img_path = image_info["img_path"]
            else:
                img_path = image_info["raw_path"]
        else:
            img_path = os.path.join(
                self.directory,
                "{}".format(self.split_name),
                image_info["file_name"],
            )

        with open(img_path, "rb") as f:
            img = f.read()
        return image_info, img

    def load_annotations(self, image_index, image_info):
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_index],
            iscrowd=None,
        )
        annotations = {
            "gt_bboxes": np.zeros((0, 4)),
            "gt_classes": np.zeros(0),
        }

        if len(annotations_ids) == 0:
            # iscrowd=1 for all bbox
            return annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)

        gt_bboxes = []
        gt_classes = []
        for _i, ann in enumerate(coco_annotations):
            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, image_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, image_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.labels:
                continue
            if ann["ignore"]:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            gt_bboxes.append(bbox)
            if ann.get("iscrowd", False):
                gt_classes.append(-1)
            else:
                gt_classes.append(self.coco_label_to_label(ann["category_id"]))

        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_classes = np.array(gt_classes, dtype=np.int64)

        annotations = {"gt_bboxes": gt_bboxes, "gt_classes": gt_classes}

        return annotations

    def load_target_image(self, image_info):
        pack_gt_img = image_info.get("pack_gt_img", False)
        if pack_gt_img:
            gt_path = image_info["img_path"]
            with open(gt_path, "rb") as f:
                gt_img = f.read()
            return gt_img
        else:
            return None

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image_info["width"]) / float(image_info["height"])


class AutoSegPacker(data.Dataset):  # noqa: D205,D400
    """Packer used to create map-style dataset for the semantic
    segmentation task.

    Your dataset should be organized according to the following directory:

    └── directory
        ├── annotations
        │   ├── train.json
        │   └── val.json
        ├── train
        └── val

    Args:
        directory: Path for dataset.
        split_name: Split name of data, such as train, val and so on.
        seg_map_dict: A dict of gt png pixel value to label id.
    """

    @require_packages("pycocotools")
    def __init__(self, directory: str, split_name: str, seg_map_dict=None):
        self.directory = directory
        self.split_name = split_name
        self.seg_map_dict = seg_map_dict

        self.coco = COCO(
            os.path.join(directory, "annotations", self.split_name + ".json")
        )
        self.image_ids = self.coco.getImgIds()
        self.load_classes()

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])

        self.labels = {}  # {0: 'road', 1: 'background', ...}
        for c in categories:
            self.labels[len(self.labels)] = c["name"]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_info, img = self.load_image(idx)
        annotations = self.load_annotations(image_info)
        results = {}
        results.update(image_info)
        results["img"] = img
        results["gt_seg"] = annotations
        gt_img = self.load_target_image(image_info)
        if gt_img is not None:
            results["gt_img"] = gt_img

        return results

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        if image_info.get("bit_nums_upper") is not None:
            if image_info["bit_nums_upper"] == 8:
                img_path = image_info["img_path"]
            else:
                img_path = image_info["raw_path"]
        else:
            img_path = os.path.join(
                self.directory,
                "{}".format(self.split_name),
                image_info["file_name"],
            )
        with open(img_path, "rb") as f:
            img = f.read()
        return image_info, img

    def load_annotations(self, image_info):
        if "label_dir" in image_info:
            label_dir = image_info["label_dir"]
            path = os.path.join(
                label_dir,
                image_info["file_name"].replace(".jpg", "_label.png"),
            )
        else:
            path = os.path.join(
                self.directory,
                "{}".format(self.split_name),
                image_info["file_name"],
            )
            path_suffix = os.path.splitext(image_info["file_name"])[-1]
            path = path.replace(path_suffix, "_label.png")

        gt_seg_map = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if self.seg_map_dict is not None:
            new_gt_semantic_seg = gt_seg_map.copy()
            assert isinstance(self.seg_map_dict, dict)
            for src_label, target_label in self.seg_map_dict.items():
                idnex = gt_seg_map == src_label
                new_gt_semantic_seg[idnex] = target_label
            gt_seg_map = new_gt_semantic_seg

        return gt_seg_map

    def load_target_image(self, image_info):
        pack_gt_img = image_info.get("pack_gt_img", False)
        if pack_gt_img:
            gt_path = image_info["img_path"]
            with open(gt_path, "rb") as f:
                gt_img = f.read()
            return gt_img
        else:
            return None

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image["width"]) / float(image["height"])


class Datum(object):
    """Class for serializing and deserializing.

    Args:
        datas: Keys inclued `img`, `height`, `width`, `gt_classes`,
        `gt_bboxes`, `gt_seg` and so on.
    """

    def __init__(self, datas: Optional[Dict] = None):
        self.datas = datas

    def SerializeToString(self):
        pack_dict = {}
        for k, v in self.datas.items():
            if k == "img":
                img = self.datas["img"][0]
                pack_dict["img"] = img
            elif k == "gt_seg":
                gt_seg = self.datas["gt_seg"][0].numpy()
                pack_dict["gt_seg"] = cv2.imencode(".png", gt_seg)[1].tobytes()
            elif k == "gt_img":
                gt_img = self.datas["gt_img"][0]
                pack_dict["gt_img"] = gt_img
            elif k in [
                "height",
                "width",
                "gt_classes",
                "id",
                "bit_nums_lower",
                "bit_nums_upper",
                "channels",
            ]:
                new_v = np.asarray(v, dtype=np.int64).tobytes()
                pack_dict[k] = new_v
            elif k in [
                "gt_bboxes",
                "dgain",
                "rg_gain",
                "bg_gain",
            ]:
                new_v = np.asarray(v, dtype=np.float64).tobytes()
                pack_dict[k] = new_v
            elif k in ["file_name", "raw_pattern"]:
                new_v = bytes(v[0], encoding="utf8")
                pack_dict[k] = new_v
            else:
                assert "wrong key"
        return msgpack.packb(pack_dict)

    def ParseFromString(self, raw_data):
        self.image_info = {}
        if raw_data.get("bit_nums_upper"):
            bit_nums = np.frombuffer(
                raw_data["bit_nums_upper"],
                dtype=np.int64,
            )
            height = np.frombuffer(raw_data["height"], dtype=np.int64)[0]
            width = np.frombuffer(raw_data["width"], dtype=np.int64)[0]
            channels = np.frombuffer(raw_data["channels"], dtype=np.int64)[0]
            if "img_type" in raw_data:
                img_type = bytes.decode(raw_data["img_type"])
                self.image = np.frombuffer(raw_data["img"], dtype=img_type)
                if bit_nums <= 8:
                    self.image = cv2.imdecode(self.image, cv2.IMREAD_COLOR)
            else:
                if bit_nums <= 8:
                    self.image = np.frombuffer(raw_data["img"], dtype=np.uint8)
                    self.image = cv2.imdecode(self.image, cv2.IMREAD_COLOR)
                elif 8 < bit_nums <= 16:
                    self.image = read_raw(
                        raw_data["img"], height, width, bit_nums_upper=bit_nums
                    )
                elif 16 < bit_nums <= 32:
                    self.image = read_raw(
                        raw_data["img"],
                        height,
                        width,
                        bit_nums_upper=bit_nums,
                        mipi=0,
                    )
                else:
                    raise ValueError(
                        "Don't support bit_nums which is bigger than 32, "
                        "current bit_nums is %d" % bit_nums
                    )
            self.image = np.reshape(self.image, (height, width, channels))
        else:
            self.image = np.frombuffer(raw_data["img"], dtype=np.uint8)
            self.image = cv2.imdecode(self.image, cv2.IMREAD_COLOR)

        for k, v in raw_data.items():
            if k == "img" or k == "img_type":
                continue
            elif k == "gt_seg":
                new_v = np.frombuffer(raw_data["gt_seg"], dtype=np.uint8)
                new_v = cv2.imdecode(new_v, cv2.IMREAD_UNCHANGED)
            elif k == "gt_img":
                new_v = np.frombuffer(raw_data["gt_img"], dtype=np.uint8)
                new_v = cv2.imdecode(new_v, cv2.IMREAD_COLOR)
            elif k in [
                "height",
                "width",
                "gt_classes",
                "id",
                "bit_nums_lower",
                "bit_nums_upper",
                "channels",
            ]:
                new_v = np.frombuffer(v, dtype=np.int64)
            elif k in [
                "rg_gain",
                "bg_gain",
                "dgain",
            ]:
                new_v = np.frombuffer(v, dtype=np.float64)
            elif k == "gt_bboxes":
                new_v = np.frombuffer(v, dtype=np.float64)
                if len(new_v) != 0:
                    new_v = new_v.reshape((int(new_v.shape[0] / 4), -1))
            elif k in ["file_name", "raw_pattern"]:
                new_v = bytes.decode(v)
                if k == "raw_pattern":
                    self.image_info["cur_pattern"] = new_v
            self.image_info[k] = new_v
        return self.image_info, self.image


def Auto2D2lmdb(
    lmdb_path: str,
    directory: str,
    split_name: str,
    task: str,
    num_workers: int,
    shuffle: Optional[bool] = True,
    **kwargs: Dict,
):
    """Pack the original data into lmdb.

    Args:
        lmdb_path: Storage path of the generated lmdb file.
        directory: Storage path of the data to be packaged.
        split_name: Split name of data, such as train, val and so on.
        task (str): Task name, such as `train`, `val` and so on.
        num_workers: The num workers for reading data, same as
            DataLoader.
        shuffle: Same as DataLoader.
        **kwargs: Receive extra parameters.
    """
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)

    if "det" in task:
        dataset = AutoDetPacker(directory=directory, split_name=split_name)
    elif "seg" in task:
        dataset = AutoSegPacker(
            directory=directory,
            split_name=split_name,
            seg_map_dict=kwargs.get("seg_map_dict", None),
        )
    else:
        assert "wrong task type"

    data_loader = DataLoader(dataset, num_workers=num_workers, shuffle=shuffle)

    db = Lmdb(
        lmdb_path,
        commit_step=500,
        map_size=1099511627776 * 50,
        meminit=False,
        map_async=True,
    )

    for idx, datas in enumerate(data_loader):
        base_data = Datum(datas=datas)
        db.write(idx, base_data.SerializeToString())

    db.close()


@OBJECT_REGISTRY.register
class Auto2dFromLMDB(data.Dataset):
    """Unpack data from lmdb.

    Args:
        data_path: Path of lmdb file.
        transforms: A list of transform.
        num_samples: As it says.
        to_rgb: Whether transform color_space of img to `RGB`.
        return_orig_img: Whether to return an extra original img,
            orig_img can usually be used on visualization.
        infer_model_type: Used in crop model, like "crop_wo_resize".
        camera_info: LMDB camera info, including yaw, pitch, roll, etc.
    """

    def __init__(
        self,
        data_path: str,
        transforms: Optional[List] = None,
        num_samples: Optional[int] = 1,
        to_rgb: Optional[bool] = False,
        return_orig_img: Optional[bool] = False,
        infer_model_type: Optional[str] = None,
        camera_info: Optional[Dict] = "",
    ):
        self.root = data_path
        self.transforms = transforms
        self.num_samples = num_samples
        self.samples = list(range(num_samples))
        self.to_rgb = to_rgb
        self.datum = Datum()
        self.return_orig_img = return_orig_img
        self.infer_model_type = infer_model_type
        self.camera_info = camera_info

        self.data_lmdb = Lmdb(
            self.root,
            writable=False,
            readahead=False,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        data = {}
        index = self.samples[item]
        raw_data = self.data_lmdb.read(index)
        raw_data = msgpack.unpackb(raw_data, raw=False)
        image_info, image = self.datum.ParseFromString(raw_data)
        if self.return_orig_img:
            data["orig_img"] = image
        color_space = "bgr"
        if self.to_rgb:
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
            color_space = "rgb"

        data.update(image_info)
        data["img_name"] = data.pop("file_name")
        data["img_height"] = data.pop("height")
        data["img_width"] = data.pop("width")
        data["img_id"] = data.pop("id")
        data["img"] = image
        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["infer_model_type"] = self.infer_model_type
        data["camera_info"] = self.camera_info

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"data_path={self.root}, "
        repr_str += f"transforms={self.transforms}, "
        repr_str += f"num_samples={self.num_samples}, "
        repr_str += f"to_rgb={self.to_rgb}, "
        repr_str += f"return_orig_img={self.return_orig_img}"
        repr_str += f"infer_model_type={self.infer_model_type}"
        repr_str += f"cam_info={self.cam_info}"
        return repr_str
