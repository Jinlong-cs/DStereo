# Copyright (c) Horizon Robotics. All rights reserved.
"""Dataset for mx-record rec data, used in phone."""

import json
import logging
import os
from typing import List, Optional

import numpy as np
import torch.utils.data as data

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type import PackTypeMapper
from hat.utils.package_helper import require_packages

try:
    import mxnet as mx
except ImportError:
    mx = None

__all__ = ["SmokeClsDataset"]
logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class SmokeClsDataset(data.Dataset):
    """Dataset for smoke classification task.

    Args:
        rec_list: List of rec_name or rec_path.
        all_rec_info_list: List of custom rec_info files.
        all_rec_imgIdx_list:
            List of custom mapping files between img label and idx in rec,
        transforms: List of transform,
            default to None.
        data_root_path: folder path of rec files,
            default to None;
        mode: train or val etc.
    """

    @require_packages("mxnet")
    def __init__(
        self,
        rec_list: List,
        all_rec_info_list: List,
        all_rec_imgIdx_list: List,
        transforms: Optional[List] = None,
        data_root_path: Optional[str] = None,
        mode: str = "train",
    ):
        self.rec_list = rec_list
        self.all_rec_info_list = all_rec_info_list
        self.all_rec_imgIdx_list = all_rec_imgIdx_list
        self.transforms = transforms
        self.data_root_path = data_root_path
        self.mode = mode

        self._prepare()

    def _prepare(self):

        # init rec_info_map and rec_imgIdx_map
        self._rec_dataset = {}
        self._all_sample_coords = []
        self.rec_info_map = {}
        self.rec_imgIdx_map = {}
        all_rec_info_map = {}
        all_rec_imgIdx_map = {}

        for rec_info_path in self.all_rec_info_list:
            with open(rec_info_path, "r", encoding="utf-8") as w:
                all_rec_info_map.update(json.load(w))
        for rec_imgIdx_path in self.all_rec_imgIdx_list:
            with open(rec_imgIdx_path, "r", encoding="utf-8") as w:
                all_rec_imgIdx_map.update(json.load(w))

        for rec in self.rec_list:
            rec_name = os.path.splitext(os.path.basename(rec))[0]
            self.rec_info_map[rec_name] = all_rec_info_map[rec_name]
            self.rec_imgIdx_map[rec_name] = all_rec_imgIdx_map[rec_name]

            # Getting rec paths more flexible
            rec_path = self.rec_info_map[rec_name]["path"]
            if not os.path.exists(rec_path):
                if self.data_root_path:
                    form_root_path = os.path.join(
                        self.data_root_path, os.path.basename(rec)
                    )
                    if os.path.exists(form_root_path):
                        rec_path = form_root_path
                elif os.path.exists(rec):
                    rec_path = rec
                else:
                    raise FileNotFoundError(f"{rec} not found")

            idx_path = rec_path.replace("rec", "idx")
            rec_dataset = PackTypeMapper["mxrecord"](
                rec_path,
                idx_path,
                writable=False,
            )
            rec_dataset.open()
            self._rec_dataset[rec_name] = rec_dataset

        self._random_enlarge()

    def _random_enlarge(self):

        # Enlarge per-subclass number through rec_info file
        sta_data = {}
        for rec_name in self.rec_info_map:
            rec_info = self.rec_info_map[rec_name]
            rec_size = rec_info["rec_size"]
            aug_type = rec_info["aug_type"]
            roi_scale = rec_info.get("roi_scale", 0.625)
            assert rec_size == len(self._rec_dataset[rec_name].record.keys), (
                f"Size inconsistent,{rec_name}, "
                f"rec:{len(self._rec_dataset[rec_name])} anno:{rec_size}"
            )

            for label_id, data_info in rec_info["data"].items():
                dst_id = int(data_info["dst_id"])
                raw_num = int(data_info["raw_num"])
                aug_num = int(data_info["aug_num"])
                if aug_num == 0:
                    continue
                img_index_list = np.array(
                    self.rec_imgIdx_map[rec_name][label_id], dtype=int
                )
                real_num = len(img_index_list)
                if real_num != raw_num:
                    aug_num = int((aug_num / raw_num) * real_num)

                tmp_sample_coords = np.zeros(0, dtype=int)
                if self.mode == "train":
                    np.random.shuffle(img_index_list)
                    cnt = aug_num
                    while cnt > 0:
                        index = min(cnt, len(img_index_list))
                        tmp_sample_coords = np.concatenate(
                            (tmp_sample_coords, img_index_list[:index])
                        )
                        cnt -= index
                else:
                    tmp_sample_coords = img_index_list

                self._all_sample_coords.extend(
                    [
                        [rec_name, i, dst_id, aug_type, roi_scale]
                        for i in tmp_sample_coords
                    ]
                )
                assert len(tmp_sample_coords) == aug_num
                sta_data.setdefault(dst_id, {})
                sta_data[dst_id]["raw_num"] = (
                    sta_data[dst_id].get("raw_num", 0) + raw_num
                )
                sta_data[dst_id]["aug_num"] = (
                    sta_data[dst_id].get("aug_num", 0) + aug_num
                )
        log_str = ""
        log_str += f"-------{self.mode} data statistic ------- \n"
        log_str += f"{'dst_id':>10}{'raw_num':>10}{'aug_num':>10} \n"
        sorted_label_id = sorted(sta_data.keys())
        count = [0] * 2
        for label_id in sorted_label_id:
            raw_num = sta_data[label_id]["raw_num"]
            aug_num = sta_data[label_id]["aug_num"]
            log_str += f"{label_id:>10}{raw_num:>10}{aug_num:>10} \n"
            count[0] += raw_num
            count[1] += aug_num
        log_str += f"{'total':>10}{count[0]:>10}{count[1]:>10} \n"
        log_str += "------------------------------ \n"
        logger.info(log_str)

    def __len__(self):
        return len(self._all_sample_coords)

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": \n"
        repr_str += f"rec_list : {self.rec_list} \n"
        repr_str += f"all_rec_info_list : {self.all_rec_info_list} \n"
        repr_str += f"all_rec_imgIdx_list : {self.all_rec_imgIdx_list} \n"
        repr_str += f"Total number of data : {len(self._all_sample_coords)} \n"
        repr_str += f"running mode : {self.mode} \n"
        return repr_str

    def __getitem__(self, idx: int):
        # get data by idx
        (
            rec_name,
            img_idx,
            dst_id,
            aug_type,
            roi_scale,
        ) = self._all_sample_coords[idx]
        data = self._rec_dataset[rec_name].read(img_idx)
        header, img = mx.recordio.unpack_img(data)
        img_loc = f"{rec_name}_{img_idx}"

        data = {
            "img": img.astype(np.uint8),
            "labels": dst_id,
            "roi_scale": np.float32(roi_scale),
            "aug_type": aug_type,
            "img_loc": img_loc,
            "layout": "hwc",
        }

        # transform
        if self.transforms is not None:
            if isinstance(self.transforms, list):
                for transform in self.transforms:
                    data = transform(data)
            else:
                data = self.transforms(data)

        return data
