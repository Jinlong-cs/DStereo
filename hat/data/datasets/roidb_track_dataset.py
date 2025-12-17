# Copyright (c) Horizon Robotics. All rights reserved.
"""Dataset for tracking mx-record data, used in auto halo."""
import logging
import pickle
from copy import deepcopy
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch.utils.data as data

from hat.core.box_utils import bbox_overlaps
from hat.data.datasets.roidb_detection_dataset import RoidbDataset
from hat.registry import OBJECT_REGISTRY

__all__ = ["RoidbTrackDataset"]


@OBJECT_REGISTRY.register
class RoidbTrackDataset(data.Dataset):
    """Dataset for tracking roidb & seq_roidb & record data.

    Args:
        data_path: Path of data relative to bucket path.
        anno_path: Path of annotation.
        anno_seq_path: Path of video annotation.
        selected_class_ids: List of selected class ids,
            classes that are not in this list will be filter out.
        transforms: List of sequence transform,
            Default to None.
        to_rgb: If convert bgr (cv2 imread) to rgb,
            Default to False.
        min_bbox_size: min size for bbox width and height,
            size smaller than this value will be filter out, default to 4.
        ign_heavy_occlusion: whether to set the heavy occlusion bbox
            track id to -1, which means track match loss on this bbox
            will be 0. If True, set track_id of heavily occluded bbox
            to -1.
            Default to True.
        only_ign_occlu_by_bbox: If only_ign_occlu_by_bbox is True and
            ign_heavy_occlusion is True, do ignore on bbox which only
            occluded by foreground bbox.
            Default to True.
        keep_ori_img: If keep original rgb image to data, used to
            visualize the predictor result.
            Default to False.
        max_sample_interval: max interval for sampling continuous frames.
            Default to 1.
        epoch_seq_length_map: sequence length changes according to the epoch.
            whenever the train epoch step to the value defined in this map
            keys, the sampling sequence length would changed to the relation
            value.
            Default to None, fixed sampling two frames per sequence.
        init_seq_length: init sequence length.
            Default to 2.
        data_desc: Data description, used in evaluation process (datasets
            will be filtered and gathered by data_desc);
            Default to ``None``.
    """

    def __init__(
        self,
        data_path: str,
        anno_path: str,
        anno_seq_path: str,
        selected_class_ids: List[int],
        transforms: Optional[List] = None,
        to_rgb: bool = False,
        min_bbox_size: int = 4,
        ign_heavy_occlusion: bool = True,
        only_ign_occlu_by_bbox: bool = True,
        keep_ori_img: bool = False,
        max_sample_interval: int = 1,
        epoch_seq_length_map: Optional[Dict] = None,
        init_seq_length: int = 2,
        data_desc: Optional[str] = None,
    ):
        self.data_path = data_path
        self.anno_path = anno_path
        self.anno_seq_path = anno_seq_path
        self.data_desc = data_desc
        self.keep_ori_img = keep_ori_img
        self.seq_transforms = transforms
        self.to_rgb = to_rgb
        self.min_bbox_size = min_bbox_size
        self.max_sample_interval = max_sample_interval
        self.epoch_seq_length_map = epoch_seq_length_map

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
        self.ign_heavy_occlusion = ign_heavy_occlusion
        self.only_ign_occlu_by_bbox = only_ign_occlu_by_bbox
        self.num_frames_per_seq = init_seq_length
        self._init_dataset()

    def _init_dataset(self):
        self.roidb_dataset = RoidbDataset(
            roidb_path=self.anno_path,
            rec_path=self.data_path,
            data_desc=self.data_desc,
        )

        if self.ign_heavy_occlusion:
            self._ign_heavy_occlu_boxes_id(self.only_ign_occlu_by_bbox)

        self._gen_seq_info()

        # default to sampling 2 frames for per seq
        self.set_num_frames_per_seq(self.num_frames_per_seq)
        self._gen_all_sample_data()

        logging.debug(
            f"dataset path: {self.data_path}, \
                {self.anno_path}, {self.anno_seq_path}"
        )
        logging.debug(f"all frames length: {len(self.all_sample_data)}")

    def __getstate__(self):
        state = self.__dict__
        state["roidb_dataset"] = None
        state["seq_info"] = None
        # state["num_frames_per_seq"] = None
        state["all_sample_data"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._init_dataset()

    def _ign_heavy_occlu_boxes_id(self, only_ign_occlu_by_bbox):
        """Set some heavy occlusion boxes track id to -1.

        set some heavy occlusion bbox (optional to do ignore only on bbox which
        is only occluded by other bbox) track id to -1, means that matched loss
        on this bbox would be 0.
        """
        for _anno in self.roidb_dataset.anno_dataset:
            if (
                ("track_id" in _anno)
                and ("occlusion" in _anno)
                and ("boxes" in _anno)
            ):  # noqa
                assert (
                    len(_anno["track_id"])
                    == len(_anno["boxes"])
                    == len(_anno["occlusion"])
                )
                occlusion_list = _anno["occlusion"]
                if len(occlusion_list) <= 1:
                    continue
                heavy_occlusion_inds = [
                    occlu == "heavily_occluded" for occlu in occlusion_list
                ]
                if only_ign_occlu_by_bbox:
                    boxes = _anno["boxes"]
                    overlaps = bbox_overlaps(boxes, boxes)
                    row, col = np.diag_indices_from(overlaps)
                    overlaps[row, col] = 0.0
                    max_overlaps = np.max(overlaps, axis=1)
                    cross_inds = max_overlaps > 0.5
                    # only ignore on this case:
                    # heavy occlusion object and IOU with other foreground
                    # objects big than 0.5, will be ignored in track loss.
                    heavy_occlusion_inds &= cross_inds
                _anno["track_id"][heavy_occlusion_inds] = -1

    def _gen_seq_info(self):
        self.seq_info = {}
        with open(self.anno_seq_path, "rb") as fn:
            anno_seq_list = pickle.load(fn, encoding="latin1")
            for seq_i, anno_seq_i in enumerate(anno_seq_list):
                seq_idx_list = anno_seq_i["image_indexes"]
                seq_anno_idx_list = [
                    self.roidb_dataset.img_idx_2_anno_idx[img_idx]
                    for img_idx in seq_idx_list
                ]
                self.seq_info[seq_i] = seq_anno_idx_list

    def _gen_all_sample_data(self):
        self.all_sample_data = []
        for seq_i in self.seq_info.keys():
            for anno_idx in self.seq_info[seq_i][
                0 : max(
                    1, len(self.seq_info[seq_i]) - self.num_frames_per_seq + 1
                )
            ]:
                self.all_sample_data.append((seq_i, anno_idx))

    def _pre_single_frame(self, index: int) -> Dict:
        data = {}
        image, anno = self.roidb_dataset[index]
        color_space = "bgr"
        if self.keep_ori_img:
            data["ori_img"] = deepcopy(image)
        if self.to_rgb:
            # cv2.cvtColor may be slow.
            # See http://wiki.hobot.cc/pages/viewpage.action?pageId=186775106 for more details. # noqa
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color_space = "rgb"

        if self.data_desc is not None:
            data["data_desc"] = anno["data_desc"]
            data["frame_index"] = anno["frame_index"]
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
        gt_ids = []
        for bbox, class_id, track_id in zip(
            anno["boxes"], anno["classes"], anno["track_id"]
        ):
            # filter invalid bbox: min_bbox_size < 4
            if (bbox[2] - bbox[0] < self.min_bbox_size) or (
                bbox[3] - bbox[1] < self.min_bbox_size
            ):
                continue

            if class_id in self.valid_selected_class_ids:
                gt_bboxes.append(bbox)
                gt_classes.append(self.class_id_map[class_id])
                gt_ids.append(track_id)
            elif -class_id in self.valid_selected_class_ids:
                gt_bboxes.append(bbox)
                gt_classes.append(-self.class_id_map[-class_id])
                gt_ids.append(track_id)

        data["gt_bboxes"] = np.array(gt_bboxes).reshape((-1, 4))
        data["gt_classes"] = np.array(gt_classes, dtype=np.int64).reshape(
            (-1,)
        )
        data["gt_ids"] = np.array(gt_ids, dtype=np.int64).reshape((-1,))
        return data

    def sample_indices(self, seq_idx, anno_idx):
        rate = np.random.randint(1, self.max_sample_interval + 1)
        tmax = self.seq_info[seq_idx][-1]
        ids = [anno_idx + rate * i for i in range(self.num_frames_per_seq)]
        return [min(i, tmax) for i in ids]

    def pre_continuous_frames(self, idxs):
        frame_data_list = [self._pre_single_frame(i) for i in idxs]
        data_seq = {"frame_data_list": frame_data_list}
        return data_seq

    def __getitem__(self, index: int) -> Dict:
        # on diff epoch, seq_length changed and the data length changed,
        # but the sampler index not changed, sampled index would out
        # data length, here deal with this case.
        if index >= len(self):
            index = np.random.randint(0, len(self))
        seq_idx, anno_idx = self.all_sample_data[index]
        idxs = self.sample_indices(seq_idx, anno_idx)
        data_seq = self.pre_continuous_frames(idxs)
        if self.seq_transforms is not None:
            data_seq = self.seq_transforms(data_seq)
        return data_seq

    def __len__(self):
        return len(self.all_sample_data)

    def __repr__(self):
        return "RoidbTrackDataset"

    def set_num_frames_per_seq(self, num):
        self.num_frames_per_seq = num
        logging.info(f"set num_frames_per_seq: {num}")

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if (
            self.epoch_seq_length_map is None
            or len(self.epoch_seq_length_map) == 0
        ):
            # fixed sampling length.
            return
        epoch_steps = sorted(self.epoch_seq_length_map.keys())
        for i in range(len(epoch_steps)):
            if epoch >= epoch_steps[i]:
                self.period_idx = i
        self.set_num_frames_per_seq(
            self.epoch_seq_length_map[epoch_steps[self.period_idx]]
        )
        # _gen_all_sample_data do on __setstate__
        # here self.seq_info is None, do _gen_all_sample_data would
        # raise Error
        # self._gen_all_sample_data()
