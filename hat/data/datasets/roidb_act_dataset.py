# Copyright (c) Horizon Robotics. All rights reserved.
"""Dataset for mx-record data and roidb data, used in gesture."""

import copy
import json
import logging
import os
import pickle
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch.utils.data as data

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type import MXRecord, PackTypeMapper
from hat.utils.pack_type.lmdb import Lmdb
from hat.utils.pack_type.mxrecord import unpack
from hat.utils.pack_type.recordio_pb2 import RecordUnit

__all__ = ["RoidbActDataset", "GesDatasetLmdb"]
logger = logging.getLogger(__name__)


class ActRoidb(object):
    """ActRoidb provides the method of reading roidb (pickle) and rec.

    Args:
        roidb_path: Gt roidb path.
        rec_path: Image record path.
        rec_lst_file_path: Image record file path,
            default to None;
            if None use: the value will be rec_path.replace('.rec', '.lst')
    """

    def __init__(
        self,
        roidb_path: str,
        rec_path: str,
        rec_lst_file_path: Optional[str] = None,
    ):

        self.roidb_path = roidb_path
        self.rec_path = rec_path

        self.rec_lst_file_path = (
            rec_lst_file_path
            if rec_lst_file_path is not None
            else rec_path.replace(".rec", ".lst")
        )
        if not os.path.exists(self.rec_lst_file_path):
            raise FileNotFoundError(
                "rec lst file (%s) not found! " % self.rec_lst_file_path
            )
        self._prepare_roidb()

    def _prepare_roidb(self):
        self.anno_dataset = self._get_anno_dataset(self.roidb_path)
        self.img_lst = self._read_lst(self.rec_lst_file_path)
        self.add_img_ind_in_rec_for_roidb()

    def _get_anno_dataset(self, anno_path: str):
        with open(anno_path, "rb") as fn:
            anno = pickle.load(fn, encoding="latin1")
            return self._rename_keys(anno)

    def _rename_keys(self, anno: Dict):
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

    def _read_lst(self, lst_path: str):
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

    def add_img_ind_in_rec_for_roidb(self):

        for _anno in self.anno_dataset:
            if "image_name" in _anno:
                image_name = _anno["image_name"]
                if image_name not in self.img_lst:
                    image_name = os.path.basename(image_name)
                    assert image_name in self.img_lst, "{} {}".format(
                        image_name, _anno["image_name"]
                    )
                index = self.img_lst[image_name]
                if "image_index" in _anno:
                    assert _anno["image_index"] == index
                else:
                    _anno["image_index"] = index
            elif "image_names" in _anno:
                image_indexes = []
                for image_name in _anno["image_names"]:
                    if image_name not in self.img_lst:
                        image_name = os.path.basename(image_name)
                        assert image_name in self.image_lst, "{} {}".format(
                            image_name, _anno["image_name"]
                        )
                    image_indexes.append(self.img_lst[image_name])
                if "image_indexes" in _anno:
                    assert len(image_indexes) == len(_anno["image_indexes"])
                    for j in range(len(image_indexes)):
                        assert image_indexes[j] == _anno["image_indexes"][j]
                else:
                    _anno["image_indexes"] = image_indexes
            else:
                raise ValueError("image_name/image_names not in _anno")

    def convert_idx_rec2roidb(self, rec_idx: int):
        # get idx-roidb(multitask/det info) by rec idx
        if not hasattr(self, "map_idx_rec2roidb"):
            self.map_idx_rec2roidb = {}
            for roidb_idx, roi_rec in enumerate(self.anno_dataset):
                image_index = roi_rec["image_index"]
                assert image_index not in self.map_idx_rec2roidb
                self.map_idx_rec2roidb[image_index] = roidb_idx
        return self.map_idx_rec2roidb[rec_idx]

    def __getitem__(self, idx):
        return self.anno_dataset[idx]

    def __len__(self):
        return len(self.anno_dataset)


class ACTRoidbRec(object):
    """ActRoidbRec for gesture.

    Read raw image,hand bbox,hand landmark,label and so on.
    Organizing the annotation information of single video.

    Args:
        roidb: Including video roidb and img roidb.
        rec_packer: Method for parsing recfile.
    """

    def __init__(
        self,
        roidb: List,
        rec_packer: MXRecord,
    ):
        self.roidb = roidb
        self.rec_packer = rec_packer

        self._all_sample_coords = []  # [(video_id, frame_id, roi_id), ...]
        self._label_dict = {}  # {0: label0_indexes, 1: label1_indexes, ...}
        self.reset()

    def reset(self):
        if len(self._all_sample_coords) < 1:
            self._gen_label_idx_dict(self.roidb)

    def _gen_label_idx_dict(self, roidb: List):
        # set flag for balance sampler

        assert isinstance(roidb[0][0], dict)
        # input roidb: [seq_roidb(video per sample) including label,
        #           roidb(img per sample)     including detinfo.]
        self._total_samples = 0
        _flag = []
        for video_id, roi_dict in enumerate(roidb[0]):
            for frame_id in range(len(roi_dict["image_names"])):
                for roi_id in range(len(roi_dict["act_label"][frame_id])):

                    self._all_sample_coords.append(
                        (video_id, frame_id, roi_id)
                    )
                    label = roi_dict["act_label"][frame_id][roi_id]
                    if label.size == 0:
                        # todo: add other filters
                        continue
                    label = int(label)

                    self._label_dict.setdefault(label, []).append(
                        self._total_samples
                    )
                    _flag.append(label)
                    self._total_samples += 1

        self.flag = np.array(_flag).astype(np.uint8)
        log_str = ""
        for label in self._label_dict:
            log_str += "There are {} samples with label {}. \n".format(
                len(self._label_dict[label]), label
            )
        logger.info(log_str)

    def _convert_idx_global2coord(self, idx: int):
        # convert global index to sample coord(video_idx, frame_idx, roi_idx)
        coord = self._all_sample_coords[idx]
        return coord

    def _sample_clip_by_coord(self, coord: Tuple):
        # sample by coord
        (video_idx, key_frame_idx, roi_idx) = coord
        video_roidb = self.roidb[0][video_idx]
        # key/center frame roi
        roi_key_frame = self._get_roi(video_idx, key_frame_idx, roi_idx)
        return video_roidb, key_frame_idx, roi_key_frame

    def _get_roi(self, video_idx: int, frame_idx: int, roi_idx: int):
        video_roidb = self.roidb[0][video_idx]
        roi = {
            "image_name": video_roidb["image_names"][frame_idx],
            "image_index": video_roidb["image_indexes"][frame_idx],
            "track_id": video_roidb["track_id"][frame_idx][roi_idx],
        }
        return roi

    def __len__(self):
        return self._total_samples

    def __getitem__(self, index: int):
        # convert idx to coord
        coord = self._convert_idx_global2coord(index)
        video_roidb, key_frame_idx, roi_key_frame = self._sample_clip_by_coord(
            coord
        )

        clip_img = []
        clip_rec_label = []

        data = {
            "video_roidb": video_roidb,
            "key_frame_idx": key_frame_idx,
            "roi_key_frame": roi_key_frame,
            "roidbs": self.roidb,
            "frames": clip_img,
            "clip_rec_label": clip_rec_label,
        }
        # add img for rgb branch
        if self.rec_packer:
            data.update(
                {
                    "coord": coord,
                    "rec_packer": self.rec_packer,
                }
            )

        return data


@OBJECT_REGISTRY.register
class RoidbActDataset(data.Dataset):
    """Dataset for action(gesture) roidb & record data.

    Args:
        rec_path: Path of recfile.
        anno_path: Path of annotation.
        roidb_path: Path of img roidb,
            including the prelabel info of each img.
        roidb_seq_path: Path of video roidb,
            including the annotation of each video.
        transforms: List of transform,
            default to None.
        is_read_rec: If read Image from rec,
            default to False;
        rec_idx_file_path: Image record index file path,
            default to None;
            if None use: the value will be rec_path.replace('.rec', '.idx')
    """

    def __init__(
        self,
        rec_path: str,
        roidb_path: str,
        roidb_seq_path: str,
        transforms: Optional[List] = None,
        is_read_rec: Optional[bool] = False,
        rec_idx_file_path: Optional[str] = None,
    ):
        self.rec_path = rec_path
        self.roidb_path = roidb_path
        self.roidb_seq_path = roidb_seq_path
        self.transforms = transforms
        self.is_read_rec = is_read_rec
        self.rec_idx_file_path = (
            rec_idx_file_path
            if rec_idx_file_path is not None
            else rec_path.replace(".rec", ".idx")
        )
        self._prepare()

    def _prepare(self):
        # init roidb including det info
        self.roidb = ActRoidb(
            roidb_path=self.roidb_path,
            rec_path=self.rec_path,
        )

        # init seq roidb including label
        self.roidb_seq = ActRoidb(
            roidb_path=self.roidb_seq_path,
            rec_path=self.rec_path,
        )

        # init dataset of img
        if self.is_read_rec:
            self._init_pack()
        else:
            self.pack_file = None

        # create act roidb dataset
        self.act_roidb_dataset = ACTRoidbRec(
            roidb=[self.roidb_seq, self.roidb],
            rec_packer=self.pack_file,
        )

        # set frame label to flag
        self._set_flag()

    def _init_pack(self):
        if not os.path.exists(self.rec_idx_file_path):
            raise FileNotFoundError(
                "rec idx file (%s) not found! " % self.rec_idx_file_path
            )
        self.pack_type = PackTypeMapper["mxrecord"]
        self.pack_file = self.pack_type(
            uri=self.rec_path, idx_path=self.rec_idx_file_path, writable=False
        )
        self.pack_file.open()

    def _set_flag(self):
        self.flag = self.act_roidb_dataset.flag
        assert self.flag.shape[0] == len(
            self
        ), "length of dataset must be equ to length of flag"

    def __len__(self):
        return len(self.act_roidb_dataset)

    def __repr__(self):
        return "RoidbActDataset"

    def __getitem__(self, idx: int):
        # get clip data by idx
        data = self.act_roidb_dataset[idx]

        # transform
        if self.transforms is not None:
            data = self.transforms(data)
        return data


@OBJECT_REGISTRY.register
class GesDatasetLmdb(data.Dataset):
    """Dataset for action(gesture) lmdb data.

    Args:
        lmdb_path: Path of lmdb file.
        transforms: data augmentation transform.
        seq_len: Sequence length for clip
            Default to 16.
        box_len: Bbox dimension
            Default to 4.
        feat_len: Sequence length for clip
            Default to 21*4, (num of hand kps * dim-x,y,z,score).
        time_stride: Time stride of clip, set by virtual frame rate.
            Example: Want to sample 32(seq_len) frame from
            0.5 seconds(duration of video clip).
            So virtual frame rate is set to 64 fps(32/0.5).
            Time stride is set to 1/64.
            Default to 0.0156.
        temporal_jitter: Whether applying temporal jitter
            during sampling frame.
            Default to False.
        temporal_jitter_type: Type of temporal jitter.
            Must be 'seq' or 'frame'.
            Default to 'seq'.
        temporal_jitter_range: Max range of temporal jitter.
            Temporal shift a whole clip in [-fps*temporal_jitter_range,
            fps*temporal_jitter_range].
            Default to 0.5.
        temporal_rand_scale: Whether applying temporal random scale.
            Change time stride by scale.
            Default to False.
        use_rgb_branch: use rgb branch or not
        to_rgb: convert img from lmdb or not
        mode: a tag, 'train' or 'val'
    """

    def __init__(
        self,
        lmdb_path: str,
        transforms: Optional[Callable] = None,
        seq_len: int = 16,
        box_len: int = 4,
        feat_len: int = 84,
        time_stride: float = 0.0156,
        temporal_jitter: bool = True,
        temporal_jitter_type: str = "seq",
        temporal_jitter_range: float = 0.1,
        temporal_rand_scale: bool = True,
        temporal_scale_range: tuple = (0.5, 2.0),
        use_rgb_branch: bool = True,
        to_rgb: bool = False,
        mode: str = "train",
    ):
        self.lmdb_path = lmdb_path
        self.transforms = transforms
        self.seq_len = seq_len
        self.box_len = box_len
        self.feat_len = feat_len
        self.time_stride = time_stride
        self.temporal_jitter = temporal_jitter
        self.temporal_jitter_type = temporal_jitter_type
        self.temporal_jitter_range = temporal_jitter_range
        self.temporal_rand_scale = temporal_rand_scale
        self.temporal_scale_range = temporal_scale_range
        self.use_rgb_branch = use_rgb_branch
        self.to_rgb = to_rgb
        self.mode = mode

        self._make_dataset()

    def _make_dataset(self):
        self.cur_fps = None

        data_base_name = os.path.basename(self.lmdb_path)
        anno_path = f"{self.lmdb_path}/annotation.json"

        with open(anno_path, "r") as f:
            anno = json.load(f)
        self.loc_list = anno["loc_list"]
        self.vid_fid_map = anno["vid_fid_map"]
        self.flag = anno["flag"]
        self.label_count = anno["label_count"]
        del anno

        self.lmdb_reader = Lmdb(
            self.lmdb_path,
            writable=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        log_str = ""
        log_str += f"\n------- {self.mode} data statistic ------- \n"
        log_str += f"{data_base_name}\n"
        log_str += f"{'label_id':>10}{'rnum':>10} \n"
        sorted_label_id = sorted(self.label_count.keys())
        total = 0
        for label_id in sorted_label_id:
            num = self.label_count[label_id]
            log_str += f"{label_id:>10}{num:>10} \n"
            total += num
        log_str += f"{'total':>10}{total:>10} \n"
        log_str += f"{'-' * 36}\n"
        logger.info(log_str)

    def __len__(self):
        return len(self.loc_list)

    def __repr__(self):
        return "GesDatasetLmdb"

    def _prepare_sample_param(self, fps):
        # sample params: time stride
        # scale

        time_stride = self.time_stride
        if self.temporal_rand_scale:
            # be careful to change video fps
            if bool(np.random.randint(0, 2)):
                # random scale of temporal stride
                ratio = np.random.uniform(
                    self.temporal_scale_range[0],
                    self.temporal_scale_range[1],
                )
                time_stride = self.time_stride * ratio
        self.frame_stride = time_stride * fps

        # jitter
        self.temporal_jitter = (
            self.temporal_jitter and np.random.randint(0, 2) > 0
        )
        temporal_jitter_range = round(fps * self.temporal_jitter_range)
        if self.temporal_jitter:
            assert self.temporal_jitter_type in [
                "seq",
                "frame",
            ], "only support seq or frame temporal jitter"
            if self.temporal_jitter_type == "seq":
                # temporal shift a whole clip
                self.frame_shift_fixed = np.random.randint(
                    -temporal_jitter_range, temporal_jitter_range + 1
                )
            else:
                assert temporal_jitter_range * 2 < self.frame_stride, (
                    "it might be out of order of a sequence if it "
                    "is not satisfied: %d < %d"
                    % (temporal_jitter_range * 2, self.frame_stride)
                )
                self.temporal_jitter_min = -temporal_jitter_range
                self.temporal_jitter_max = temporal_jitter_range + 1

    def _get_clip_data(self, loc):
        data = {}
        video_id, center_frame_id, center_track_id = [
            int(i) for i in loc.split("_")
        ]
        person_box_list = []
        person_pose_list = []
        person_hand_left_right_attr_list = []
        clip_idxs = []
        all_labels = []
        track_id_list = []
        occlusion_list = []
        ignore_list = []
        frame_list = []
        width = -1
        height = -1
        clip_rec_label = []

        center_data_item = self.lmdb_reader.get(
            f"{video_id}_{center_frame_id}".encode()
        )
        center_data_item = pickle.loads(center_data_item)

        fps = center_data_item["fps"]
        if self.cur_fps != fps:
            self._prepare_sample_param(fps)
            self.cur_fps == fps  # noqa
        max_frame_id = self.vid_fid_map[str(video_id)] - 1
        for frame_offset in range(self.seq_len):
            select_frame_id = self._get_frame_id(
                center_frame_id, frame_offset, max_frame_id
            )
            select_frame_loc = f"{video_id}_{select_frame_id}"
            select_loc = f"{video_id}_{select_frame_id}_{center_track_id}"

            data_item = self.lmdb_reader.get(select_frame_loc.encode())
            data_item = pickle.loads(data_item)

            if self.use_rgb_branch:
                raw_rec_data = data_item["rec_data"]
                rec_data = self._parser_rec(raw_rec_data)
                if len(rec_data) == 2:
                    frame, rec_label = rec_data[0], rec_data[1]
                else:
                    frame, rec_label = rec_data, None
                clip_rec_label.append(rec_label)

            # if the roi of this track id exist
            if select_loc in self.loc_list:

                data_track_id_list = [int(i) for i in data_item["track_id"]]
                roi_idx = data_track_id_list.index(int(center_track_id))

                person_box = data_item["boxes"][
                    roi_idx, : self.box_len
                ].astype(np.float32)
                person_pose = data_item["keypoints"][roi_idx, :].astype(
                    np.float32
                )

                if "hand_left_right_attr" in data_item:
                    hand_left_right_attr = data_item["hand_left_right_attr"][
                        roi_idx, :
                    ].astype(np.float32)
                elif "hand_attr" in data_item:
                    if isinstance(data_item["hand_attr"], list):
                        assert len(data_item["hand_attr"]) == 1
                        data_item["hand_attr"] = data_item["hand_attr"][
                            0
                        ].reshape((-1, 3))
                    hand_left_right_attr = data_item["hand_attr"][
                        roi_idx, :
                    ].astype(np.float32)

                assert (
                    len(hand_left_right_attr.shape) == 1
                    and hand_left_right_attr.shape[0] == 3
                )

                # label for frame
                labels = data_item["act_label"][roi_idx]

                if "occlusion" in data_item:
                    occlusion = data_item["occlusion"][roi_idx]
                    if not isinstance(occlusion, str):
                        occlusion = "full_visible"
                # ignore
                ignore = None
                if "ignore" in data_item:
                    ignore = data_item["ignore"][roi_idx]
                    if not isinstance(ignore, str):
                        ignore = "no"
                # img width and height
                if width == -1 and height == -1:
                    width = data_item["image_width"]
                    height = data_item["image_height"]
                else:
                    assert (
                        width == data_item["image_width"]
                        and height == data_item["image_height"]
                    )
                track_id = center_track_id
            else:
                # get bbox, kps, label, etc.
                person_box = np.full((self.box_len,), -1, dtype=np.float32)
                person_pose = np.full(
                    (self.feat_len + 1,), -1, dtype=np.float32
                )
                hand_left_right_attr = np.full((3,), -1, dtype=np.float32)
                labels = np.full(1, -1)

                ignore = "no"
                occlusion = "full_visible"
                track_id = -1

                # del data_item
            if person_pose.shape[0] == self.feat_len:
                person_pose = np.append(person_pose, np.float32(-1.0))
            else:
                assert (
                    person_pose.shape[0] == self.feat_len + 1
                ), person_pose.shape

            person_box_list.append(copy.deepcopy(person_box))
            person_pose_list.append(copy.deepcopy(person_pose))
            person_hand_left_right_attr_list.append(
                copy.deepcopy(hand_left_right_attr)
            )
            all_labels.append(copy.deepcopy(labels))
            clip_idxs.append(select_frame_id)
            track_id_list.append(track_id)
            occlusion_list.append(occlusion)
            ignore_list.append(ignore)
            if self.use_rgb_branch:
                frame_list.append(np.array(frame))

        clip_idxs = np.array(clip_idxs, np.int32)
        person_boxes = np.array(person_box_list).astype(np.float32)
        person_keypoints = np.array(person_pose_list).astype(np.float32)
        person_hand_left_right_attrs = np.array(
            person_hand_left_right_attr_list
        ).astype(np.float32)
        all_labels = np.array(all_labels).astype(np.int32)
        seq_labels = all_labels.copy()
        assert seq_labels.shape[1] == 1
        seq_labels = seq_labels.reshape(-1)

        # get label for video clip
        all_labels = all_labels[int(self.seq_len / 2), :]

        clip_kps_meta_info = {
            "clip_idx": clip_idxs,
            "act_label": all_labels,
            "seq_label": seq_labels,
            "clip_boxes": person_boxes,
            "clip_keypoints": person_keypoints,
            "attrs": {
                "hand_left_right_attr": person_hand_left_right_attrs,
            },
            "img_shape": (height, width),
            "track_ids": track_id_list,
            "ignore": ignore_list,
            "occlusion": occlusion_list,
            "kps_layout": "hwc",
        }
        data["clip_kps_meta_info"] = clip_kps_meta_info
        data["frames"] = frame_list
        data["frame_layout"] = "hwc"
        data["clip_rec_label"] = clip_rec_label
        return data

    def _get_frame_id(self, ctr_frame_idx, frame_offset, max_frame_id):
        # center_idx + offset*frame_stride(fps*duration/expected-feat-len)
        # e.g. 30+(-12)*(15[oms camera fps]*0.5[0.5 sec]/32[seq len])
        frame_id = int(
            ctr_frame_idx
            + (frame_offset - self.seq_len / 2) * self.frame_stride
        )
        if self.temporal_jitter:
            if self.temporal_jitter_type == "seq":
                frame_id += self.frame_shift_fixed
            else:
                frame_id += np.random.randint(
                    self.temporal_jitter_min, self.temporal_jitter_max
                )

        frame_id = min(max(0, frame_id), max_frame_id)
        return frame_id

    def _parser_rec(self, raw_record: str):
        _, string_record = unpack(raw_record)
        rec_data = RecordUnit()
        rec_data.ParseFromString(string_record)
        rec_data = rec_data.body
        num_hand = len(rec_data.data)
        handimgs_per_frame = []
        for idx in range(num_hand - 1):
            hand_img = cv2.imdecode(
                np.frombuffer(rec_data.data[idx].value, dtype=np.uint8),
                flags=1,
            )
            if self.to_rgb:
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
            handimgs_per_frame.append(hand_img)
        # label:
        #  hand_boxes\crop_boxes\raw_shape\track_id\
        #  keypoints\image_name\image_shapes
        label = json.loads(bytes.decode(rec_data.data[-1].value))

        return handimgs_per_frame, label

    def __getitem__(self, idx: int):

        loc = self.loc_list[idx]
        data = self._get_clip_data(loc)
        if self.transforms is not None:
            data = self.transforms(data)

        return data
