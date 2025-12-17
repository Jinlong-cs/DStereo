# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import json
import logging
from typing import Dict, List

import cv2
import numpy as np

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.mxrecord import unpack
from hat.utils.pack_type.recordio_pb2 import RecordUnit

logger = logging.getLogger(__name__)

__all__ = [
    "ActionGetMetaData",
    "ActionGetImgClip",
]


@OBJECT_REGISTRY.register
class ActionGetMetaData(object):
    """A tool to sample meta data of one clip from video roidb.

    Args:
        tasks: A list of tasks names,
            Default to ['fall'].
        seq_len: Sequence length for clip
            Default to 32.
        feat_len: Sequence length for clip
            Default to 21*4, (num of hand kps * dim-x,y,z,score).
        box_len: Bbox dimension
            Default to 4.
        stride: Time stride of clip, set by virtual frame rate.
            Example: Want to sample 32(seq_len) frame from
            0.5 seconds(duration of video clip).
            So virtual frame rate is set to 64 fps(32/0.5).
            Time stride is set to 1/64.
            Default to 0.0156.
        label_assign_method: Mode of parsing recfile, support "key_frame".
            Default to key_frame.
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
        temporal_scale_range: Temporal jitter range.
            Example: Before temporal rand scale, want to sample 32(seq_len)
            frame from 0.5 seconds(duration of video clip).
            Suppose temporal scale is 2, actually sample 32(seq_len) frame
            from 0.5*2 seconds(duration of video clip).
            Default to [0.5, 2.0].

    Returns:
        clip_kps_meta_info: meta info of clip.
            clip_idx: frame idxs in video clip, such as 0 idx(first frame).
            act_label: label for video clip.
            seq_label: label for per img in clip.
            clip_boxes: list of roi bboxes, such as hand boxes for gesture.
            clip_keypoints: list of keypoints, such as hand kps for gesture.
            hand_left_right_attr: Left and right properties of the hand.
            img_shape: (height, width).
            image_name: name of image.
            image_path: path of image.
            track_ids: list of track id, such as hand trackid for gesure.
            ignore: list of bool,each element indicate
            whether this frame needs to be ignored.
            occlusion: list of bool,each element indicates
            whether there is occlusion.
    """

    def __init__(
        self,
        tasks: List = ("hgr",),
        seq_len: int = 32,
        feat_len: int = 84,
        box_len: int = 4,
        time_stride: float = 0.0156,
        label_assign_method: str = "key_frame",
        temporal_jitter: bool = False,
        temporal_jitter_type: str = "seq",
        temporal_jitter_range: float = 0.5,
        temporal_rand_scale: bool = False,
        temporal_scale_range: List = (0.5, 2.0),
    ):
        self._tasks = list(tasks)

        # params of seq
        self._seq_len = seq_len  # temporal
        self._feat_len = feat_len  # spatial
        self._box_len = box_len
        self._time_stride = time_stride
        # params of define gt
        self._label_assign_method = label_assign_method
        # params of temporal jitter
        self._temporal_jitter = temporal_jitter
        self._temporal_jitter_type = temporal_jitter_type
        self._temporal_jitter_range = temporal_jitter_range
        self._temporal_rand_scale = temporal_rand_scale
        self._temporal_scale_range = list(temporal_scale_range)

    def _prepare_sample_param(self, video_roidb):
        # sample params: time stride
        # scale
        time_stride = self._time_stride
        if self._temporal_rand_scale:
            # be careful to change video fps
            if bool(np.random.randint(0, 2)):
                # random scale of temporal stride
                ratio = np.random.uniform(
                    self._temporal_scale_range[0],
                    self._temporal_scale_range[1],
                )
                time_stride = self._time_stride * ratio
        self.frame_stride = time_stride * video_roidb["fps"]

        # jitter
        self.temporal_jitter = (
            self._temporal_jitter and np.random.randint(0, 2) > 0
        )
        temporal_jitter_range = round(
            video_roidb["fps"] * self._temporal_jitter_range
        )
        if self.temporal_jitter:
            assert self._temporal_jitter_type in [
                "seq",
                "frame",
            ], "only support seq or frame temporal jitter"
            if self._temporal_jitter_type == "seq":
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

    def _get_frame_id(self, ctr_frame_idx, frame_offset, max_frame_id):
        # center_idx + offset*frame_stride(fps*duration/expected-feat-len)
        # e.g. 30+(-12)*(15[oms camera fps]*0.5[0.5 sec]/32[seq len])
        frame_id = int(
            ctr_frame_idx
            + (frame_offset - self._seq_len / 2) * self.frame_stride
        )
        if self.temporal_jitter:
            if self._temporal_jitter_type == "seq":
                frame_id += self.frame_shift_fixed
            else:
                frame_id += np.random.randint(
                    self.temporal_jitter_min, self.temporal_jitter_max
                )

        frame_id = min(max(0, frame_id), max_frame_id)
        return frame_id

    def _lista_elems_in_listb(self, lista: List, listb: List):
        return set(lista).issubset(set(listb))

    def _get_clip_meta_info(
        self,
        video_roidb: Dict,
        ctr_frame_idx: int,
        this_roi: Dict,
        roidbs: List,
    ):
        assert self._lista_elems_in_listb(self._tasks, video_roidb["tasks"])
        # Get meta info of video clip
        person_box_list = []
        person_pose_list = []
        person_hand_left_right_attr_list = []
        clip_idxs = []
        all_labels = []
        track_id_list = []
        occlusion_list = []
        ignore_list = []
        width = -1
        height = -1
        label_idxs = np.array(
            [video_roidb["tasks"].index(_cls) for _cls in self._tasks]
        )
        max_frame_id = len(video_roidb["track_id"]) - 1
        box_len = self._box_len
        for frame_offset in range(self._seq_len):
            # get idx
            frame_id = self._get_frame_id(
                ctr_frame_idx, frame_offset, max_frame_id
            )

            frame_track_ids = video_roidb["track_id"][frame_id]
            frame_image_index = video_roidb["image_indexes"][frame_id]

            # get bbox, kps, label, etc.
            person_box = np.full((box_len,), -1, dtype=np.float32)
            person_pose = np.full((self._feat_len,), -1, dtype=np.float32)
            hand_left_right_attr = np.full((3,), -1, dtype=np.float32)
            labels = np.full(len(self._tasks), -1)
            track_id = -1
            roi_idx = -1
            ignore = "no"
            occlusion = "full_visible"

            if this_roi["track_id"] in frame_track_ids:
                if isinstance(frame_track_ids, np.ndarray):
                    roi_idx = np.where(
                        frame_track_ids == this_roi["track_id"]
                    )[0][0]
                elif isinstance(frame_track_ids, list):
                    roi_idx = frame_track_ids.index(this_roi["track_id"])
            if roi_idx >= 0:
                idx2 = roidbs[1].convert_idx_rec2roidb(frame_image_index)
                roi_rec_i = roidbs[1][idx2]
                assert roi_idx <= len(roi_rec_i["boxes"]) and roi_idx <= len(
                    roi_rec_i["keypoints"]
                ), "num trackid not match num boxes or num kps"
                # get bbox kps, such as hand bbox and kps for gesture task
                person_box = roi_rec_i["boxes"][roi_idx, :box_len].astype(
                    np.float32
                )
                person_pose = roi_rec_i["keypoints"][roi_idx, :].astype(
                    np.float32
                )
                # get Left and right hand properties
                if "hand_left_right_attr" in roi_rec_i:
                    hand_left_right_attr = roi_rec_i["hand_left_right_attr"][
                        roi_idx, :
                    ].astype(np.float32)
                elif "hand_attr" in roi_rec_i:
                    if isinstance(roi_rec_i["hand_attr"], list):
                        assert len(roi_rec_i["hand_attr"]) == 1
                        roi_rec_i["hand_attr"] = roi_rec_i["hand_attr"][
                            0
                        ].reshape((-1, 3))
                    hand_left_right_attr = roi_rec_i["hand_attr"][
                        roi_idx, :
                    ].astype(np.float32)
                assert (
                    len(hand_left_right_attr.shape) == 1
                    and hand_left_right_attr.shape[0] == 3
                )
                # label for frame
                labels = video_roidb["act_label"][frame_id][roi_idx][
                    label_idxs
                ]
                # track id
                track_id = this_roi["track_id"]
                if "occlusion" in roi_rec_i:
                    occlusion = roi_rec_i["occlusion"][roi_idx]
                    if not isinstance(occlusion, str):
                        occlusion = "full_visible"
                # ignore
                ignore = None
                if "ignore" in roi_rec_i:
                    ignore = roi_rec_i["ignore"][roi_idx]
                    if not isinstance(ignore, str):
                        ignore = "no"
                # img width and height
                if width == -1 and height == -1:
                    width = roi_rec_i["image_width"]
                    height = roi_rec_i["image_height"]
                else:
                    assert (
                        width == roi_rec_i["image_width"]
                        and height == roi_rec_i["image_height"]
                    )
            if person_pose.shape[0] == self._feat_len:
                person_pose = np.append(person_pose, np.float32(-1.0))
            else:
                assert person_pose.shape[0] == self._feat_len + 1

            person_box_list.append(copy.deepcopy(person_box))
            person_pose_list.append(copy.deepcopy(person_pose))
            person_hand_left_right_attr_list.append(
                copy.deepcopy(hand_left_right_attr)
            )
            all_labels.append(copy.deepcopy(labels))
            clip_idxs.append(frame_id)
            track_id_list.append(track_id)
            occlusion_list.append(occlusion)
            ignore_list.append(ignore)
            # be careful to shollow copy

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
        if self._label_assign_method == "key_frame":
            # get label for video clip
            all_labels = all_labels[int(self._seq_len / 2), :]
        else:
            raise NotImplementedError

        frame_idx2 = roidbs[1].convert_idx_rec2roidb(
            video_roidb["image_indexes"][ctr_frame_idx]
        )
        frame_idx_roi_rec = roidbs[1][frame_idx2]
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
            "image_name": video_roidb["image_names"][ctr_frame_idx],
            "image_path": frame_idx_roi_rec.get("image_path", ""),
            "track_ids": track_id_list,
            "ignore": ignore_list,
            "occlusion": occlusion_list,
            "kps_layout": "hwc",
        }
        return clip_kps_meta_info

    def __call__(self, data):
        assert "video_roidb" in data
        assert "roidbs" in data

        self._prepare_sample_param(data["video_roidb"])
        clip_kps_meta_info = self._get_clip_meta_info(
            video_roidb=data["video_roidb"],
            ctr_frame_idx=data["key_frame_idx"],
            this_roi=data["roi_key_frame"],
            roidbs=data["roidbs"],
        )
        data["clip_kps_meta_info"] = clip_kps_meta_info
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"tasks={self._tasks}"
        repr_str += f"seq_len={self._seq_len}"
        repr_str += f"feat_len={self._feat_len}"
        repr_str += f"box_len={self._box_len}"
        repr_str += f"time_stride={self._time_stride}"
        repr_str += f"label_assign_method={self._label_assign_method}"
        repr_str += f"temporal_jitter={self._temporal_jitter}"
        repr_str += f"temporal_jitter_type={self._temporal_jitter_type}"
        repr_str += f"temporal_jitter_range={self._temporal_jitter_range}"
        repr_str += f"temporal_rand_scale={self._temporal_rand_scale}"
        repr_str += f"temporal_scale_range={self._temporal_scale_range}"
        return repr_str


@OBJECT_REGISTRY.register
class ActionGetImgClip(object):
    """If need rgb branch, use this class to get img data.

    Args:
        to_rgb: whether convert bgr (cv2 imread) to rgb,
            Default to False.
        parser_rec_mode: Mode of parsing recfile, support "roirec".
            Default to 'roirec'.
    """

    def __init__(self, to_rgb: bool = False, parser_rec_mode: str = "roirec"):
        self._to_rgb = to_rgb
        self._parser_rec_mode = parser_rec_mode

    def _parser_roirec(self, raw_record: str):
        _, string_record = unpack(raw_record)
        rec_data = RecordUnit()
        rec_data.ParseFromString(string_record)
        rec_data = rec_data.body
        # rec_data: [hand1 hand2 ...]
        num_hand = len(rec_data.data)
        handimgs_per_frame = []
        for idx in range(num_hand - 1):
            hand_img = cv2.imdecode(
                np.frombuffer(rec_data.data[idx].value, dtype=np.uint8),
                flags=1,
            )
            if self._to_rgb:
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
            handimgs_per_frame.append(hand_img)
        # label:
        #  hand_boxes\crop_boxes\raw_shape\track_id\
        #  keypoints\image_name\image_shapes
        label = json.loads(bytes.decode(rec_data.data[-1].value))
        return handimgs_per_frame, label

    def _parser_rec(self, raw_record: bytes, mode: str = "roirec"):
        if mode == "roirec":
            return self._parser_roirec(raw_record)
        else:
            raise NotImplementedError(
                f"Parse mode support roirec. Not support {mode}"
            )

    def __call__(self, data):
        if "rec_packer" in data:
            video_idx = data["coord"][0]
            for frame_id in data["clip_kps_meta_info"]["clip_idx"]:
                image_index = data["roidbs"][0][video_idx]["image_indexes"][
                    frame_id
                ]
                raw_record = data["rec_packer"].read(image_index)
                rec_data = self._parser_rec(raw_record, self._parser_rec_mode)
                if len(rec_data) == 2:
                    img, rec_label = rec_data[0], rec_data[1]
                else:
                    img, rec_label = rec_data, None
                data["frames"].append(img)
                data["clip_rec_label"].append(rec_label)
            data["frame_layout"] = "hwc"

        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"to_rgb={self._to_rgb}"
        repr_str += f"parser_rec_mode={self._parser_rec_mode}"
        return repr_str
