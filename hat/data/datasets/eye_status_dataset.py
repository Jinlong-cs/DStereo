# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Tuple

import numpy as np
import yaml
from torch.utils.data import ConcatDataset, Dataset

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type import PackTypeMapper
from hat.utils.pack_type.mxrecord import unpack_img

__all__ = ["EyeStatusDataset"]


EYE_STATUS_FIX_SIZE = 8


class EyeVisDataset(Dataset):
    """EyeVisDataset reads eye visual data from mx rec.

    Args:
        rec_path: Path to eye status rec.
        idx_path: Path to eye status idx.
        inf_path: Path to eye status info.
        transfroms : Transfroms of data before using.
    """

    def __init__(
        self,
        rec_path: str,
        idx_path: str,
        inf_path: str,
        transforms: Optional[List] = None,
    ):
        super(EyeVisDataset, self).__init__()
        self.rec_path = rec_path
        self.idx_path = idx_path
        self.inf_path = inf_path
        self.transforms = transforms
        self.pack_type = PackTypeMapper["mxrecord"]

        self._init_pack()

    def _init_pack(self):
        self.pack_file = self.pack_type(
            uri=self.rec_path, idx_path=self.idx_path, writable=False
        )
        self.pack_file.open()

        self._size = len(self.pack_file.record.keys)

        self.id2range = {}
        self.id_sample_num = {}
        data_offset = 0
        with open(self.inf_path, "r") as fr:
            while True:
                line = fr.readline()
                if not line or "total" in line:
                    break
                tmp_res = yaml.safe_load(line.strip()[:-1])
                raw_num = int(tmp_res["raw_num"])
                dst_id = tmp_res["dst_id"]
                aug_num = tmp_res.get("aug_num", raw_num)
                self.id2range[dst_id] = (data_offset, data_offset + raw_num)
                self.id_sample_num[dst_id] = int(aug_num)
                data_offset += raw_num
        assert (
            data_offset == self._size
        ), f"info raw num: {data_offset} not equal {self._size}"

    def _get_eye_vis_label(self, eye_vis_label: List[float]):
        left_mask = eye_vis_label[1]
        left_vis = eye_vis_label[3]
        right_mask = eye_vis_label[2]
        right_vis = eye_vis_label[4]

        left_weight = [0, 0]
        right_weight = [0, 0]
        if int(left_mask) != 0:
            left_weight[int(left_vis)] = 1
        if int(right_mask) != 0:
            right_weight[int(right_vis)] = 1
        return left_weight, right_weight

    def __getitem__(self, index: int):
        record = self.pack_file.read(self.pack_file.record.keys[index])
        header, img = unpack_img(record, iscolor=1)
        label = header.label
        left_weight, right_weight = self._get_eye_vis_label(label)
        data = {
            "img": img,
            "layout": "hwc",
            "eye_vis_labels": label,
            "gt_eye_cls_labels": np.concatenate(
                [left_weight, right_weight],
            ).astype(np.float32),
        }
        if self.transforms:
            data = self.transforms(data)
        return data

    def __len__(self):
        return self._size


class EyeStatusRecDataset(Dataset):
    """EyeStatusRecDataset reads eye status data from mx rec.

    Args:
        rec_path: Path to eye status rec.
        idx_path: Path to eye status idx.
        transfroms : Transfroms of data before using.
    """

    def __init__(
        self,
        rec_path: str,
        idx_path: str,
        inf_path: str,
        transforms: Optional[List] = None,
    ):
        super(EyeStatusRecDataset, self).__init__()
        self.rec_path = rec_path
        self.idx_path = idx_path
        self.inf_path = inf_path
        self.transforms = transforms
        self.pack_type = PackTypeMapper["mxrecord"]

        self._init_pack()

    def _init_pack(self):
        self.pack_file = self.pack_type(
            uri=self.rec_path, idx_path=self.idx_path, writable=False
        )
        self.pack_file.open()

        self._size = len(self.pack_file.record.keys)

        self.id2range = {}
        self.id_sample_num = {}
        data_offset = 0
        with open(self.inf_path, "r") as fr:
            while True:
                line = fr.readline()
                if not line:
                    break
                tmp_res = yaml.safe_load(line.strip()[:-1])
                raw_num = int(tmp_res["raw_num"])
                dst_id = tmp_res["dst_id"]
                aug_num = tmp_res.get("aug_num", raw_num)
                self.id2range[dst_id] = (data_offset, data_offset + raw_num)
                self.id_sample_num[dst_id] = int(aug_num)
                data_offset += raw_num
        assert (
            data_offset == self._size
        ), f"info raw num: {data_offset} not equal {self._size}"

    def _get_each_part(self, label: List):
        """Get label recursively since ldmk len is not always the same.

        Args:
            label: label of eye status and eye ldmk, arranged like
                   [len(labels), labels, len(labels), labels, ...]

        Returns:
            splited labels.
        """
        ans_len = int(label.pop(0))
        ans = []
        for _ in range(ans_len):
            ans.append(label.pop(0))
        if not label:
            return [ans]
        return [ans] + self._get_each_part(label)

    def _fix_eye_status(self, status: List[float]):
        eye_status = (
            np.ones(
                EYE_STATUS_FIX_SIZE,
                dtype=np.int32,
            )
            * -1
        )

        status_len = len(status)

        if status_len == EYE_STATUS_FIX_SIZE:
            eye_status[:] = status
        else:
            eye_status[: status_len // 2] = status[: status_len // 2]
            start_pos = EYE_STATUS_FIX_SIZE // 2
            eye_status[start_pos : start_pos + status_len // 2] = status[
                status_len // 2 :
            ]

        return eye_status

    def _get_eye_status_label(self, eye_status: List[float]):
        """Eye status label parsing.

        for more info, please refer
            http://biaozhu.horizon.ai/mindoc/docs/zby#oxmh7
        eye_dict = {
            'eye_closed': 0,
            'eye_open': 1,
            'eye_occluded': 2,
            'eye_narrow': 3,
            'eye_lookdown': 4,
            'eye_hard': 5
        }

        feature_dict = {
            'clear_eye_feature': 0,
            'unsure_eye_feature': 1,
            'no_eye_feature': 2,
        }

        occluded_dict = {
            'full_visible': 0,
            'occluded': 1,
            'spot_occluded': 2,
            'heavily_occluded': 3,
            'sot_heavily_occluded': 4,
        }

        ignore_dict = {
            'no': 0,
            'yes': 1
        }

        Returns:
            left_weight:
            right_weight:

        """
        eye_status_len = len(eye_status)
        left_status = eye_status[: eye_status_len // 2]
        right_status = eye_status[eye_status_len // 2 :]
        left_weight = self._get_softlabel_weights(left_status)
        right_weight = self._get_softlabel_weights(right_status)
        return left_weight, right_weight

    def _get_softlabel_weights(self, eye_status: List[float]):
        eye_status = [int(x) for x in eye_status]
        dst_weight = [0 for i in range(5)]
        src_label = eye_status[0]

        # former labeled process
        if len(eye_status) != 4:
            if src_label > 4:
                return dst_weight

            dst_weight[src_label] = 1
            return dst_weight

        # if eye_hard or ignore
        if src_label == 5 or eye_status[-1] == 1:

            # invalid label, treat it as masked
            return dst_weight

        # if occulde
        if src_label == 2 or eye_status[2] in (3, 4):
            dst_weight[2] = 1
            return dst_weight

        # if narrow_no_eyeball, treat it as close
        if src_label == 3 and eye_status[1] == 2:
            dst_weight[0] = 1
            return dst_weight

        # softlabel for close
        if src_label == 0:
            if eye_status[2] == 2:
                dst_weight[0] = 0.97
                dst_weight[2] = 0.03
            elif eye_status[2] == 1:
                dst_weight[0] = 0.93
                dst_weight[2] = 0.07
            else:
                dst_weight[0] = 1
            return dst_weight

        dst_weight[src_label] = 1
        return dst_weight

    def _get_ldmk_info(self, ldmk):

        ldmk_loc = [0 for _ in range(34)]
        ldmk_weight = [0 for _ in range(34)]

        if len(ldmk) == 0:
            ldmk_weight = ldmk_weight[:16]
            ldmk_loc = ldmk_loc[:16]
            return ldmk_loc, ldmk_weight

        ldmk_len = len(ldmk) // 3

        x_list, y_list, w_list = (
            ldmk[: 2 * ldmk_len : 2],
            ldmk[1 : 2 * ldmk_len : 2],
            ldmk[2 * ldmk_len :],
        )

        # parse loc and weight for 1d,ls
        for idx, (x, y, w) in enumerate(zip(x_list, y_list, w_list)):
            ldmk_loc[idx * 2] = x
            ldmk_loc[idx * 2 + 1] = y

            ldmk_weight[idx * 2] = w
            ldmk_weight[idx * 2 + 1] = w

        # When labeling, 8 pts is located for eye ldmk if former model outputs
        # close for this single eye. otherwise 17 pts is located. Thus we can
        # use the number of ldmks to judge whether this eye is closed.
        # TODO: This judgement is not so accuracy because former eye model
        # outputs are not always correct. Try use eye status label.
        is_close = False
        if np.sum(ldmk_weight[16:]) == 0:
            is_close = True

        # using 8 ldmk pts for eye lid
        ldmk_weight = ldmk_weight[:16]
        ldmk_loc = ldmk_loc[:16]

        # doubles weights
        if is_close:
            ldmk_weight = [v * 2 for v in ldmk_weight]

        return ldmk_loc, ldmk_weight

    def _parse_label(self, img_shape: Tuple, label: List[float]):
        label = list(label)
        img_height, img_width = img_shape[:2]
        image_scale = np.array(
            [[img_width, img_height, 1.0]], dtype=np.float32
        )

        # get eye_status, left_ldmk, right_ldmk annos form list
        eye_status, left_ldmk, right_ldmk = self._get_each_part(label)

        # get weights for each eye
        left_weight, right_weight = self._get_eye_status_label(eye_status)

        # get coords and their weight for 8 pts eye ldmk of eyelid for each eye
        left_ldmk_loc, left_ldmk_weight = self._get_ldmk_info(left_ldmk)
        right_ldmk_loc, right_ldmk_weight = self._get_ldmk_info(right_ldmk)

        ldmk_weight = np.concatenate(
            [left_ldmk_weight, right_ldmk_weight]
        ).astype(np.float32)
        ldmk_weight = ldmk_weight.reshape((-1))
        ldmk = np.concatenate([left_ldmk_loc, right_ldmk_loc]).astype(
            np.float32
        )
        ldmk = ldmk.reshape((-1, 2))
        gt_ldmk = np.concatenate(
            [ldmk, np.zeros(shape=(ldmk.shape[0], 1), dtype=np.float32)],
            axis=-1,
        )
        eye_status = self._fix_eye_status(eye_status)

        res = {
            "img_height": img_height,
            "img_width": img_width,
            "eye_status": eye_status,
            "gt_eye_cls_labels": np.concatenate(
                [left_weight, right_weight],
            ).astype(np.float32),
            "gt_ldmk": gt_ldmk * image_scale,
            "gt_ldmk_attr": ldmk_weight[::2],
        }
        res["ldmk_pairs"] = [
            (0, 12),
            (1, 11),
            (2, 10),
            (3, 9),
            (4, 8),
            (5, 15),
            (6, 14),
            (7, 13),
        ]

        return res

    def __getitem__(self, index: int):
        record = self.pack_file.read(self.pack_file.record.keys[index])
        header, img = unpack_img(record, iscolor=1)
        label = self._parse_label(img.shape, header.label)
        data = {
            "img": img,
            "layout": "hwc",
        }
        data.update(label)
        if self.transforms:
            data = self.transforms(data)
        return data

    def __len__(self):
        return self._size


@OBJECT_REGISTRY.register
class EyeStatusDataset(ConcatDataset):
    """EyeStatusDataset reads eye data from mx rec list.

    Args:
        rec_paths : Path list to eye status recs.
        idx_paths : Path list to eye status idx.
        inf_paths : Path list to eye status inf.
        dataset_type : Whether eye vis or eye status dataset.
        transfroms : Transfroms of data before using.
    """

    def __init__(
        self,
        rec_paths: List[str],
        idx_paths: List[str] = None,
        inf_paths: List[str] = None,
        dataset_type: str = "eye_status",
        transforms: Optional[List] = None,
    ):
        self.transforms = transforms
        self.rec_paths = rec_paths
        if idx_paths is None:
            self.idx_paths = [rec.replace(".rec", ".idx") for rec in rec_paths]
        else:
            self.idx_paths = idx_paths

        if inf_paths is None:
            self.inf_paths = [rec.replace(".rec", ".inf") for rec in rec_paths]
        else:
            self.inf_paths = inf_paths

        assert len(self.rec_paths) == len(self.idx_paths)
        assert len(self.rec_paths) == len(self.inf_paths)
        self.dataset_type = dataset_type
        self._init_pack()

        super(EyeStatusDataset, self).__init__(self.dataset_list)

    def _init_pack(self):
        self.dataset_list = []
        for rec, idx, inf in zip(
            self.rec_paths, self.idx_paths, self.inf_paths
        ):
            if self.dataset_type == "eye_status":
                self.dataset_list.append(
                    EyeStatusRecDataset(
                        rec_path=rec,
                        idx_path=idx,
                        inf_path=inf,
                        transforms=self.transforms,
                    )
                )
            elif self.dataset_type == "eye_vis":
                self.dataset_list.append(
                    EyeVisDataset(
                        rec_path=rec,
                        idx_path=idx,
                        inf_path=inf,
                        transforms=self.transforms,
                    )
                )
            else:
                raise TypeError(f"not support type {self.dataset_type}")
