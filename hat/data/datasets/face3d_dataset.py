# Copyright (c) Horizon Robotics. All rights reserved.
import copy
from typing import Dict, List, Optional

import cv2
import msgpack
import numpy as np
import torch
import torch.utils.data as data

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.utils import get_packtype_from_path

__all__ = ["Face3dDataset"]


def _to_numpy(data):
    return np.array(data, np.float32)


@OBJECT_REGISTRY.register
class Face3dSingleDataset(data.Dataset):
    """Dataset for 3D Face Reconstruction.

    Read raw image, bbox, 3D face landmark(68pts) and face mask from single
    lmdb file. Return cropped face roi and landmark. Target image and mask are
    also required in `finetune` stage.

    Args:
        image_path: The path of image lmdb file.
        mask_path: The path of mask lmdb file.
        anno_path: The path of anno lmdb file.
        transforms: Transforms of data augmentation.
        stage: Data stage. Only `pretrain`, `finetune` and `predict` are
            supported. In `pretrain` stage, we try to optimize global_pose
            and transl use 2D landmark constraints or ground truth supervision.
            In `finetune` stage, model is trained with both landmark loss and
            image-based reconstruction loss for better head pose,
            head position and face geometry. Face mask and target image are
            necessary in this stage. In `predict` stage, only image and
            gt_bboxes are required for inference.
        pack_kwargs: Kwargs for pack type.
        modeltype: Set as wpp or pp. When set as pp, dataset will include
            camera params for perspective projection. Default to wpp.
        only_y_channel: use y channel of gray image or not. Default to False.
        independent_transform_list: Transforms list of data aug. Do transform
            in the list seperately after `transforms` and concat/stack results
            temporarily.

    """

    OPTIONAL_ANNO_KEYS = [
        "gt_pose",
        "gt_ldmk",
        "gt_ldmk3d",
        "gt_verts",
        "gt_depth",
        "intrinsic",
        "distortion",
        "eye3d_left",
        "eye3d_right",
        "global_pose",
        "transl",
    ]
    # fmt:off
    LDMK_PAIRS = [
        # contour
        [0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],
        # eyebrow
        [17, 26], [18, 25], [19, 24], [20, 23], [21, 22],
        # nose
        [31, 35], [32, 34],
        # eyes
        [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],
        # mouse
        [48, 54], [49, 53], [50, 52], [61, 63], [60, 64], [67, 65], [58, 56], [59, 55]  # noqa
    ]
    # fmt: on

    def __init__(
        self,
        image_path: str,
        mask_path: str,
        anno_path: str,
        transforms: Optional[List] = None,
        stage: Optional[str] = "finetune",
        pack_kwargs: Optional[dict] = None,
        modeltype: Optional[str] = "wpp",
        only_y_channel: Optional[bool] = False,
        independent_transform_list: Optional[List] = None,
    ):
        self.transforms = transforms
        self.independent_transform_list = independent_transform_list
        self.kwargs = {} if pack_kwargs is None else pack_kwargs
        self.pack_type = get_packtype_from_path(image_path)
        self.data_pack = self.pack_type(
            image_path, writable=False, **self.kwargs
        )
        self.data_pack.open()
        self.anno_pack = self.pack_type(
            anno_path, writable=False, **self.kwargs
        )
        self.anno_pack.open()

        self.stage = stage.lower()
        if self.stage == "finetune":
            self.mask_pack = self.pack_type(
                mask_path, writable=False, **self.kwargs
            )
            self.mask_pack.open()

        self.samples = self.anno_pack.get_keys()
        assert self.stage in [
            "finetune",
            "pretrain",
            "predict",
        ], "stage must be pretrain or finetune."
        self.modeltype = modeltype.lower()
        self.only_y_channel = only_y_channel

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return "Face3dSingleDataset"

    def _merge_trans_results(self, data, trans_results):
        for k, v in trans_results[0].items():
            value_list = []
            for res in trans_results:
                assert res.get(k) is not None
                value_list.append(res[k])

            if k in ["img", "grid_map", "position_map"]:
                data[k] = torch.cat(value_list, 0)
            elif k in ["gt_bboxes"]:
                data[k] = torch.stack(value_list, 0)
            elif k in ["gt_mask", "gt_img"]:
                data[k] = np.concatenate(value_list, 0)
            elif isinstance(v, np.ndarray):
                data[k] = np.stack(value_list, 0)
        return data

    def get_anno(self, anno_data: Dict):
        anno = {}
        anno["gt_bboxes"] = _to_numpy(anno_data["gt_bboxes"][:4])
        anno["gt_bboxes_local"] = _to_numpy(anno_data["gt_bboxes"][:4])
        anno["ldmk_pairs"] = self.LDMK_PAIRS
        for key in self.OPTIONAL_ANNO_KEYS:
            if key in anno_data and anno_data[key] is not None:
                anno[key] = _to_numpy(anno_data[key])

        anno["roi_offset"] = np.array([0, 0])
        anno["save_crop"] = False
        if anno_data.get("save_crop", False):
            assert anno_data.get("crop_bbox") is not None
            assert anno_data.get("raw_img_shape") is not None
            anno["save_crop"] = True
            anno["raw_img_shape"] = anno_data["raw_img_shape"]
            roi_offset_x, roi_offset_y = anno_data["crop_bbox"][:2]
            anno["roi_offset"] = np.array([[roi_offset_x, roi_offset_y]])
            anno["gt_bboxes"] += np.array(
                [roi_offset_x, roi_offset_y, roi_offset_x, roi_offset_y]
            )
            if "gt_ldmk" in anno:
                anno["gt_ldmk"][:, :2] += anno["roi_offset"]

        if "gt_ldmk" in anno and self.stage != "predict":
            index = anno["gt_ldmk"][:, 2] < 0.4
            anno["gt_ldmk"][index, 2] = 0.0

        return anno

    def __getitem__(self, index):
        data = {}
        key = self.samples[index]
        img_data = msgpack.unpackb(self.data_pack.read(key), raw=False)
        img = cv2.imdecode(
            np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert img.ndim == 3
        if self.only_y_channel:
            h, w = img.shape[:2]
            yuv420 = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_I420).flatten()
            img = yuv420[: h * w].reshape(h, w, 1)
        data["img"] = img
        data["img_shape"] = data["img"].shape
        data["raw_img_shape"] = data["img"].shape
        data["layout"] = "hwc"
        anno_data = msgpack.unpackb(self.anno_pack.read(key), raw=False)
        data["img_path"] = anno_data["img_path"]
        data.update(self.get_anno(anno_data))
        if self.modeltype == "pp":
            assert "intrinsic" in data and "distortion" in data
        if self.stage == "finetune":
            mask_data = msgpack.unpackb(self.mask_pack.read(key), raw=False)
            data["gt_mask"] = cv2.imdecode(
                np.frombuffer(mask_data, dtype=np.uint8), -1
            )
            data["gt_img"] = data["img"].copy()

        if self.transforms is not None:
            data = self.transforms(data)

        if (
            self.independent_transform_list is not None
            and len(self.independent_transform_list) > 0
        ):
            trans_results = []
            for indp_trans in self.independent_transform_list:
                data_cur = copy.deepcopy(data)
                trans_results.append(indp_trans(data_cur))
            data = self._merge_trans_results(data, trans_results)

        return data


@OBJECT_REGISTRY.register
class Face3dDataset(data.ConcatDataset):
    """
    Read image, bbox, 3D face landmarks and face mask from lmdb file list.

    Args:
        image_path_list: List of path to image lmdb file.
        mask_path_list: List of path to mask lmdb file.
        anno_path_list: List of path to anno lmdb file.
        transforms: Transforms of data augmentation.
        stage: Data stage. Only `pretrain`, `finetune` and `predict`
            are supported.
        pack_kwargs (dict): Kwargs for pack type.
        modeltype: Set as wpp or pp. When set as pp, dataset will include
            camera params for perspective projection. Default to wpp.
        only_y_channel: use y channel of gray image or not. Default to False.
    """

    def __init__(
        self,
        image_path_list: List[str],
        mask_path_list: List[str],
        anno_path_list: List[str],
        transforms: Optional[List] = None,
        stage: Optional[str] = "finetune",
        pack_kwargs: Optional[dict] = None,
        modeltype: Optional[str] = "wpp",
        only_y_channel: Optional[bool] = False,
        independent_transform_list: Optional[List] = None,
    ):
        self.image_path_list = image_path_list
        self.mask_path_list = mask_path_list
        self.anno_path_list = anno_path_list
        self.transforms = transforms
        self.satge = stage
        self.pack_kwargs = pack_kwargs
        self.modeltype = modeltype.lower()
        self.only_y_channel = only_y_channel
        self.independent_transform_list = independent_transform_list

        self._init_pack()
        super(Face3dDataset, self).__init__(self.dataset_list)

    def __repr__(self):
        return "Face3dDataset"

    def _init_pack(self):
        assert len(self.image_path_list) == len(self.mask_path_list)
        assert len(self.image_path_list) == len(self.anno_path_list)
        self.dataset_list = []
        for image_path, mask_path, anno_path in zip(
            self.image_path_list, self.mask_path_list, self.anno_path_list
        ):
            self.dataset_list.append(
                Face3dSingleDataset(
                    image_path=image_path,
                    mask_path=mask_path,
                    anno_path=anno_path,
                    transforms=self.transforms,
                    stage=self.satge,
                    pack_kwargs=self.pack_kwargs,
                    modeltype=self.modeltype,
                    only_y_channel=self.only_y_channel,
                    independent_transform_list=self.independent_transform_list,
                )
            )
