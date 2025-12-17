from __future__ import division
from typing import List, Optional

import cv2
import msgpack
import numpy as np
import torch
from torch.utils.data import Dataset

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.utils import get_packtype_from_path


@OBJECT_REGISTRY.register
class Human3dDataset(Dataset):
    """Human3d dataset, which only process one lmdb pack.

    Args:
        dataset: Name of dataset.
        image_path: Image pack path.
        anno_path: Anno pack path.
        ldmk_pairs: 2d keypoints pairs.
        smpl_pose_pairs: SMPL keypoints pairs.
        ignore_smpl: Whether ignore smpl label. Defaults to False.
        ignore_3d: Whether ignore 3d keypoints label. Defaults to False.
        transforms: Transfroms of data before using. Defaults to None.
    """

    def __init__(
        self,
        dataset: str,
        image_path: str,
        anno_path: str,
        ldmk_pairs: List[List[int]],
        smpl_pose_pairs: List[List[int]],
        ignore_smpl: bool = False,
        ignore_3d: bool = False,
        transforms: Optional[List] = None,
    ):
        super(Human3dDataset, self).__init__()
        self.dataset = dataset
        self.image_path = image_path
        self.anno_path = anno_path
        self.ldmk_pairs = ldmk_pairs
        self.smpl_pose_pairs = smpl_pose_pairs
        self.transforms = transforms
        self.ignore_3d = ignore_3d
        self.ignore_smpl = ignore_smpl
        self.pack_type = get_packtype_from_path(image_path)
        self.kwargs = {}
        self.image_pack = self.pack_type(
            image_path, writable=False, **self.kwargs
        )
        self.image_pack.open()
        self.anno_pack = self.pack_type(
            anno_path, writable=False, **self.kwargs
        )
        self.anno_pack.open()

        self.samples = self.image_pack.get_keys()

    def __getitem__(self, index):
        item = {}
        key = self.samples[index]
        image_data = msgpack.unpackb(self.image_pack.read(key), raw=False)
        img = cv2.imdecode(
            np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        orig_shape = np.array(img.shape)[:2]
        anno_data = msgpack.unpackb(self.anno_pack.read(key), raw=False)

        # Get SMPL parameters, if available
        if (
            anno_data["gt_smpl_pose"] is not None
            and anno_data["gt_smpl_shape"] is not None
        ):
            pose = np.array(anno_data["gt_smpl_pose"])
            betas = np.array(anno_data["gt_smpl_shape"])
            item["has_smpl"] = True
        else:
            pose = np.zeros(72)
            betas = np.zeros(11)
            item["has_smpl"] = False
        if self.ignore_smpl:
            item["has_smpl"] = False

        if anno_data["gt_cam"] is not None:
            camera = np.array(anno_data["gt_cam"])
        else:
            camera = np.zeros(3)
        item["gt_cam"] = camera.astype(np.float32)

        item["img"] = img
        item["gt_smpl_pose"] = pose.astype(np.float32)
        item["gt_smpl_betas"] = betas.astype(np.float32)
        item["img_name"] = anno_data["img_name"]

        # Get 3D pose, if available
        if anno_data["gt_ldmk_3d"] is not None:
            gt_ldmk_3d = np.array(anno_data["gt_ldmk_3d"])
            item["gt_ldmk_3d"] = gt_ldmk_3d.astype(np.float32)[:, :-1]
            item["gt_ldmk_3d_attr"] = gt_ldmk_3d.astype(np.float32)[:, -1]
            item["has_ldmk_3d"] = True
        else:
            item["gt_ldmk_3d"] = np.zeros((24, 3), dtype=np.float32)
            item["gt_ldmk_3d_attr"] = np.zeros((24, 1), dtype=np.float32)
            item["has_ldmk_3d"] = False
        if self.ignore_3d:
            item["has_ldmk_3d"] = False

        # Get 2D keypoints
        if anno_data["gt_ldmk"] is not None:
            keypoints_gt = np.array(anno_data["gt_ldmk"])
        else:
            keypoints_gt = np.zeros((24, 3))

        keypoints = keypoints_gt

        # -1:ignore; 0:invisible; 1:occluded; 2:full visible
        keypoints[keypoints[:, 2] <= 0, 2] = 0
        keypoints[keypoints[:, 2] > 0, 2] = 1

        item["gt_ldmk"] = keypoints.astype(np.float32)[:, :-1]
        item["gt_ldmk_attr"] = keypoints.astype(np.float32)[:, -1]

        item["img_shape"] = orig_shape
        item["is_flipped"] = False
        item["rot_angle"] = 0

        # Get gender data, if available
        if anno_data["gender"] is not None:
            gender = anno_data["gender"]
            gender = 0 if str(gender) == "m" else 1
        else:
            gender = -1

        # Get face ldmk68
        if (
            "gt_face_ldmk68" in anno_data
            and anno_data["gt_face_ldmk68"] is not None
        ):  # noqa
            face_ldmk68 = anno_data["gt_face_ldmk68"]
        else:
            face_ldmk68 = []
        if (
            "gt_lhand_ldmk" in anno_data
            and anno_data["gt_lhand_ldmk"] is not None
        ):  # noqa
            lhand_ldmk = anno_data["gt_lhand_ldmk"]
        else:
            lhand_ldmk = []
        if (
            "gt_rhand_ldmk" in anno_data
            and anno_data["gt_rhand_ldmk"] is not None
        ):  # noqa
            rhand_ldmk = anno_data["gt_rhand_ldmk"]
        else:
            rhand_ldmk = []
        item["gender"] = gender
        item["sample_index"] = index
        item["dataset_name"] = self.dataset
        item["layout"] = "hwc"
        item["ldmk_pairs"] = self.ldmk_pairs
        item["smpl_pose_pairs"] = self.smpl_pose_pairs
        item["gt_face_ldmk68"] = face_ldmk68
        item["gt_lhand_ldmk"] = lhand_ldmk
        item["gt_rhand_ldmk"] = rhand_ldmk

        if self.transforms is not None:
            item = self.transforms(item)

        return item

    def __len__(self):
        return len(self.samples)


@OBJECT_REGISTRY.register
class Human3dMixedDataset(torch.utils.data.Dataset):
    """Dataset for list of human3d data.

    Different data are sampled proportionally within the dataset.
    Such a batch will have data from different datasets
    to help improve model performance.

    Args:
        dataset_list: List of dataset name.
        img_path_list: List of image pack path.
        anno_path_list: List of image anno path.
        ldmk_pairs: 2d keypoints pairs.
        smpl_pose_pairs: SMPL keypoints pairs.
        ignore_smpl: Whether ignore smpl label. Defaults to False.
        ignore_3d: Whether ignore keypoints 3d label. Defaults to False.
        mode: Mode to calculate the length of dataset . Defaults to "sum".
        transforms: Transfroms of data before using. Defaults to None.
    """

    def __init__(
        self,
        dataset_list: List[str],
        img_path_list: List[str],
        anno_path_list: List[str],
        ldmk_pairs: List[List[int]],
        smpl_pose_pairs: List[List[int]],
        ignore_smpl: bool = False,
        ignore_3d: bool = False,
        mode: str = "sum",
        transforms: Optional[List] = None,
    ):
        self.dataset_list = dataset_list
        self.img_path_list = img_path_list
        self.anno_path_list = anno_path_list
        self.datasets = []
        for data, img_path, anno_path in zip(
            dataset_list, img_path_list, anno_path_list
        ):
            self.datasets.append(
                Human3dDataset(
                    data,
                    img_path,
                    anno_path,
                    ldmk_pairs=ldmk_pairs,
                    smpl_pose_pairs=smpl_pose_pairs,
                    ignore_smpl=ignore_smpl,
                    ignore_3d=ignore_3d,
                    transforms=transforms,
                )
            )
        length_itw = sum([len(ds) for ds in self.datasets])
        if mode == "max":
            self.length = max([len(ds) for ds in self.datasets])
        elif mode == "sum":
            self.length = sum([len(ds) for ds in self.datasets])
        else:
            assert "mode must in [max, sum]"

        # Data distribution inside each batch
        self.partition = [
            len(dataset) / length_itw for dataset in self.datasets
        ]

        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(len(self.datasets)):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
