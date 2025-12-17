# Copyright (c) Horizon Robotics. All rights reserved.
import copy
from typing import List, Optional, Tuple

import cv2
import msgpack
import numpy as np
import torch.utils.data as data

from hat.data.transforms.landmark import GenerateGaussianHeatmap
from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.utils import get_packtype_from_path

__all__ = ["Hand3dLmdbSingleDataset"]


@OBJECT_REGISTRY.register
class Hand3dLmdbSingleDataset(data.Dataset):
    """
    Hand3d dataset for loading images, annotations, masks, and depths \
    from LMDB files for 3D hand pose estimation.

    Args:
        image_path:
            The path to the LMDB file containing the RGB images.
        anno_path:
            The path to the LMDB file containing the hand joint annotations.
        mask_path:
            The path to the LMDB file containing the hand masks.
            Defaults to None.
        depth_path: T
            he path to the LMDB file containing the depth maps.
            Defaults to None.
        transforms:
            A list of PyTorch transforms to apply to the data.
            Defaults to None.
        pack_kwargs:
            A dictionary of arguments to pass to the `pack_input` function.
            Defaults to None.
        set_name:
            The name of the dataset split. Defaults to "training".
        expansion_factor:
            Expansion factor of bounding box. Defaults to 1.6.
        enable_inshape_uniform:
            Whether to enable in-shape uniform sampling. Defaults to False.
        virtual_img_shape:
            The virtual image shape. Defaults to (1080, 1920, 3).
        normalized_length_joint09:
            Normalized length of joint 0-9.
            Defaults to None means no normalization.
        return_in_millimetres:
            Return 2d joints in pixels and 3d joints in millimetres.
            Defaults to True.
    """

    def __init__(
        self,
        image_path: str,
        anno_path: str,
        mask_path: str = None,
        depth_path: str = None,
        transforms: Optional[List] = None,
        pack_kwargs: Optional[dict] = None,
        set_name: Optional[str] = "training",
        expansion_factor: Optional[float] = 1.6,
        enable_inshape_uniform: Optional[bool] = False,
        virtual_img_shape: Optional[Tuple] = (1080, 1920, 3),
        normalized_length_joint09: Optional[float] = None,
        return_in_millimetres: Optional[bool] = True,
    ):
        self.transforms = transforms
        self.pack_kwargs = {} if pack_kwargs is None else pack_kwargs

        # load image lmdb.
        self.pack_type = get_packtype_from_path(image_path)
        self.data_pack = self.pack_type(
            image_path, writable=False, **self.pack_kwargs
        )
        self.data_pack.open()

        # load anno lmdb.
        self.anno_pack = self.pack_type(
            anno_path, writable=False, **self.pack_kwargs
        )
        self.anno_pack.open()

        # load mask lmdb.
        if mask_path is not None:
            self.mask_pack = self.pack_type(
                mask_path, writable=False, **self.pack_kwargs
            )
            self.mask_pack.open()
        else:
            self.mask_pack = None

        # load depth lmdb.
        if depth_path is not None:
            self.depth_pack = self.pack_type(
                depth_path, writable=False, **self.pack_kwargs
            )
            self.depth_pack.open()
        else:
            self.depth_pack = None

        # sample
        self.samples = self.sampler(self.anno_pack)

        self.set_name = set_name
        self.root_idx = 0
        self.expansion_factor = 1.6
        self.enable_inshape_uniform = enable_inshape_uniform
        self.virtual_img_shape = virtual_img_shape
        self.normalized_length_joint09 = normalized_length_joint09
        self.return_in_millimetres = return_in_millimetres

        self.enable_heatmap_head = False
        self.generator = GenerateGaussianHeatmap(
            num_ldmk=21,
            feat_stride=4,
            heatmap_shape=(64, 64),
            sigma=3,
            encoding_method="unbiased",
        )

    def sampler(self, anno_pack):
        all_samples = anno_pack.get_keys()
        return all_samples

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return "Hand3dLmdbSingleDataset"

    def get_bbox(
        self,
        joint_img,
        joint_valid,
        origin_width,
        origin_height,
        expansion_factor=1.6,
        aspect_ratio=1,
    ):
        x_img, y_img = joint_img[:, 0], joint_img[:, 1]
        x_img = x_img[joint_valid == 1]
        y_img = y_img[joint_valid == 1]
        xmin, ymin = min(x_img), min(y_img)
        xmax, ymax = max(x_img), max(y_img)

        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = (xmax - xmin + 1) * expansion_factor
        height = (ymax - ymin + 1) * expansion_factor
        width = max(width, 100)
        height = max(height, 100)

        if width > aspect_ratio * height:
            height = width / aspect_ratio
        elif width < aspect_ratio * height:
            width = height * aspect_ratio

        xmin = x_center - width // 2
        ymin = y_center - height // 2
        xmax = xmin + width
        ymax = ymin + height

        bbox = np.array([xmin, ymin, xmax, ymax])
        return bbox.astype(np.int64)

    def __read_ann(self, img, ann, mask, depth):
        # is_right and gesture
        is_right = int(ann.get("is_right", 1))
        gesture = 0 if ann.get("gesture", "index") == "other" else 1
        mask_vis = 0 if mask is None else 1
        depth_vis = 0 if depth is None else 1

        # crop_roi
        crop_roi = np.array(ann.get("crop_roi", []), dtype=np.float32)
        crop_roi = list(map(int, crop_roi))

        # joint2d
        gt_ldmk = np.array(
            ann.get("joint2d", np.zeros((21, 2), dtype=np.float32)),
            dtype=np.float32,
        ).reshape((21, -1))
        if gt_ldmk.shape[-1] == 2:
            gt_ldmk = np.concatenate(
                [gt_ldmk, np.ones_like(gt_ldmk[:, :1])], axis=-1
            )
        ldmk_vis = (
            np.ones((21,), dtype=np.float32)
            if ann.get("ldmk_vis") is None
            else np.array(ann.get("ldmk_vis"), dtype=np.float32).reshape((21,))
        )

        # ldmk3d
        if ann.get("joint3d") is not None:
            gt_ldmk3d = np.array(ann["joint3d"], dtype=np.float32).reshape(
                (21, 3)
            )
            ldmk3d_vis = 1
        else:
            gt_ldmk3d = np.zeros((21, 3), dtype=np.float32)
            ldmk3d_vis = 0

        # gt_verts
        if ann.get("verts3d") is not None:
            gt_verts = np.array(ann["verts3d"], dtype=np.float32).reshape(
                (-1, 3)
            )
            vert3d_vis = 1
        else:
            gt_verts = np.zeros((778, 3), dtype=np.float32)
            vert3d_vis = 0

        # instri
        if ann.get("instri") is None:
            estimate_f = np.sqrt(np.square(self.raw_img_shape).sum()) / 3.464
            intrinsic = np.diag([estimate_f, estimate_f, 1])
        else:
            intrinsic = np.array(ann["instri"], dtype=np.float32).reshape(
                (3, 3)
            )
        distortion = (
            np.zeros((5,), dtype=np.float32)
            if ann.get("dist") is None
            else np.array(ann["dist"], dtype=np.float32).reshape((5,))
        )

        # mano_shape
        if ann.get("mano_shape") is None:
            mano_shape = np.zeros((10,), dtype=np.float32)
            shape_vis = 0
        else:
            mano_shape = np.array(
                ann.get("mano_shape"), dtype=np.float32
            ).reshape((10,))
            shape_vis = 1

        # mano_pose
        if ann.get("mano_pose") is None:
            mano_pose = np.zeros((48,), dtype=np.float32)
            pose_vis = 0
        else:
            mano_pose = np.array(
                ann.get("mano_pose"), dtype=np.float32
            ).reshape((48,))
            pose_vis = is_right

        rmat_cami2s = (
            np.eye(3, dtype=np.float32)
            if ann.get("rmat_cami2s") is None
            else np.array(ann["rmat_cami2s"])
        )
        tvec_cami2s = (
            np.zeros((3,), dtype=np.float32)
            if ann.get("tvec_cami2s") is None
            else np.array(ann["tvec_cami2s"])
        )

        gt_ldmk[:, :2] -= np.array([[crop_roi[0], crop_roi[1]]])
        intrinsic[:2, -1] -= np.array([crop_roi[0], crop_roi[1]])

        if self.enable_inshape_uniform:
            raw_img_shape = self.virtual_img_shape
            image = np.zeros(raw_img_shape, dtype=np.float32)
            image[
                crop_roi[1] : crop_roi[3], crop_roi[0] : crop_roi[2], :
            ] = img
            gt_ldmk[:, :2] += np.array(crop_roi[:2]).reshape((-1, 2))
            intrinsic[:2, -1] += crop_roi[:2]
        else:
            image = img.astype(np.float32)
            raw_img_shape = image.shape

        if not is_right:
            image = np.fliplr(image)
            gt_ldmk[:, 0] = raw_img_shape[1] - 1 - gt_ldmk[:, 0]
            rot_aug_mat = np.array(
                [[-1, 0, raw_img_shape[1] - 1], [0, 1, 0], [0, 0, 1]],
                dtype=np.float32,
            )
            rot_aug_mat = np.dot(
                np.linalg.inv(intrinsic), np.dot(rot_aug_mat, intrinsic)
            )
            gt_ldmk3d = np.dot(rot_aug_mat, gt_ldmk3d.swapaxes(1, 0)).swapaxes(
                1, 0
            )
            gt_verts = np.dot(rot_aug_mat, gt_verts.swapaxes(1, 0)).swapaxes(
                1, 0
            )

        if self.normalized_length_joint09 is not None:
            len_hand = np.sqrt(
                np.sum((gt_ldmk3d[9, :] - gt_ldmk3d[0, :]) ** 2)
            )
            gt_ldmk3d = (
                gt_ldmk3d * self.normalized_length_joint09 / (len_hand + 1e-6)
            )
            gt_verts = (
                gt_verts * self.normalized_length_joint09 / (len_hand + 1e-6)
            )
        if self.return_in_millimetres:
            gt_ldmk3d *= 1000
            gt_verts *= 1000

        ldmk3d_relat = copy.deepcopy(gt_ldmk3d)
        gt_root = copy.deepcopy(
            ldmk3d_relat[self.root_idx : self.root_idx + 1]
        )
        ldmk3d_relat -= ldmk3d_relat[self.root_idx, None, :]

        bbox = self.get_bbox(
            gt_ldmk[:, :2],
            np.ones_like(gt_ldmk[:, 0]),
            raw_img_shape[1],
            raw_img_shape[0],
            expansion_factor=self.expansion_factor,
        )

        outputs = {
            # inputs
            "img": image,
            "gt_bboxes": bbox,
            "gt_mask": mask,
            "mask_vis": mask_vis,
            "gt_depth": depth,
            "depth_vis": depth_vis,
            # targets
            "gt_ldmk": gt_ldmk,
            "ldmk_vis": ldmk_vis,
            "gt_ldmk3d": gt_ldmk3d,
            "ldmk3d_vis": ldmk3d_vis,
            "ldmk3d_relat": ldmk3d_relat,
            "gt_verts": gt_verts,
            "vert3d_vis": vert3d_vis,
            "gt_pose": mano_pose,
            "pose_vis": pose_vis,
            "gt_shape": mano_shape,
            "shape_vis": shape_vis,
            "intrinsic": intrinsic,
            "distortion": distortion,
            "gt_root": gt_root,
            "rmat_cami2s": rmat_cami2s,
            "tvec_cami2s": tvec_cami2s,
            # meta_info
            "layout": "hwc",
            "color_space": "rgb",
            "is_right": is_right,
            "gesture": gesture,
            "img_shape": raw_img_shape,
            "virtual_img_shape": self.virtual_img_shape,
            "root_idx": self.root_idx,
        }

        return outputs

    def __getitem__(self, index):
        # get lmdb key
        key = self.samples[index]

        img_data = msgpack.unpackb(self.data_pack.read(key), raw=False)
        img = cv2.imdecode(
            np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        anno_data = msgpack.unpackb(self.anno_pack.read(key), raw=False)

        if self.mask_pack is not None:
            mask_data = msgpack.unpackb(self.mask_pack.read(key), raw=False)
            mask = cv2.imdecode(
                np.frombuffer(mask_data, dtype=np.uint8), cv2.IMREAD_COLOR
            )
        else:
            mask = np.zeros_like(img[..., 0])

        if self.depth_pack is not None:
            depth_data = msgpack.unpackb(self.depth_pack.read(key), raw=False)
            depth = cv2.imdecode(
                np.frombuffer(depth_data, dtype=np.uint8), cv2.IMREAD_COLOR
            )
        else:
            depth = np.zeros_like(img[..., 0])

        data = self.__read_ann(img, anno_data, mask, depth)

        # transform
        if self.transforms is not None:
            if isinstance(self.transforms, list):
                for transform in self.transforms:
                    data = transform(data)
            else:
                data = self.transforms(data)

        return data
