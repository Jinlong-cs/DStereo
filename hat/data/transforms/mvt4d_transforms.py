import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

try:
    from hatbc.message import Attribute, CameraFrame, CameraParam
    from hatbc.message import Image as HoImage
    from hatbc.message import MessageMeta, SyncMessages, VCSParam
except ImportError:
    Attribute, CameraFrame, CameraParam, HoImage = None, None, None, None
    MessageMeta, SyncMessages, VCSParam = None, None, None
from numpy import random
from PIL import Image

try:
    from pycocotools import mask as coco_mask
except ImportError:
    coco_mask = None

from hat.data.datasets.bev3d_multiview_dataset import get_mat_from_trans_rpy
from hat.data.datasets.real3d_dataset import Real3DDataset
from hat.data.transforms.detection import ToTensor
from hat.data.transforms.grid_mask import GridMask
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import check_packages_available, require_packages

try:
    import mmcv
except ImportError:
    mmcv = None

import logging

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class RawDataLoading(object):
    def __init__(self, keys, output_key):
        self.keys = keys
        self.output_key = output_key

    def __call__(self, data):
        output = []
        dfs_path = []
        self._dfs_loading(data, 0, dfs_path, output)
        if len(output) == 1:
            output = output[0]
        data[self.output_key] = output
        return data

    def _dfs_loading(self, data, index, dfs_path, output):
        if index == len(self.keys):
            output.append(self._load_data(dfs_path))
            return
        key = self.keys[index]
        if isinstance(data[key], (list, tuple)):
            for path in data[key]:
                dfs_path.append(path)
                self._dfs_loading(data, index + 1, dfs_path, output)
                dfs_path.pop()
        else:
            dfs_path.append(data[key])
            self._dfs_loading(data, index + 1, dfs_path, output)
            dfs_path.pop()

    def _load_data(self, path):
        if isinstance(path, (list, tuple)):
            path = os.path.join(*path)
        return path


@OBJECT_REGISTRY.register
class RawImageLoading(RawDataLoading):
    def __init__(self, keys=("dataroot", "img_path"), output_key="imgs"):
        super().__init__(keys, output_key)

    def _load_data(self, path):
        if isinstance(path, (list, tuple)):
            path = os.path.join(*path)
        return cv2.imread(path)


@OBJECT_REGISTRY.register
class RawLidarPointsLoading(RawDataLoading):
    def __init__(
        self,
        keys=("dataroot", "lidar_pts_path"),
        output_key="points",
        load_dim=5,
        use_dim=5,
        dtype="float32",
    ):
        super().__init__(keys, output_key)
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.dtype = dtype

    def _load_data(self, path):
        if isinstance(path, (list, tuple)):
            path = os.path.join(*path)
        points = np.fromfile(path, self.dtype).reshape(-1, self.load_dim)
        return points[:, : self.use_dim]


@OBJECT_REGISTRY.register
class ConvertToSyncMessages(object):
    """Convert multiview dict to SyncMessages."""

    @require_packages("hatbc")
    def __call__(self, imgs_meta: Dict[str, Any]) -> Dict[str, Any]:
        imgs = imgs_meta["imgs"]
        label = imgs_meta["meta"]

        assert len(label["img_orders"]) == len(
            imgs
        ), "please check num of imgs or img_orders"

        camera_messages = []
        attribute_messages = []

        T_lidar2vcs = None
        for view, img in zip(label["img_orders"], imgs):

            img = HoImage(
                data=img,
                color_space="rgb",
                layout="hwc",
                format="jpg",
            )

            calib = label["view_anno"][view]["meta"].get("calib", None)
            camera_param = None
            if calib is not None:
                camera_param = CameraParam(
                    camera_type=view,
                    focal_u=calib.get("focal_u"),
                    focal_v=calib.get("focal_v"),
                    center_u=calib.get("center_u"),
                    center_v=calib.get("center_v"),
                    camera_x=calib.get("camera_x"),
                    camera_y=calib.get("camera_y"),
                    camera_z=calib.get("camera_z"),
                    pitch=calib.get("pitch"),
                    yaw=calib.get("yaw"),
                    roll=calib.get("roll"),
                    fov=calib.get("fov"),
                    vcs=VCSParam(
                        rotation=calib.get("vcs", {}).get("rotation"),
                        translation=calib.get("vcs", {}).get("translation"),
                    ),
                    distort=calib["distort"]["param"]
                    if "param" in calib["distort"]
                    else calib["distort"],
                    image_height=calib["img_wh"][1]
                    if "img_wh" in calib
                    else img.shape[0],
                    image_width=calib["img_wh"][0]
                    if "img_wh" in calib
                    else img.shape[1],
                )
                if T_lidar2vcs is None and calib.get("lidar", None):
                    T_lidar2vcs = get_mat_from_trans_rpy(calib["lidar"], "xyz")

            camera_messages.append(
                CameraFrame(
                    topic="camera_frame",
                    image=img,
                    camera_param=camera_param,
                )
            )

        if T_lidar2vcs is None:
            T_lidar2vcs = np.identity(4)

        if "ego_pose" in label:
            T_lidar2global = np.array(label["ego_pose"], dtype=np.float32)
            T_vcs2lidar = np.linalg.inv(T_lidar2vcs)
            T_vcs2global = np.dot(T_lidar2global, T_vcs2lidar)
        else:
            T_vcs2global = np.eye(4, dtype=np.float32)

        attribute_messages.append(
            Attribute(
                topic="ego_pose",
                value=T_vcs2global,
            )
        )

        messages = camera_messages + attribute_messages

        meta = MessageMeta(timestamp=int(label["timestamp"]))

        return SyncMessages(
            meta=meta,
            messages=messages,
        )


def sync_messages_to_multiview_dict(data: "SyncMessages"):
    """Convert SyncMessages to multiview dict to."""

    camera_messages = data.get_messages(topics="camera_frame")
    ego_messages = data.get_messages(topics="ego_pose")
    if len(ego_messages) > 0:
        ego_pose = ego_messages[0].value
    else:
        ego_pose = np.eye(4, dtype=np.float32)

    timestamp = data.meta.timestamp

    imgs = []
    img_orders = []
    view_anno = {}
    for camera_frame in camera_messages:
        imgs.append(camera_frame.image.as_numpy())

        calib_dict = camera_frame.camera_param.to_dict()["CameraParam"]
        view = calib_dict.pop("camera_type")
        img_orders.append(view)
        calib_dict["vcs"] = calib_dict["vcs"].pop("VCSParam")
        view_anno[view] = {"meta": {"calib": calib_dict}}
    meta = {
        "timestamp": timestamp,
        "img_orders": img_orders,
        "view_anno": view_anno,
        "ego_pose": ego_pose,
    }

    return {"imgs": imgs, "meta": meta}


@OBJECT_REGISTRY.register
class MultiViewRecPadView(object):
    """Pad view with empty image for missing cam image.

    Args:
        pad_value: The value for pad.
        camera_view_names: The name corresponding to the
            current camera of each view.
        view_shapes: The shape of each view.
        drop_view: The view to drop.
        drop_view_ratio: The ratio to drop view.
    """

    def __init__(
        self,
        pad_value: int = 128,
        camera_view_names: Optional[Sequence[str]] = None,
        view_shapes: Optional[Dict[str, Sequence[int]]] = None,
        drop_view: Optional[Sequence[str]] = None,
        drop_view_ratio: float = 0.0,
    ):
        self.camera_view_names = camera_view_names
        self.pad_value = pad_value
        self.view_shapes = view_shapes
        self.drop_view = drop_view
        self.drop_view_ratio = drop_view_ratio

    def __call__(self, imgs_meta: Dict[str, Any]) -> Dict[str, Any]:
        imgs = imgs_meta["imgs"]
        label = imgs_meta["meta"]
        if random.random() < self.drop_view_ratio:
            drop_view = self.drop_view
        else:
            drop_view = []

        imgs_list = []
        cam2idx = {cam: idx for idx, cam in enumerate(label["img_orders"])}
        ts = label["timestamp"]
        label["views_pad"] = []
        for cam in self.camera_view_names:
            if cam not in cam2idx or cam in drop_view:
                logger.debug(
                    f"{cam} image missed in timestamp {ts} or {cam} is dropped, padded with empty image"  # noqa
                    f"{''.join(label['img_orders'])}"
                )
                img_h, img_w = self.view_shapes[f"camera_{cam}"]
                imgs_list.append(
                    np.full((img_h, img_w, 3), self.pad_value, dtype=np.uint8)
                )
                label["views_pad"].append(self.camera_view_names.index(cam))
                continue
            idx = cam2idx[cam]
            imgs_list.append(imgs[idx])

        imgs_meta["imgs"] = imgs_list
        imgs_meta["meta"] = label
        return imgs_meta


@OBJECT_REGISTRY.register
class MultiViewDrawIgnoreMask(object):
    def __init__(
        self,
        pad_value: int = 128,
    ):
        self.pad_value = pad_value

    def __call__(self, imgs_meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
        imgs = imgs_meta["imgs"]
        label = imgs_meta["meta"]

        for img, cam in zip(imgs, label["img_orders"]):
            ignore_mask = label["view_anno"][cam]["meta"]["ignore_mask"]
            if not isinstance(ignore_mask, np.ndarray):
                if coco_mask is None:
                    check_packages_available("pycocotools")
                ignore_mask = coco_mask.decode(ignore_mask)
            ignore_mask = ignore_mask.astype(np.uint8)
            ignore_mask = np.where(ignore_mask > 0)
            img[ignore_mask] = self.pad_value

        return imgs_meta


@OBJECT_REGISTRY.register
class MultiViewRecTransform(object):
    def __init__(
        self,
        category_id_dict: Dict[int, int],
        camera_view_names: List[str],
    ):
        self.category_id_dict = category_id_dict
        self.camera_view_names = camera_view_names

    def __call__(self, imgs_meta: Dict[str, Any]) -> Dict[str, np.ndarray]:

        annotations = imgs_meta.pop("meta")
        gt_bboxes_3d = []
        gt_labels_3d = []
        drop_view = [
            self.camera_view_names[idx] for idx in annotations["views_pad"]
        ]

        if "objects" in annotations:
            for anno in annotations["objects"]:
                uid = anno["uid"]
                in_vcs = anno["in_vcs"]
                ignore = True
                for cam_name, view_anno in annotations["view_anno"].items():
                    if cam_name in drop_view:
                        continue
                    if (
                        uid in view_anno["objects"]
                        and not view_anno["objects"][uid]["ignore"]
                    ) or (
                        str(uid) in view_anno["objects"]
                        and not view_anno["objects"][str(uid)]["ignore"]
                    ):
                        ignore = False
                        break
                if ignore:
                    continue

                if self.category_id_dict[anno["category_id"]] < 0:
                    continue

                dim = in_vcs["dim"]  # [l, w, h]
                loc = in_vcs["location"]  # [x, y, bottom_z]
                loc[2] += dim[2] * 0.5  # convert to real center

                gt_labels_3d.append(self.category_id_dict[anno["category_id"]])

                gt_bboxes_3d.append([*loc, *dim, in_vcs["yaw"]])

        gt_bboxes_3d = np.array(gt_bboxes_3d, dtype=np.float32)
        gt_labels_3d = np.array(gt_labels_3d, dtype=np.int64)

        # img instric K
        cam_intrinsic_list = []
        d_coef_list = []
        T_vcs2cam_list = []
        T_local2cam_list = []
        T_local2vcs_list = []
        for camera_name in self.camera_view_names:
            if camera_name in drop_view:
                cam_intrinsic_list.append(np.identity(3))
                d_coef_list.append(np.zeros((8,)).astype(float))
                T_vcs2cam_list.append(np.identity(4))
                T_local2cam_list.append(np.identity(4))
                T_local2vcs_list.append(np.identity(4))
            else:
                cam_intrinsic_list.append(
                    imgs_meta["calib_params"][f"camera_{camera_name}"]["K"]
                )
                d_coef_list.append(
                    imgs_meta["calib_params"][f"camera_{camera_name}"]["d"]
                )
                T_vcs2cam_list.append(
                    imgs_meta["calib_params"][f"camera_{camera_name}"][
                        "T_vcs2cam"
                    ]
                )
                if (
                    "T_local2cam"
                    in imgs_meta["calib_params"][
                        f"camera_{camera_name}"
                    ].keys()
                ):
                    T_local2cam_list.append(
                        imgs_meta["calib_params"][f"camera_{camera_name}"][
                            "T_local2cam"
                        ]
                    )
                    T_local2vcs_list.append(
                        imgs_meta["calib_params"][f"camera_{camera_name}"][
                            "T_local2vcs"
                        ]
                    )

        d_coef = np.array(d_coef_list, dtype=np.float32)

        imgs_meta["ori_camera_matrix"] = np.stack(cam_intrinsic_list)
        imgs_meta["ori_distcoeffs"] = d_coef
        imgs_meta["ori_vcs2cam"] = np.stack(T_vcs2cam_list)
        imgs_meta["ori_image_size"] = np.stack(
            [[x.shape[1], x.shape[0]] for x in imgs_meta["imgs"]]
        )

        # undistort image
        undistort_cam_intrinsic_list = []
        undistort_imgs = []
        for img_i in range(len(imgs_meta["imgs"])):
            raw_img = imgs_meta["imgs"][img_i]
            undistort_cam_intrinsic = cv2.getOptimalNewCameraMatrix(
                cam_intrinsic_list[img_i],
                d_coef[img_i],
                (raw_img.shape[1], raw_img.shape[0]),
                0,
                (raw_img.shape[1], raw_img.shape[0]),
            )[0]

            undistort_cam_intrinsic_list.append(undistort_cam_intrinsic)

            undistort_img = cv2.undistort(
                raw_img,
                cameraMatrix=cam_intrinsic_list[img_i],
                distCoeffs=d_coef[img_i],
                newCameraMatrix=undistort_cam_intrinsic,
            )
            undistort_imgs.append(undistort_img)

        cam_intrinsic = np.stack(undistort_cam_intrinsic_list)
        vcs2_cam = np.stack(T_vcs2cam_list)
        if len(T_local2vcs_list) > 0:
            local2vcs = np.stack(T_local2vcs_list)
            local2cam = np.stack(T_local2cam_list)
            imgs_meta["T_local2vcs"] = local2vcs
            imgs_meta["T_local2cam"] = local2cam

        if "ego_pose" in annotations:
            T_lidar2global = np.array(
                annotations["ego_pose"], dtype=np.float32
            )
            T_lidar2vcs = imgs_meta["calib_params"]["T_lidar2vcs"]
            T_vcs2lidar = np.linalg.inv(T_lidar2vcs)
            T_global2lidar = np.linalg.inv(T_lidar2global)
            T_vcs2global = np.dot(T_lidar2global, T_vcs2lidar)
            T_global2vcs = np.dot(T_lidar2vcs, T_global2lidar)
        else:
            T_vcs2global = np.eye(4, dtype=np.float32)
            T_global2vcs = np.eye(4, dtype=np.float32)

        imgs_meta["imgs"] = undistort_imgs
        imgs_meta["T_vcs2cam"] = vcs2_cam
        imgs_meta["camera_matrix"] = cam_intrinsic
        imgs_meta["gt_bboxes_3d"] = gt_bboxes_3d
        imgs_meta["gt_labels_3d"] = gt_labels_3d
        imgs_meta["timestamp"] = np.float64(annotations["timestamp"]) / 1e3
        imgs_meta["views_pad"] = annotations["views_pad"]
        imgs_meta["T_vcs2global"] = T_vcs2global
        imgs_meta["T_global2vcs"] = T_global2vcs
        imgs_meta.pop("calib_params")

        return imgs_meta


@OBJECT_REGISTRY.register
class MultiViewFlipResizeCrop(object):
    """Resize, crop and flip for Multiview images.

    Args:
        resize_target_dim: (resize_target_h, resize_target_w).
        keep_ratio: whether to keep aspect ratio.
        horizontal_flip_ratio: ratio to apply horizontal flip.
    """

    def __init__(
        self,
        resize_target_dim: Tuple[int, int],
        keep_ratio: bool = True,
        horizontal_flip_ratio: float = -1.0,
    ):
        self.resize_target_dim = resize_target_dim
        self.keep_ratio = keep_ratio
        self.horizontal_flip_ratio = horizontal_flip_ratio

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:

        imgs = data["imgs"]

        n_cams = len(imgs)

        target_h, target_w = self.resize_target_dim

        resize_imgs = []
        for i in range(n_cams):
            ori_img = imgs[i]

            ori_h, ori_w = ori_img.shape[:2]

            if self.keep_ratio:
                resize_rate = [
                    float(target_w / ori_w),
                    float(target_w / ori_w),
                ]
            else:
                resize_rate = [
                    float(target_h / ori_h),
                    float(target_w / ori_w),
                ]

            y_size = int(ori_h * resize_rate[0])
            x_size = int(ori_w * resize_rate[1])

            resize_img = mmcv.imresize(
                ori_img, (x_size, y_size), return_scale=False
            )

            target_h = min(target_h, y_size)
            crop_h = y_size - target_h
            crop = (0, crop_h, target_w, crop_h + target_h)

            resize_img = resize_img[
                crop_h : crop_h + target_h, 0:target_w, ...
            ]

            resize_imgs.append(resize_img)

            ida_rot = torch.eye(2)
            ida_tran = torch.zeros(2)
            ida_rot *= torch.Tensor(resize_rate[::-1]).unsqueeze(-1)
            ida_tran -= torch.Tensor(crop[:2])

            ida_mat = torch.eye(3)
            ida_mat[:2, :2] = ida_rot
            ida_mat[:2, 2] = ida_tran

            data["camera_matrix"][i, :3, :3] = (
                ida_mat @ data["camera_matrix"][i, :3, :3]
            )

        data["T_vcs2img"] = data["camera_matrix"] @ data["T_vcs2cam"][:, :3]

        if np.random.rand() < self.horizontal_flip_ratio:
            resize_imgs = [
                mmcv.imflip(img_, "horizontal") for img_ in resize_imgs
            ]
            data["img_horizontal_flip"] = True

        resize_imgs = [img_.astype(np.float32) for img_ in resize_imgs]
        data["imgs"] = resize_imgs

        data["img_shape"] = np.array([x.shape[:2] for x in resize_imgs])

        return data


@OBJECT_REGISTRY.register
class MultiViewPhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially. \
    every transformation is applied with a probability of 0.5. \
    The position of random contrast is in second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta: int = 32,
        contrast_range: tuple = (0.5, 1.5),
        saturation_range: tuple = (0.5, 1.5),
        hue_delta: int = 18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data["imgs"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the "
                "input image of dtype np.float32"
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(
                    -self.brightness_delta, self.brightness_delta
                )
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)

        data["imgs"] = new_imgs
        return data


@OBJECT_REGISTRY.register
class MultiViewNormalize(object):
    def __init__(
        self,
        img_norm_cfg: Dict[str, Any],
        category_view: List[str] = None,
    ):
        # self.img_norm_cfg = img_norm_cfg
        self.img_mean = np.array(img_norm_cfg["mean"], dtype=np.float32)
        self.img_std = np.array(img_norm_cfg["std"], dtype=np.float32)
        self.to_rgb = img_norm_cfg["to_rgb"]
        self.category_view = category_view

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.category_view is None:
            img_keys = ["imgs"]
        else:
            img_keys = [f"{_}_imgs" for _ in self.category_view.keys()]
        for k in img_keys:
            norm_imgs = [
                mmcv.imnormalize(
                    img_, self.img_mean, self.img_std, self.to_rgb
                )
                for img_ in data[k]
            ]

            data[k] = norm_imgs

        return data


@OBJECT_REGISTRY.register
class MultiViewPadImage(object):
    def __init__(
        self,
        size: tuple = None,
        size_divisor: int = None,
        pad_val: int = 0,
    ):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:

        if self.size is not None:
            padded_img = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
                for img in data["imgs"]
            ]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(
                    img, self.size_divisor, pad_val=self.pad_val
                )
                for img in data["imgs"]
            ]

        data["img_shape"] = np.array([x.shape[:2] for x in padded_img])
        data["imgs"] = padded_img

        return data


@OBJECT_REGISTRY.register
class MultiViewGridMask(object):
    """GridMask for grid masking augmentation."""

    def __init__(
        self,
        category_view: List[str] = None,
        **kwargs,
    ):
        """Generate GridMask."""
        super(MultiViewGridMask, self).__init__()
        self.grid_mask = GridMask(**kwargs)
        self.category_view = category_view

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.category_view is None:
            img_keys = ["imgs"]
        else:
            img_keys = [f"{_}_imgs" for _ in self.category_view.keys()]
        for k in img_keys:
            data[k] = [self.grid_mask(x) for x in data[k]]

        return data


@OBJECT_REGISTRY.register
class MVT4DImgFormat(object):
    def __init__(self, category_view: List[str] = None):
        self.category_view = category_view

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.category_view is None:
            imgs = [img_.transpose(2, 0, 1) for img_ in data["imgs"]]
            imgs = np.ascontiguousarray(np.stack(imgs, axis=0))

            data["img"] = ToTensor._to_tensor(imgs)
        else:
            img_keys = [f"{_}_imgs" for _ in self.category_view.keys()]
            for k in img_keys:
                imgs = [img_.transpose(2, 0, 1) for img_ in data[k]]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))

                # adapt to network input
                data[k.replace("imgs", "img")] = ToTensor._to_tensor(imgs)

        return data


@OBJECT_REGISTRY.register
class MultiViewRangeFliter(object):
    def __init__(
        self,
        point_cloud_range: List[float],
        max_num: int = 300,
        state_dims: int = 7,
    ):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)
        self.max_num = max_num
        self.state_dims = state_dims

    def limit_period(
        self, val: np.ndarray, offset: float = 0.5, period: float = np.pi
    ) -> np.ndarray:
        limited_val = val - np.floor(val / period + offset) * period
        return limited_val

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:

        if "gt_labels_3d" in data:

            gt_bboxes_3d = data["gt_bboxes_3d"]
            gt_labels_3d = data["gt_labels_3d"]
            if gt_bboxes_3d.shape[0] > 0:
                bev_bboxes = gt_bboxes_3d[:, [0, 1, 3, 4, 6]]
                if len(self.pcd_range) >= 5:
                    bev_range = self.pcd_range[[0, 1, 3, 4]]
                    bev_range_mask = (
                        (bev_bboxes[:, 0] > bev_range[0])
                        & (bev_bboxes[:, 1] > bev_range[1])
                        & (bev_bboxes[:, 0] < bev_range[2])
                        & (bev_bboxes[:, 1] < bev_range[3])
                    )
                else:
                    bev_range_mask = (
                        np.linalg.norm(bev_bboxes[:, :2], axis=-1)
                        < self.pcd_range[0]
                    )

                gt_labels_3d = gt_labels_3d[bev_range_mask]
                gt_bboxes_3d = gt_bboxes_3d[bev_range_mask]
                gt_bboxes_3d[:, 6] = self.limit_period(
                    gt_bboxes_3d[:, 6], offset=0.5, period=2 * np.pi
                )

            else:
                gt_bboxes_3d = np.zeros((0, self.state_dims), dtype=np.float32)

            gt_bboxes_3d = np.pad(
                gt_bboxes_3d,
                ((0, self.max_num - gt_bboxes_3d.shape[0]), (0, 0)),
                "constant",
                constant_values=(0.0, 0.0),
            )
            gt_labels_3d = np.pad(
                gt_labels_3d,
                (0, self.max_num - gt_labels_3d.shape[0]),
                "constant",
                constant_values=(-1, -1),
            )

            data["gt_bboxes_3d"] = ToTensor._to_tensor(gt_bboxes_3d)
            data["gt_labels_3d"] = ToTensor._to_tensor(gt_labels_3d)

        return data


@OBJECT_REGISTRY.register
class MultiViewCollect3D(object):
    def __init__(
        self,
        keep_keys: tuple = (
            "img",
            "gt_labels_3d",
            "gt_bboxes_3d",
            "img_metas",
        ),
        img_metas_keys: tuple = (
            "img_shape",
            "T_vcs2img",
            "timestamp",
            "img_horizontal_flip",
        ),
    ):
        self.keep_keys = keep_keys
        self.img_metas_keys = img_metas_keys

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image_metas = {}
        for key_ in self.img_metas_keys:
            if key_ in data:
                image_metas[key_] = data[key_]
        data["img_metas"] = image_metas

        output = {}
        for key_ in self.keep_keys:
            if key_ in data:
                output[key_] = data[key_]
        return output


@OBJECT_REGISTRY.register
class MultiScaleDepthMapGenerator(object):
    def __init__(self, downsample=1, max_depth=60, project_key="lidar2img"):
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample
        self.max_depth = max_depth
        self.project_key = project_key

    def __call__(self, data):
        points = data["points"][..., :3, None]
        gt_depth = []
        for i, projection_mat in enumerate(data[self.project_key]):
            H, W = data["img_shape"][i][:2]

            pts_2d = (
                np.squeeze(projection_mat[:3, :3] @ points, axis=-1)
                + projection_mat[:3, 3]
            )
            pts_2d[:, :2] /= pts_2d[:, 2:3]
            U = np.round(pts_2d[:, 0]).astype(np.int32)
            V = np.round(pts_2d[:, 1]).astype(np.int32)
            depth = pts_2d[:, 2]
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depth >= 0.1,
                    depth <= self.max_depth,
                ]
            )
            V, U, depth = V[mask], U[mask], depth[mask]
            sort_idx = np.argsort(depth)[::-1]
            V, U, depth = V[sort_idx], U[sort_idx], depth[sort_idx]

            for j, downsample in enumerate(self.downsample):
                if len(gt_depth) < j + 1:
                    gt_depth.append([])
                h, w = (int(H / downsample), int(W / downsample))
                u = np.floor(U / downsample).astype(np.int32)
                v = np.floor(V / downsample).astype(np.int32)
                depth_map = np.ones([h, w], dtype=np.float32) * -1
                depth_map[v, u] = depth
                gt_depth[j].append(depth_map)
        data["gt_depth"] = [np.stack(x) for x in gt_depth]
        return data


@OBJECT_REGISTRY.register
class ResizeCropFlipImage(object):
    def __init__(
        self,
        transform_matrix_key=("lidar2img", "cam_intrinsic"),
        data_aug_configs=None,
        test_mode=True,
        return_ori_imgs=False,
        data_aug_config_list=None,
    ):
        self.transform_matrix_key = transform_matrix_key
        self.data_aug_configs = data_aug_configs
        self.test_mode = test_mode
        self.return_ori_imgs = return_ori_imgs
        if data_aug_configs is not None:
            self.data_aug_configs = {
                "resize_range": (1.0, 1.0),
                "rotation_range": (0.0, 0.0),
                "rand_flip": False,
                "crop_vertical_range": (0.0, 0.0),
            }
            self.data_aug_configs.update(data_aug_configs)
        self.data_aug_config_list = data_aug_config_list

    def __call__(self, data):
        imgs = data.get("imgs")
        if imgs is None or len(imgs) == 0:
            return data
        aug_configs = data.get("aug_configs")
        if aug_configs is None and self.data_aug_config_list is None:
            assert self.data_aug_configs is not None
            aug_configs = self.get_aug_configs(imgs[0].shape[:2])
        new_imgs = []
        for i, img in enumerate(imgs):
            if self.data_aug_config_list:
                aug_configs = self.data_aug_config_list[i]
            img, extend_matrix = self._img_transform(img, aug_configs)
            new_imgs.append(img)
            for key in self.transform_matrix_key:
                D = data[key][i].shape[0]
                data[key][i] = extend_matrix[:D, :D] @ data[key][i]

        data["imgs"] = new_imgs
        if self.return_ori_imgs:
            data["ori_imgs"] = np.stack(new_imgs).copy()
        data["img_shape"] = [x.shape[:2] for x in data["imgs"]]
        data["img_horizontal_flip"] = aug_configs.get("flip", False)
        return data

    def _img_transform(self, img, aug_configs):
        H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = (int(W * resize), int(H * resize))
        crop = aug_configs.get("crop", [0, 0, *resize_dims])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        # recommended to do data augmentation on uint8 img
        # rather than normalized img.
        origin_dtype = img.dtype
        if origin_dtype != np.uint8:
            min_value = img.min()
            max_vaule = img.max()
            scale = 255 / (max_vaule - min_value)
            img = (img - min_value) * scale
            img = np.uint8(img)

        img = Image.fromarray(np.uint8(img))
        img = img.resize(resize_dims).crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        img = np.array(img).astype(np.float32)

        if origin_dtype != np.uint8:
            img = img.astype(np.float32)
            img = img / scale + min_value

        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.array(crop[:2])
        if flip:
            flip_matrix = np.array(
                [[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]]
            )
            transform_matrix = flip_matrix @ transform_matrix
        rotate = rotate / 180 * np.pi
        rot_matrix = np.array(
            [
                [np.cos(rotate), np.sin(rotate), 0],
                [-np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1],
            ]
        )
        rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
        transform_matrix = rot_matrix @ transform_matrix
        extend_matrix = np.eye(4)
        extend_matrix[:3, :3] = transform_matrix
        return img, extend_matrix

    def get_aug_configs(self, origin_img_hw=None):
        if origin_img_hw is not None:
            H, W = origin_img_hw
        else:
            H, W = self.data_aug_configs["origin_img_hw"]
        fH, fW = self.data_aug_configs["output_img_hw"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_configs["resize_range"])
            resize_dims = (int(W * resize), int(H * resize))
            crop_top_proportion = np.random.uniform(
                *self.data_aug_configs["crop_vertical_range"]
            )
            crop_w = int(np.random.uniform(0, max(0, resize_dims[0] - fW)))
            flip = self.data_aug_configs["rand_flip"]
            flip = flip and np.random.choice([True, False])
            rotate = np.random.uniform(
                *self.data_aug_configs["rotation_range"]
            )
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            crop_top_proportion = np.mean(
                self.data_aug_configs["crop_vertical_range"]
            )
            crop_h = int((1 - crop_top_proportion) * resize_dims[1]) - fH
            crop_w = int(max(0, resize_dims[0] - fW) / 2)
            flip = False
            rotate = 0
        crop_h = int((1 - crop_top_proportion) * resize_dims[1]) - fH
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        aug_configs = {
            "resize": resize,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
        }
        return aug_configs


@OBJECT_REGISTRY.register
class BBoxRotation(object):
    def __init__(
        self,
        transform_matrix_key=("lidar2img",),
        global_key=("lidar2global",),
        rotation_3d_range=None,
    ):
        self.transform_matrix_key = transform_matrix_key
        self.global_key = global_key
        self.rotation_3d_range = rotation_3d_range

    def __call__(self, data):
        aug_configs = data.get("aug_configs")
        if aug_configs is None or "rotate_3d" not in aug_configs:
            assert self.rotation_3d_range is not None
            angle = np.random.uniform(*self.rotation_3d_range)
        else:
            angle = aug_configs["rotate_3d"]
        angle = angle / 180 * np.pi
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)
        for key in self.transform_matrix_key:
            assert key in data, f"miss key {key} in data"
            for i, mat in enumerate(data[key]):
                data[key][i] = mat @ rot_mat_inv
        for key in self.global_key:
            assert key in data, f"miss key {key} in data"
            data[key] = data[key] @ rot_mat_inv

        if "gt_bboxes_3d" in data and data["gt_bboxes_3d"].shape[0] > 0:
            data["gt_bboxes_3d"] = self.box_rotate(data["gt_bboxes_3d"], angle)
        return data

    @staticmethod
    def box_rotate(bbox_3d, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array(
            [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T
        bbox_3d[:, 6] += angle
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d[:, 7:].shape[-1]
            bbox_3d[:, 7:] = bbox_3d[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]
        return bbox_3d


@OBJECT_REGISTRY.register
class Sparse4DAdaptor(object):
    def __init__(
        self,
        projection_key="lidar2img",
        img_shape_key="img_shape",
        ego_pose_key="lidar2global",
        cam_intrinsic_key="cam_intrinsic",
        category_view: List[str] = None,
    ):
        self.projection_key = projection_key
        self.img_shape_key = img_shape_key
        self.ego_pose_key = ego_pose_key
        self.cam_intrinsic_key = cam_intrinsic_key
        self.category_view = category_view

    def __call__(self, data):
        if self.category_view is None:
            keys = [""]
        else:
            keys = [f"{_}_" for _ in self.category_view.keys()]
        for k in keys:
            data[f"{k}projection_mat"] = np.float32(
                np.stack(data[k + self.projection_key])
            )
            if k + self.img_shape_key in data:
                data[f"{k}image_wh"] = np.ascontiguousarray(
                    np.array(data[k + self.img_shape_key], dtype=np.float32)[
                        :, :2
                    ][:, ::-1]
                )
            if k + self.cam_intrinsic_key in data:
                data[f"{k}cam_intrinsic"] = np.float32(
                    np.stack(data[k + self.cam_intrinsic_key])
                )
                data[f"{k}focal"] = np.sqrt(
                    np.abs(np.linalg.det(data[f"{k}cam_intrinsic"][:, :2, :2]))
                )
        if self.ego_pose_key in data:
            data["T_global_inv"] = np.linalg.inv(data[self.ego_pose_key])
            data["T_global"] = data[self.ego_pose_key]

        return data


@OBJECT_REGISTRY.register
class ViewPadMask(object):
    """View pad mask generation.

    If the corresponding perspective is filled, the corresponding mask is 1.

    Args:
        num_cam: The number of corresponding model input perspectives.
    """

    def __init__(self, num_cam):
        self.num_cam = num_cam

    def __call__(self, data):
        views_pad = data["views_pad"]
        mask = np.zeros(self.num_cam, dtype=np.int8)
        mask[views_pad] = 1
        data["view_pad_mask"] = mask
        return data


@OBJECT_REGISTRY.register
class VirtualCropCamera(object):
    """Virtual crop camera.

    Current only support contrust virtual camera for front or rear or
    front_30fov views. More detail see
    https://horizonrobotics.feishu.cn/wiki/NhyUwTVdziRkJtkRWjVckchPnBa.

    Args:
        virtual_view: Which perspectives are added to the virtual crop camera.
        crop_roi: The crop roi corresponding to each virtual crop camera.
        camera_view_names: The name corresponding to the current camera of
            each view.
        prob: Whether to replace front30fov camera with front virtual crop
            camera.
    """

    def __init__(
        self,
        virtual_view: List[str] = None,
        crop_roi: List[list] = None,
        camera_view_names: List = None,
        prob: float = 0.0,
    ):
        if virtual_view is not None:
            for _ in virtual_view:
                assert _ == "front" or _ == "rear" or _ == "front_30fov", (
                    "Current only support contrust virtual camera for front or"
                    + " rear or front_30fov views"
                )
        self.virtual_view = virtual_view
        self.crop_roi = crop_roi
        self.camera_view_names = camera_view_names
        self.prob = prob

    def __call__(self, data):
        for i, virtual in enumerate(self.virtual_view):
            assert virtual in self.camera_view_names
            bbox = self.crop_roi[i]
            x1, y1, x2, y2 = bbox
            camera_id = self.camera_view_names.index(virtual)

            ida_rot = torch.eye(2)
            ida_tran = torch.zeros(2)
            crop = [x1, y1]
            ida_tran -= torch.Tensor(crop[:2])

            ida_mat = torch.eye(3)
            ida_mat[:2, :2] = ida_rot
            ida_mat[:2, 2] = ida_tran

            camera_matrix = data.get("camera_matrix", None)
            ori_camera_matrix = data.get("ori_camera_matrix", None)

            if camera_id in data["views_pad"]:
                ida_mat = np.identity(3)
            if camera_matrix is not None:
                data["camera_matrix"] = np.concatenate(
                    (
                        camera_matrix,
                        (ida_mat @ camera_matrix[camera_id])[None, ...],
                    ),
                    axis=0,
                )
            if ori_camera_matrix is not None:
                data["ori_camera_matrix"] = np.concatenate(
                    (
                        ori_camera_matrix,
                        (ida_mat @ ori_camera_matrix[camera_id])[None, ...],
                    ),
                    axis=0,
                )
            virtual_img = data["imgs"][camera_id][y1:y2, x1:x2, :]
            data["imgs"].append(virtual_img)

            ori_image_size = data.get("ori_image_size", None)
            views_pad = data.get("views_pad", None)

            for tmp_key in [
                "ori_distcoeffs",
                "ori_vcs2cam",
                "T_local2vcs",
                "T_local2cam",
                "T_vcs2cam",
            ]:
                tmp_data = data.get(tmp_key, None)
                if tmp_data is not None:
                    data[tmp_key] = np.concatenate(
                        (tmp_data, tmp_data[camera_id][None, ...]), axis=0
                    )

            if ori_image_size is not None:
                data["ori_image_size"] = np.concatenate(
                    (
                        ori_image_size,
                        np.array(virtual_img.shape[:2])[None, ...],
                    ),
                    axis=0,
                )
            if views_pad is not None:
                if camera_id in views_pad:
                    data["views_pad"].append(len(self.camera_view_names) + i)

        # 如果存在虚拟front crop相机
        # 1. 若front 30fov存在，则删除front crop相机
        # 2. 若front 30fov不存在，则用front crop相机替代front 30fov
        if "front" in self.virtual_view:
            if "front_30fov" in self.camera_view_names:
                front_narrow_index = self.camera_view_names.index(
                    "front_30fov"
                )
                front_virtual_index = len(
                    self.camera_view_names
                ) + self.virtual_view.index("front")
                if front_narrow_index in data["views_pad"]:
                    delete_index = front_narrow_index
                else:
                    if np.random.rand() < self.prob:
                        delete_index = front_narrow_index
                    else:
                        delete_index = front_virtual_index

                for tmp_key in [
                    "camera_matrix",
                    "ori_camera_matrix",
                    "ori_distcoeffs",
                    "ori_vcs2cam",
                    "ori_image_size",
                    "T_local2vcs",
                    "T_local2cam",
                    "T_vcs2cam",
                ]:
                    if tmp_key in data.keys():
                        if delete_index == front_narrow_index:
                            data[tmp_key][front_narrow_index] = data[tmp_key][
                                front_virtual_index
                            ]
                        data[tmp_key] = np.delete(
                            data[tmp_key], delete_index, 0
                        )
                if "imgs" in data.keys():
                    if delete_index == front_narrow_index:
                        data["imgs"][front_narrow_index] = data["imgs"][
                            front_virtual_index
                        ]
                    data["imgs"].pop(delete_index)
                if "views_pad" in data.keys():
                    new_views_pad = []
                    if front_narrow_index in data["views_pad"]:
                        for views_pad_id in data["views_pad"]:
                            if views_pad_id == delete_index:
                                continue
                            if views_pad_id <= len(self.camera_view_names) - 1:
                                new_views_pad.append(views_pad_id)
                            else:
                                new_views_pad.append(views_pad_id - 1)
                    else:
                        virtual_front_id = self.virtual_view.index(
                            "front"
                        ) + len(self.camera_view_names)
                        for views_pad_id in data["views_pad"]:
                            if views_pad_id == delete_index:
                                continue
                            if views_pad_id < virtual_front_id:
                                new_views_pad.append(views_pad_id)
                            else:
                                # 因为front是在前视窄角不存在时自动填充，所以实际num_cam不会增加
                                # 虚拟相机中位于front后的虚拟相机pad_id需要减-1
                                new_views_pad.append(views_pad_id - 1)
                    data["views_pad"] = new_views_pad
        return data


@OBJECT_REGISTRY.register
class SplitMultiView(object):
    """Divide multiple backbone inputs.

    Support the use of different backbones from different perspectives.
    More details see this link:
    https://horizonrobotics.feishu.cn/wiki/WRwBw7ZmFixlOakwtsxcStWqnEe.

    Args:
        category_view: The perspective name corresponding to each backbone.
        camera_view_names: The name corresponding to the current camera of
            each view.
        split_key: Which keys need to be split according to category_view.
    """

    def __init__(
        self,
        category_view: List[str] = None,
        camera_view_names: List[str] = None,
        split_key: List[str] = None,
    ):
        self.category_view = category_view
        self.camera_view_names = camera_view_names
        assert split_key is not None
        self.split_key = split_key

    def __call__(self, data):
        if self.category_view is None:
            return data

        for k, v in self.category_view.items():
            tmp_index = []
            for _ in v:
                if _ in self.camera_view_names:
                    tmp_index.append(self.camera_view_names.index(_))
            setattr(self, f"{k}_index", tmp_index)

        output = {}

        for k, v in data.items():
            if k in self.split_key:
                if isinstance(v, list):
                    tmp_v = [
                        [v[_] for _ in getattr(self, f"{i}_index")]
                        for i in self.category_view.keys()
                    ]
                elif isinstance(v, np.ndarray):
                    tmp_v = [
                        np.concatenate(
                            [
                                v[_][None, :]
                                for _ in getattr(self, f"{i}_index")
                            ],
                            axis=0,
                        )
                        for i in self.category_view.keys()
                    ]
                else:
                    continue
                for index, _ in enumerate(self.category_view.keys()):
                    output[f"{_}_{k}"] = tmp_v[index]
            else:
                output[k] = v
        return output


@OBJECT_REGISTRY.register
class SplitMultiViewFlipResizeCrop(object):
    """Resize, crop and flip for Multiview images.

    Args:
        resize_target_dim: (resize_target_h, resize_target_w).
        keep_ratio: whether to keep aspect ratio.
        horizontal_flip_ratio: ratio to apply horizontal flip.
        category_view: The perspective name corresponding to each backbone.
    """

    def __init__(
        self,
        resize_target_dim: Tuple[int, int],
        keep_ratio: bool = True,
        horizontal_flip_ratio: float = -1.0,
        category_view: List[str] = None,
    ):
        self.resize_target_dim = resize_target_dim
        self.keep_ratio = keep_ratio
        self.horizontal_flip_ratio = horizontal_flip_ratio
        self.category_view = category_view

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip_flag = np.random.rand() < self.horizontal_flip_ratio

        for view in self.category_view.keys():
            imgs = data[f"{view}_imgs"]

            n_cams = len(imgs)

            target_h, target_w = self.resize_target_dim[view]

            resize_imgs = []
            for i in range(n_cams):
                ori_img = imgs[i]

                ori_h, ori_w = ori_img.shape[:2]

                if self.keep_ratio:
                    resize_rate = [
                        float(target_w / ori_w),
                        float(target_w / ori_w),
                    ]
                else:
                    resize_rate = [
                        float(target_h / ori_h),
                        float(target_w / ori_w),
                    ]

                y_size = int(ori_h * resize_rate[0])
                x_size = int(ori_w * resize_rate[1])

                resize_img = mmcv.imresize(
                    ori_img, (x_size, y_size), return_scale=False
                )

                target_h = min(target_h, y_size)
                crop_h = y_size - target_h
                crop = (0, crop_h, target_w, crop_h + target_h)

                resize_img = resize_img[
                    crop_h : crop_h + target_h, 0:target_w, ...
                ]

                resize_imgs.append(resize_img)

                ida_rot = torch.eye(2)
                ida_tran = torch.zeros(2)
                ida_rot *= torch.Tensor(resize_rate[::-1]).unsqueeze(-1)
                ida_tran -= torch.Tensor(crop[:2])

                ida_mat = torch.eye(3)
                ida_mat[:2, :2] = ida_rot
                ida_mat[:2, 2] = ida_tran

                data[f"{view}_camera_matrix"][i, :3, :3] = (
                    ida_mat @ data[f"{view}_camera_matrix"][i, :3, :3]
                )

            data[f"{view}_T_vcs2img"] = (
                data[f"{view}_camera_matrix"]
                @ data[f"{view}_T_vcs2cam"][:, :3]
            )

            if flip_flag:
                resize_imgs = [
                    mmcv.imflip(img_, "horizontal") for img_ in resize_imgs
                ]
                data["img_horizontal_flip"] = True

            resize_imgs = [img_.astype(np.float32) for img_ in resize_imgs]
            data[f"{view}_imgs"] = resize_imgs

            data[f"{view}_img_shape"] = np.array(
                [x.shape[:2] for x in resize_imgs]
            )

        return data


@OBJECT_REGISTRY.register
class SplitMultiViewPadImage(MultiViewPadImage):
    """Inherited from MultiViewPadImage, supports multiple backbones.

    Args:
        category_view:  The perspective name corresponding to each backbone.
    """

    def __init__(
        self,
        size: tuple = None,
        size_divisor: int = None,
        pad_val: int = 0,
        category_view: List[str] = None,
    ):
        super(SplitMultiViewPadImage, self).__init__(
            size,
            size_divisor,
            pad_val,
        )
        self.category_view = category_view

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        img_keys = [f"{_}_imgs" for _ in self.category_view.keys()]
        self.target_size = [self.size[_] for _ in self.category_view.keys()]
        for index, k in enumerate(img_keys):
            if self.target_size is not None:
                padded_img = [
                    mmcv.impad(
                        img,
                        shape=self.target_size[index],
                        pad_val=self.pad_val,
                    )
                    for img in data[k]
                ]
            elif self.size_divisor is not None:
                padded_img = [
                    mmcv.impad_to_multiple(
                        img, self.size_divisor, pad_val=self.pad_val
                    )
                    for img in data[k]
                ]
            prefix = k.split("_")[0] + "_"
            data[f"{prefix}img_shape"] = np.array(
                [x.shape[:2] for x in padded_img]
            )
            data[k] = padded_img

        return data


@OBJECT_REGISTRY.register
class SplitMultiViewCollect3D(MultiViewCollect3D):
    """Inherited from MultiViewCollect3D, supports multiple backbones.

    Args:
        category_view: The perspective name corresponding to each backbone.
    """

    def __init__(
        self,
        keep_keys,
        img_metas_keys,
        category_view: List[str] = None,
    ):
        super(SplitMultiViewCollect3D, self).__init__(
            keep_keys, img_metas_keys
        )
        self.category_view = category_view

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image_metas = {}
        for key_ in self.img_metas_keys:
            if key_ in data:
                image_metas[key_] = data[key_]
            else:
                collect_tensor_list = []
                for view in self.category_view.keys():
                    if f"{view}_{key_}" in data:
                        collect_tensor_list.append(data[f"{view}_{key_}"])
                    image_metas[key_] = np.concatenate(
                        collect_tensor_list, axis=0
                    )
        data["img_metas"] = image_metas
        output = {}
        for key_ in self.keep_keys:
            if key_ in data:
                output[key_] = data[key_]
            else:
                collect_tensor_list = []
                for view in self.category_view.keys():
                    if f"{view}_{key_}" in data:
                        collect_tensor_list.append(data[f"{view}_{key_}"])
                if key_ == "img":
                    output[key_] = []
                    for index, _ in enumerate(self.category_view.keys()):
                        output["img"].append(collect_tensor_list[index])
                elif collect_tensor_list:
                    output[key_] = np.concatenate(collect_tensor_list, axis=0)
        return output


@OBJECT_REGISTRY.register
class SplitMultiViewPhotoMetricDistortion(MultiViewPhotoMetricDistortion):
    """Inherited from MultiViewCollect3D, supports multiple backbones.

    Args:
        category_view: The perspective name corresponding to each backbone.
    """

    def __init__(
        self,
        brightness_delta: int = 32,
        contrast_range: tuple = (0.5, 1.5),
        saturation_range: tuple = (0.5, 1.5),
        hue_delta: int = 18,
        category_view: List[str] = None,
    ):
        super(SplitMultiViewPhotoMetricDistortion, self).__init__(
            brightness_delta,
            contrast_range,
            saturation_range,
            hue_delta,
        )
        self.category_view = category_view

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        img_keys = [f"{_}_imgs" for _ in self.category_view.keys()]
        for k in img_keys:
            tmp_data = {"imgs": data[k]}
            tmp_data = super().__call__(tmp_data)
            data[k] = tmp_data["imgs"]
        return data


@OBJECT_REGISTRY.register
class Real3DDatasetAdaptor(object):
    def __init__(
        self,
        num_classes=1,
        state_dims=7,
        skip_ignore=True,
        extend_intrinsic_matrix=True,
    ):
        self.category_id_dict = Real3DDataset.get_category_id_dict(num_classes)
        self.state_dims = state_dims
        self.skip_ignore = skip_ignore
        self.extend_intrinsic_matrix = extend_intrinsic_matrix

    def __call__(self, data):
        out = dict()  # noqa: C408
        out["imgs"] = data["imgs"]

        camera_matrix = data["calibration"]
        out["cam_intrinsic"] = np.array([camera_matrix])
        out["cam_distcoeffs"] = np.array([data["dist_coeffs"]])

        viewpad = np.eye(4)
        out["identity_trans_matrix"] = np.array([viewpad], dtype=np.float32)
        if self.extend_intrinsic_matrix:
            viewpad[
                : camera_matrix.shape[0], : camera_matrix.shape[1]
            ] = camera_matrix
            out["cam_intrinsic"] = np.array([viewpad])
        # ms -> s
        out["timestamp"] = float(data["timestamp"]) / 1000

        gt_labels_3d = []
        gt_bboxes_3d = []
        gt_ignore = []
        for obj in data["annotations"]:
            if obj["ignore"] and self.skip_ignore:
                continue

            category_id = self.category_id_dict[obj["category_id"]]
            if category_id == -99:
                continue

            gt_ignore.append(obj["ignore"])
            gt_labels_3d.append(category_id)

            if "in_camera" in obj:
                obj.update(obj.pop("in_camera"))
            dim_cam = [obj["dim"][2], obj["dim"][0], obj["dim"][1]]
            gt_bboxes_3d.append(
                obj["location"] + dim_cam + [obj["rotation_y"]]
            )

        out["gt_labels_3d"] = np.array(gt_labels_3d)
        if gt_bboxes_3d:
            out["gt_bboxes_3d"] = np.array(gt_bboxes_3d)
        else:
            out["gt_bboxes_3d"] = np.zeros(
                (0, self.state_dims), dtype=np.float32
            )

        if not self.skip_ignore:
            assert len(gt_ignore) == len(gt_labels_3d) == len(gt_bboxes_3d)
            out["gt_ignore"] = np.array(gt_ignore, dtype=np.bool)

        return out
