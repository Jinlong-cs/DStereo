# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import uuid
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from hat.core.box3d_utils import (
    compute_box_3d,
    corners_to_local_rot_y,
    get_3dboxcorner_in_velo,
)
from hat.core.box_utils import xywh_to_x1y1x2y2
from hat.core.differentiable_camera.utils import VIRTUAL_CAMERA_MAP
from hat.core.virtual_camera import (
    CameraBase,
    CylindricalCamera,
    FisheyeCamera,
    PinholeCamera,
    SphericalCamera,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list, multi_apply
from hat.utils.package_helper import check_packages_available

try:
    from pycocotools import mask as coco_mask
except ImportError:
    coco_mask = None


__all__ = [
    "CreateVirtualCamera",
    "FCOS3DCameraPlugin",
    "VCWarpImageRec",
    "CameraParamResize",
    "CameraPresetParamCrop",
    "CameraParamHorizontalFlip2D",
    "CameraParamVerticalTranslation",
    "DiffCamImageProjector",
]

logger = logging.getLogger(__name__)

CAMERA_TYPE = ["Cylindrical", "Pinhole", "Fisheye", "Spherical"]

CLASS_PED = 0
CLASS_CYC = 1
CLASS_CAR = 2


@OBJECT_REGISTRY.register
class CreateVirtualCamera(object):
    """Generate CreateVirtualCamera.

    Args:
        task_type: task type for choice init camera type.
            Default is "real3d".
        source_cam: original camera type. Default: Fisheye.
        virtual_cam: target camera type. Default: Cylindrical
        image_size: image size for [width, height]
        cameraMatrix: cameraMatrix the same as opencv usage
        poseMat: camera extrinic param matrix, shape=(4,4).
            Default is None.
        distcoeffs: camera distort coeffs.
        is_virtual: is virtual camera. Default: False.
        calib_json: horizon camera calib json file.
    """

    def __init__(
        self,
        task_type: str = "real3d",
        source_cam: str = "Fisheye",
        virtual_cam: str = "Cylindrical",
        image_size: Optional[List] = None,
        cameraMatrix: Optional[np.ndarray] = None,
        poseMat: Optional[np.ndarray] = None,
        distcoeffs: Optional[np.ndarray] = None,
        is_virtual: bool = False,
        calib_json: Optional[str] = None,
    ):
        self.task_type = task_type
        assert virtual_cam in CAMERA_TYPE, f"camera_type must in {CAMERA_TYPE}"
        assert source_cam in CAMERA_TYPE, f"camera_type must in {CAMERA_TYPE}"
        self.src_cam_type = source_cam
        self.dst_cam_type = virtual_cam
        self.image_size = image_size
        self.cameraMatrix = cameraMatrix
        self.is_virtual = is_virtual
        self.poseMat = poseMat
        self.distcoeffs = distcoeffs
        self.calib_json = calib_json
        if task_type == "inference":
            assert calib_json.endswith(
                "json"
            ), "test mode must set calib json path"
        self.uv_mat_cached = {}

    def _create_camera(
        self,
        camera_type,
        image_size=None,
        poseMat=None,
        cameraMatrix=None,
        distCoeffs=None,
        is_virtual=False,
        calib_json=None,
    ):
        if poseMat is None:
            poseMat = np.identity(4, np.float32)

        if camera_type == "Cylindrical":
            camera = CylindricalCamera()
            distCoeffs = np.zeros(4)
        elif camera_type == "Pinhole":
            camera = PinholeCamera()
        elif camera_type == "Fisheye":
            camera = FisheyeCamera()
        elif camera_type == "Spherical":
            camera = SphericalCamera()
            distCoeffs = np.zeros(4)
        else:
            raise NotImplementedError
        if calib_json:
            camera = camera.init_cam_param_by_file(calib_json, is_virtual)
            if is_virtual:
                if image_size is not None:
                    camera.image_size = image_size
                if cameraMatrix is not None:
                    camera.camera_matrix = cameraMatrix

        else:
            camera = camera.init_cam_param_by_matrix(
                image_size, cameraMatrix, poseMat, distCoeffs, is_virtual
            )
        return camera

    def _add_real3d_camera(self, label: Dict):
        """Generate real3d camera.

        Args:
            label:
                The dict contains at leaset below keys:
                    image_width, image_height, calibration
                    dist_coeffs, Tr_vcs2cam
        """

        image_size = np.array([label["image_width"], label["image_height"]])
        cameraMatrix = np.array(label["calibration"])[:3, :3]
        distCoeffs = label["dist_coeffs"]
        poseMat_vcs2cam = label.get("Tr_vcs2cam", np.eye(4))
        if label.get("Tr_vcs2cam") is None:
            label["Tr_vcs2cam"] = np.eye(4)
        # source camera default real camera
        if "camera_model" in label:
            src_cam_type = label.pop("camera_model")
        else:
            src_cam_type = self.src_cam_type

        self.src_cam = self._create_camera(
            src_cam_type,
            deepcopy(image_size),
            deepcopy(poseMat_vcs2cam),
            deepcopy(cameraMatrix),
            deepcopy(distCoeffs),
            False,
        )
        self.src_cam.poseMat_lidar2cam = deepcopy(label["Tr_vel2cam"])
        if self.image_size is not None:
            image_size = self.image_size
        if self.cameraMatrix is not None:
            cameraMatrix = self.cameraMatrix

        if (
            self.src_cam_type == "Pinhole"
            and self.dst_cam_type == "Cylindrical"
        ):
            is_virtual = True
        else:
            is_virtual = self.is_virtual

        self.dst_cam = self._create_camera(
            self.dst_cam_type,
            deepcopy(image_size),
            deepcopy(poseMat_vcs2cam),
            deepcopy(cameraMatrix),
            deepcopy(distCoeffs),
            is_virtual,
        )
        if is_virtual:
            # transform lidar2cam of src_cam to dst_cam
            src_lidar2vcs = (
                np.linalg.inv(self.src_cam.poseMat_vcs2cam)
                @ label["Tr_vel2cam"]
            )
            self.dst_cam.poseMat_lidar2cam = np.dot(
                self.dst_cam.poseMat_vcs2cam, src_lidar2vcs
            )
        else:
            self.dst_cam.poseMat_lidar2cam = deepcopy(label["Tr_vel2cam"])

        label["source_cam"] = self.src_cam
        label["virtual_cam"] = self.dst_cam
        return label

    def _add_inference_camera(self, label, calib_json):
        """Generate labels.

        Args:
            label: default format labels
            calib_json: calib file path.
        """
        assert calib_json, "must set calib_json path."
        # source camera default real camera
        self.src_cam = self._create_camera(
            self.src_cam_type,
            is_virtual=False,
            calib_json=calib_json,
        )
        self.dst_cam = self._create_camera(
            self.dst_cam_type,
            image_size=self.image_size,
            cameraMatrix=self.cameraMatrix,
            is_virtual=self.is_virtual,
            calib_json=calib_json,
        )
        label["source_cam"] = self.src_cam
        label["virtual_cam"] = self.dst_cam

        # warp image
        image_name = label["image_name"]
        # only usful for aidi default exat frame
        date_loc = image_name.split("__")[0]
        # Warning: only for aidi default exat frame
        # else: Watch out! out of memery!
        if not self.uv_mat_cached.get(date_loc):
            uv_map = self.src_cam.generate_mapping(self.dst_cam)
            self.uv_mat_cached[date_loc] = uv_map

        label["img"] = self.src_cam.project_image2dstCam(
            self.dst_cam, label["img"], self.uv_mat_cached[date_loc]
        )
        label["image_height"] = self.dst_cam.image_size[1]
        label["image_width"] = self.dst_cam.image_size[0]
        label["uv_map"] = self.uv_mat_cached[date_loc]
        return label

    def __call__(self, label):
        if self.task_type == "real3d":
            label = self._add_real3d_camera(label)
        elif self.task_type == "inference":
            self._add_inference_camera(label, self.calib_json)
        return label


@OBJECT_REGISTRY.register
class WarpImagePlugin(nn.Module):
    def __init__(
        self,
    ):
        super(WarpImagePlugin, self).__init__()

    @staticmethod
    def image_to_tensor(
        image: Union["np.ndarray", "torch.Tensor"], keepdim: bool = True
    ) -> torch.Tensor:
        """Convert a numpy image to a PyTorch 4d tensor image.

        Args:
            image: image of the form :math:`(H, W, C)`, :math:`(H, W)` or
                :math:`(B, H, W, C)`.
            keepdim: If ``False`` unsqueeze the input image to match the shape
                :math:`(B, H, W, C)`.

        Returns:
            tensor of the form :math:`(B, C, H, W)` if keepdim is ``False``,
                :math:`(C, H, W)` otherwise.

        """

        if len(image.shape) > 4 or len(image.shape) < 2:
            raise ValueError(
                "Input size must be a two, three or four dimensional array"
            )

        input_shape = image.shape

        if isinstance(image, np.ndarray):
            tensor: torch.Tensor = torch.from_numpy(image).float()
        elif isinstance(image, torch.Tensor):
            tensor: torch.Tensor = image.float()
        else:
            raise TypeError("image type only support numpy or Tensor!")

        if not tensor.is_cuda:
            tensor = tensor.cuda(
                device=torch.cuda.current_device(), non_blocking=True
            )

        # default layout "hwc"
        if len(input_shape) == 2:
            # (H, W) -> (1, H, W)
            tensor = tensor.unsqueeze(0)
        elif len(input_shape) == 3:
            # (H, W, C) -> (C, H, W)
            tensor = tensor.permute(2, 0, 1)
        elif len(input_shape) == 4:
            # (B, H, W, C) -> (B, C, H, W)
            tensor = tensor.permute(0, 3, 1, 2)
            keepdim = True  # no need to unsqueeze
        else:
            raise ValueError(f"Cannot process image with shape {input_shape}")

        return tensor.unsqueeze(0) if not keepdim else tensor

    @torch.no_grad()
    def forward_single(self, x: torch.Tensor, uv_map: torch.Tensor):
        x = self.image_to_tensor(x, keepdim=False)

        x = F.grid_sample(x, uv_map, mode="bilinear")
        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor, uv_map: Union[List, torch.Tensor]):
        if isinstance(uv_map, torch.Tensor):
            x = self.forward_single(x, uv_map)
        elif isinstance(uv_map, Sequence):
            x = _as_list(x)
            uv_map = _as_list(uv_map)
            x = multi_apply(
                self.forward_single,
                x,
                uv_map,
            )
            x = torch.cat(x, dim=0)
        else:
            raise NotImplementedError(
                f"Not supported un_map type is {type(uv_map)}."
            )

        return x.type(torch.uint8)


@OBJECT_REGISTRY.register
class VCWarpImageRec(object):
    """Virtual Camera warp image transform.

    Args:
        cache_static_map: cache_static_map flag.
        uv_map_key: uv_map key in label.
        use_gpu: whether to use gpu warp.
        verbose: whether to print infomation.
    """

    def __init__(
        self,
        cache_static_map: bool = False,
        uv_map_key: str = None,
        use_gpu: bool = True,
        verbose: bool = False,
    ):
        self.cache_static_map = cache_static_map
        self.uv_mat_cached = {}
        self.uv_map_key = uv_map_key
        self.use_gpu = use_gpu
        self.warp_model = WarpImagePlugin().cuda()
        self.normlization = False
        if self.use_gpu:
            self.normlization = True
        self.verbose = verbose

    @staticmethod
    def get_cam_uuid(camera):
        def _get_cam_uuid_single(cam):
            distcoeffs = str(cam.distcoeffs)
            poseMat_vcs2cam = str(cam.poseMat_vcs2cam)
            camera_matrix = str(cam.camera_matrix)
            camera_param_str = distcoeffs + poseMat_vcs2cam + camera_matrix
            cam_uuid = uuid.uuid3(uuid.NAMESPACE_DNS, camera_param_str)
            return cam_uuid

        if isinstance(camera, CameraBase):
            return _get_cam_uuid_single(camera)
        elif isinstance(camera, Sequence):
            cam_uuid = list(map(_get_cam_uuid_single, camera))
            return cam_uuid
        else:
            raise NotImplementedError(
                "only support single camera or List of camera!"
            )

    @staticmethod
    def normalize_pixel_coordinates(
        pixel_coordinates: torch.Tensor,
        height: int,
        width: int,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        r"""Normalize pixel coordinates between -1 and 1.

        Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1).

        Args:
            pixel_coordinates: the grid with pixel coordinates.
                Shape can be :math:`(*, 2)`.
            width: the maximum width in the x-axis.
            height: the maximum height in the y-axis.
            eps: safe division by zero.

        Return:
            the normalized pixel coordinates with shape :math:`(*, 2)`.

        Examples:
            >>> coords = torch.tensor([[50., 100.]])
            >>> normalize_pixel_coordinates(coords, 100, 50)  # noqa
            tensor([[1.0408, 1.0202]])
        """
        if pixel_coordinates.shape[-1] != 2:
            raise ValueError(
                "Input pixel_coordinates must be of shape (*, 2). "
                "Got {}".format(pixel_coordinates.shape)
            )

        # compute normalization factor
        hw: torch.Tensor = torch.stack(
            [
                torch.tensor(
                    width,
                    device=pixel_coordinates.device,
                    dtype=pixel_coordinates.dtype,
                ),
                torch.tensor(
                    height,
                    device=pixel_coordinates.device,
                    dtype=pixel_coordinates.dtype,
                ),
            ]
        )

        factor: torch.Tensor = torch.tensor(
            2.0, device=pixel_coordinates.device, dtype=pixel_coordinates.dtype
        ) / (hw - 1).clamp(eps)

        return factor * pixel_coordinates - 1

    def get_uv_map(self, cam_uuids, src_cams, dst_cams):
        def _get_uv_map_single(cam_uuid, src_cam, dst_cam):
            if not self.cache_static_map:
                return None
            if cam_uuid not in self.uv_mat_cached:
                assert isinstance(
                    src_cam, CameraBase
                ), "Camera instance check fail!"
                assert isinstance(
                    dst_cam, CameraBase
                ), "Camera instance check fail!"
                _uv_map = src_cam.generate_mapping(dst_cam)
                if self.use_gpu:
                    _uv_map = torch.from_numpy(
                        np.concatenate(_uv_map, axis=-1)
                    ).cuda(
                        device=torch.cuda.current_device(),
                        non_blocking=True,
                    )[
                        None, ...
                    ]
                    _uv_map = self.normalize_pixel_coordinates(
                        _uv_map, src_cam.height, src_cam.width
                    )
                self.uv_mat_cached[cam_uuid] = _uv_map
            else:
                _uv_map = self.uv_mat_cached[cam_uuid]
            return _uv_map

        if isinstance(src_cams, (Sequence, List)):
            uv_map = list(
                map(_get_uv_map_single, cam_uuids, src_cams, dst_cams)
            )
            if self.use_gpu:
                uv_map = torch.cat(
                    uv_map,
                    dim=0,
                )
        elif isinstance(src_cams, CameraBase):
            uv_map = _get_uv_map_single(cam_uuids, src_cams, dst_cams)
        else:
            raise NotImplementedError(
                "only support single camera or List of camera!"
            )
        return uv_map

    def batch_project_image2dstCam(
        self,
        source_cam,
        virtual_cam,
        uv_map,
        image,
    ):
        assert isinstance(source_cam, CameraBase), "Need CameraBase Instance!"
        assert isinstance(virtual_cam, CameraBase), "Need CameraBase Instance!"

        warpped_img = source_cam.project_image2dstCam(
            virtual_cam,
            image,
            uv_map if self.cache_static_map else None,
        )
        return warpped_img

    def warp_image(self, label):

        src_cam_key = self.get_cam_uuid(self.source_cam)

        if "virtual_cam_flip2d" in label:
            self.virtual_cam_2d: CameraBase = label.pop("virtual_cam_flip2d")
            warp_virtual_cam = self.virtual_cam_2d
        else:
            warp_virtual_cam = self.virtual_cam

        dst_cam_key = self.get_cam_uuid(warp_virtual_cam)
        uv_map_key = [
            str(x[0]) + "_" + str(x[1]) for x in zip(src_cam_key, dst_cam_key)
        ]

        if len(self.uv_mat_cached) > 32:
            self.uv_mat_cached = {}
            if self.verbose:
                logger.info(
                    "\n" + "#" * 50 + "\nreset uv_mat_cached\n" + "#" * 50
                )

        uv_map = self.get_uv_map(uv_map_key, self.source_cam, warp_virtual_cam)

        if self.use_gpu and self.cache_static_map:

            with torch.no_grad():
                # to batch Tensor with same shape.
                label["img"] = self.warp_model(label["img"], uv_map)

        else:
            images = multi_apply(
                self.batch_project_image2dstCam,
                self.source_cam,
                self.virtual_cam,
                uv_map,
                label["img"],
            )
            label["img"] = (
                torch.cat(
                    torch.from_numpy(np.array(images)).transpose(2, 0, 1),
                    dim=1,
                )
                .type(torch.uint8)
                .cuda(device=torch.cuda.current_device(), non_blocking=True)
            )

        if "uv_map" in label:
            label.pop("uv_map")

        return label

    def __call__(self, label):
        if "source_cam" not in label:
            return label
        self.source_cam: CameraBase = label["source_cam"]
        self.virtual_cam: CameraBase = label["virtual_cam"]
        label = self.warp_image(label)
        return label


@OBJECT_REGISTRY.register
class CameraParamResize(object):
    """Generate CameraParamResize.

    Args:
        task_type: task type for choice init camera type.
            Default is "real3d".
        scale_x: scale factor along x axis
        scale_y: scale factor along y axis
        img_scale: A list of target image shape in hw format.
            Only supported in MultiBatchProcessor.
        select_ratio: A list of select ratio corresponding to image scale.
            Only supported in MultiBatchProcessor.

    """

    def __init__(
        self,
        task_type: str = "real3d",
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        img_scale: Optional[List] = None,
        select_ratio: Optional[List] = None,
    ):
        self.task_type = task_type
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.img_scale = img_scale
        self.select_ratio = select_ratio

    def __call__(self, label):
        virtual_cam: CameraBase = label["virtual_cam"]
        if self.img_scale is not None:
            assert isinstance(self.img_scale, list)
            if self.select_ratio is None:
                self.select_ratio = [1 / len(self.img_scale)] * len(
                    self.img_scale
                )
            else:
                assert len(self.img_scale) == len(
                    self.select_ratio
                ), "Only support matched scale and ratio"
            assert isinstance(self.select_ratio, list)
            idx = np.random.choice(
                range(len(self.img_scale)), p=self.select_ratio
            )
            image_size = self.img_scale[idx]
            self.scale_x = image_size[1] / virtual_cam.width
            self.scale_y = image_size[0] / virtual_cam.height

        virtual_cam.camera_matrix = virtual_cam.camera_matrix * np.array(
            [
                [self.scale_x, 1, self.scale_x],
                [0, self.scale_y, self.scale_y],
                [0, 0, 1],
            ]
        )

        virtual_cam.width = virtual_cam.width * self.scale_x
        virtual_cam.height = virtual_cam.height * self.scale_y
        label["virtual_cam"] = virtual_cam
        return label


@OBJECT_REGISTRY.register
class CameraPresetParamCrop(object):
    """Generate Preset Crop Parameter Camera.

    Args:
        task_type: task type for choice init camera type.
            Default is "real3d".
        crop_top: crop size from top boundary
        crop_bottom: crop size from bottom boundary
        crop_left: crop size from left boundary
        crop_right: crop size from right boundary
    """

    def __init__(
        self,
        task_type: str = "real3d",
        crop_top: int = 0,
        crop_bottom: int = 0,
        crop_left: int = 0,
        crop_right: int = 0,
    ):
        self.task_type = task_type
        self.crop_top = int(crop_top)
        self.crop_bottom = int(crop_bottom)
        self.crop_left = int(crop_left)
        self.crop_right = int(crop_right)

    def __call__(self, label):
        self.virtual_cam: CameraBase = label["virtual_cam"]
        self.virtual_cam.camera_matrix[0, 2] -= self.crop_left
        self.virtual_cam.camera_matrix[1, 2] -= self.crop_top

        crop_width = self.crop_left + self.crop_right
        crop_height = self.crop_top + self.crop_bottom
        self.virtual_cam.width -= crop_width
        self.virtual_cam.height -= crop_height
        return label


@OBJECT_REGISTRY.register
class CameraParamHorizontalFlip2D(object):
    """Generate CameraParamFlip.

    Args:
        flip_ratio: flip prob.
        WIP: virtual camera should add mirror projection.
    """

    def __init__(self, flip_ratio: float = 0.0):
        assert 0 <= flip_ratio <= 1, "flip_ratios must in (0, 1.0)"
        self.flip_ratio = flip_ratio

    def __call__(self, label):
        assert "virtual_cam" in label, "only support virtual camera"
        virtual_cam: CameraBase = label["virtual_cam"]

        non_flip_ratio = 1 - self.flip_ratio
        flip_ratio_list = [non_flip_ratio, self.flip_ratio]
        flip_flag = np.random.choice([False, True], p=flip_ratio_list)
        virtual_cam_flip2d = deepcopy(virtual_cam)

        if flip_flag:
            virtual_cam_flip2d.fx = -virtual_cam_flip2d.fx
            virtual_cam_flip2d.cx = (
                virtual_cam_flip2d.width - virtual_cam_flip2d.cx
            )
        label["virtual_cam_flip2d"] = virtual_cam_flip2d
        label["flip_flag"] = flip_flag
        return label


@OBJECT_REGISTRY.register
class CameraParamVerticalTranslation(object):
    """Generate Vertical Translation Camera.

    Args:
        norm_camera_z: normalization camera_z to translate.
        designated_alignmnet_depth: appoint alignment depth to translate,
            only needed when random_trans is False. The virtual camera will
            align the pixel to the same height on when the pixel is projected
            at the designated_alignmnet_depth.
        random_trans: Whether random translation is required.
        random_type: the random translation type, only support
            ["normal", "uniform"], only needed when random_trans is True.
        random_upper_boundary: only needed when random_trans is True.
            When random_type = normal, the random_upper_boundary express the
            upper boundary of 0.95 confidence of normal distribution. When
            random_type = uniform, the random_upper_boundary express the upper
            interval of uniform distribution.
        random_lower_boundary: only needed when random_trans is True. When
            random_type = normal, the random_lower_boundary express the lower
            boundary of 0.95 confidence of normal distribution. When
            random_type = uniform, the random_lower_boundary express the lower
            interval of uniform distribution.
    """

    def __init__(
        self,
        norm_camera_z: float,
        designated_alignmnet_depth: Optional[float] = None,
        random_trans: bool = False,
        random_type: Optional[str] = None,
        random_upper_boundary: float = 2,
        random_lower_boundary: float = 20,
    ):
        self.norm_camera_z = norm_camera_z
        self.random_trans = random_trans

        if self.random_trans:
            assert random_type in ["normal", "uniform"]
            self.random_type = random_type
            self.random_upper_boundary = random_upper_boundary
            self.random_lower_boundary = random_lower_boundary
        else:
            assert designated_alignmnet_depth is not None
            self.designated_alignmnet_depth = designated_alignmnet_depth

    def __call__(self, label):
        virtual_cam: CameraBase = label["virtual_cam"]
        virtual_cam_translation = deepcopy(virtual_cam)
        origin_camera_z = float(virtual_cam_translation.poseMat_vcs2cam[1, 3])
        camera_z_diff = origin_camera_z - self.norm_camera_z
        if self.random_trans:
            if self.random_type == "normal":
                diff_up_boundary = int(
                    virtual_cam_translation.fy
                    / self.random_upper_boundary
                    * camera_z_diff
                )
                diff_low_boundary = int(
                    virtual_cam_translation.fy
                    / self.random_lower_boundary
                    * camera_z_diff
                )
                mean_value = (diff_up_boundary + diff_low_boundary) / 2
                sigma = abs(diff_up_boundary - mean_value) / 1.96
                random_diff = np.random.normal(
                    loc=mean_value, scale=sigma, size=None
                )
            elif self.random_type == "uniform":
                random_alignment_depth = np.random.uniform(
                    low=self.random_upper_boundary,
                    high=self.random_lower_boundary,
                    size=None,
                )
                random_diff = int(
                    virtual_cam_translation.fy
                    / random_alignment_depth
                    * camera_z_diff
                )
            virtual_cam_translation.cy = virtual_cam_translation.cy - int(
                random_diff
            )
        else:
            diff = int(
                virtual_cam_translation.fy
                / self.designated_alignmnet_depth
                * camera_z_diff
            )
            virtual_cam_translation.cy = virtual_cam_translation.cy - diff
        label["virtual_cam"] = virtual_cam_translation
        return label


@OBJECT_REGISTRY.register
class FCOS3DCameraPlugin(object):
    """Generate FCOS3D gound truth labels for real3d.

    Args:
        is_train: whether is train mode. Default is True.
        num_classes: num of classes. Default: 3.
        category_id_dict: map of category. Default is None.
        bbox_ct: whether prj bbox_2d to dst cam by center line.
            Default: True.
        rescale: whether rescale to original image size. Default is False.
        max_depth: max gt depth which will be kept.
        mask_depth: max gt depth will be masked.
        depth_type: depth coord type. default: Cartesian.
        camera_names: camera_names in image name.
            Must be set when cache_static_map.
        cache_static_map: whether to cache static uv_map. Default is False.
        is_warp_image: whether to warp image to dst_cam. Default is False.
        keep_org_img: wheter to keep original image. Default is False.
        verbose: whether to print messages. Default is False.
    """

    uv_mat_cached = {}

    def __init__(
        self,
        is_train: bool = True,
        num_classes: int = 3,
        category_id_dict: Optional[Dict] = None,
        bbox_ct: bool = True,
        rescale: bool = False,
        max_depth: float = 30.0,
        mask_depth: float = 200.0,
        depth_type: str = "Cartesian",
        camera_names: Optional[str] = None,
        cache_static_map: bool = False,
        is_warp_image: bool = False,
        keep_org_img: bool = False,
        uv_mat_cache_size: int = 128,
        verbose: bool = False,
    ):
        self.bbox_ct = bbox_ct
        self.rescale = rescale
        self.cache_static_map = cache_static_map
        if self.cache_static_map:
            assert (
                camera_names
            ), "if cache uv_map, camera_names must be setted."
        self.camera_names = camera_names
        self.is_train = is_train
        if category_id_dict is None:
            self.category_id_dict = self._get_category_id_dict(num_classes)
        else:
            self.category_id_dict = category_id_dict
        self.max_depth = max_depth
        self.mask_depth = mask_depth
        self.is_warp_image = is_warp_image
        self.keep_org_img = keep_org_img
        self.depth_type = depth_type
        self.uv_mat_cache_size = uv_mat_cache_size
        self.verbose = verbose
        assert self.depth_type in [
            "Cartesian",
            "Cylindrical",
            "Spherical",
        ], "depth encode type must in [Cartesian, Cylindrical, Spherical]"

    def _reset_camera_param(self, camera):
        # 重置虚拟相机
        raise NotImplementedError

    def _get_category_id_dict(self, num_classes):
        assert (
            num_classes == 3 or num_classes == 7
        ), f"currently the number of classes must be [3, 7], \
            but got {num_classes}"

        if num_classes == 3:
            category_id_dict = {
                1: CLASS_PED,  # Pedestrian -> Pedestrian
                2: CLASS_CAR,  # Car        -> Car
                3: CLASS_CYC,  # Cyclist    -> Cyclist
                4: CLASS_CAR,  # Bus        -> Car
                5: CLASS_CAR,  # Truck      -> Car
                6: CLASS_CAR,  # SpecialCar -> Car
                7: CLASS_CAR,  # Tricycle   -> Car
                8: -1,  # Other    -> ignore
            }  # 'Dontcare' -> Ignore
        elif num_classes == 7:
            category_id_dict = {
                1: 0,  # Pedestrian -> Pedestrian
                2: 2,  # Car        -> Car
                3: 1,  # Cyclist    -> Cyclist
                4: 3,  # Bus        -> Bus
                5: 4,  # Truck      -> Truck
                6: 5,  # SpecialCar -> SpecialCar
                7: 6,  # Tricycle     -> Tricycle
                8: -1,  # Other    -> ignore
            }  # 'Dontcare' -> Ignore
        return category_id_dict

    def _parse_annotations(self, annotations):
        if len(annotations) == 0:
            return annotations
        for obj in annotations:
            if "in_camera" in obj:
                obj.update(obj.pop("in_camera"))
        return annotations

    def xywh_to_center(self, bboxes):
        if isinstance(bboxes, (list, tuple)):
            bboxes = np.array(bboxes)
        ctr = bboxes[..., :2] + bboxes[..., 2:] / 2
        return ctr

    @staticmethod
    def fill_mask_by_bbox(mask, bbox, value=1.0):
        bbox = list(map(int, bbox))
        mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = value
        return mask

    @staticmethod
    def _to_tensor(
        data: Union[torch.Tensor, np.ndarray, Sequence, int, float]
    ):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data.copy())
        elif isinstance(data, Sequence) and not isinstance(data, str):
            return torch.tensor(data)
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        else:
            raise TypeError(
                f"type {type(data)} cannot be converted to tensor."
            )

    def _bbox_projection(
        self, source_cam, virtual_cam, x1y1x2y2: list, bbox_ct=True
    ):
        x1, y1, x2, y2 = x1y1x2y2
        if not bbox_ct:
            points_in = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        else:
            points_in = [
                (x1, (y2 + y1) // 2),
                ((x1 + x2) // 2, y1),
                (x2, (y2 + y1) // 2),
                ((x1 + x2) // 2, y2),
            ]

        virtual_points = source_cam.project_pixel2dstCam(
            virtual_cam, points_in
        )
        x, y, w, h = cv2.boundingRect(np.int32(virtual_points))
        bbox = xywh_to_x1y1x2y2([x, y, w, h])
        return torch.tensor(bbox).float()

    @staticmethod
    def bboxes_3d_to_2d(camera, bboxes_3d, is_voxel=True, xyxy_out=False):
        """Project 3d bbox to 2d.

        Args:
            camera: target camera instance
            bboxes_3d: x,y,z,h,w,l,roty
            is_voxel: True: xyz is voxel center else False: btm ctr

        Returns:
            bbox_2d: x, y, w, h
        """

        nums = len(bboxes_3d)
        loc = deepcopy(bboxes_3d[:, :3])
        dim = deepcopy(bboxes_3d[:, 3:6])
        rot_y = deepcopy(bboxes_3d[:, 6])
        bboxes_2d = np.zeros((nums, 4))
        if is_voxel:
            # only correct when pitch=0
            # voxel ctr to bottom ctr
            loc[:, 1] += dim[:, 0] / 2.0
        for idx in range(nums):
            corner_3d = compute_box_3d(
                dim[idx],
                loc[idx],
                rot_y[idx],
                pitch=0,
            )
            bbox_2d = FCOS3DCameraPlugin.project_corners_to_2d(
                corner_3d, camera
            )
            if not xyxy_out:
                bbox_2d[2:] -= bbox_2d[:2]
            bboxes_2d[idx] = bbox_2d
        return bboxes_2d

    @staticmethod
    def _roty_projection(camera, in_lidar):
        """Project lidar annos templete."""
        in_lidar = deepcopy(in_lidar)
        dim_hwl = in_lidar["dim"]
        loc = in_lidar["location"]
        yaw = -in_lidar["yaw"]
        loc[2] += dim_hwl[2] / 2  # to voxel ctr
        # x,y,z,w,l,h,yaw
        bboxes3d = loc + dim_hwl + [yaw]
        corners_lidar = get_3dboxcorner_in_velo(bboxes3d)
        corners_cam = camera.project_lidar2cam(corners_lidar)
        roty_c = corners_to_local_rot_y(corners_cam)
        return roty_c

    def _erase_ignore_region(self, label):
        if coco_mask is None:
            check_packages_available("pycocotools")
        ignore_mask = coco_mask.decode(label["ignore_mask"]).astype(np.uint8)
        mask = (ignore_mask > 0).astype(np.uint8)
        mask = np.where(mask > 0)
        label["img"][mask] *= 0

        return label

    @staticmethod
    def project_corners_to_2d(corners_cam, camera: CameraBase):

        box3d_prj_bev = camera.project_cam2pixel(corners_cam[:4, :])
        # get bev bbox width
        x_bev, _, w_bev, _ = cv2.boundingRect(np.int32(box3d_prj_bev))

        corners_cam_ctr = np.zeros((6, 3))
        # bootom-left
        corners_cam_ctr[0, 0] = corners_cam[[0, 3], 0].mean(axis=0)
        corners_cam_ctr[0, 1] = corners_cam[[0, 3], 1].mean(axis=0)
        corners_cam_ctr[0, 2] = corners_cam[[0, 3], 2].mean(axis=0)
        # bootom-front
        corners_cam_ctr[1, 0] = corners_cam[[0, 1], 0].mean(axis=0)
        corners_cam_ctr[1, 1] = corners_cam[[0, 1], 1].mean(axis=0)
        corners_cam_ctr[1, 2] = corners_cam[[0, 1], 2].mean(axis=0)
        # bootom-right
        corners_cam_ctr[2, 0] = corners_cam[[1, 2], 0].mean(axis=0)
        corners_cam_ctr[2, 1] = corners_cam[[1, 2], 1].mean(axis=0)
        corners_cam_ctr[2, 2] = corners_cam[[1, 2], 2].mean(axis=0)
        # bootom-back
        corners_cam_ctr[3, 0] = corners_cam[[2, 3], 0].mean(axis=0)
        corners_cam_ctr[3, 1] = corners_cam[[2, 3], 1].mean(axis=0)
        corners_cam_ctr[3, 2] = corners_cam[[2, 3], 2].mean(axis=0)
        # top
        corners_cam_ctr[4, 0] = corners_cam[[6, 7, 5, 4], 0].mean(axis=0)
        corners_cam_ctr[4, 1] = corners_cam[[6, 7, 5, 4], 1].mean(axis=0)
        corners_cam_ctr[4, 2] = corners_cam[[6, 7, 5, 4], 2].mean(axis=0)
        # bootom
        corners_cam_ctr[5, 0] = corners_cam[[0, 1, 2, 3], 0].mean(axis=0)
        corners_cam_ctr[5, 1] = corners_cam[[0, 1, 2, 3], 1].mean(axis=0)
        corners_cam_ctr[5, 2] = corners_cam[[0, 1, 2, 3], 2].mean(axis=0)
        box3d_ctr_prj = camera.project_cam2pixel(corners_cam_ctr)

        # get height
        _, y_used, _, h_used = cv2.boundingRect(np.int32(box3d_ctr_prj))
        # x1,y1,x2,y2
        return np.array(
            [x_bev, y_used, x_bev + w_bev, y_used + h_used]
        ).astype(float)

    def _project_fcos3d_annos_to_virtual_camera(self, label):
        vc_h = self.virtual_cam.image_size[1]
        vc_w = self.virtual_cam.image_size[0]
        h_offset = label.pop("h_offset") if "h_offset" in label else 0

        annos = label["annotations"]
        if self.is_train:
            if coco_mask is None:
                check_packages_available("pycocotools")
            ignore_mask = coco_mask.decode(label["ignore_mask"]).astype(
                np.uint8
            )

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_3d = []
        gt_labels_3d = []
        centers2d = []
        depths = []
        is_valid = False

        # not process 2d bbox with h_offset
        # cause not used.
        for obj in annos:
            cls_id = int(self.category_id_dict[obj["category_id"]])
            # other cls is bg cls
            if cls_id <= -1:
                continue
            # pos cls ignore by flag
            if obj.get("ignore", False):
                # ignore sample has been masked
                continue

            if self.is_train:
                # ignore mask in original image
                if obj["bbox_2d"] is not None:
                    ign_bbox = xywh_to_x1y1x2y2(obj["bbox_2d"])
                else:
                    ign_bbox = xywh_to_x1y1x2y2(obj["bbox"])

            dim_hwl = deepcopy(obj["dim"])
            loc = deepcopy(np.array(obj["location"]))
            # must recover to voxel ctr
            loc[1] -= dim_hwl[0] / 2
            # roty_2_virtual_camera
            if "in_lidar" in obj and self.virtual_cam.is_virtual:
                roty = self._roty_projection(self.virtual_cam, obj["in_lidar"])
                obj["rotation_y"] = roty
            else:
                roty = deepcopy(obj["rotation_y"])

            loc = self.source_cam.project_cam2dstCam(
                self.virtual_cam, loc[None, ...]
            )[0].tolist()

            if self.flip_flag:
                # flip 3d gt to flip image
                roty = -roty + np.pi
                loc[0] = -loc[0]
                obj["rotation_y"] = roty

            # x,y,z,h,w,l,roty
            bboxes3d = deepcopy(np.array(loc + dim_hwl + [roty]))

            if not isinstance(self.source_cam, CylindricalCamera):
                # default voxel ctr
                bbox_prj = self.bboxes_3d_to_2d(
                    self.virtual_cam, bboxes3d[None, ...], is_voxel=True
                )
                bbox_prj = bbox_prj[0].tolist()
                obj["bbox"] = bbox_prj
            else:
                bbox_prj = obj["bbox"]
                bbox_prj = xywh_to_x1y1x2y2(bbox_prj)
                bbox_prj = self._bbox_projection(
                    self.source_cam, self.virtual_cam_2d, bbox_prj
                )
                bbox_prj[2:] -= bbox_prj[:2]
                obj["bbox"] = bbox_prj.numpy().tolist()

            # back to original anno style
            loc[1] += dim_hwl[0] / 2
            # project annos
            obj["location"] = loc
            obj["depth"] = loc[2]
            if obj["bbox_2d"] is not None:
                # direct project 2d bbox to virtual camera
                obj["bbox_2d"][1] = obj["bbox_2d"][1] - h_offset
                bbox_prj_2d = xywh_to_x1y1x2y2(obj["bbox_2d"])
                bbox_prj_2d = self._bbox_projection(
                    self.source_cam, self.virtual_cam_2d, bbox_prj_2d
                )
                bbox_prj_2d[2:] -= bbox_prj_2d[:2]
            else:
                bbox_prj_2d = bbox_prj
            obj["bbox_2d"] = bbox_prj_2d

            if self.depth_type == "Cartesian":
                enc_dep = loc[2]
            elif self.depth_type == "Cylindrical":
                enc_dep = np.sqrt(loc[0] ** 2 + loc[2] ** 2)
            elif self.depth_type == "Spherical":
                raise NotImplementedError("not support Spherical yet!")
            else:
                raise NotImplementedError

            # back to original anno style
            # round 2m range use 3d prjed bbox
            # match dist error target use 3d prjd bbox
            if enc_dep < 2.0 and (
                abs(bboxes3d[6]) < np.deg2rad(30)
                or abs(bboxes3d[6]) > np.deg2rad(150)
            ):
                obj["bbox_2d"] = bbox_prj

            # ignore by ρ range
            if enc_dep > self.max_depth:
                # will ctr eval
                obj["ignore"] = True
                if self.is_train:
                    if enc_dep > self.mask_depth:
                        self.fill_mask_by_bbox(ignore_mask, ign_bbox, 1)
                        continue
                    else:
                        cls_id = -1

            if max(obj["bbox_2d"][2:4]) < 8:
                obj["ignore"] = True
                cls_id = -1

            # pinhole model not support fov over 180°
            if (
                enc_dep <= 0.05
                and isinstance(self.virtual_cam, PinholeCamera)
                and self.is_train
            ):
                self.fill_mask_by_bbox(ignore_mask, ign_bbox, 1)
                continue
            if cls_id >= 0:
                is_valid = True

            # projected 2d bbox
            bbox = xywh_to_x1y1x2y2(obj["bbox_2d"])
            # clip
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, vc_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, vc_h - 1)
            bbox = bbox.tolist()

            # generate fcos3d
            gt_labels.append(cls_id)
            gt_labels_3d.append(cls_id)

            gt_bboxes.append(bbox)
            # bboxes3d:x,y,z,h,w,l,rot_y
            gt_bboxes_3d.append(bboxes3d.tolist())
            # Warning:encoding depth diff with bboxes3d[2]
            depths.append(enc_dep)
        label["gt_bboxes"] = self._to_tensor(gt_bboxes)
        label["gt_classes"] = self._to_tensor(gt_labels)
        label["gt_bboxes_3d"] = self._to_tensor(gt_bboxes_3d)
        label["gt_classes_3d"] = self._to_tensor(gt_labels_3d)
        label["centers2d_prj"] = self._to_tensor(centers2d)
        label["depths"] = self._to_tensor(depths)
        label["valid"] = is_valid
        label["depth_type"] = self.depth_type
        if label["valid"]:
            label["centers2d_prj"] = torch.from_numpy(
                self.virtual_cam.project_cam2pixel(
                    np.array(gt_bboxes_3d)[:, :3]
                )
            )
        if self.is_train:
            # begin with h_offset
            mask = (ignore_mask[h_offset:, ...] > 0).astype(np.uint8)
            mask = np.where(mask > 0)
            label["img"][mask] *= 0

        label["image_height"] = vc_h
        label["image_width"] = vc_w
        return label

    def _fcos3d_from_rec_annos(self, label):
        vc_h = self.virtual_cam.image_size[1]
        vc_w = self.virtual_cam.image_size[0]
        h_offset = label.pop("h_offset") if "h_offset" in label else 0
        annos = label["annotations"]
        if self.is_train:
            if coco_mask is None:
                check_packages_available("pycocotools")
            ignore_mask = coco_mask.decode(label["ignore_mask"]).astype(
                np.uint8
            )

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_3d = []
        gt_labels_3d = []
        centers2d = []
        depths = []
        is_valid = False

        # not process 2d bbox with h_offset
        # cause not used.
        for obj in annos:
            cls_id = int(self.category_id_dict[obj["category_id"]])
            # other cls is bg cls
            if cls_id <= -1:
                continue
            # pos cls ignore by flag
            if obj.get("ignore", False):
                # ignore sample has be masked
                continue

            if self.is_train:
                # ignore mask in original image
                if obj["bbox_2d"] is not None:
                    ign_bbox = xywh_to_x1y1x2y2(obj["bbox_2d"])
                else:
                    ign_bbox = xywh_to_x1y1x2y2(obj["bbox"])

            dim_hwl = deepcopy(obj["dim"])
            loc = deepcopy(obj["location"])
            # must recover to voxel ctr
            loc[1] -= dim_hwl[0] / 2
            roty = deepcopy(obj["rotation_y"])

            # x,y,z,h,w,l,roty
            bboxes3d = deepcopy(np.array(loc + dim_hwl + [roty]))

            if self.depth_type == "Cartesian":
                enc_dep = loc[2]
            elif self.depth_type == "Cylindrical":
                enc_dep = np.sqrt(loc[0] ** 2 + loc[2] ** 2)
            elif self.depth_type == "Spherical":
                enc_dep = np.sqrt(loc[0] ** 2 + loc[1] ** 2 + loc[2] ** 2)
                raise NotImplementedError(f"Not support {self.depth_type}")
            else:
                raise NotImplementedError(f"Not support {self.depth_type}")
            # back to original anno style
            # round 2m range use 3d prjed bbox
            if enc_dep < 2.0 or obj["bbox_2d"] is None:
                obj["bbox_2d"] = deepcopy(obj["bbox"])

            # ignore by ρ range
            if enc_dep >= self.max_depth:
                # will ctr eval
                obj["ignore"] = True
                if self.is_train:
                    if enc_dep >= self.mask_depth:
                        self.fill_mask_by_bbox(ignore_mask, ign_bbox, 1)
                        continue
                    else:
                        cls_id = -1
            if cls_id >= 0:
                is_valid = True
            # generate fcos3d
            gt_labels.append(cls_id)
            gt_labels_3d.append(cls_id)
            # projected 2d bbox
            bbox = xywh_to_x1y1x2y2(obj["bbox_2d"]).astype(float)
            # clip
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, vc_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, vc_h - 1)
            bbox = bbox.tolist()
            gt_bboxes.append(bbox)
            # bboxes3d:x,y,z,h,w,l,rot_y
            gt_bboxes_3d.append(bboxes3d.tolist())
            # Warning:encoding depth diff with bboxes3d[2]
            depths.append(enc_dep)
        label["gt_bboxes"] = self._to_tensor(gt_bboxes)
        label["gt_classes"] = self._to_tensor(gt_labels)
        label["gt_bboxes_3d"] = self._to_tensor(gt_bboxes_3d)
        label["gt_classes_3d"] = self._to_tensor(gt_labels_3d)
        label["centers2d_prj"] = self._to_tensor(centers2d)
        label["depths"] = self._to_tensor(depths)
        label["valid"] = is_valid
        label["depth_type"] = self.depth_type
        if label["valid"]:
            # fix centers2d_prj not equal to 3d location projection
            label["centers2d_prj"] = torch.from_numpy(
                self.virtual_cam.project_cam2pixel(
                    np.array(gt_bboxes_3d)[:, :3]
                )
            )
        if self.is_train:
            # begin with h_offset
            mask = (ignore_mask[h_offset:, ...] > 0).astype(np.uint8)
            mask = np.where(mask > 0)
            label["img"][mask] *= 0

        label["image_height"] = vc_h
        label["image_width"] = vc_w
        return label

    @staticmethod
    def get_camera_location(image_name, camera_names):
        if camera_names is not None:
            for name in camera_names:
                if name in image_name:
                    return name
        else:
            raise NotImplementedError("please set camera_names!")

    @staticmethod
    def get_cam_uuid(camera):
        distcoeffs = str(camera.distcoeffs)
        poseMat_vcs2cam = str(camera.poseMat_vcs2cam)
        camera_matrix = str(camera.camera_matrix)
        camera_param_str = distcoeffs + poseMat_vcs2cam + camera_matrix
        cam_uuid = uuid.uuid3(uuid.NAMESPACE_DNS, camera_param_str)
        return cam_uuid

    def _assign_src_image_to_min_height(self, label, h_min=1440):
        # Not Process 2D bbox
        # only consider original image process
        # default crop bottom region, all y axis pixel - h_offest
        # for assign all image region.
        if label["image_height"] > h_min:
            h_offset = (
                self.source_cam.image_size[1] - h_min
            )  # balance orignal img region

            # fix src camera height
            self.source_cam.image_size[1] = h_min
            # move cy assign when h_min
            self.source_cam.camera_matrix[1, 2] -= h_offset
            label["image_height"] = h_min
            # crop (0, h_offset, 1920, 1536)
            label["img"] = label["img"][h_offset:, ...]
            # processing by h_offset
            label["h_offset"] = h_offset
        else:
            label["h_offset"] = 0

    def warp_image(self, label):

        source_cam_uuid = self.get_cam_uuid(self.source_cam)
        virtual_cam_uuid = self.get_cam_uuid(self.virtual_cam_2d)
        uv_map_key = f"{str(source_cam_uuid)}_{str(virtual_cam_uuid)}"

        # get camera location
        image_name = label["image_name"]
        camera_name = self.get_camera_location(image_name, self.camera_names)

        if uv_map_key not in self.uv_mat_cached:
            self.uv_mat_cached[uv_map_key] = {}
        if not self.cache_static_map:
            self.uv_mat_cached[uv_map_key][camera_name] = None
        elif camera_name not in self.uv_mat_cached[uv_map_key]:
            uv_map = self.source_cam.generate_mapping(self.virtual_cam_2d)
            self.uv_mat_cached[uv_map_key][camera_name] = uv_map
        else:
            uv_map = self.uv_mat_cached[uv_map_key][camera_name]

        label["img"] = self.source_cam.project_image2dstCam(
            self.virtual_cam_2d,
            label["img"],
            uv_map,
        )

        if len(self.uv_mat_cached) > self.uv_mat_cache_size:
            self.uv_mat_cached = {}
            if self.verbose:
                logger.info(
                    "\n" + "#" * 50 + "\nreset uv_mat_cached\n" + "#" * 50
                )
        return label

    def _calib_param_to_tensor(self, label):
        # ****************** sensor calib to tensor ******************
        label["calibration"] = self._to_tensor(label["calibration"])
        label["dist_coeffs"] = self._to_tensor(label["dist_coeffs"])
        label["Tr_vel2cam"] = self._to_tensor(label["Tr_vel2cam"])
        label["Tr_vcs2cam"] = self._to_tensor(label["Tr_vcs2cam"])
        label["image_height"] = self._to_tensor(int(label["image_height"]))
        label["image_width"] = self._to_tensor(int(label["image_width"]))
        return label

    def __call__(self, label):
        """Generate labels.

        Args:
            label (dict): Type is ndarray
                The dict contains at leaset below keys:
                    annotations, calibration, image_transform, ignore_mask,
                    source_cam, virtual_cam

        Returns (dict): label on virtual camera
        """
        self.source_cam: CameraBase = label["source_cam"]
        self.virtual_cam: CameraBase = label["virtual_cam"]
        self.virtual_cam_2d: CameraBase = self.virtual_cam
        self.flip_flag = False

        if "virtual_cam_flip2d" in label:
            self.virtual_cam_2d: CameraBase = label["virtual_cam_flip2d"]
            self.flip_flag = label.pop("flip_flag")
        self.virtual_cam_2d.init_cam_param_by_matrix(
            image_size=(self.virtual_cam_2d.width, self.virtual_cam_2d.height),
            camera_matrix=self.virtual_cam_2d.camera_matrix,
            poseMat_vcs2cam=self.source_cam.poseMat_vcs2cam,
            is_virtual=self.virtual_cam_2d.is_virtual,
        )

        if self.keep_org_img:
            label["org_image"] = deepcopy(label["img"])
            color_space = label["color_space"]
            if color_space == "rgb":
                label["org_image"] = cv2.cvtColor(
                    label["org_image"], cv2.COLOR_RGB2BGR
                )
            elif color_space == "bgr":
                pass
            else:
                raise NotImplementedError(
                    f"unexcepted color space:{color_space}"
                )

        if "annotations" in label:
            # lift annotations
            _ = self._parse_annotations(label["annotations"])
            # whether transform labels to virtual camera
            # default train mode must do project
            # val model maybe do not need project(few used)
            if not self.rescale:
                label = self._project_fcos3d_annos_to_virtual_camera(label)

        # cpu warp: need after erase ignore region
        if self.is_warp_image:
            label = self.warp_image(label)
        else:
            label["warp_cam"] = self.virtual_cam_2d

        if "virtual_cam_flip2d" in label:
            label.pop("virtual_cam_flip2d")
        # Fisheye mix Pinhole distcoeffs to Cylindrical
        if isinstance(self.virtual_cam, CylindricalCamera):
            label["dist_coeffs"] = np.zeros(8)
        if "mat_vcsgnd2img" in label:
            label.pop("mat_vcsgnd2img")
        return label


@OBJECT_REGISTRY.register
class DiffCamImageProjector:
    """Project image from using Diff Cam interface.

    This transform allows input batch of images with different camera types.
    Powered by differential virtual camera(DiffCam) @xudong.he.

    Args:
        num_cal_iters: number of iterations for DVC.
        merge_data_by_camera: whether to merge data by camera type.
            Default: True.
        src_cam_key: key to gather source camera from data.
            Default: "src_cam".
        dst_cam_key: key to gather dest camera from data.
            Default: "dst_cam".
        from_meta: whether to gather data from meta.
            Default: False.
        image_key: key to gather image from data.
            Default: "img".
    """

    def __init__(
        self,
        num_cal_iters: int = 10,
        merge_data_by_camera: bool = True,
        src_cam_key: str = "src_cam",
        dst_cam_key: str = "dst_cam",
        from_meta: bool = False,
        image_key: bool = "img",
    ):
        self.num_cal_iters = num_cal_iters
        self.merge_data_by_camera = merge_data_by_camera
        self.pad_shape = False
        self.src_cam_key = src_cam_key
        self.dst_cam_key = dst_cam_key
        self.from_meta = from_meta
        self.image_key = image_key

    def _warp(
        self,
        src_cams: List[CameraBase],
        dst_cams: List[CameraBase],
        image: torch.Tensor,
    ):
        """Warp image from src_cam to dst_cam.

        Args:
            src_cams: list of source cameras with same class names.
            dst_cams: list of dest cameras with same class names.
            image: image tensor with shape (N, C, H, W) or (C, H, W).
        """
        if len(image.shape) == 3:
            self.pad_shape = True
            image = image.unsqueeze(0)
        diff_src_cam = (
            VIRTUAL_CAMERA_MAP[src_cams[0].__class__]
            .init_from_cameras_list(src_cams)
            .to(image.device)
        )
        diff_src_cam.num_iters = self.num_cal_iters
        diff_dst_cam = (
            VIRTUAL_CAMERA_MAP[dst_cams[0].__class__]
            .init_from_cameras_list(dst_cams)
            .to(image.device)
        )
        diff_dst_cam.num_iters = self.num_cal_iters
        dst_image = diff_src_cam.project_image2dstCam(
            diff_dst_cam, image.float()
        ).byte()
        return dst_image

    def _get_data(self, data: dict):
        """To get meta data from data.

        Args:
            data: data to get meta from.

        Returns:
            src_cams: list of source cameras.
            dst_cams: list of dest cameras.
            image_ori: original image.
        """
        if not (self.src_cam_key in data and self.dst_cam_key in data):
            return None
        src_cams = self._gather_camera(self.src_cam_key, data)
        dst_cams = self._gather_camera(self.dst_cam_key, data)
        assert len(src_cams) == len(dst_cams)
        image_ori = self._gather_image(self.image_key, data)
        return src_cams, dst_cams, image_ori

    def _gather_camera(
        self, cam_key: str, data: dict
    ) -> Union[CameraBase, List[CameraBase]]:
        """To gather camera from data.

        Args:
            cam_key: key to gather camera from data.
            data: data to gather camera from.
        """
        if self.from_meta:
            data = data["meta"]
        return _as_list(data[cam_key])

    def _gather_image(
        self, image_key: str, data: dict
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """To gather image from data.

        Args:
            image_key: key to gather image from data.
            data: data to gather image from.
        """
        return data[image_key]

    def __call__(self, data: dict):
        """Transform oncall function.

        Args:
            data: Image and gt data to transform. Should contain keys given in
                self.src_cam_key, self.dst_cam_key and self.image_key(if need).

        Returns:
            data: Transformed data.
        """
        camera_data = self._get_data(data)
        if camera_data is None:
            return data
        src_cams, dst_cams, image_ori = camera_data
        if isinstance(image_ori, list):
            dst_images = []
            if self.merge_data_by_camera:
                camera_pair_2_image_group = defaultdict(list)
                for ii, imagei in enumerate(image_ori):
                    src_cam = src_cams[ii]
                    dst_cam = dst_cams[ii]
                    group_key = (
                        src_cam.__class__,
                        (src_cam.height, src_cam.width),
                        dst_cam.__class__,
                        (dst_cam.height, dst_cam.width),
                    )
                    camera_pair_2_image_group[group_key].append(
                        (imagei, src_cam, dst_cam, ii)
                    )
                results = [None for _ in range(len(image_ori))]
                for group_data in camera_pair_2_image_group.values():
                    image_group, src_cam_group, dst_cam_group, idxs = zip(
                        *group_data
                    )
                    src_image = torch.stack(image_group, dim=0)
                    dst_image = self._warp(
                        src_cam_group,
                        dst_cam_group,
                        src_image,
                    )
                    for ii, idx in enumerate(idxs):
                        results[idx] = dst_image[ii : ii + 1]
                assert all([r is not None for r in results])
                data["img"] = torch.cat(results, dim=0)
            else:
                for ii, imagei in enumerate(image_ori):
                    src_cam = src_cams[ii]
                    dst_cam = dst_cams[ii]
                    dst_image = dst_image = self._warp(
                        [src_cam],
                        [dst_cam],
                        imagei,
                    )
                    dst_images.append(dst_image)
                data["img"] = torch.cat(dst_images, dim=0)
        else:
            assert (
                len({type(c) for c in src_cams}) == 1
            ), "source_cam must be same type"
            assert (
                len({type(c) for c in dst_cams}) == 1
            ), "warp_cam must be same type"
            data["img"] = self._warp(src_cams, dst_cams, image_ori)
        if self.pad_shape:
            data["img"] = data["img"].squeeze(0)

        return data
