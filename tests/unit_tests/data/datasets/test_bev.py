import json
import os
import pickle

# Note: Here will be strange inconsistency in calculations on AIDI CI:
# https://stackoverflow.com/questions/30203803/using-numpy-in-different-platforms  # noqa
# Temporary addition of version check and skip
from distutils.version import LooseVersion

import cv2
import numpy as np
import pytest
import torch

from hat.core.bev_elevation_utils import decimal_div
from hat.core.virtual_camera.camera_base import CameraParam
from hat.data.collates.collates import collate_3d
from hat.data.datasets.bev import (
    ANCCamPrenorm,
    ANCGenerateBEVHomOffset,
    ANCRpyParamAug,
    HomoGenerator,
    HomoNoise,
)
from hat.data.transforms.auto_3dv import (
    ANCConvertPackDataTo3DV,
    ANCToTensor3DV,
)
from hat.models.task_modules.bev import (
    SpatialTransfomer,
    SpatialTransfomerFixedOffset,
    SpatialTransfomerWithOffset,
)
from hat.utils.apply_func import img_array2tensor, img_tensor2array
from hat.utils.trace import get_part_dict
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH

try:
    import hatbc
    import pytorch3d
except ImportError:
    hatbc = None
    pytorch3d = None

need_skip_cause_np = not LooseVersion(np.__version__) >= LooseVersion("1.21.1")


class HomoGeneratorV2(HomoGenerator):
    def _get_calib_parameters(
        self,
        camera_view_name,
        lidar_param_file,
        camera_param_file,
        calib_param_file=None,
        attribute_param_file=None,
    ):
        """Dummy function to pass through params"""
        file_name = "/".join(lidar_param_file.split("/")[:-1])
        with open(file_name, "rb") as f:
            calib_dict = pickle.load(f)
        calib = calib_dict[camera_view_name]
        T_vcs2cam = calib["T_vcs2cam"]
        K = calib["K"]
        d_coef = calib["d"].copy()
        return (
            T_vcs2cam,
            K,
            d_coef,
            None,
            None,
            np.eye(4, dtype=np.float32),
            None,
        )


def get_common_data():
    bucket_path = HAT_BUCKET_PATH

    data_root = os.path.join(
        bucket_path, "users/jianglei.huang/multiview_dataset/imgs_json"
    )
    anno_json_file = os.path.join(data_root, "anno_test.json")

    camera_view_names = [
        "camera_front_left",
        "camera_front_right",
        "camera_rear_left",
        "camera_rear_right",
        "camera_rear",
    ]
    view_names = [name.replace("camera_", "") for name in camera_view_names]
    per_view_shape = {
        "camera_front_left": (1280, 1920),
        "camera_front_right": (1280, 1920),
        "camera_rear_left": (1280, 1920),
        "camera_rear_right": (1280, 1920),
        "camera_rear": (1280, 1920),
    }
    vcs_range = (-70.0, -50.0, 30.0, 50.0)
    ipm_output_size = (256, 256)  # (height, witdh)
    spatial_resolution = (
        abs(vcs_range[2] - vcs_range[0]) / ipm_output_size[0],
        abs(vcs_range[3] - vcs_range[1]) / ipm_output_size[1],
    )  # (height, witdh)
    homo_cfg = dict(
        homo_path=None,
        calib_path=None,
        spatial_resolution=spatial_resolution,
        vcs_range=vcs_range,
        camera_view_names=camera_view_names,
        per_view_shape=per_view_shape,
        task_camera_view_names=per_view_shape.keys(),
        use_distorted_offset=True,
        homo_transforms={
            view: {"Resize": (640, 960)} for view in camera_view_names
        },
    )

    with open(anno_json_file, "r") as f:
        line = f.readline()
        anno = json.loads(line)
    calib_dict = {}
    for i, cam_name in enumerate(view_names):
        cam_k = camera_view_names[i]
        calib_params = anno["view_anno"][cam_name]["meta"]["calib"]
        calib_getter = CameraParam.init_cam_param_by_dict(
            calib_params, is_virtual=False
        )
        calib_dict[cam_k] = {
            "K": np.array(calib_getter.camera_matrix, dtype=np.float32),
            "d": np.array(calib_getter.distcoeffs, dtype=np.float32),
            "T_vcs2cam": np.array(
                calib_getter.poseMat_vcs2cam, dtype=np.float32
            ),
        }
    return homo_cfg, calib_dict, camera_view_names


@pytest.mark.skipif(True, reason="need update")
def test_homography_offset():
    homo_cfg, calib_dict, camera_view_rename = get_common_data()

    calib_path = os.getenv("PWD") + "/tmp_calib_params.pkl"
    with open(calib_path, "wb") as f:
        pickle.dump(calib_dict, f)

    homo_cfg["calib_path"] = calib_path
    homo_gen1 = HomoGeneratorV2(**homo_cfg)
    homography1 = homo_gen1.get_homography()
    homo_offset1 = homo_gen1.get_homo_offset()

    homo_cfg["calib_path"] = None
    homo_gen2 = HomoGenerator(**homo_cfg)
    homography2 = homo_gen2.get_homography2(calib_dict)
    homo_offset2 = homo_gen2.get_homo_offset2(calib_dict)

    assert np.all(np.abs(homography1 - homography2) < 1e-6)
    assert np.all(np.abs(homo_offset1 - homo_offset2) < 1e-6)

    pkl_name = os.path.join(
        HAT_BUCKET_PATH,
        "users/jianglei.huang/multiview_dataset/homography_and_offset.pkl",
    )
    with open(pkl_name, "rb") as f:
        exp_data = pickle.load(f)

    assert np.all(np.abs(exp_data["homography"] - homography2) < 1e-6)
    assert np.all(np.abs(exp_data["homo_offset"] - homo_offset2) < 1e-6)

    os.remove(calib_path)


@pytest.mark.parametrize(
    "use_horizon_grid_sample",
    [
        True,
        False,
    ],
)
def test_spatial_transformer(use_horizon_grid_sample):
    ipm_out_size = (16, 16)
    multi_warp_nums = 2
    feature = torch.randn((1, 4, 16, 24))
    homography = [torch.rand((1, 3, 3)) for _ in range(multi_warp_nums)]
    # single warp
    st = SpatialTransfomer(
        ipm_out_size[0],
        ipm_out_size[1],
        use_horizon_grid_sample=use_horizon_grid_sample,
    )
    ipm, _ = st(feature, homography[0])
    assert ipm.shape == (1, 4, ipm_out_size[0], ipm_out_size[1])
    # multiple warp
    multi_st = SpatialTransfomer(
        ipm_out_size[0],
        ipm_out_size[1],
        multi_warp_nums=multi_warp_nums,
        use_horizon_grid_sample=use_horizon_grid_sample,
    )
    ipm_mul, _ = multi_st(feature, homography)
    assert ipm_mul.shape == (
        1,
        4 * multi_warp_nums,
        ipm_out_size[0],
        ipm_out_size[1],
    )
    ipm_mul = ipm_mul.split(4, dim=1)
    assert ipm.equal(ipm_mul[0])


def test_spaitial_transformer_with_offset():
    ipm_out_size = (16, 16)
    multi_warp_nums = 2
    feature = torch.randn((1, 4, 16, 24))
    homo_offset = [
        torch.rand((1, ipm_out_size[0], ipm_out_size[1], 2))
        for _ in range(multi_warp_nums)
    ]
    # single warp
    st = SpatialTransfomerWithOffset(ipm_out_size[0], ipm_out_size[1])
    ipm = st(feature, homo_offset[0])
    assert ipm.shape == (1, 4, ipm_out_size[0], ipm_out_size[1])
    # multiple warp
    multi_st = SpatialTransfomerWithOffset(
        ipm_out_size[0], ipm_out_size[1], multi_warp_nums=multi_warp_nums
    )
    ipm_mul = multi_st(feature, homo_offset)
    assert ipm_mul.shape == (
        1,
        4 * multi_warp_nums,
        ipm_out_size[0],
        ipm_out_size[1],
    )
    ipm_mul = ipm_mul.split(4, dim=1)
    assert ipm.equal(ipm_mul[0])


@pytest.mark.parametrize(
    "homo_type",
    [
        "homo_offset",
        "homography",
    ],
)
def test_spaitial_transformer_fixed_offset(homo_type):
    ipm_out_size = (16, 16)
    multi_warp_nums = 2
    feature = torch.randn((1, 4, 16, 24))
    if homo_type == "homo_offset":
        homo_offset = [
            torch.rand((1, ipm_out_size[0], ipm_out_size[1], 2))
            for _ in range(multi_warp_nums)
        ]
        st = SpatialTransfomerFixedOffset(
            ipm_out_size[0],
            ipm_out_size[1],
            homo_offset=homo_offset[0],
        )
        multi_st = SpatialTransfomerFixedOffset(
            ipm_out_size[0],
            ipm_out_size[1],
            homo_offset=homo_offset,
            multi_warp_nums=multi_warp_nums,
        )
    else:
        homography = [torch.rand((1, 3, 3)) for _ in range(multi_warp_nums)]
        st = SpatialTransfomerFixedOffset(
            ipm_out_size[0],
            ipm_out_size[1],
            homography=homography[0],
        )
        multi_st = SpatialTransfomerFixedOffset(
            ipm_out_size[0],
            ipm_out_size[1],
            homography=homography,
            multi_warp_nums=multi_warp_nums,
        )

    # single warp
    ipm = st(feature)
    assert ipm.shape == (1, 4, ipm_out_size[0], ipm_out_size[1])

    # multiple warp
    ipm_mul = multi_st(feature)
    assert ipm_mul.shape == (
        1,
        4 * multi_warp_nums,
        ipm_out_size[0],
        ipm_out_size[1],
    )
    ipm_mul = ipm_mul.split(4, dim=1)
    assert ipm.equal(ipm_mul[0])


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_homogenerator_from_calib():
    bucket_path = HAT_BUCKET_PATH

    vcs_range = (-30.0, -51.2, 72.4, 51.2)  # (bottom, right, top, left)
    spatial_resolution = (0.2, 0.2)  # (height, witdh)

    data_root = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/ben.hu/OnlineHomography/bev3d/65U3D/FSD_BEV_v1/65U3D_20211007_D/20211007-132116_096",  # noqa
    )
    img_front_left_path = os.path.join(
        data_root, "camera_front_left/1633584136098.jpg"
    )
    img_front_path = os.path.join(data_root, "camera_front/1633584136113.jpg")
    img_front_right = os.path.join(
        data_root, "camera_front_right/1633584136098.jpg"
    )
    img_rear_left_path = os.path.join(
        data_root, "camera_rear_left/1633584136098.jpg"
    )
    img_rear_path = os.path.join(data_root, "camera_rear/1633584136098.jpg")
    img_rear_right_path = os.path.join(
        data_root, "camera_rear_right/1633584136099.jpg"
    )

    sync_imgs = {
        "camera_front_left": img_front_left_path,
        "camera_front": img_front_path,
        "camera_front_right": img_front_right,
        "camera_rear_left": img_rear_left_path,
        "camera_rear": img_rear_path,
        "camera_rear_right": img_rear_right_path,
    }
    per_view_shape = {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 2048),
        "camera_front_right": (1280, 2048),
        "camera_rear_left": (1280, 2048),
        "camera_rear_right": (1280, 2048),
        "camera_rear": (1280, 2048),
    }

    # ipm from calibration params
    calib_path = "unit_test_data/J5FSD/users/ben.hu/OnlineHomography/bev3d/65U3D/65U3D_params_20210927"  # noqa
    calib_path = os.path.join(bucket_path, calib_path)
    H_persp_view_scale = 0.25
    homo_transforms = {
        "camera_front": {
            "Resize": (540, 960),
            "Crop": (0, 0, 512, 960),
        },
        "camera_front_left": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_front_right": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_rear_left": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_rear_right": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_rear": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
    }
    homo_path = None
    homo_gen = HomoGenerator(
        homo_path=homo_path,
        calib_path=calib_path,
        spatial_resolution=spatial_resolution,
        vcs_range=vcs_range,
        H_persp_view_scale=H_persp_view_scale,
        camera_view_names=per_view_shape.keys(),
        task_camera_view_names=per_view_shape.keys(),
        per_view_shape=per_view_shape,
        homo_transforms=homo_transforms,
        vcs_plane_heights=(0, 0.5, 1.0, 1.5),
    )

    bev_ipm_img = homo_gen.visualize_bev_homography(
        sync_imgs,
    )
    homo_offset = homo_gen.get_homo_offset()
    assert bev_ipm_img is not None
    assert homo_offset is not None


# read Homo_mat from stored path
def test_homogenerator_from_homo_mat():
    bucket_path = HAT_BUCKET_PATH

    vcs_range = (-30.0, -51.2, 72.4, 51.2)  # (bottom, right, top, left)
    spatial_resolution = (0.2, 0.2)  # (height, witdh)

    data_root = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/ben.hu/OnlineHomography/bev3d/65U3D/FSD_BEV_v1/65U3D_20211007_D/20211007-132116_096",  # noqa
    )
    img_front_left_path = os.path.join(
        data_root, "camera_front_left/1633584136098.jpg"
    )
    img_front_path = os.path.join(data_root, "camera_front/1633584136113.jpg")
    img_front_right = os.path.join(
        data_root, "camera_front_right/1633584136098.jpg"
    )
    img_rear_left_path = os.path.join(
        data_root, "camera_rear_left/1633584136098.jpg"
    )
    img_rear_path = os.path.join(data_root, "camera_rear/1633584136098.jpg")
    img_rear_right_path = os.path.join(
        data_root, "camera_rear_right/1633584136099.jpg"
    )

    sync_imgs = {
        "camera_front_left": img_front_left_path,
        "camera_front": img_front_path,
        "camera_front_right": img_front_right,
        "camera_rear_left": img_rear_left_path,
        "camera_rear": img_rear_path,
        "camera_rear_right": img_rear_right_path,
    }
    per_view_shape = {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 2048),
        "camera_front_right": (1280, 2048),
        "camera_rear_left": (1280, 2048),
        "camera_rear_right": (1280, 2048),
        "camera_rear": (1280, 2048),
    }

    # ipm from stored homography
    calib_path = "unit_test_data/J5FSD/users/ben.hu/OnlineHomography/bev3d/65U3D/65U3D_params_20210927"  # noqa
    homo_path = "unit_test_data/J5FSD/users/ben.hu/OnlineHomography/bev3d/65U3D/homography_6v_ipm512_range102p4m/65U3D_calibration_20210927"  # noqa
    calib_path = os.path.join(bucket_path, calib_path)
    homo_path = os.path.join(bucket_path, homo_path)
    H_persp_view_scale = 0.25
    homo_transforms = {
        "camera_front": {
            "Resize": (540, 960),
            "Crop": (0, 0, 512, 960),
        },
        "camera_front_left": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_front_right": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_rear_left": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_rear_right": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_rear": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
    }
    homo_gen = HomoGenerator(
        homo_path=homo_path,
        calib_path=calib_path,
        spatial_resolution=spatial_resolution,
        vcs_range=vcs_range,
        camera_view_names=per_view_shape.keys(),
        task_camera_view_names=per_view_shape.keys(),
        H_persp_view_scale=H_persp_view_scale,
        per_view_shape=per_view_shape,
        homo_transforms=homo_transforms,
        vcs_plane_heights=(0,),  # the homo_mat is for the ground plane
    )
    bev_ipm_img = homo_gen.visualize_bev_homography(
        sync_imgs,
    )
    homo_offset = homo_gen.get_homo_offset()
    assert bev_ipm_img is not None
    assert homo_offset is not None


# Provide the calibra parameter directly when building HomoGenerator
@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_homogenerator_from_calib_para():
    bucket_path = HAT_BUCKET_PATH

    vcs_range = (-30.0, -51.2, 72.4, 51.2)  # (bottom, right, top, left)
    spatial_resolution = (0.2, 0.2)  # (height, witdh)

    data_root = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/ben.hu/OnlineHomography/bev3d/65U3D/FSD_BEV_v1/65U3D_20211007_D/20211007-132116_096",  # noqa
    )
    img_front_left_path = os.path.join(
        data_root, "camera_front_left/1633584136098.jpg"
    )
    img_front_path = os.path.join(data_root, "camera_front/1633584136113.jpg")
    img_front_right = os.path.join(
        data_root, "camera_front_right/1633584136098.jpg"
    )
    img_rear_left_path = os.path.join(
        data_root, "camera_rear_left/1633584136098.jpg"
    )
    img_rear_path = os.path.join(data_root, "camera_rear/1633584136098.jpg")
    img_rear_right_path = os.path.join(
        data_root, "camera_rear_right/1633584136099.jpg"
    )

    sync_imgs = {
        "camera_front_left": img_front_left_path,
        "camera_front": img_front_path,
        "camera_front_right": img_front_right,
        "camera_rear_left": img_rear_left_path,
        "camera_rear": img_rear_path,
        "camera_rear_right": img_rear_right_path,
    }
    per_view_shape = {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 2048),
        "camera_front_right": (1280, 2048),
        "camera_rear_left": (1280, 2048),
        "camera_rear_right": (1280, 2048),
        "camera_rear": (1280, 2048),
    }

    # ipm from calibration params list
    calib_path = "unit_test_data/J5FSD/users/ben.hu/OnlineHomography/bev3d/65U3D/65U3D_params_20210927"  # noqa
    calib_path = os.path.join(bucket_path, calib_path)
    calib_json = os.path.join(calib_path, "calibration.json")
    with open(calib_json, "r") as f:
        calib_param = json.load(f)
    view_keys = [
        "camera_front_json",
        "camera_frontleft_json",
        "camera_frontright_json",
        "camera_rearleft_json",
        "camera_rearright_json",
        "camera_rear_json",
    ]
    # calib_para={"camera_front": {"K": K, "d_coef": d_coef, "T_vcs2cam": T_vcs2cam} # noqa
    calib_para = {}
    convert = ANCConvertPackDataTo3DV()
    for view, view_key in zip(list(per_view_shape.keys()), view_keys):
        calib_para[view] = convert.reformat_calibration(calib_param[view_key])
    H_persp_view_scale = 0.25
    homo_transforms = {
        "camera_front": {
            "Resize": (540, 960),
            "Crop": (0, 0, 512, 960),
        },
        "camera_front_left": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_front_right": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_rear_left": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_rear_right": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_rear": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
    }
    homo_path = None
    calib_path = None
    homo_gen = HomoGenerator(
        homo_path=homo_path,
        calib_path=calib_path,
        calib_para=calib_para,
        spatial_resolution=spatial_resolution,
        vcs_range=vcs_range,
        camera_view_names=list(per_view_shape.keys()),
        task_camera_view_names=list(per_view_shape.keys()),
        per_view_shape=per_view_shape,
        H_persp_view_scale=H_persp_view_scale,
        homo_transforms=homo_transforms,
        vcs_plane_heights=(0,),  # the homo_mat is for the ground plane
    )

    bev_ipm_img = homo_gen.visualize_bev_homography(
        sync_imgs,
    )
    homo_offset = homo_gen.get_homo_offset()
    assert bev_ipm_img is not None
    assert homo_offset is not None


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
@pytest.mark.parametrize(
    [
        "use_distorted_offset",
    ],
    [
        pytest.param(False),
        pytest.param(True),
    ],
)
def test_homo_offset(use_distorted_offset):
    bucket_path = HAT_BUCKET_PATH

    vcs_range = (-30.0, -51.2, 72.4, 51.2)  # (bottom, right, top, left)
    spatial_resolution = (0.2, 0.2)  # (height, witdh)
    h = int(decimal_div((vcs_range[2] - vcs_range[0]), spatial_resolution[0]))
    w = int(decimal_div((vcs_range[3] - vcs_range[1]), spatial_resolution[1]))

    data_root = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/ben.hu/OnlineHomography/bev3d/65U3D/FSD_BEV_v1/65U3D_20211007_D/20211007-132116_096",  # noqa
    )
    img_front_left_path = os.path.join(
        data_root, "camera_front_left/1633584136098.jpg"
    )
    img_front_path = os.path.join(data_root, "camera_front/1633584136113.jpg")
    img_front_right = os.path.join(
        data_root, "camera_front_right/1633584136098.jpg"
    )
    img_rear_left_path = os.path.join(
        data_root, "camera_rear_left/1633584136098.jpg"
    )
    img_rear_path = os.path.join(data_root, "camera_rear/1633584136098.jpg")
    img_rear_right_path = os.path.join(
        data_root, "camera_rear_right/1633584136099.jpg"
    )

    sync_imgs = {
        "camera_front_left": img_front_left_path,
        "camera_front": img_front_path,
        "camera_front_right": img_front_right,
        "camera_rear_left": img_rear_left_path,
        "camera_rear": img_rear_path,
        "camera_rear_right": img_rear_right_path,
    }

    per_view_shape = {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 2048),
        "camera_front_right": (1280, 2048),
        "camera_rear_left": (1280, 2048),
        "camera_rear_right": (1280, 2048),
        "camera_rear": (1280, 2048),
    }

    camera_view_names = per_view_shape.keys()
    homo_transforms = {
        "camera_front": {
            "Resize": (2160, 3840),
            "Crop": (0, 0, 2160, 3840),
        },
        "camera_front_left": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
        "camera_front_right": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
        "camera_rear_left": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
        "camera_rear_right": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
        "camera_rear": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
    }
    # ipm from calibration params
    calib_path = "unit_test_data/J5FSD/users/ben.hu/OnlineHomography/bev3d/65U3D/65U3D_params_20210927"  # noqa
    calib_path = os.path.join(bucket_path, calib_path)
    homo_path = None
    vcs_plane_heights = (0, 0.5, 1.0, 1.5)
    homo_gen = HomoGenerator(
        homo_path=homo_path,
        calib_path=calib_path,
        spatial_resolution=spatial_resolution,
        vcs_range=vcs_range,
        camera_view_names=camera_view_names,
        task_camera_view_names=camera_view_names,
        per_view_shape=per_view_shape,
        use_distorted_offset=use_distorted_offset,
        homo_transforms=homo_transforms,
        vcs_plane_heights=vcs_plane_heights,
    )
    homo_offset = homo_gen.get_homo_offset()

    st = SpatialTransfomerWithOffset(h, w)
    multi_st = SpatialTransfomerWithOffset(
        h, w, multi_warp_nums=len(vcs_plane_heights)
    )
    for _, name in enumerate(camera_view_names):
        cur_homo_offset = torch.from_numpy(homo_offset[name]).float()
        cur_homo_offset = cur_homo_offset.split(1, dim=0)

        img = cv2.imread(sync_imgs[name], -1)
        img = cv2.resize(img, homo_transforms[name]["Resize"][::-1])
        img_torch = img_array2tensor(img)
        # offset is a tensor
        ipm_ground = st(img_torch, cur_homo_offset[0])
        ipm_ground = img_tensor2array(ipm_ground)
        # offset is a list
        ipm_multi_plane = multi_st(img_torch, cur_homo_offset)
        ipm_multi_plane = np.split(
            img_tensor2array(ipm_multi_plane), len(vcs_plane_heights), axis=2
        )
        assert np.equal(ipm_ground, ipm_multi_plane[0]).all()


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_fisheye_homo_offset():
    bucket_path = HAT_BUCKET_PATH

    vcs_range = (-8.0, -10.5, 13.0, 10.5)  # (bottom, right, top, left)
    spatial_resolution = (0.041, 0.041)  # (height, witdh)
    h = int(decimal_div((vcs_range[2] - vcs_range[0]), spatial_resolution[0]))
    w = int(decimal_div((vcs_range[3] - vcs_range[1]), spatial_resolution[1]))

    pad_top = int(decimal_div(abs(vcs_range[2]), spatial_resolution[0]))
    pad_bottom = int(decimal_div(abs(vcs_range[0]), spatial_resolution[0]))
    pad_left = int(decimal_div(abs(vcs_range[3]), spatial_resolution[1]))
    pad_right = int(decimal_div(abs(vcs_range[1]), spatial_resolution[1]))

    fisheye_block_warp_padding = [
        (0, 0, 0, pad_bottom),
        (0, 0, pad_top, 0),
        (0, pad_right, 0, 0),
        (pad_left, 0, 0, 0),
    ]

    data_root = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/mengyuan.wang/fisheye_img_calibration/",
    )

    img_front_path = os.path.join(data_root, "fisheye_front/1620874477113.jpg")
    img_rear_path = os.path.join(data_root, "fisheye_rear/1620874477097.jpg")
    img_left_path = os.path.join(data_root, "fisheye_left/1620874477097.jpg")
    img_right_path = os.path.join(data_root, "fisheye_right/1620874477097.jpg")

    sync_imgs = {
        "fisheye_front": img_front_path,
        "fisheye_rear": img_rear_path,
        "fisheye_left": img_left_path,
        "fisheye_right": img_right_path,
    }
    camera_view_names = sync_imgs.keys()

    homo_path = None
    calib_path = os.path.join(
        data_root,
        "calibration/calibration",
    )
    per_view_shape = {
        "fisheye_front": (1080, 1920),
        "fisheye_rear": (1080, 1920),
        "fisheye_left": (1080, 1920),
        "fisheye_right": (1080, 1920),
    }
    use_distorted_offset = True
    H_persp_view_scale = 0.25
    fisheye_homo_transforms = {
        "fisheye_front": {
            "Resize": (540, 960),
            "Pad": (0, 100, 0, 0),
        },
        "fisheye_rear": {
            "Resize": (540, 960),
            "Pad": (0, 100, 0, 0),
        },
        "fisheye_left": {
            "Resize": (540, 960),
            "Pad": (0, 100, 0, 0),
        },
        "fisheye_right": {
            "Resize": (540, 960),
            "Pad": (0, 100, 0, 0),
        },
    }
    vcs_plane_heights = (0, 0.5, 1.0, 1.5)
    homo_gen = HomoGenerator(
        homo_path=homo_path,
        calib_path=calib_path,
        spatial_resolution=spatial_resolution,
        vcs_range=vcs_range,
        camera_view_names=camera_view_names,
        task_camera_view_names=camera_view_names,
        per_view_shape=per_view_shape,
        H_persp_view_scale=H_persp_view_scale,
        use_distorted_offset=use_distorted_offset,
        homo_transforms=fisheye_homo_transforms,
        vcs_plane_heights=vcs_plane_heights,
    )
    homo_offset = homo_gen.get_homo_offset()
    # homo_offset = load_homo_offset_path(homo_offset_path, camera_view_names)

    st = []
    multi_st = []
    for i in range(len(fisheye_block_warp_padding)):
        st_i = SpatialTransfomerWithOffset(
            h, w, block_warp_padding=fisheye_block_warp_padding[i]
        )
        st.append(st_i)
        multi_st_i = SpatialTransfomerWithOffset(
            h,
            w,
            block_warp_padding=fisheye_block_warp_padding[i],
            multi_warp_nums=len(vcs_plane_heights),
        )
        multi_st.append(multi_st_i)
    for i, name in enumerate(camera_view_names):
        cur_homo_offset = torch.from_numpy(homo_offset[name]).float()
        cur_homo_offset = cur_homo_offset.split(1, dim=0)

        img = cv2.imread(sync_imgs[name], -1)
        img = cv2.resize(img, (960, 540))
        tmp_img = np.zeros((640, 960, 3), dtype="uint8")
        tmp_img[100:, :, :] = img
        img = cv2.resize(tmp_img, (240, 160))
        img_torch = img_array2tensor(img)
        # offset is a tensor
        ipm_ground = st[i](img_torch, cur_homo_offset[0])
        ipm_ground = img_tensor2array(ipm_ground)
        # offset is a list
        ipm_multi_plane = multi_st[i](img_torch, cur_homo_offset)
        ipm_multi_plane = np.split(
            img_tensor2array(ipm_multi_plane), len(vcs_plane_heights), axis=2
        )
        assert np.equal(ipm_ground, ipm_multi_plane[0]).all()


def test_homography():
    bucket_path = HAT_BUCKET_PATH

    vcs_range = (-30.0, -51.2, 72.4, 51.2)  # (bottom, right, top, left)
    spatial_resolution = (0.2, 0.2)  # (height, witdh)
    h = int(decimal_div((vcs_range[2] - vcs_range[0]), spatial_resolution[0]))
    w = int(decimal_div((vcs_range[3] - vcs_range[1]), spatial_resolution[1]))

    data_root = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/ben.hu/OnlineHomography/bev3d/65U3D/FSD_BEV_v1/65U3D_20211007_D/20211007-132116_096",  # noqa
    )
    img_front_left_path = os.path.join(
        data_root, "camera_front_left/1633584136098.jpg"
    )
    img_front_path = os.path.join(data_root, "camera_front/1633584136113.jpg")
    img_front_right = os.path.join(
        data_root, "camera_front_right/1633584136098.jpg"
    )
    img_rear_left_path = os.path.join(
        data_root, "camera_rear_left/1633584136098.jpg"
    )
    img_rear_path = os.path.join(data_root, "camera_rear/1633584136098.jpg")
    img_rear_right_path = os.path.join(
        data_root, "camera_rear_right/1633584136099.jpg"
    )

    sync_imgs = {
        "camera_front_left": img_front_left_path,
        "camera_front": img_front_path,
        "camera_front_right": img_front_right,
        "camera_rear_left": img_rear_left_path,
        "camera_rear": img_rear_path,
        "camera_rear_right": img_rear_right_path,
    }

    per_view_shape = {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 2048),
        "camera_front_right": (1280, 2048),
        "camera_rear_left": (1280, 2048),
        "camera_rear_right": (1280, 2048),
        "camera_rear": (1280, 2048),
    }

    camera_view_names = per_view_shape.keys()
    homo_transforms = {
        "camera_front": {
            "Resize": (2160, 3840),
            "Crop": (0, 0, 2160, 3840),
        },
        "camera_front_left": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
        "camera_front_right": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
        "camera_rear_left": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
        "camera_rear_right": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
        "camera_rear": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
    }
    # ipm from calibration params
    calib_path = "unit_test_data/J5FSD/users/ben.hu/OnlineHomography/bev3d/65U3D/65U3D_params_20210927"  # noqa
    calib_path = os.path.join(bucket_path, calib_path)
    homo_path = None
    vcs_plane_heights = (0, 0.5, 1.0, 1.5)
    homo_gen = HomoGenerator(
        homo_path=homo_path,
        calib_path=calib_path,
        spatial_resolution=spatial_resolution,
        vcs_range=vcs_range,
        camera_view_names=camera_view_names,
        task_camera_view_names=camera_view_names,
        per_view_shape=per_view_shape,
        homo_transforms=homo_transforms,
        vcs_plane_heights=vcs_plane_heights,
    )
    homography = homo_gen.get_homography()

    st = SpatialTransfomer(h, w)
    multi_st = SpatialTransfomer(h, w, multi_warp_nums=len(vcs_plane_heights))
    for _, name in enumerate(camera_view_names):
        cur_homography = torch.from_numpy(homography[name]).float()
        cur_homography = cur_homography.split(1, dim=0)

        img = cv2.imread(sync_imgs[name], -1)
        img = cv2.resize(img, homo_transforms[name]["Resize"][::-1])
        img_torch = img_array2tensor(img)
        # homography is a tensor
        ipm_ground = st(img_torch, cur_homography[0])[0]
        ipm_ground = img_tensor2array(ipm_ground)
        # homography is a list
        ipm_multi_plane = multi_st(img_torch, cur_homography)[0]
        ipm_multi_plane = np.split(
            img_tensor2array(ipm_multi_plane), len(vcs_plane_heights), axis=2
        )
        assert np.equal(ipm_ground, ipm_multi_plane[0]).all()


def test_homo_noise():
    import math

    camera_view_names = {
        "camera_front",
        "camera_front_left",
        "camera_front_right",
        "camera_rear_left",
        "camera_rear_right",
        "camera_rear",
    }
    ori_T_vcs2cam = [
        0.00504,
        -0.99995,
        0.00819,
        -0.02260,
        -0.00091,
        -0.00820,
        -0.99997,
        1.53975,
        0.99999,
        0.00503,
        -0.00095,
        -1.98151,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    ori_T_vcs2cam = np.array(ori_T_vcs2cam).reshape(4, 4)

    homo_noise = {
        "noise_value": (0.2, 0.2, 0.2, 0.0, 0.0, 0.04),
        "noise_type": "random_cam",
        "noise_view_names": camera_view_names,
    }
    homo_noise = HomoNoise(**homo_noise)
    homo_noise.get_homo_noise()

    homo_noise.get_noise_matrix(
        ori_T_vcs2cam, homo_noise.view2noise["camera_front"]
    )

    homo_noise = {
        "noise_value": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        "noise_type": "random_vcs",
        "noise_view_names": camera_view_names,
    }
    homo_noise = HomoNoise(**homo_noise)
    homo_noise.get_homo_noise()

    assert (
        homo_noise.view2noise["camera_front"][0]
        == homo_noise.view2noise["camera_rear"][0]
    )
    assert (
        homo_noise.view2noise["camera_front"][1]
        == homo_noise.view2noise["camera_rear"][1]
    )

    homo_noise.get_noise_matrix(
        ori_T_vcs2cam, homo_noise.view2noise["camera_front"]
    )

    homo_noise = {
        "noise_value": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        "noise_type": "specific_cam",
        "noise_view_names": camera_view_names,
    }
    homo_noise = HomoNoise(**homo_noise)
    homo_noise.get_homo_noise()
    assert homo_noise.view2noise["camera_front"][0] == [
        0.0,
        1.0 / 180 * math.pi,
        0.0,
    ]
    assert homo_noise.view2noise["camera_front"][1] == [0.0, 0.0, 0.0]

    homo_noise.get_noise_matrix(
        ori_T_vcs2cam, homo_noise.view2noise["camera_front"]
    )


def test_11v_from_attribute_json():
    """For bev 11V homo_ffset using attribute.json."""
    bucket_path = HAT_BUCKET_PATH
    use_distorted_offset = True

    vcs_range = (-153.6, -76.8, 153.6, 76.8)  # (bottom, right, top, left)
    spatial_resolution = (0.8, 0.8)  # (height, witdh)
    h = int(decimal_div((vcs_range[2] - vcs_range[0]), spatial_resolution[0]))
    w = int(decimal_div((vcs_range[3] - vcs_range[1]), spatial_resolution[1]))

    data_root = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/xiangyu.li/test_attribute/20220611-141510_348",  # noqa
    )
    img_front_path = os.path.join(data_root, "camera_front/1654928408600.jpg")
    img_front_left_path = os.path.join(
        data_root, "camera_front_left/1654928408600.jpg"
    )
    img_front_right = os.path.join(
        data_root, "camera_front_right/1654928408600.jpg"
    )
    img_rear_left_path = os.path.join(
        data_root, "camera_rear_left/1654928408600.jpg"
    )
    img_rear_right_path = os.path.join(
        data_root, "camera_rear_right/1654928408600.jpg"
    )
    img_rear_path = os.path.join(data_root, "camera_rear/1654928408600.jpg")
    fisheye_img_front_path = os.path.join(
        data_root, "fisheye_front/1654928408614.jpg"
    )
    fisheye_img_rear_path = os.path.join(
        data_root, "fisheye_rear/1654928408614.jpg"
    )
    fisheye_img_left_path = os.path.join(
        data_root, "fisheye_left/1654928408614.jpg"
    )
    fisheye_img_right_path = os.path.join(
        data_root, "fisheye_right/1654928408614.jpg"
    )
    narrow_img_path = os.path.join(
        data_root, "camera_front_30fov/1654928408599.jpg"
    )

    sync_imgs = {
        "camera_front": img_front_path,
        "camera_front_left": img_front_left_path,
        "camera_front_right": img_front_right,
        "camera_rear_left": img_rear_left_path,
        "camera_rear": img_rear_path,
        "camera_rear_right": img_rear_right_path,
        "fisheye_front": fisheye_img_front_path,
        "fisheye_rear": fisheye_img_rear_path,
        "fisheye_left": fisheye_img_left_path,
        "fisheye_right": fisheye_img_right_path,
        "camera_front_30fov": narrow_img_path,
    }

    per_view_shape = {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 2048),
        "camera_front_right": (1280, 2048),
        "camera_rear_left": (1280, 2048),
        "camera_rear_right": (1280, 2048),
        "camera_rear": (1280, 2048),
        "fisheye_front": (1536, 1920),
        "fisheye_rear": (1536, 1920),
        "fisheye_left": (1536, 1920),
        "fisheye_right": (1536, 1920),
        "camera_front_30fov": (2160, 3840),
    }

    camera_view_names = per_view_shape.keys()
    homo_transforms = {
        "camera_front": {
            "Resize": (2160, 3840),
            "Crop": (0, 0, 2160, 3840),
        },
        "camera_front_left": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
        "camera_front_right": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
        "camera_rear_left": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
        "camera_rear_right": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
        "camera_rear": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
        "fisheye_front": {
            "Resize": (1536, 1920),
            "Pad": (0, 0, 0, 0),
        },
        "fisheye_rear": {
            "Resize": (1536, 1920),
            "Pad": (0, 0, 0, 0),
        },
        "fisheye_left": {
            "Resize": (1536, 1920),
            "Pad": (0, 0, 0, 0),
        },
        "fisheye_right": {
            "Resize": (1536, 1920),
            "Pad": (0, 0, 0, 0),
        },
        "camera_front_30fov": {
            "Resize": (2160, 3840),
            "Crop": (0, 0, 2160, 3840),
        },
    }
    # ipm from calibration params
    calib_path = "Calibration_params/UTHS6"  # noqa
    calib_path = os.path.join(data_root, calib_path)
    homo_path = None
    vcs_plane_heights = (0, 0.5, 1.0, 1.5)
    homo_gen = HomoGenerator(
        homo_path=homo_path,
        calib_path=calib_path,
        spatial_resolution=spatial_resolution,
        vcs_range=vcs_range,
        camera_view_names=camera_view_names,
        task_camera_view_names=camera_view_names,
        per_view_shape=per_view_shape,
        use_distorted_offset=use_distorted_offset,
        homo_transforms=homo_transforms,
        vcs_plane_heights=vcs_plane_heights,
    )
    homo_offset = homo_gen.get_homo_offset()

    st = SpatialTransfomerWithOffset(h, w)
    multi_st = SpatialTransfomerWithOffset(
        h, w, multi_warp_nums=len(vcs_plane_heights)
    )
    for _, name in enumerate(camera_view_names):
        cur_homo_offset = torch.from_numpy(homo_offset[name]).float()
        cur_homo_offset = cur_homo_offset.split(1, dim=0)

        img = cv2.imread(sync_imgs[name], -1)
        img = cv2.resize(img, homo_transforms[name]["Resize"][::-1])
        img_torch = img_array2tensor(img)
        # offset is a tensor
        ipm_ground = st(img_torch, cur_homo_offset[0])
        ipm_ground = img_tensor2array(ipm_ground)
        # offset is a list
        ipm_multi_plane = multi_st(img_torch, cur_homo_offset)
        ipm_multi_plane = np.split(
            img_tensor2array(ipm_multi_plane), len(vcs_plane_heights), axis=2
        )
        assert np.equal(ipm_ground, ipm_multi_plane[0]).all()


@pytest.mark.skipif(hatbc is None, reason="need update")
def test_generatebevhomoffset():
    """Test for GenerateBEVHomOffset."""
    bucket_path = HAT_BUCKET_PATH
    use_distorted_offset = True

    vcs_range = (-153.6, -76.8, 153.6, 76.8)  # (bottom, right, top, left)
    spatial_resolution = (0.8, 0.8)  # (height, witdh)

    data_root = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/xiangyu.li/test_attribute/20220611-141510_348",  # noqa
    )

    per_view_shape = {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 2048),
        "camera_front_right": (1280, 2048),
        "camera_rear_left": (1280, 2048),
        "camera_rear_right": (1280, 2048),
        "camera_rear": (1280, 2048),
        "fisheye_front": (1536, 1920),
        "fisheye_rear": (1536, 1920),
        "fisheye_left": (1536, 1920),
        "fisheye_right": (1536, 1920),
        "camera_front_30fov": (2160, 3840),
    }

    camera_view_names = per_view_shape.keys()
    homo_transforms = {
        "camera_front": {
            "Resize": (2160, 3840),
            "Crop": (0, 0, 2160, 3840),
        },
        "camera_front_left": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
        "camera_front_right": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
        "camera_rear_left": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
        "camera_rear_right": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
        "camera_rear": {
            "Resize": (1280, 2048),
            "Crop": (0, 0, 1280, 2048),
        },
        "fisheye_front": {
            "Resize": (1536, 1920),
            "Pad": (0, 0, 0, 0),
        },
        "fisheye_rear": {
            "Resize": (1536, 1920),
            "Pad": (0, 0, 0, 0),
        },
        "fisheye_left": {
            "Resize": (1536, 1920),
            "Pad": (0, 0, 0, 0),
        },
        "fisheye_right": {
            "Resize": (1536, 1920),
            "Pad": (0, 0, 0, 0),
        },
        "camera_front_30fov": {
            "Resize": (2160, 3840),
            "Crop": (0, 0, 2160, 3840),
        },
    }
    # ipm from calibration params
    calib_path = "Calibration_params/UTHS6"  # noqa
    calib_path = os.path.join(data_root, calib_path)
    homo_path = None
    vcs_plane_heights = (0, 0.5, 1.0, 1.5)

    homo_gen = HomoGenerator(
        homo_path=homo_path,
        calib_path=calib_path,
        spatial_resolution=spatial_resolution,
        vcs_range=vcs_range,
        camera_view_names=camera_view_names,
        task_camera_view_names=camera_view_names,
        per_view_shape=per_view_shape,
        use_distorted_offset=use_distorted_offset,
        homo_transforms=homo_transforms,
        vcs_plane_heights=vcs_plane_heights,
    )

    # ! homo_offset_on_cpu
    # `return_offset_in_meta_info` in HomoGenerator is default True
    # we can get the homo_offset from meta_info or through
    # `homo_gen.get_dist_homo_offset`
    homo_offset_cpu = homo_gen.get_homo_offset()
    homo_offset_cpu = list(homo_offset_cpu.values())
    homo_offset_cpu = np.concatenate(homo_offset_cpu, axis=0)

    # ! homo_offset_on_gpu
    # for gpu calculate homo_offset, the `return_offset_in_meta_info`
    # should be False to skip calculate homooffset and save it in
    # memory.
    setattr(homo_gen, "return_offset_in_meta_info", False)  # noqa
    meta_info_new = homo_gen.get_meta_info()
    assert "homo_offset" not in meta_info_new

    for mate_name, meta_value in meta_info_new.items():
        if isinstance(meta_value, list):
            meta_info_new[mate_name] = [
                torch.from_numpy(param) for param in meta_value
            ]
        elif isinstance(meta_value, str):
            meta_info_new[mate_name] = meta_value
        elif isinstance(meta_value, np.ndarray):
            meta_info_new[mate_name] = torch.from_numpy(meta_value)

    meta_info_keys = [
        "T_vcs2cam",
        "intrinsics",
        "distort_coeffs",
        "transformats",
        "ipm_img_sizes",
        "img_shape",
        "fake_homo_flag",
        "aug_flag",
    ]
    meta_info_new = get_part_dict(meta_info_new, meta_info_keys)
    collate_meta_info = collate_3d([meta_info_new])
    temporal_info = {
        "num_frames_per_iter": 1,
    }
    collate_temporal_info = collate_3d([temporal_info])
    viewsdomain2_nums = {
        "front": (1,),
        "side": (5,),
        "round": (4,),
        "narrow": (1,),
    }
    generate_offset = ANCGenerateBEVHomOffset(
        vcs_range=vcs_range,
        spatial_resolution=spatial_resolution,
        vcs_plane_heights=vcs_plane_heights,
    )
    homo_offset_torch = generate_offset(
        collate_meta_info, collate_temporal_info, viewsdomain2_nums
    )
    homo_offset_gpu = homo_offset_torch.numpy()

    homo_offset_diff = homo_offset_cpu - homo_offset_gpu

    assert homo_offset_diff.max() == 0.0007186126554188377
    assert homo_offset_diff.min() == -0.0023099231912055984


@pytest.mark.skipif(
    not (hatbc and pytorch3d), reason="need install hatbc and pytorch3d"
)
def test_rpy_param_aug():
    bucket_path = HAT_BUCKET_PATH
    use_distorted_offset = True

    vcs_range = (-153.6, -76.8, 153.6, 76.8)  # (bottom, right, top, left)
    spatial_resolution = (0.8, 0.8)  # (height, witdh)
    vcs_range_small = (-12.8, -12.8, 25.6, 12.8)  # (bottom, right, top, left)
    spatial_resolution_small = (0.2, 0.2)  # (height, witdh)

    data_root = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/xiangyu.li/test_attribute/20220611-141510_348",  # noqa
    )

    img_front_path = os.path.join(data_root, "camera_front/1654928408600.jpg")
    img_front_left_path = os.path.join(
        data_root, "camera_front_left/1654928408600.jpg"
    )
    img_front_right = os.path.join(
        data_root, "camera_front_right/1654928408600.jpg"
    )
    img_rear_left_path = os.path.join(
        data_root, "camera_rear_left/1654928408600.jpg"
    )
    img_rear_right_path = os.path.join(
        data_root, "camera_rear_right/1654928408600.jpg"
    )
    img_rear_path = os.path.join(data_root, "camera_rear/1654928408600.jpg")
    fisheye_img_front_path = os.path.join(
        data_root, "fisheye_front/1654928408614.jpg"
    )
    fisheye_img_rear_path = os.path.join(
        data_root, "fisheye_rear/1654928408614.jpg"
    )
    fisheye_img_left_path = os.path.join(
        data_root, "fisheye_left/1654928408614.jpg"
    )
    fisheye_img_right_path = os.path.join(
        data_root, "fisheye_right/1654928408614.jpg"
    )
    narrow_img_path = os.path.join(
        data_root, "camera_front_30fov/1654928408599.jpg"
    )

    sync_imgs = {
        "camera_front": img_front_path,
        "camera_front_left": img_front_left_path,
        "camera_front_right": img_front_right,
        "camera_rear_left": img_rear_left_path,
        "camera_rear": img_rear_path,
        "camera_rear_right": img_rear_right_path,
        "fisheye_front": fisheye_img_front_path,
        "fisheye_rear": fisheye_img_rear_path,
        "fisheye_left": fisheye_img_left_path,
        "fisheye_right": fisheye_img_right_path,
        "camera_front_30fov": narrow_img_path,
    }

    per_view_shape = {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 2048),
        "camera_front_right": (1280, 2048),
        "camera_rear_left": (1280, 2048),
        "camera_rear_right": (1280, 2048),
        "camera_rear": (1280, 2048),
        "fisheye_front": (1536, 1920),
        "fisheye_rear": (1536, 1920),
        "fisheye_left": (1536, 1920),
        "fisheye_right": (1536, 1920),
        "camera_front_30fov": (2160, 3840),
    }

    camera_view_names = per_view_shape.keys()
    homo_transforms = {
        "camera_front": {
            "Resize": (540, 960),
            "Crop": (0, 0, 512, 960),
        },
        "camera_front_left": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_front_right": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_rear_left": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_rear_right": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_rear": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "fisheye_front": {
            "Resize": (768, 960),
            "Crop": (0, 0, 768, 960),
        },
        "fisheye_rear": {
            "Resize": (768, 960),
            "Crop": (0, 0, 768, 960),
        },
        "fisheye_left": {
            "Resize": (768, 960),
            "Crop": (0, 0, 768, 960),
        },
        "fisheye_right": {
            "Resize": (768, 960),
            "Crop": (0, 0, 768, 960),
        },
        "camera_front_30fov": {
            "Resize": (540, 960),
            "Crop": (0, 0, 512, 960),
        },
    }
    # ipm from calibration params
    calib_path = "Calibration_params/UTHS6"  # noqa
    calib_path = os.path.join(data_root, calib_path)
    homo_path = None
    vcs_plane_heights = (0, 0.5, 1.0, 1.5)

    homo_gen = HomoGenerator(
        homo_path=homo_path,
        calib_path=calib_path,
        spatial_resolution=spatial_resolution,
        vcs_range=vcs_range,
        camera_view_names=camera_view_names,
        task_camera_view_names=camera_view_names,
        per_view_shape=per_view_shape,
        use_distorted_offset=use_distorted_offset,
        homo_transforms=homo_transforms,
        vcs_plane_heights=vcs_plane_heights,
        return_offset_in_meta_info=False,
    )
    homo_gen_small = HomoGenerator(
        homo_path=homo_path,
        calib_path=calib_path,
        spatial_resolution=spatial_resolution_small,
        vcs_range=vcs_range_small,
        camera_view_names=camera_view_names,
        task_camera_view_names=camera_view_names,
        per_view_shape=per_view_shape,
        use_distorted_offset=use_distorted_offset,
        homo_transforms=homo_transforms,
        vcs_plane_heights=vcs_plane_heights,
        return_offset_in_meta_info=False,
    )

    meta_info = homo_gen.get_meta_info()
    meta_info_small = homo_gen_small.get_meta_info()

    for mate_name, meta_value in meta_info.items():
        meta_info[mate_name] = ANCToTensor3DV._meta_to_tensor(meta_value)
    for mate_name, meta_value in meta_info_small.items():
        meta_info_small[mate_name] = ANCToTensor3DV._meta_to_tensor(meta_value)

    collate_meta_info = collate_3d([meta_info])
    collate_meta_info_small = collate_3d([meta_info_small])
    temporal_info = {
        "num_frames_per_iter": 1,
    }
    collate_temporal_info = collate_3d([temporal_info])
    collate_views_domain2nums = {
        "front": (1,),
        "side": (5,),
        "round": (4,),
        "narrow": (1,),
    }

    torch_imgs = []
    for _, name in enumerate(camera_view_names):
        img = cv2.imread(sync_imgs[name], -1)
        img = cv2.resize(img, homo_transforms[name]["Resize"][::-1])
        img = img[
            : homo_transforms[name]["Crop"][2],
            : homo_transforms[name]["Crop"][3],
            :,
        ]
        img_torch = img_array2tensor(img)
        torch_imgs.append(img_torch)

    data = {
        "img": [torch_imgs[0]],
        "side_img": [torch.cat(torch_imgs[1:6])],
        "round_img": [torch.cat(torch_imgs[6:10])],
        "narrow_img": [torch_imgs[10]],
        "meta_info": collate_meta_info,
        "meta_info_small": collate_meta_info_small,
        "view": collate_views_domain2nums,
        "aug_transforms": {
            "rpy_augmentation": {"prob": [1], "aug_degree": [1]}
        },
        "temporal_info": collate_temporal_info,
    }
    rpy_aug_module = ANCRpyParamAug(
        prob=1,
        aug_degree=5,  # deg
    )
    data = rpy_aug_module(data)

    assert "img" in data
    assert "side_img" in data
    assert "round_img" in data
    assert "narrow_img" in data
    assert "T_vcs2cam" in data["meta_info"]
    assert "T_vcs2cam" in data["meta_info_small"]
    assert (
        data["meta_info"]["T_vcs2cam"] == data["meta_info_small"]["T_vcs2cam"]
    )
    assert (
        data["meta_info"]["cam2local_rot"]
        == data["meta_info_small"]["cam2local_rot"]
    )


@pytest.mark.skipif(
    not (hatbc and pytorch3d), reason="need install hatbc and pytorch3d"
)
def test_cam_prenorm():

    bucket_path = HAT_BUCKET_PATH
    use_distorted_offset = True

    vcs_range = (-153.6, -76.8, 153.6, 76.8)  # (bottom, right, top, left)
    spatial_resolution = (0.8, 0.8)  # (height, witdh)
    vcs_range_small = (-12.8, -12.8, 25.6, 12.8)  # (bottom, right, top, left)
    spatial_resolution_small = (0.2, 0.2)  # (height, witdh)

    data_root = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/xiangyu.li/test_attribute/20220611-141510_348",  # noqa
    )

    img_front_path = os.path.join(data_root, "camera_front/1654928408600.jpg")
    img_front_left_path = os.path.join(
        data_root, "camera_front_left/1654928408600.jpg"
    )
    img_front_right = os.path.join(
        data_root, "camera_front_right/1654928408600.jpg"
    )
    img_rear_left_path = os.path.join(
        data_root, "camera_rear_left/1654928408600.jpg"
    )
    img_rear_right_path = os.path.join(
        data_root, "camera_rear_right/1654928408600.jpg"
    )
    img_rear_path = os.path.join(data_root, "camera_rear/1654928408600.jpg")
    fisheye_img_front_path = os.path.join(
        data_root, "fisheye_front/1654928408614.jpg"
    )
    fisheye_img_rear_path = os.path.join(
        data_root, "fisheye_rear/1654928408614.jpg"
    )
    fisheye_img_left_path = os.path.join(
        data_root, "fisheye_left/1654928408614.jpg"
    )
    fisheye_img_right_path = os.path.join(
        data_root, "fisheye_right/1654928408614.jpg"
    )
    narrow_img_path = os.path.join(
        data_root, "camera_front_30fov/1654928408599.jpg"
    )

    sync_imgs = {
        "camera_front": img_front_path,
        "camera_front_left": img_front_left_path,
        "camera_front_right": img_front_right,
        "camera_rear_left": img_rear_left_path,
        "camera_rear": img_rear_path,
        "camera_rear_right": img_rear_right_path,
        "fisheye_front": fisheye_img_front_path,
        "fisheye_rear": fisheye_img_rear_path,
        "fisheye_left": fisheye_img_left_path,
        "fisheye_right": fisheye_img_right_path,
        "camera_front_30fov": narrow_img_path,
    }

    per_view_shape = {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 2048),
        "camera_front_right": (1280, 2048),
        "camera_rear_left": (1280, 2048),
        "camera_rear_right": (1280, 2048),
        "camera_rear": (1280, 2048),
        "fisheye_front": (1536, 1920),
        "fisheye_rear": (1536, 1920),
        "fisheye_left": (1536, 1920),
        "fisheye_right": (1536, 1920),
        "camera_front_30fov": (2160, 3840),
    }

    camera_view_names = list(per_view_shape.keys())
    homo_transforms = {
        "camera_front": {
            "Resize": (540, 960),
            "Crop": (0, 0, 512, 960),
        },
        "camera_front_left": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_front_right": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_rear_left": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_rear_right": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "camera_rear": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
        },
        "fisheye_front": {
            "Resize": (768, 960),
            "Crop": (0, 0, 768, 960),
        },
        "fisheye_rear": {
            "Resize": (768, 960),
            "Crop": (0, 0, 768, 960),
        },
        "fisheye_left": {
            "Resize": (768, 960),
            "Crop": (0, 0, 768, 960),
        },
        "fisheye_right": {
            "Resize": (768, 960),
            "Crop": (0, 0, 768, 960),
        },
        "camera_front_30fov": {
            "Resize": (540, 960),
            "Crop": (0, 0, 512, 960),
        },
    }
    # ipm from calibration params
    calib_path = "Calibration_params/UTHS6"  # noqa
    calib_path = os.path.join(data_root, calib_path)
    homo_path = None
    vcs_plane_heights = (0, 0.5, 1.0, 1.5)

    homo_gen = HomoGenerator(
        homo_path=homo_path,
        calib_path=calib_path,
        spatial_resolution=spatial_resolution,
        vcs_range=vcs_range,
        camera_view_names=camera_view_names,
        task_camera_view_names=camera_view_names,
        per_view_shape=per_view_shape,
        use_distorted_offset=use_distorted_offset,
        homo_transforms=homo_transforms,
        vcs_plane_heights=vcs_plane_heights,
        return_offset_in_meta_info=False,
    )
    homo_gen_small = HomoGenerator(
        homo_path=homo_path,
        calib_path=calib_path,
        spatial_resolution=spatial_resolution_small,
        vcs_range=vcs_range_small,
        camera_view_names=camera_view_names,
        task_camera_view_names=camera_view_names,
        per_view_shape=per_view_shape,
        use_distorted_offset=use_distorted_offset,
        homo_transforms=homo_transforms,
        vcs_plane_heights=vcs_plane_heights,
        return_offset_in_meta_info=False,
    )

    meta_info = homo_gen.get_meta_info()
    meta_info_small = homo_gen_small.get_meta_info()

    for mate_name, meta_value in meta_info.items():
        meta_info[mate_name] = ANCToTensor3DV._meta_to_tensor(meta_value)
    for mate_name, meta_value in meta_info_small.items():
        meta_info_small[mate_name] = ANCToTensor3DV._meta_to_tensor(meta_value)

    collate_meta_info = collate_3d([meta_info])
    collate_meta_info_small = collate_3d([meta_info_small])
    temporal_info = {
        "num_frames_per_iter": 1,
    }
    collate_temporal_info = collate_3d([temporal_info])
    collate_views_domain2nums = {
        "front": (1,),
        "side": (5,),
        "round": (4,),
        "narrow": (1,),
    }

    torch_imgs = []
    for _, name in enumerate(camera_view_names):
        img = cv2.imread(sync_imgs[name], -1)
        img = cv2.resize(img, homo_transforms[name]["Resize"][::-1])
        img = img[
            : homo_transforms[name]["Crop"][2],
            : homo_transforms[name]["Crop"][3],
            :,
        ]
        img_torch = img_array2tensor(img)
        torch_imgs.append(img_torch)

    data = {
        "img": [torch_imgs[0]],
        "side_img": [torch.cat(torch_imgs[1:6])],
        "round_img": [torch.cat(torch_imgs[6:10])],
        "narrow_img": [torch_imgs[10]],
        "meta_info": collate_meta_info,
        "meta_info_small": collate_meta_info_small,
        "temporal_info": collate_temporal_info,
        "view": collate_views_domain2nums,
        "aug_transforms": {
            "rpy_augmentation": {"prob": [1], "aug_degree": [1]}
        },
    }

    camera_prenorm_setting = {
        "camera_front_right": [0, 0, 0],
    }
    cam_prenorm_module = ANCCamPrenorm(
        camera_prenorm_setting=camera_prenorm_setting,
        camera_view_names=camera_view_names,
    )
    data = cam_prenorm_module(data)

    rpy_aug_module = ANCRpyParamAug(
        prob=1,
        aug_degree=5,  # deg
        filter_cameras=list(camera_prenorm_setting.keys()),
        camera_view_names=camera_view_names,
    )
    data = rpy_aug_module(data)

    assert "side_img_uv_map" in data


if __name__ == "__main__":
    pytest.main(["-s", __file__])
