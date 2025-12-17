import copy
import json
import logging
import os
from glob import glob

import numpy as np

from hat.utils.package_helper import require_packages

try:
    from pyramid_resizer.image_cvt_utils import convert_nv12_to_yuv444_uint8
except ImportError:
    convert_nv12_to_yuv444_uint8 = None

from torch.utils.data import Dataset

from hat.core.position_embedding_utils import PositionEncoder
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)


def read_yuv_with_shape(yuv_file, im_hw):
    frame_len = im_hw[0] * im_hw[1] * 3 // 2
    with open(yuv_file, "rb") as f:
        raw = f.read(frame_len)
    yuv = np.frombuffer(raw, dtype=np.uint8)
    yuv = yuv.reshape(im_hw[0] * 3 // 2, im_hw[1])
    return yuv


def load_calib_and_dist_coeffs(calib_path):
    with open(calib_path, "r") as rf:
        calib_dict = json.load(rf)

    fu, fv = calib_dict["focal_u"], calib_dict["focal_v"]
    cu, cv = calib_dict["center_u"], calib_dict["center_v"]
    calib = np.array(
        [
            [fu, 0, cu, 0],
            [0, fv, cv, 0],
            [0, 0, 1, 0],
        ]
    )
    distCoeffs = np.array(calib_dict["distort"])

    return calib, distCoeffs, calib_dict


@OBJECT_REGISTRY.register
class YUVFrames(Dataset):
    @require_packages("pyramid_resizer")
    def __init__(
        self,
        img_dir,
        im_hw,
        calib_path=None,
        transforms=None,
        pe_config=None,
        enable_calib_all=False,
    ):
        self.img_paths = glob(os.path.join(img_dir, "*.yuv"))
        self.im_hw = im_hw

        if calib_path is not None:
            calib, dist_coeffs, self.pe_params = load_calib_and_dist_coeffs(
                calib_path
            )
            self.calib_dict = {"calib": calib, "distCoeffs": dist_coeffs}
        else:
            self.pe_params = None
            self.calib_dict = None

        self.position_encoder = None
        if pe_config is not None:
            logger.warning("Please use the PEGenerator transform instead.")
            self.position_encoder = PositionEncoder(
                pe_stride=pe_config["pe_stride"],
                input_hw=pe_config["input_hw"],
                img_resize=pe_config["img_resize"],
                pe_h=pe_config["pe_h"],
                pe_w=pe_config["pe_w"],
                default_intrinsic_mat=pe_config["default_intrinsic_mat"],
                default_distort=pe_config["default_distort"],
                default_pitch=pe_config["default_pitch"],
                default_roll=pe_config["default_roll"],
                default_camera_z=pe_config["default_camera_z"],
                crop_roi=pe_config["crop_roi_3d"],
                verbose=pe_config["verbose"],
            )

        self.enable_calib_all = enable_calib_all
        self.transform = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        nv12 = read_yuv_with_shape(img_path, self.im_hw)

        yuv = convert_nv12_to_yuv444_uint8(nv12, self.im_hw)
        data = {
            "img": yuv.astype(np.float32).transpose(2, 0, 1),
            "img_id": img_path.split("/")[-1].split(".")[0],
            "img_height": self.im_hw[0],
            "img_width": self.im_hw[1],
            "layout": "chw",
        }
        if self.pe_params is not None and self.enable_calib_all:
            data["calib_all"] = copy.deepcopy(self.pe_params)

        if self.calib_dict is not None:
            data.update(self.calib_dict)
        if callable(self.position_encoder):
            coordinate3d_map = self.position_encoder(self.pe_params)
            data.update(coordinate_map=coordinate3d_map)

        data = data if self.transform is None else self.transform(data)
        return data

    def __len__(self):
        return len(self.img_paths)


@OBJECT_REGISTRY.register
class YUVMultiCamera(Dataset):
    @require_packages("pyramid_resizer")
    def __init__(
        self,
        data_dir,
        view_img_hw,
        view_homo_hw,
        grid_quant_scale,
        json_path=None,
        transforms=None,
    ):
        """Multi-camera yuv Dataset.

        data_dir format example:
            data_dir
                - 1000
                    - 1000_homo_0.dat
                    - 1000_homo_1.dat
                    - 1000_image_0.yuv
                    - 1000_image_1.yuv
                - 1001
                    - 1001_homo_0.dat
                    - 1001_homo_1.dat
                    - 1001_image_0.yuv
                    - 1001_image_1.yuv
        Folders of data_dir represent each sample

        json format example:
            {
                key_0: {
                    view_0: {
                        "image": path_relative_to_data_dir,
                        "homo": path_relative_to_data_dir
                    },
                    view_1: {
                        "image": path_relative_to_data_dir,
                        "homo": path_relative_to_data_dir
                    },
                    ...
                },
                ...
            }
        Element of json represent each sample

        Args:
            data_dir: data path.
            view_img_hw: img hw of each view.
            view_homo_hw: homo hw of each view,
                note whether crop for homo offset.
            grid_quant_scale: quant scale of homo_offset.
            json_path: json path with data info.
            transforms: transform function.
        """

        sample_all = {}
        if json_path is not None:
            with open(json_path, "r") as f_in:
                sample_info = json.load(f_in)
                for s_id, s_info in sample_info.items():
                    sample_all[s_id] = {"sample_id": s_id}
                    for v, paths in s_info.items():
                        sample_all[s_id][v] = {}
                        homo_path = os.path.join(data_dir, paths["homo"])
                        sample_all[s_id][v].update({"homo_offset": homo_path})
                        image_path = os.path.join(data_dir, paths["image"])
                        sample_all[s_id][v].update({"image": image_path})
        else:
            for name in os.listdir(data_dir):
                path = os.path.join(data_dir, name)
                if os.path.isdir(path):
                    sample_idx = name
                    for file in os.listdir(path):
                        split = file.split(".")[0].split("_")
                        idx, dtype = split[0], split[1]
                        camera_view = "_".join(split[2:])
                        if idx != sample_idx:
                            continue
                        if idx not in sample_all:
                            sample_all[idx] = {"sample_id": idx}
                        if camera_view not in sample_all[idx]:
                            sample_all[idx][camera_view] = {}
                        if dtype == "homo":
                            homo = os.path.join(data_dir, name, file)
                            sample_all[idx][camera_view].update(
                                {"homo_offset": homo}
                            )
                        if dtype == "image":
                            image = os.path.join(data_dir, name, file)
                            sample_all[idx][camera_view].update(
                                {"image": image}
                            )
        self.sample_all = list(sample_all.values())
        self.view_img_hw = view_img_hw
        self.view_homo_hw = view_homo_hw
        self.grid_quant_scale = grid_quant_scale

        self.transform = transforms

    def __getitem__(self, index):
        sample = self.sample_all[index]
        data = {
            "sample_id": sample["sample_id"],
            "layout": "chw",
        }
        img_id_list = []

        for idx, (view, hw) in enumerate(self.view_img_hw.items()):
            img_id_list.append(
                sample[view]["image"].split("/")[-1].split(".")[0]
            )
            nv12 = read_yuv_with_shape(sample[view]["image"], hw)
            yuv = (
                convert_nv12_to_yuv444_uint8(nv12, hw)
                .astype(np.float32)
                .transpose(2, 0, 1)
            )
            homo_offset = np.fromfile(
                sample[view]["homo_offset"], dtype=np.int16
            ).astype(np.float32) * (1 / 64.0)
            homo_offset = homo_offset.reshape(
                (2,) + self.view_homo_hw[view]
            ).transpose(1, 2, 0)
            frame = {
                "img": yuv,
                "layout": "chw",
            }
            frame = frame if self.transform is None else self.transform(frame)
            data.update(
                {
                    f"img_{idx}": frame["img"],
                    f"homo_offset_{idx}": homo_offset,
                }
            )
        data["img_id"] = tuple(img_id_list)
        return data

    def __len__(self):
        return len(self.sample_all)
