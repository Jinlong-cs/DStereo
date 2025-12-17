import json
import logging

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

from hat.core.position_embedding_utils import PositionEncoder
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class FrameDataset(Dataset):
    def __init__(
        self,
        img_path,
        calib_path,
        to_rgb=False,
        buf_only=False,
        transforms=None,
        pe_config=None,
    ):
        data = {}
        image = cv.imread(img_path)

        color_space = "bgr"
        if to_rgb:
            cv.cvtColor(image, cv.COLOR_BGR2RGB, image)
            color_space = "rgb"

        data["color_space"] = color_space
        data["ori_img"] = image
        data["img_id"] = 0

        if buf_only:
            with open(img_path, "rb") as rf:
                data["img_buf"] = rf.read()
        else:
            data["img"] = data["ori_img"]

        assert calib_path.endswith(".json")

        with open(calib_path, "r") as rf:
            calib_dict = json.load(rf)

        data["calib"] = np.array(calib_dict["calib"])
        data["distCoeffs"] = np.array(calib_dict["distCoeffs"])
        if "calib_all" in calib_dict:
            data["calib_all"] = calib_dict["calib_all"]

        data["layout"] = "hwc"
        self.data = data
        self.transforms = transforms
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
            pe_params = calib_dict.get("calib_all", None)
            coordinate3d_map = self.position_encoder(pe_params)
            data["coordinate_map"] = coordinate3d_map

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if self.transforms is not None:
            data = self.transforms(self.data)
        return data
