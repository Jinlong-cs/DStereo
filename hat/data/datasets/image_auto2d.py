# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import glob
import json
import os
from typing import Callable, List, Optional

import cv2
import numpy as np
import torch.utils.data as data

from hat.registry import OBJECT_REGISTRY

__all__ = ["Auto2dFromImage"]


@OBJECT_REGISTRY.register
class Auto2dFromImage(data.Dataset):
    """Dataset which gets img data from the data_path.

    This dataset can used for inference on unlabeled data.

    Args:
        data_path: The path where the image is stored.
        transforms: List of transform.
        infer_model_type: Inference model type, currently only the
            crop model is supported, optional values include
            [`crop_with_resize_quarter`, `crop_wo_resize`].
        to_rgb: Whether to convert to `rgb` color_space.
        return_orig_img: Whether to return an extra original img,
            orig_img can usually be used on visualization.
        return_orig_hw: Whether to return height and width of original image.
        image_types: The format list of images that needs to read.
        skip_first_frame: Skip first frame to get compatible with resflow task.
        return_img_buf: Whether to return undecoded image buffer.
            Default is False.

    """

    def __init__(
        self,
        data_path: str,
        transforms: Optional[List[Callable]] = None,
        infer_model_type: Optional[str] = None,
        to_rgb: bool = False,
        return_orig_img: bool = False,
        return_orig_hw: bool = False,
        image_types: Optional[List[str]] = None,
        skip_first_frame: bool = False,
        return_img_buf: bool = False,
    ):
        self.data_path = data_path
        self.transforms = transforms
        if image_types is None:
            image_types = [".jpeg", ".png", ".jpg"]
        self.image_types = copy.deepcopy(image_types)
        # sometimes image_name like this
        # '594975_2/data/ADAS_20210306-112355_482_0__135656_1615001259481_0.jpg',
        # we can't just get the base name of this image or we'll get an error
        # in the eval process.
        (
            self.image_path_list,
            self.image_name_list,
            self.image_annos,
        ) = self.get_image_info(
            self.data_path, self.image_types, infer_model_type
        )
        if skip_first_frame:
            self.image_name_list = self.image_name_list[1:]
        self.num_samples = len(self.image_name_list)
        self.to_rgb = to_rgb
        self.return_orig_img = return_orig_img
        self.return_orig_hw = return_orig_hw
        self.infer_model_type = infer_model_type
        self.return_img_buf = return_img_buf

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, item):
        data = {}
        image_path = self.image_path_list[item]
        image = cv2.imread(image_path)
        if self.return_img_buf:
            with open(image_path, "rb") as rf:
                data["img_buf"] = rf.read()
        if self.return_orig_img:
            data["orig_img"] = image.copy()
        if self.return_orig_hw:
            data["orig_hw"] = image.shape[:2]
        color_space = "bgr"
        if self.to_rgb:
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
            color_space = "rgb"
        data["img_name"] = self.image_name_list[item]
        data["img_height"] = image.shape[0]
        data["img_width"] = image.shape[1]
        data["img_id"] = item
        data["img"] = image
        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image.shape
        data["pad_shape"] = image.shape
        if self.infer_model_type in [
            "crop_with_resize_quarter",
            "crop_wo_resize",
        ]:
            data["camera_info"] = (
                0
                if len(self.image_annos) == 0
                else self.image_annos[image_path]
            )
            data["infer_model_type"] = self.infer_model_type
        elif self.infer_model_type == "real3d":
            cam_info = self.image_annos[image_path]
            data["calibration"] = cam_info["calibration"]
            data["dist_coeffs"] = cam_info["dist_coeffs"]
            data["Tr_vel2cam"] = np.array(
                cam_info["lidar2camera"]["Tr_vel2cam"]
            )
            Tr_vcs2cam = np.eye(4)
            if (
                "rotMat" in cam_info["lidar2camera"]
                and "transMat" in cam_info["lidar2camera"]
            ):
                Tr_vcs2cam[:3, :3] = np.array(
                    cam_info["lidar2camera"]["rotMat"]
                )
                Tr_vcs2cam[:3, 3] = np.array(
                    cam_info["lidar2camera"]["transMat"]
                ).T
            data["Tr_vcs2cam"] = Tr_vcs2cam
            data["ignore_mask"] = cam_info["ignore_mask"]
        elif self.infer_model_type == "rcnn":
            data["gt_boxes"] = np.zeros((1, 5))

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"data_path={self.data_path}, "
        repr_str += f"to_rgb={self.to_rgb}, "
        repr_str += f"return_orig_img={self.return_orig_img}"
        return repr_str

    @staticmethod
    def get_image_info(data_path, image_types, model_type=None):
        """Get the path, name and annotation list of all images under the \
        data path."""

        image_path_list = []
        image_name_list = []
        image_annos = {}
        if data_path.endswith(".json"):
            jsonfiles = [data_path]
        else:
            jsonfiles = glob.glob(os.path.join(data_path, "*/*.json"))
        if not jsonfiles:
            images = glob.glob(os.path.join(data_path, "*/images"))
            jsonfile = None
            if len(images) == 1:
                images = images[0]
        else:
            jsonfile = jsonfiles[0]
            images = os.path.splitext(jsonfile)[0]
        # When the dir don't have jsonfile or images dir
        # We try to get images from 'self.data_path'
        if jsonfile is None and len(images) == 0:
            for file in sorted(os.listdir(data_path)):
                if os.path.splitext(file)[1] in image_types:
                    image_path_list.append(os.path.join(data_path, file))
                    image_name_list.append(file)
        else:
            if jsonfile is not None:
                with open(jsonfile, "r") as fread:
                    annos = fread.readlines()
                img_anno_list = [json.loads(anno_i) for anno_i in annos]

                img_url_list = []
                for anno_i in img_anno_list:
                    if "image_key" not in anno_i:
                        anno_i["image_key"] = anno_i["img_key"]
                    camera_info = 0
                    if (
                        "attrs" in anno_i
                        and "camera_default" in anno_i["attrs"]
                    ):
                        camera_info = anno_i["attrs"]["camera_default"]
                    elif model_type == "real3d":
                        camera_info = {
                            "calibration": anno_i["calib"],
                            "dist_coeffs": anno_i["distCoeffs"],
                            "lidar2camera": anno_i["lidar_to_camera"],
                            "ignore_mask": anno_i["ignore_mask"],
                        }
                    image_annos[
                        os.path.join(images, anno_i["image_key"])
                    ] = camera_info
                    img_url_list.append(
                        os.path.join(images, anno_i["image_key"])
                    )
            else:
                img_url_list = []
                img_key_list = []
                for curdir, _dirnames, filenames in os.walk(images):
                    for filename in filenames:
                        if os.path.splitext(filename)[-1] in image_types:
                            img_url = os.path.join(curdir, filename)
                            img_url_list.append(img_url)
                            img_key_list.append(img_url[len(images) + 1 :])
            for img in img_url_list:
                assert (
                    os.path.splitext(img)[1] in image_types
                ), "%s type must in %s" % (img.split(".")[-1], image_types)
                image_path_list.append(img)
                image_name_list.append(img.replace(images + "/", ""))

        return image_path_list, image_name_list, image_annos
