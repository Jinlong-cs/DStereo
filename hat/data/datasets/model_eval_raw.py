# Copyright (c) Horizon Robotics. All rights reserved.

import glob
import json
import logging
import os
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from hat.core.position_embedding_utils import PositionEncoder
from hat.registry import OBJECT_REGISTRY
from hat.utils.bucket import url_to_local_path
from .image_auto2d import Auto2dFromImage

logger = logging.getLogger(__name__)


__all__ = ["ModelEvalRawDataset"]


@OBJECT_REGISTRY.register
class ModelEvalRawDataset(Auto2dFromImage):
    """Dataset which gets img data from the data_path.

    This dataset can used for inference on unlabeled data.

    Args:
        data_path: The path where the image is stored.
        transforms: List of transform.
        infer_model_type: Inference model type, currently only the
            crop model is supported, optional values include
            [`crop_with_resize_quarter`, `crop_wo_resize`].
        buf_only: Whether to read data buf or decode.
        to_rgb: Whether to convert to `rgb` color_space.
        return_orig_img: Whether to return an extra original img,
            orig_img can usually be used on visualization.
        image_types: The format list of images that needs to
            read.
        skip_first_frame: Skip first frame to get compatible with resflow task.
        pe_config: config params of position encoding (optional).
        use_dataset_extrinsic: use extrinsic in dataset or not.
    """

    def __init__(
        self,
        data_path: str,
        transforms: Optional[List[Callable]] = None,
        infer_model_type: Optional[str] = None,
        buf_only: bool = False,
        to_rgb: bool = False,
        return_orig_img: bool = False,
        image_types: Optional[List[str]] = None,
        skip_first_frame: bool = False,
        pe_config: Optional[Dict] = None,
        use_dataset_extrinsic: bool = False,
        enable_calib_all: bool = False,
    ):
        super().__init__(
            data_path,
            transforms=transforms,
            infer_model_type=infer_model_type,
            to_rgb=to_rgb,
            return_orig_img=return_orig_img,
            image_types=image_types,
            skip_first_frame=skip_first_frame,
        )
        self.buf_only = buf_only

        self.use_dataset_extrinsic = use_dataset_extrinsic
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
            self.pe_verbose = pe_config["verbose"]
        self.enable_calib_all = enable_calib_all

    def __getitem__(self, item):
        data = {}
        image_path = self.image_path_list[item]
        image = cv2.imread(image_path)
        if self.return_orig_img:
            data["ori_img"] = image
        color_space = "bgr"
        if self.to_rgb:
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
            color_space = "rgb"
        data["img_name"] = self.image_name_list[item]
        data["img_height"] = image.shape[0]
        data["img_width"] = image.shape[1]
        data["img_id"] = item
        if self.buf_only:
            with open(image_path, "rb") as rf:
                data["img_buf"] = rf.read()
        else:
            data["img"] = image

        pe_params = None
        crop_roi = None
        if image_path in self.image_annos:
            img_anno = self.image_annos[image_path]
            if isinstance(img_anno, dict):
                if "dynamic_region" in img_anno:
                    crop_roi = img_anno["dynamic_region"]
                if "calib" in img_anno:
                    data["calib"] = np.array(img_anno["calib"])
                if "distCoeffs" in img_anno:
                    data["distCoeffs"] = np.array(img_anno["distCoeffs"])
                if "calib_all" in img_anno:
                    if self.enable_calib_all:
                        data["calib_all"] = img_anno["calib_all"]
                    if self.use_dataset_extrinsic:
                        pe_params = img_anno["calib_all"]
                elif "extrinsic" in img_anno:
                    if self.enable_calib_all:
                        data["calib_all"] = img_anno["extrinsic"]
                    if self.use_dataset_extrinsic:
                        pe_params = img_anno["extrinsic"]

        if callable(self.position_encoder):
            # TODO(xinjie.wang): avoid multiple verbose assignment
            self.position_encoder.verbose = self.pe_verbose
            coordinate3d_map = self.position_encoder(pe_params)
            data["coordinate_map"] = coordinate3d_map

        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image.shape
        data["pad_shape"] = image.shape
        if crop_roi:
            data["crop_roi"] = crop_roi
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
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    @staticmethod
    def get_image_info(data_path, image_types, model_type=None):
        """Get the path, name and annotation list of all images under the \
        data path."""

        image_path_list = []
        image_name_list = []
        image_annos = {}
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
                    try:
                        annos = {
                            "calib": anno_i["calib"],
                            "distCoeffs": anno_i["distCoeffs"],
                        }
                        if "calib_all" in anno_i:
                            annos.update(calib_all=anno_i["calib_all"])
                        elif "extrinsic" in anno_i:
                            annos.update(extrinsic=anno_i["extrinsic"])
                    except KeyError:
                        annos = ModelEvalRawDataset.parse_2d_anno(anno_i)

                    if (
                        isinstance(anno_i, dict)
                        and "attrs" in anno_i
                        and isinstance(anno_i["attrs"], dict)
                        and "dynamic_region" in anno_i["attrs"]
                    ):
                        if isinstance(annos, dict):
                            annos.update(
                                dynamic_region=anno_i["attrs"][
                                    "dynamic_region"
                                ]
                            )
                        else:
                            annos = {
                                "dynamic_region": anno_i["attrs"][
                                    "dynamic_region"
                                ]
                            }
                    # using image source as image path
                    # support any bucket image store.
                    # if image already on auto_eval, use it.
                    # else using image source path.
                    img_url = url_to_local_path(
                        os.path.join(images, anno_i["image_key"])
                    )
                    image_name_list.append(anno_i["image_key"])
                    if "image_source" in anno_i and not os.path.exists(
                        img_url
                    ):
                        img_url = anno_i["image_source"]
                        if img_url.startswith("dmpv2:"):
                            img_url = url_to_local_path(img_url)
                    img_url_list.append(img_url)
                    image_annos[img_url] = annos
            else:
                img_url_list = []
                img_key_list = []
                for curdir, _dirnames, filenames in os.walk(images):
                    for filename in filenames:
                        if os.path.splitext(filename)[-1] in image_types:
                            img_url = os.path.join(curdir, filename)
                            img_url_list.append(img_url)
                            img_key_list.append(img_url[len(images) + 1 :])
                            image_name_list.append(os.path.basename(img_url))
            for img in img_url_list:
                assert (
                    os.path.splitext(img)[1] in image_types
                ), "%s type must in %s" % (img.split(".")[-1], image_types)
                image_path_list.append(img)

        return image_path_list, image_name_list, image_annos

    @staticmethod
    def parse_2d_anno(anno):
        try:
            attr = anno["attrs"]["camera_default"]
            calib_all = {
                "image_width": anno["width"],
                "image_height": anno["height"],
                "camera_x": attr["cameraX"],
                "camera_y": attr["cameraY"],
                "camera_z": attr["cameraZ"],
                "roll": attr["roll"],
                "pitch": attr["pitch"],
                "yaw": attr["yaw"],
                "distort": attr["distort"],
                "focal_u": attr["focalU"],
                "focal_v": attr["focalV"],
                "center_u": attr["centerU"],
                "center_v": attr["centerV"],
                "vcs": attr["vcs"],
            }
            return {"calib_all": calib_all}
        except KeyError:
            return 0
