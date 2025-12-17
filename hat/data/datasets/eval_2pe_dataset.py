# Copyright (c) Horizon Robotics. All rights reserved.
import glob
import json
import logging
import os
from typing import Callable, List, Optional

import cv2
import numpy as np

from hat.registry import OBJECT_REGISTRY
from .image_auto2d import Auto2dFromImage

logger = logging.getLogger(__name__)


__all__ = ["MtlEvalRaw2PEDataset"]


@OBJECT_REGISTRY.register
class MtlEvalRaw2PEDataset(Auto2dFromImage):
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
    """

    def __init__(
        self,
        data_path: str,
        task_type: str,
        transforms: Optional[List[Callable]] = None,
        infer_model_type: Optional[str] = None,
        buf_only: bool = False,
        to_rgb: bool = False,
        return_orig_img: bool = False,
        image_types: Optional[List[str]] = None,
        skip_first_frame: bool = False,
        instance_key: str = "common_box",
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
        self.task_type = task_type
        self.instance_key = instance_key
        (
            self.image_path_list,
            self.image_name_list,
            self.image_annos,
        ) = self.get_image_info(
            self.data_path,
            self.image_types,
            infer_model_type,
            self.instance_key,
        )

    def _filter(self, instance):
        # Filter side and back lights
        if instance["attrs"].get("direction", "unk") != "front":
            return True
        # remove light which is not on
        if instance["attrs"].get("Type", "unk") != "on":
            return True
        # remove light which is not target
        if instance["attrs"].get("NonTarget", "unk") != "No":
            return True
        # remove light which is ignore
        if instance["attrs"].get("ignore", "unk") != "no":
            return True
        # Only fully visible and partially obscured data is retained
        if instance["attrs"].get("occlusion", "unk") not in [
            "full_visible",
            "occluded",
        ]:
            return True
        # remove lightboxes with short sides less than 8
        x1, y1, x2, y2 = instance["data"]
        if x2 - x1 < 8 or y2 - y1 < 8:
            return True
        return False

    def __getitem__(self, index):
        data = {}
        image_path = self.image_path_list[index]
        image = cv2.imread(image_path)
        if self.return_orig_img:
            data["ori_img"] = image
        color_space = "bgr"
        if self.to_rgb:
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
            color_space = "rgb"
        data["img_name"] = self.image_name_list[index]
        data["img_height"] = image.shape[0]
        data["img_width"] = image.shape[1]
        data["img_id"] = index
        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img"] = image
        if self.buf_only:
            with open(image_path, "rb") as rf:
                data["img_buf"] = rf.read()
        else:
            data["ori_img"] = image.copy()
        results = []
        dummy_data = data.copy()
        if image_path in self.image_annos:
            img_anno = self.image_annos[image_path]
            if "instances" not in img_anno.keys():
                data["crop_roi"] = np.array(
                    [0, 0, image.shape[1], image.shape[0]], dtype=np.float32
                )
                data["obj_id"] = 0
                if self.transforms is not None:
                    results.append(self.transforms(data))
                return results
            for instance_idx in range(0, len(img_anno["instances"])):
                data = dummy_data.copy()
                ins = img_anno["instances"][instance_idx]
                # For traffic lens detection and evaluation, only the light box
                # that meets the frame cutting condition is retained.
                if self.task_type == "detection":
                    if self._filter(ins):
                        continue
                crop_roi = ins["data"]
                data["crop_roi"] = np.array(crop_roi, dtype=np.float32)
                data["obj_id"] = instance_idx
                if self.transforms is not None:
                    data = self.transforms(data)
                    if self.task_type == "detection":
                        data["inverse_affine_aug_param"] = np.linalg.inv(
                            data["affine_aug_param"]
                        )
                results.append(data)
        return results

    @staticmethod
    def get_image_info(
        data_path, image_types, model_type=None, instance_key="common_box"
    ):
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
            assert jsonfile is not None
            with open(jsonfile, "r") as fread:
                annos = fread.readlines()
            img_anno_list = [json.loads(anno_i) for anno_i in annos]

            img_url_list = []
            for anno_i in img_anno_list:
                annos = {}
                if instance_key in anno_i.keys():
                    annos.update(instances=anno_i[instance_key])
                if "attrs" in anno_i and "camera_default" in anno_i["attrs"]:
                    annos.update(camera_info=anno_i["attrs"]["camera_default"])
                image_annos[os.path.join(images, anno_i["image_key"])] = annos
                img_url_list.append(os.path.join(images, anno_i["image_key"]))
            for img in img_url_list:
                assert (
                    os.path.splitext(img)[1] in image_types
                ), "%s type must in %s" % (img.split(".")[-1], image_types)
                image_path_list.append(img)
                image_name_list.append(img.replace(images + "/", ""))

        return image_path_list, image_name_list, image_annos
