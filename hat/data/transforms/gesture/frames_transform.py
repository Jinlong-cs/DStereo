# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import inspect
import logging
from typing import Dict, List

import horizon_plugin_pytorch.nn.bgr_to_yuv444 as b2y
import numpy as np
import torch

from hat.data.transforms.detection import ToTensor
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

try:
    from torchvision.transforms import functional as F
except ImportError:
    F = None

try:
    import albumentations
except ImportError:
    albumentations = None

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ActionImgClipAlbuTrans(object):
    """Img clip is augmented by albumentations.

    Args:
        albu_same_in_seq (bool): whether to apply the same data
            augmentation to all data. Defaults to True.
        albu_params (List[Dict]): params for albumentations.
            Defaults to None.
    """

    @require_packages("albumentations==0.4.6")
    def __init__(
        self, albu_same_in_seq: bool = True, albu_params: List[Dict] = None
    ):
        self.albu_same_in_seq = albu_same_in_seq
        self.albu_transform = self.compose_albu_transform(albu_params)

    def compose_albu_transform(self, albu_params):
        transforms = []
        if isinstance(albu_params, list):
            for albu_param in albu_params:
                param = copy.deepcopy(albu_param)
                name = param.pop("name")
                transform = getattr(albumentations, name)(**param)
                self.check_transform(transform)
                transforms.append(transform)
        elif isinstance(albu_params, dict):
            logging.info(
                "albu_params is dict, please do not " "use ordered transforms."
            )
            for name in albu_params:
                param = copy.deepcopy(albu_params[name])
                transform = getattr(albumentations, name)(**param)
                self.check_transform(transform)
                transforms.append(transform)
        else:
            raise ValueError
        return albumentations.Compose(transforms)

    def check_transform(self, transform):
        # Check transform is ImageOnlyTransform
        base_list = inspect.getmro(type(transform))
        if albumentations.ImageOnlyTransform not in base_list:
            raise ValueError("%s is not ImageOnlyTransform" % transform)

    def __call__(self, data):
        if "frames" in data:
            if data["frames"] is not None:
                if self.albu_same_in_seq:
                    len_roi_imgs = len(data["frames"])
                    data["frames"] = np.concatenate(data["frames"], axis=0)
                    data["frames"] = self.albu_transform(image=data["frames"])[
                        "image"
                    ]
                    data["frames"] = np.clip(data["frames"], 0, 255)
                    data["frames"] = np.array_split(
                        data["frames"], len_roi_imgs, axis=0
                    )
                else:
                    for idx in range(len(data["frames"])):
                        data["frames"][idx] = self.albu_transform(
                            image=data["frames"][idx]
                        )["image"]
                        data["frames"][idx] = np.clip(
                            data["frames"][idx], 0, 255
                        )
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"albu_same_in_seq={self.albu_same_in_seq}"
        repr_str += f"albu_params={self.albu_params}"
        return repr_str


@OBJECT_REGISTRY.register
class ActionImgClipToTensor(ToTensor):
    """Convert objects of img(np) to torch.Tensor.

    Convert the img to yuv444 format if to_yuv is True.

    Args:
        to_yuv: If true, convert the img to yuv444 format.
             Defaults to True.
        tensor_layout: layout of img tensor, support 'chw','hwc'.
            Defaults to 'chw'.
        img_layout: layout of input image, support 'bgr','rgb'.
            Defaults to "rgb".
        mean: sequence of means for each channel.
            Defaults to (128.0, 128.0, 128.0).
        std: sequence of standard deviations for each channel.
            Defaults to (128.0, 128.0, 128.0).
    """

    @require_packages("torchvision")
    def __init__(
        self,
        to_yuv: bool = True,
        tensor_layout: str = "chw",
        img_layout: str = "rgb",
        mean: List[float] = (128.0, 128.0, 128.0),
        std: List[float] = (128.0, 128.0, 128.0),
    ):
        super(ActionImgClipToTensor, self).__init__()
        self.to_yuv = to_yuv
        self.tensor_layout = tensor_layout
        self.channel_reversal = img_layout == "rgb"
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, data):
        if "frames" in data:
            if data["frames"] is not None:
                # step1 stack and to torch
                data["frames"] = [
                    self._to_tensor(image) for image in data["frames"]
                ]
                data["frames"] = torch.stack(data["frames"], dim=0)
                # step2 transpose
                if "frame_layout" not in data:
                    data["frame_layout"] = "hwc"
                if (
                    self.tensor_layout == "chw"
                    and data["frame_layout"] == "hwc"
                ):
                    data["frames"] = data["frames"].permute(0, 3, 1, 2)
                # step3 to yuv
                if self.to_yuv:
                    # be careful to set channel_reversal
                    data["frames"] = b2y.bgr_to_yuv444(
                        data["frames"], self.channel_reversal
                    )
                # step4 normalize and img type uint -> float
                # todo: need to check .float()
                data["frames"] = F.normalize(
                    data["frames"].float(), self.mean, self.std
                )
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"to_yuv={self.to_yuv}"
        repr_str += f"tensor_layout={self.tensor_layout}"
        repr_str += f"channel_reversal={self.channel_reversal}"
        repr_str += f"mean={self.mean}"
        repr_str += f"std={self.std}"
        return repr_str
