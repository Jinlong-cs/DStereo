# Copyright (c) Horizon Robotics. All rights reserved.
import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from hat.registry import OBJECT_REGISTRY

__all__ = [
    "RandomFlipVideo",
    "UniformCropVideo",
    "RandomCropVideo",
    "JitterScaleVideo",
    "NormalizeVideo",
]


@OBJECT_REGISTRY.register
class RandomFlipVideo(object):
    """Random flip images.

    Args:
        px: Horizontal flip probability, range between [0, 1].
    """

    def __init__(self, px: Optional[float] = 0.5):
        assert px >= 0 and px <= 1, "px must range between [0, 1]"
        self.px = px

    def __call__(self, data):
        img = data["img"]
        px = self.px
        flip = np.random.choice([False, True], p=[1 - px, px])
        if flip:
            img = img.flip(3)  # [c,t,h,w]
        else:
            img = img
        data["img"] = img
        return data


@OBJECT_REGISTRY.register
class UniformCropVideo(object):
    """Perform uniform spatial sampling on the images.

    Perform uniform spatial sampling on the image select the
    two ends of the long side and the middle position
    (left middle right or top middle bottom) 3 regions.

    Args:
        target_size(Union[int, Tuple[int]]): (w, h) of target size for crop.
    """

    def __init__(self, target_size: Union[int, Tuple[int]]):
        if isinstance(target_size, tuple):
            self.target_size = target_size
        elif isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            raise TypeError(
                f"target_size must be int or tuple[int], \
                    but got {type(target_size)}"
            )

    def __call__(self, data):

        img = data["img"]
        img_h, img_w = img.shape[2:]
        crop_w, crop_h = self.target_size
        if img_h > img_w:
            offsets = [
                (0, 0),
                (0, int(math.ceil((img_h - crop_h) / 2))),
                (0, img_h - crop_h),
            ]
        else:
            offsets = [
                (0, 0),
                (int(math.ceil((img_w - crop_w) / 2)), 0),
                (img_w - crop_w, 0),
            ]
        img_crops = []
        # [c,t,h,w]
        for x_offset, y_offset in offsets:
            crop = img[
                :,
                :,
                y_offset : y_offset + crop_h,
                x_offset : x_offset + crop_w,
            ]
            img_crops.append(crop)
        img_crops = torch.cat(img_crops, dim=1)
        data["img"] = img_crops
        return data


@OBJECT_REGISTRY.register
class RandomCropVideo(object):
    """Random crop images.

    Args:
        target_size(int): Random crop a square with the
            target_size from an image.
    """

    def __init__(self, target_size: int):
        self.target_size = target_size

    def __call__(self, data):
        img = data["img"]
        # [c,t,h,w]
        h, w = img.shape[2:]
        th, tw = self.target_size, self.target_size

        assert (w >= self.target_size) and (
            h >= self.target_size
        ), "image width({}) and height({}) should be larger than \
            crop size".format(
            w, h
        )

        crop_images = []
        x1 = np.random.randint(0, w - tw) if w - tw > 0 else 0
        y1 = np.random.randint(0, h - th) if h - th > 0 else 0
        crop_images = img[:, :, y1 : y1 + th, x1 : x1 + tw]  # [C, T, th, tw]
        data["img"] = crop_images
        return data


@OBJECT_REGISTRY.register
class JitterScaleVideo(object):
    """Scale image.

    Scale image while the target short size is randomly select between
    min_size and max_size.

    Args:
        min_size: Lower bound for random sampler.
        max_size: Higher bound for random sampler.
    """

    def __init__(
        self,
        min_size: int,
        max_size: int,
        short_cycle_factors: Tuple = (0.5, 0.7071),
        default_min_size: int = 256,
    ):
        self.default_min_size = default_min_size
        self.orig_min_size = self.min_size = min_size
        self.max_size = max_size
        self.short_cycle_factors = short_cycle_factors

    def __call__(self, data):
        short_cycle_idx = data.get("short_cycle_idx")
        if short_cycle_idx in [0, 1]:
            self.min_size = int(
                round(
                    self.short_cycle_factors[short_cycle_idx]
                    * self.default_min_size
                )
            )
        else:
            self.min_size = self.orig_min_size

        img = data["img"]
        size = int(round(np.random.uniform(self.min_size, self.max_size)))
        assert len(img) >= 1, "len(imgs): {} should be larger than 1".format(
            len(img)
        )

        height, width = img.shape[2:]
        if (width <= height and width == size) or (
            height <= width and height == size
        ):
            return data

        new_width = size
        new_height = size
        if width < height:
            new_height = int(math.floor((float(height) / width) * size))
        else:
            new_width = int(math.floor((float(width) / height) * size))

        frames_resize = F.interpolate(
            img,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
        data["img"] = frames_resize
        return data


@OBJECT_REGISTRY.register
class NormalizeVideo(object):
    """Normalization.

    Args:
        mean(Sequence[float]): mean values of different channels.
        std(Sequence[float]): std values of different channels.
        tensor_shape(list): size of mean, default [3,1,1].
            For slowfast, [1,1,1,3]
    """

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        tensor_shape: Tuple = (3, 1, 1),
    ):
        if not isinstance(mean, Sequence):
            raise TypeError(
                f"Mean must be list, tuple or np.ndarray, but got {type(mean)}"
            )
        if not isinstance(std, Sequence):
            raise TypeError(
                f"Std must be list, tuple or np.ndarray, but got {type(std)}"
            )
        self.mean = np.array(mean).reshape(tensor_shape).astype(np.float32)
        self.std = np.array(std).reshape(tensor_shape).astype(np.float32)

    def __call__(self, data):
        img = data["img"]

        norm_img = img / 255.0
        norm_img -= self.mean
        norm_img /= self.std
        norm_img = torch.as_tensor(norm_img).float()
        # thwc -> cthw
        norm_img = norm_img.permute(3, 0, 1, 2)
        data["img"] = norm_img
        return data
