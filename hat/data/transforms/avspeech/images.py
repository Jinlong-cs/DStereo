# Copyright (c) Horizon Robotics, All rights reserved.
# type: ignore
"""AVSPEECH(多模语音)处理图像数据转换.

目前该模块下共有十二个类:

ImageInterface          : 图像数据处理基类, 检查字段是否存在以及数据是否为空.
BrightnessContrast      : 调整图片的亮度和对比度.
CoarseDropout           : 随机丢弃序列图片矩形区域的信息.
HueSaturateValue        : 调整图片的 HSV.
FancyPCA                : 对图片做 PCA 数据扩充.
ToGray                  : 将图片转为灰度图.
ImageListStack          : 负责将图片的 tensor 或者 array 列表转换成 T*C*H*W 的 Tensor.
ImageListRandomFlip     : 将图片水平翻转.
ImageListRandomCrop     : 将图片随机裁剪.
TimeMask                : 对图片序列随机 Mask.
ImageListToYUV444       : 负责将 T*C*H*W BGR|RGB 格式的 Tensor 转换成 YUV44 格式的 Tensor.
ImageListNormalize      : 负责对 T*C*H*W 的 Tensor 进行归一化操作.

在调用方式上, 以 ImageListStack 为分界线.
BrightnessContrast、CoarseDropout、HueSaturateValue、FancyPCA、ToGray
在 ImageListStack 之前调用, 依次对图片列表中的每个 array 或 tensor 执行数据增强和变换.
TimeMask、ImageListToYUV444、ImageListRandomCrop、ImageListNormalize
在 ImageListStack 之后调用, 对 stack 后的大 tensor 做统一的数据增强和处理.
ImageListRandomFlip 在 ImageListStack 前后调用均可, 后者效率更高.
"""

import random
import warnings
from typing import List, Mapping, Optional

import cv2
import horizon_plugin_pytorch.nn.bgr_to_yuv444 as b2y
import numpy as np
import torch

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages
from .utils import (
    brightness_contrast_adjust,
    is_grayscale_image,
    is_rgb_image,
    to_tuple,
)

try:
    import torchvision.transforms as T
    from torchvision.transforms import functional as F
except ImportError:
    T = None
    F = None


@OBJECT_REGISTRY.register
class ImageInterface(object):
    """图像处理基类.

    要求调用时传入的data包含 "images" 键,
    要求子类实现transform接口.
    """

    def __call__(self, data):
        assert "images" in data, f"{__class__.__name__} use ``images`` in data"
        if data["images"] is None:
            return data
        data = self.transform(data)
        return data

    def transform(self, data):
        """所有子类都应该实现该接口."""
        raise NotImplementedError


@OBJECT_REGISTRY.register
class BrightnessContrast(ImageInterface):
    """调整序列图片的亮度和对比度.

    以一定概率对序列图片的亮度和对比度进行调整.
    **在``ImageListStack``之前调用.**

    Args:
        brightness_limit: 亮度调整幅度, 范围(-brightness_limit, +brightness_limit).
        contrast_limit: 对比度调整幅度, 范围(-contrast_limit, +contrast_limit).
        brightness_by_max: 如果设置为True, 按图片dtype的最大值调整.
                           如果设置为False, 按图片均值调整.
                           默认为True.
        p: 应用变换的概率值.
    """

    def __init__(
        self,
        brightness_limit=0.2,
        contrast_limit=0.2,
        brightness_by_max=True,
        p=0.5,
    ):
        self.brightness_limit = to_tuple(brightness_limit)
        self.contrast_limit = to_tuple(contrast_limit)
        self.brightness_by_max = brightness_by_max
        self.p = p

    def transform(
        self, data: Mapping[str, Optional[torch.Tensor]]
    ) -> Mapping[str, Optional[torch.Tensor]]:
        if random.random() > self.p:
            return data

        images = data["images"]
        self.alpha = 1.0 + random.uniform(
            self.contrast_limit[0], self.contrast_limit[1]
        )
        self.beta = 0.0 + random.uniform(
            self.brightness_limit[0], self.brightness_limit[1]
        )

        images = [
            brightness_contrast_adjust(
                img, self.alpha, self.beta, self.brightness_by_max
            )
            for img in images
        ]
        data["images"] = images
        data["images_lens"] = len(images)
        return data


@OBJECT_REGISTRY.register
class CoarseDropout(ImageInterface):
    """随机丢弃序列图片矩形区域的信息.

    以一定概率对序列图片在面积大小可选定、位置随机的矩形区域上丢失信息实现转换.
    **在``ImageListStack``之前调用.**

    Args:
        max_holes: 矩形块的最大个数.
        max_height: 矩形块的最大高度.
        max_width: 矩形块的最大宽度.
        min_holes: 矩形块的最小个数.
        min_height: 矩形块的最小高度.
        min_width: 矩形块的最小宽度.
        fill_value: 矩形块的填充值.
        p: 应用变换的概率值.
    """

    def __init__(
        self,
        max_holes=8,
        max_height=8,
        max_width=8,
        min_holes=None,
        min_height=None,
        min_width=None,
        fill_value=0,
        p=0.5,
    ):
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes if min_holes is not None else max_holes
        self.min_height = min_height if min_height is not None else max_height
        self.min_width = min_width if min_width is not None else max_width
        self.fill_value = fill_value
        self.p = p
        if not 0 < self.min_holes <= self.max_holes:
            raise ValueError(
                "Invalid combination of min_holes and max_holes. Got: {}".format(  # noqa E501
                    [min_holes, max_holes]
                )
            )

        self.check_range(self.max_height)
        self.check_range(self.min_height)
        self.check_range(self.max_width)
        self.check_range(self.min_width)

        if not 0 < self.min_height <= self.max_height:
            raise ValueError(
                "Invalid combination of min_height \
                    and max_height. Got: {}".format(
                    [min_height, max_height]
                )
            )
        if not 0 < self.min_width <= self.max_width:
            raise ValueError(
                "Invalid combination of min_width \
                    and max_width. Got: {}".format(
                    [min_width, max_width]
                )
            )

    def check_range(self, dimension):
        if isinstance(dimension, float) and not 0 <= dimension < 1.0:
            raise ValueError(
                "Invalid value {}. If using floats, the value \
                    should be in the range [0.0, 1.0)".format(
                    dimension
                )
            )

    def get_params_dependent_on_targets(self, img):
        height, width = img.shape[:2]

        holes = []
        for _n in range(random.randint(self.min_holes, self.max_holes)):
            if all(
                [
                    isinstance(self.min_height, int),
                    isinstance(self.min_width, int),
                    isinstance(self.max_height, int),
                    isinstance(self.max_width, int),
                ]
            ):
                hole_height = random.randint(self.min_height, self.max_height)
                hole_width = random.randint(self.min_width, self.max_width)
            elif all(
                [
                    isinstance(self.min_height, float),
                    isinstance(self.min_width, float),
                    isinstance(self.max_height, float),
                    isinstance(self.max_width, float),
                ]
            ):
                hole_height = int(
                    height * random.uniform(self.min_height, self.max_height)
                )
                hole_width = int(
                    width * random.uniform(self.min_width, self.max_width)
                )
            else:
                raise ValueError(
                    "Min width, max width, \
                    min height and max height \
                    should all either be ints or floats. \
                    Got: {} respectively".format(
                        [
                            type(self.min_width),
                            type(self.max_width),
                            type(self.min_height),
                            type(self.max_height),
                        ]
                    )
                )

            y1 = random.randint(0, height - hole_height)
            x1 = random.randint(0, width - hole_width)
            y2 = y1 + hole_height
            x2 = x1 + hole_width
            holes.append((x1, y1, x2, y2))

        return holes

    def cutout(self, img, holes, fill_value=0):
        img = img.copy()
        for x1, y1, x2, y2 in holes:
            img[y1:y2, x1:x2] = fill_value
        return img

    def transform(
        self, data: Mapping[str, Optional[torch.Tensor]]
    ) -> Mapping[str, Optional[torch.Tensor]]:
        images = data["images"]
        holes = self.get_params_dependent_on_targets(images[0])
        if random.random() < self.p:
            images = [
                self.cutout(img, holes, self.fill_value) for img in images
            ]
        data["images"] = images
        return data


@OBJECT_REGISTRY.register
class HueSaturateValue(ImageInterface):
    """调整序列图片的HSV.

    以一定概率对序列图片的色调和饱和度进行调整.
    **在``ImageListStack``之前调用.**

    Args:
        hue_shift_limit: 色相调整幅度.
        sat_shift_limit: 饱和度调整幅度.
        val_shift_limit: 明度调整幅度.
        p: 应用变换的概率值.
    """

    def __init__(
        self,
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.5,
    ):
        self.hue_shift_limit = to_tuple(hue_shift_limit)
        self.sat_shift_limit = to_tuple(sat_shift_limit)
        self.val_shift_limit = to_tuple(val_shift_limit)
        self.p = p

    def _shift_hsv_uint8(self, img, hue_shift, sat_shift, val_shift):
        dtype = img.dtype
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue, sat, val = cv2.split(img)

        if hue_shift != 0:
            lut_hue = np.arange(0, 256, dtype=np.int16)
            lut_hue = np.mod(lut_hue + hue_shift, 180).astype(dtype)
            hue = cv2.LUT(hue, lut_hue)

        if sat_shift != 0:
            lut_sat = np.arange(0, 256, dtype=np.int16)
            lut_sat = np.clip(lut_sat + sat_shift, 0, 255).astype(dtype)
            sat = cv2.LUT(sat, lut_sat)

        if val_shift != 0:
            lut_val = np.arange(0, 256, dtype=np.int16)
            lut_val = np.clip(lut_val + val_shift, 0, 255).astype(dtype)
            val = cv2.LUT(val, lut_val)

        img = cv2.merge((hue, sat, val)).astype(dtype)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img

    def _shift_hsv_non_uint8(self, img, hue_shift, sat_shift, val_shift):
        dtype = img.dtype
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue, sat, val = cv2.split(img)

        if hue_shift != 0:
            hue = cv2.add(hue, hue_shift)
            hue = np.mod(hue, 360)
            hue = hue.astype(dtype)

        if sat_shift != 0:
            sat = np.clip(cv2.add(sat, sat_shift), 0, 1.0).astype(dtype)

        if val_shift != 0:
            val = np.clip(cv2.add(val, val_shift), 0, 1.0).astype(dtype)

        img = cv2.merge((hue, sat, val)).astype(dtype)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img

    def shift_hsv(self, img, hue_shift, sat_shift, val_shift):
        if hue_shift == 0 and sat_shift == 0 and val_shift == 0:
            return img
        is_gray = is_grayscale_image(img)
        if is_gray:
            if hue_shift != 0 or sat_shift != 0:
                hue_shift = 0
                sat_shift = 0
                warnings.warn(
                    "HueSaturateValue: hue_shift and sat_shift \
                        are not applicable to grayscale image. "
                    "Set them to 0 or use RGB image"
                )
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if img.dtype == np.uint8:
            img = self._shift_hsv_uint8(img, hue_shift, sat_shift, val_shift)
        else:
            img = self._shift_hsv_non_uint8(
                img, hue_shift, sat_shift, val_shift
            )

        if is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        return img

    def transform(
        self, data: Mapping[str, Optional[torch.Tensor]]
    ) -> Mapping[str, Optional[torch.Tensor]]:
        if random.random() > self.p:
            return data

        images = data["images"]
        self.hue_shift = random.uniform(
            self.hue_shift_limit[0], self.hue_shift_limit[1]
        )
        self.sat_shift = random.uniform(
            self.sat_shift_limit[0], self.sat_shift_limit[1]
        )
        self.val_shift = random.uniform(
            self.val_shift_limit[0], self.val_shift_limit[1]
        )
        images = [
            self.shift_hsv(
                img, self.hue_shift, self.sat_shift, self.val_shift
            ).astype(np.uint8)
            for img in images
        ]
        data["images"] = images
        return data


@OBJECT_REGISTRY.register
class FancyPCA(ImageInterface):
    """对图片做PCA图像扩充.

    以一定概率对序列图片用PCA方法进行图像扩充.
    **在``ImageListStack``之前调用.**

    Args:
        alpha: 高斯分布的标准差, 默认为0.1.
        p: 应用变换的概率值.
    """

    def __init__(self, alpha=0.1, p=0.5):
        self.alpha = alpha
        self.p = p

    def fancy_pca(self, img, alpha=0.1):
        if not is_rgb_image(img) or img.dtype != np.uint8:
            raise TypeError("Image must be RGB image in uint8 format.")

        orig_img = img.astype(float).copy()
        img = img / 255.0
        img_rs = img.reshape(-1, 3)
        img_centered = img_rs - np.mean(img_rs, axis=0)
        img_cov = np.cov(img_centered, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(img_cov)
        sort_perm = eig_vals[::-1].argsort()
        eig_vals[::-1].sort()
        eig_vecs = eig_vecs[:, sort_perm]
        m1 = np.column_stack((eig_vecs))
        m2 = np.zeros((3, 1))
        m2[:, 0] = alpha * eig_vals[:]
        add_vect = np.matrix(m1) * np.matrix(m2)

        for idx in range(3):  # RGB
            orig_img[..., idx] += add_vect[idx] * 255
        orig_img = np.clip(orig_img, 0.0, 255.0)
        orig_img = orig_img.astype(np.uint8)

        return orig_img

    def transform(
        self, data: Mapping[str, Optional[torch.Tensor]]
    ) -> Mapping[str, Optional[torch.Tensor]]:
        if random.random() < self.p:
            images = data["images"]
            alpha = random.gauss(0, self.alpha)
            images = [self.fancy_pca(img, alpha) for img in images]
            data["images"] = images
        return data


@OBJECT_REGISTRY.register
class ToGray(ImageInterface):
    """将 RGB 或 BGR 格式的图片转换为灰度图.

    将序列中 RGB 或 BGR 格式的图片转换为灰度图.
    **在``ImageListStack``之前调用.**

    Args:
        p: 应用变换的概率值.
        rgb_data: 如果设置为True, 传入的图片是RGB格式.
                  如果设置为False, 传入的图片是BGR格式.
                  默认为True.
    """

    def __init__(self, p: float = 0.2, rgb_data: bool = True):
        self.p = p
        self.rgb_data = rgb_data

    def _do_gray(self, images):
        new_images = []
        for img in images:
            new_img = np.zeros(img.shape, dtype=np.uint8)
            if self.rgb_data:
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for i in range(new_img.shape[-1]):
                new_img[:, :, i] = gray_img
            new_images.append(new_img)
        return new_images

    def transform(self, data):
        images = data["images"]
        do_gray = np.random.choice([False, True], p=[1 - self.p, self.p])
        if images is not None and do_gray:
            images = self._do_gray(images)
        data["images"] = images
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"p_dogray={self.p}"
        return repr_str


@OBJECT_REGISTRY.register
class ImageListStack(ImageInterface):
    """将图片列表堆叠成Tensor.

    通过 "images" 获取到保存图片的array或者tensor列表.
    将表示图片列表的 List[np.array] 或者 List[Tensor] 的每一个元素堆叠到一起.
    在 dim=0 的维度上进行扩充 T*... 维度的 Tensor.

    Args:
        hwc2chw: 如果设置为True, 图片数组的维度含义是 (Tx)HxWxC,
                 并将其维度转变成 (Tx)CxHxW.
                 如果设置为False, 不做任何变换.
                 默认为True.
    """

    def __init__(self, hwc2chw: bool = True):
        self.hwc2chw = hwc2chw

    def transform(
        self, data: Mapping[str, Optional[torch.Tensor]]
    ) -> Mapping[str, Optional[torch.Tensor]]:
        images = [
            image if isinstance(image, torch.Tensor) else torch.tensor(image)
            for image in data["images"]
        ]
        images = torch.stack(images, dim=0)
        if self.hwc2chw:
            images = images.permute(0, 3, 1, 2)
        data["images"] = images
        data["images_lens"] = images.size(0)
        return data


class ImageListRandomFlip(ImageInterface):
    """将图片水平翻转.

    以一定概率将图片序列水平翻转.
    **在``ImageListStack``前后都可调用, 后者效率更高.**

    Args:
        p: 应用变换的概率值.
    """

    @require_packages("torchvision")
    def __init__(self, p):
        self.p = p

    def transform(self, data):
        if random.random() > self.p:
            return data
        images = data["images"]
        if isinstance(images, list):
            images = [F.hflip(img) for img in images]
        elif isinstance(images, torch.Tensor):
            n, c, h, w = images.shape
            images = images.reshape(n * c, h, w)
            images = F.hflip(images)
            images = images.reshape(n, c, h, w)
        data["images"] = images
        return data


class ImageListRandomCrop(ImageInterface):
    """将图片随机裁剪.

    以一定概率将图片序列随机裁剪, 并调整为指定的大小.
    **在``ImageListStack``之后调用.**

    Args:
        p: 应用变换的概率值.
        scale: 随机裁剪区域的上下界比例.
        ratio: 随机裁剪长宽比的上下界比例.
        size: 目标图片的形状.
    """

    @require_packages("torchvision")
    def __init__(
        self, p, scale=(0.9, 1.0), ratio=(3 / 4, 4 / 3), size=(96, 96)
    ):
        assert p >= 0.0 and p <= 1.0
        self.p = p
        self._transform = T.RandomResizedCrop(
            size=size, scale=scale, ratio=ratio
        )

    def transform(self, data):
        if random.random() > self.p:
            return data
        images = data["images"]
        assert isinstance(images, torch.Tensor)
        n, c, h, w = images.shape
        images = images.reshape(n * c, h, w)
        images = self._transform(images)
        images = images.reshape(n, c, h, w)
        data["images"] = images
        return data


class TimeMask(ImageInterface):
    """图片序列随机Mask.

    将图片序列在时间维度以一定概率随机Mask.
    **在``ImageListStack``之后调用.**

    Args:
        p: 应用变换的概率值.
        max_frame: Mask的最大长度.
        num_mask: 随机Mask的次数.
        replace_with_zero: 如果设置为True, Mask值设置为128.
                           如果设置为False, Mask值设置为图片数据的均值.
                           默认为False.
    """

    def __init__(
        self, p=0.5, max_frame=30, num_mask=2, replace_with_zero=False
    ):
        self.max_frame = max_frame
        self.num_mask = num_mask
        self.replace_with_zero = replace_with_zero
        self.p = p

    def transform(self, data):
        if random.random() > self.p:
            return data
        images = data["images"]
        if self.replace_with_zero:
            mask_value = 128
        else:
            mask_value = images.float().mean()

        # time mask
        total_frames = images.shape[0]
        for _ in range(self.num_mask):
            start = random.randint(0, total_frames - 1)
            length = random.randint(1, self.max_frame)
            end = min(total_frames, start + length)
            images[start:end, :] = mask_value

        data["images"] = images
        return data


@OBJECT_REGISTRY.register
class ImageListToYUV444(ImageInterface):
    """将图片序列转成YUV444格式.

    将用RGB或者BGR通道格式表示的图片Tensor(TxCxHxW)转变成YUV444图片格式的图片Tensor.
    **在``ImageListStack``之后调用.**

    Args:
        layout: 图片的R,G,B三个通道的排布顺序, 取值范围是 'bgr' 或者 'rgb'.
                默认是'bgr'.
    """

    def __init__(self, layout: str = "bgr"):
        layout = layout.lower()
        assert layout in ["bgr", "rgb"], f"{layout} not include ['bgr', 'rgb']"
        self.layout = layout
        self.channel_reversal = layout == "rgb"

    def transform(
        self, data: Mapping[str, Optional[torch.Tensor]]
    ) -> Mapping[str, Optional[torch.Tensor]]:
        data["images"] = b2y.bgr_to_yuv444(
            data["images"], self.channel_reversal
        )
        return data


@OBJECT_REGISTRY.register
class ImageListNormalize(ImageInterface):
    r"""图片序列归一化.

    对表示图片序列的Tensor(TxCxHxW)按照维度进行减均值除标准差的归一化操作.
    **在``ImageListStack``之后调用.**

    .. math::

        x = \frac{x - mean}{std}

    Args:
        mean: 均值列表, 长度为3.
        std: 标准差列表, 长度为3.
        inplace: 当设置为True时, 图片做完变换后仍保存在原存储空间.
                 当设置为False时, 图片做完变换后会保存到新的存储空间.
                 默认为False.
    """

    @require_packages("torchvision")
    def __init__(
        self, mean: List[float], std: List[float], inplace: bool = False
    ):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def transform(
        self, data: Mapping[str, Optional[torch.Tensor]]
    ) -> Mapping[str, Optional[torch.Tensor]]:
        images = data["images"]
        if not images.is_floating_point():
            images = images.to(dtype=torch.float32)
        data["images"] = F.normalize(
            images, self.mean, self.std, inplace=self.inplace
        )
        return data


class DownSampleAugment(object):
    """图片序列下采样.

    将图片下采样然后再上采样，以此来模拟低分辨率的效果。

    Args:
        p: 应用变换的概率值.
        min_size: 下采样的最小尺寸.
        max_size: 下采样的最大尺寸.
    """

    def __init__(self, p: float, min_size: tuple, max_size) -> None:
        self.p = p
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, data):
        if random.random() > self.p:
            return data

        images = data["images"]
        origin_size = tuple(images[0].shape[:2])
        resize_size = (
            int(random.uniform(self.min_size[0], self.max_size[0])),
            int(random.uniform(self.min_size[1], self.max_size[1])),
        )
        new_images = [
            cv2.resize(img, resize_size, interpolation=cv2.INTER_AREA)
            for img in images
        ]
        new_images = [
            cv2.resize(img, origin_size, interpolation=cv2.INTER_LINEAR)
            for img in new_images
        ]
        data["images"] = new_images

        return data


class ImageListSpatialMask(object):
    """图片序列空间遮挡.

    对图片序列随机位置（前后帧保持一致）进行遮挡，以模拟实际有物体遮挡的情况。
    **在``ImageListStack``之后调用.**

    Args:
        p: 应用变换的概率值.
        mode: 遮挡模式，可选值为 ['zero', 'mean', 'random'].
        min_width: 遮挡的最小宽度.
        max_width: 遮挡的最大宽度.
    """

    def __init__(
        self, p: float, mode: str, min_width: int = 20, max_width: int = 40
    ):
        self.p = p
        self.mode = mode
        self.min_width = min_width
        self.max_width = max_width

    def __call__(self, data):
        if random.random() > self.p:
            return data

        images = data["images"]

        assert isinstance(images, torch.Tensor)
        n, c, h, w = images.shape
        mask_width = int(random.uniform(self.min_width, self.max_width))
        assert mask_width < w
        if random.random() > 0.5:
            left, right = 0, mask_width
        else:
            left, right = w - mask_width, w

        if self.mode == "zero":
            images[:, :, :, left:right] = 0
        elif self.mode == "mean":
            value = images.to(torch.float).mean(dim=(2, 3), keepdim=True)
            images[:, :, :, left:right] = value.to(torch.uint8)
        elif self.mode == "random":
            images[:, :, :, left:right] = torch.randint(
                0, 256, size=(n, c, h, mask_width)
            )
        else:
            raise NotImplementedError

        data["images"] = images

        return data
