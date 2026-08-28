#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os, copy, math
import operator
import traceback
import logging
import random
import time
import os.path as osp
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data.dataset import Dataset, ConcatDataset
from torchvision import transforms

from .list_dataset import ListDataset, DrivingStereoDataset
from .resize_aware import ResizeAwareStereo

logger = logging.getLogger(__name__)
__all__ = ["AugDataset", "Augmentor", "Resizor", "Cropper", "Normalizor", "Identity"]


class Augmentor:
    def __init__(
        self,
        gray_p,
        color_p,
        spatial_p,
        occlusion_p,
        seed=0,
    ):
        super().__init__()
        self.rng = np.random.RandomState(seed)
        self.gray_p = gray_p
        self.color_p = color_p
        self.spatial_p = spatial_p
        self.occlusion_p = occlusion_p

    def chromatic_augmentation(self, img):
        random_brightness = np.random.uniform(0.8, 1.2)
        random_contrast = np.random.uniform(0.8, 1.2)
        random_gamma = np.random.uniform(0.8, 1.2)
        img = Image.fromarray(img)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random_brightness)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random_contrast)
        gamma_map = [
            255 * 1.0 * pow(ele / 255.0, random_gamma) for ele in range(256)
        ] * 3
        # use PIL's point-function to accelerate this part
        img = img.point(gamma_map)
        img_ = np.array(img)
        return img_

    def __call__(self, x):
        left_img, right_img, disp = x
        # 1. chromatic augmentation
        if random.random() < self.gray_p:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            left_img = np.repeat(left_img[..., None], 3, axis=-1)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            right_img = np.repeat(right_img[..., None], 3, axis=-1)
        if random.random() < self.color_p:
            left_img = self.chromatic_augmentation(left_img)
            right_img = self.chromatic_augmentation(right_img)

        # 2. spatial augmentation,rotate & vertical shift for right image
        if random.random() < self.spatial_p:
            angle, pixel = 0.1, 2
            px = self.rng.uniform(-pixel, pixel)
            ag = self.rng.uniform(-angle, angle)
            image_center = (
                self.rng.uniform(0, right_img.shape[0]),
                self.rng.uniform(0, right_img.shape[1]),
            )
            rot_mat = cv2.getRotationMatrix2D(image_center, ag, 1.0)
            right_img = cv2.warpAffine(
                right_img, rot_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            )
            trans_mat = np.float32([[1, 0, 0], [0, 1, px]])
            right_img = cv2.warpAffine(
                right_img, trans_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            )

        # 3. add random occlusion to right image
        if random.random() < self.occlusion_p:
            sx = int(self.rng.uniform(3, 20))  # change the value
            sy = int(self.rng.uniform(3, 20))
            cx = int(self.rng.uniform(sx, right_img.shape[0] - sx))
            cy = int(self.rng.uniform(sy, right_img.shape[1] - sy))
            right_img[cx - sx : cx + sx, cy - sy : cy + sy] = np.mean(
                np.mean(right_img, 0), 0
            )[np.newaxis, np.newaxis]

        return left_img, right_img, disp


class Resizor:
    def __init__(
        self, nh, nw, rand_resize=False, scale=1.0, min_scale=None, max_scale=None
    ):
        assert isinstance(nh, int)
        assert isinstance(nw, int)
        assert isinstance(rand_resize, bool)
        if rand_resize:
            assert nh, nw == (-1, -1)
        self.nh = nh
        self.nw = nw
        self.scale = scale
        self.rand_resize = rand_resize
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, x):
        left, right, disp_left = x
        h, w = left.shape[:2]
        if self.rand_resize:
            if self.min_scale is not None:
                s = np.random.uniform(self.min_scale, self.max_scale)
            else:
                s = np.random.uniform(1.0, 1.3) * self.scale
            nh, nw = int(round(h * s)), int(round(w * s))
        else:
            # 1088 * 864 --> 640 * 508
            ratio = w / h
            # 根据宽高比计算目标高度（保持比例情况下的对应高度）
            nh = int(self.nw / ratio)
            nw = self.nw
            # print("origin %d * %d, to %d * %d" % (h, w, nh, nw))
        left = cv2.resize(left, (nw, nh))
        right = cv2.resize(right, (nw, nh))
        disp_left = cv2.resize(disp_left, (nw, nh), interpolation=cv2.INTER_NEAREST)
        disp_left = disp_left * nw / w
        return left, right, disp_left


class Normalizor:
    @staticmethod
    def _img_zscore(img, eps=1e-5):
        img = np.array(img).astype(np.float32)
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        # 220904: add eps to avoid the input is nan
        r = (r - np.mean(r[:])) / (np.std(r[:]) + eps)
        g = (g - np.mean(g[:])) / (np.std(g[:]) + eps)
        b = (b - np.mean(b[:])) / (np.std(b[:]) + eps)
        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
        return img

    def __init__(self, norm_type=None):
        assert norm_type in [
            None,
            "z-score",
            "imagenet-rgb",
            "imagenet-bgr",
            "MixVarGENet",
        ]
        self.norm_type = norm_type
        if self.norm_type == "imagenet-rgb":
            normalize = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
            self.T = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize),
                ]
            )
        elif self.norm_type == "imagenet-bgr":
            normalize = {"mean": [0.406, 0.456, 0.485], "std": [0.225, 0.224, 0.229]}
            self.T = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize),
                ]
            )

    def __call__(self, x):
        left, right, disp = x
        if self.norm_type is None:
            pass
        elif self.norm_type == "z-score":
            left = self._img_zscore(left)
            right = self._img_zscore(right)
        elif self.norm_type == "imagenet-rgb" or self.norm_type == "imagenet-bgr":
            assert left.dtype == np.uint8
            left = self.T(left)
            right = self.T(right)
            left = left.numpy()
            right = right.numpy()
            left = np.transpose(left, (1, 2, 0))
            right = np.transpose(right, (1, 2, 0))
        elif self.norm_type == "MixVarGENet":
            left = np.array(left).astype(np.float32)
            left = left / 128.0 - 1.0
            right = np.array(right).astype(np.float32)
            right = right / 128.0 - 1.0
        else:
            raise ValueError
        return left, right, disp


class Cropper:
    def __init__(self, crop_type, crop_height, crop_width):
        assert crop_type in ["center", "random"]
        self.crop_type = crop_type
        self.crop_size = crop_height, crop_width

    def random_crop(self, inputs):
        inputs = list(inputs)
        crop_height, crop_width = self.crop_size
        height, width = inputs[0].shape[:2]
        if height - crop_height > 0:
            top = random.randint(0, height - crop_height)
        else:
            top = 0
        if width - crop_width > 0:
            left = random.randint(0, width - crop_width)
        else:
            left = 0
        right, bottom = left + crop_width, top + crop_height
        for i in range(len(inputs)):
            inputs[i] = inputs[i][top:bottom, left:right]
            assert (
                inputs[i].shape[:2] == self.crop_size
            ), f"{inputs[i].shape[:2]},{self.crop_size}"
        return inputs

    def center_crop(self, inputs):
        inputs = list(inputs)
        crop_height, crop_width = self.crop_size
        height, width = inputs[0].shape[:2]
        if height - crop_height > 0:
            top = int((height - crop_height + 1) * 0.5)
        else:
            top = 0
        if width - crop_width > 0:
            left = int((width - crop_width + 1) * 0.5)
        else:
            left = 0
        right, bottom = left + crop_width, top + crop_height
        for i in range(len(inputs)):
            inputs[i] = inputs[i][top:bottom, left:right]
            assert inputs[i].shape[:2] == self.crop_size
        return inputs

    def padding(self, x):
        x = list(x)
        crop_height, crop_width = self.crop_size
        left, right, disp = x
        h, w = left.shape[:2]
        if h >= crop_height and w >= crop_width:
            return x
        # logger.info("padding H: %d, W: %d" % (crop_height - h, crop_width - w))
        if h < crop_height:
            for i in range(3):
                xi_shape = list(x[i].shape)
                xi_shape[0] = crop_height
                temp = np.zeros(xi_shape, dtype=x[i].dtype)
                temp[: x[i].shape[0], : x[i].shape[1]] = x[i]
                x[i] = temp
        if w < crop_width:
            for i in range(3):
                xi_shape = list(x[i].shape)
                xi_shape[1] = crop_width
                temp = np.zeros(xi_shape, dtype=x[i].dtype)
                temp[: x[i].shape[0], : x[i].shape[1]] = x[i]
                x[i] = temp
        return x

    def __call__(self, x):
        x = self.padding(x)
        if self.crop_type == "center":
            x = self.center_crop(x)
        elif self.crop_type == "random":
            x = self.random_crop(x)
        return x


class Identity:
    def __call__(self, x):
        return x


class AugDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        test_mode,
        max_disp,
        aug_args=None,
        res_args=None,
        norm_args=None,
        crop_args=None,
        debug=False,
        img_open_mode="bgr",
        skip=False,
        *,
        resize_aware_args=None,
    ):
        super().__init__()
        if isinstance(base_dataset, str):
            self.base_dataset = ListDataset(
                base_dataset, debug=debug, img_open_mode=img_open_mode
            )
        else:
            self.base_dataset = base_dataset
        self.test_mode = test_mode
        self.augmentor = Augmentor(*aug_args) if aug_args else Identity()
        self.resizor = Resizor(*res_args) if res_args else Identity()
        self.normalizer = Normalizor(*norm_args) if norm_args else Identity()
        self.cropper = Cropper(*crop_args) if crop_args else Identity()
        if resize_aware_args is not None:
            resize_aware_args = dict(resize_aware_args)
            resize_max_disp = float(
                resize_aware_args.get("max_disp", max_disp)
            )
            if not math.isclose(resize_max_disp, float(max_disp)):
                raise ValueError(
                    "resize-aware max_disp must match AugDataset max_disp: "
                    f"{resize_max_disp} != {max_disp}"
                )
            resize_aware_args["max_disp"] = max_disp
            self.resize_aware = ResizeAwareStereo(**resize_aware_args)
            canonical_shape = (
                self.resize_aware.base_height,
                self.resize_aware.base_width,
            )
            if (
                not isinstance(self.cropper, Identity)
                and self.cropper.crop_size != canonical_shape
            ):
                raise ValueError(
                    "resize-aware crop size must match canonical shape: "
                    f"{self.cropper.crop_size} != {canonical_shape}"
                )
        else:
            self.resize_aware = None
        self.debug = debug
        self.max_disp = max_disp

    def __getitem__(self, i):
        sample_index, resize_scale = self._split_sample_index(i)
        data = {}
        x = self.base_dataset[sample_index]
        x = self.resizor(x)
        if type(self.cropper) == Identity:
            data["origin_shape"] = x[0].shape[:2]
            l = self.base_dataset.pad_image_to_multiple_of_32(x[0])
            r = self.base_dataset.pad_image_to_multiple_of_32(x[1])
            d = self.base_dataset.pad_image_to_multiple_of_32(x[2], value=0)
            x = (l, r, d)
        else:
            x = self.cropper(x)
            data["origin_shape"] = x[0].shape[:2]
        x = self.augmentor(x)

        resize_mask_flag = None
        metric_gt_disp = None
        if self.resize_aware is not None:
            if resize_scale is None:
                if len(self.resize_aware.scales) != 1:
                    raise ValueError(
                        "multi-scale resize-aware AugDataset requires a "
                        "(sample_index, scale) index"
                    )
                resize_scale = self.resize_aware.scales[0]
            if self.test_mode:
                metric_gt_disp = np.asarray(x[2], dtype=np.float32).copy()
            resize_mask_flag = self._resize_mask_flag(x[2])
            left, right, disparity, metadata = self.resize_aware(
                x[0], x[1], x[2], resize_scale
            )
            x = (left, right, disparity)
            data.update(metadata)
            data["origin_shape"] = metadata["resize_content_shape"]
        elif resize_scale is not None:
            raise ValueError(
                "received a scale-tagged index while resize-aware mode "
                "is disabled"
            )

        left_x5_nv12 = self._bgr2nv12(x[0])
        left_x5_nv12 = np.ascontiguousarray(left_x5_nv12)
        left = self._nv12Toyuv444(left_x5_nv12, *x[0].shape[:2])
        right_x5_nv12 = self._bgr2nv12(x[1])
        right_x5_nv12 = np.ascontiguousarray(right_x5_nv12)
        right = self._nv12Toyuv444(right_x5_nv12, *x[0].shape[:2])

        data["left_img"] = x[0]  # cv2.cvtColor(x[0], cv2.COLOR_RGB2BGR)
        data["right_img"] = x[1]  # cv2.cvtColor(x[1], cv2.COLOR_RGB2BGR)
        data["left_img_yuv"] = left  # cv2.cvtColor(x[0], cv2.COLOR_RGB2BGR)
        data["right_img_yuv"] = right  # cv2.cvtColor(x[1], cv2.COLOR_RGB2BGR)
        data["sample_idx"] = sample_index
        data["left_img_name"] = (
            self.base_dataset.file_list[sample_index][0]
            if isinstance(self.base_dataset.file_list[sample_index], list)
            else self.base_dataset.file_list[sample_index]
        )
        data["right_img_name"] = (
            self.base_dataset.file_list[sample_index][1]
            if isinstance(self.base_dataset.file_list[sample_index], list)
            else self.base_dataset.file_list[sample_index]
        )
        data["data_root"] = self.base_dataset.root_dir

        x = self.normalizer((left, right, x[2]))
        l, r, g = x

        g[np.isnan(g)] = 0
        g[np.isinf(g)] = 0
        l = torch.from_numpy(np.transpose(l, (2, 0, 1))).float().contiguous()
        r = torch.from_numpy(np.transpose(r, (2, 0, 1))).float().contiguous()
        g = torch.from_numpy(g).float().contiguous()
        if self.resize_aware is not None:
            # Float DStereo predictions are [N, 1, H, W].  Keep this
            # experiment's labels channel-aligned without changing the legacy
            # dataset contract used by the canonical training profile.
            g = g.unsqueeze(0)
            if metric_gt_disp is not None:
                metric_gt_disp = np.nan_to_num(
                    metric_gt_disp,
                    copy=False,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                data["metric_gt_disp"] = (
                    torch.from_numpy(metric_gt_disp)
                    .float()
                    .contiguous()
                    .unsqueeze(0)
                )

        # assert l.shape[-2:] == self.cropper.crop_size, self.base_dataset.file_list[i]
        # assert r.shape[-2:] == self.cropper.crop_size, self.base_dataset.file_list[i]
        # assert g.shape == self.cropper.crop_size, self.base_dataset.file_list[i]
        img = torch.stack([l, r], dim=0)
        data["img"] = img
        data["gt_disp"] = g
        data["mask_flag"] = (
            resize_mask_flag if resize_mask_flag is not None else True
        )
        data["dataset_name"] = self.base_dataset.name

        if resize_mask_flag is None:
            # 判断小于0或大于100的元素
            condition = (g <= 0) | (g > self.max_disp)

            # 计算符合条件的元素个数
            count = torch.sum(condition).item()

            # 计算矩阵中的总元素个数
            total_elements = g.numel()

            # 计算符合条件的元素占比
            ratio = count / total_elements

            if ratio > 0.33 and not self.test_mode:
                data["mask_flag"] = False

        return data

    def __len__(self):
        return len(self.base_dataset)

    @staticmethod
    def _split_sample_index(index):
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError(
                    "scale-tagged indices must be "
                    "(sample_index, scale) pairs"
                )
            return operator.index(index[0]), float(index[1])
        return operator.index(index), None

    def _resize_mask_flag(self, disparity):
        if self.test_mode:
            return True
        disparity = np.asarray(disparity)
        invalid = (
            (~np.isfinite(disparity))
            | (disparity <= 0)
            | (disparity >= self.max_disp)
        )
        return np.count_nonzero(invalid) / disparity.size <= 0.33

    def _test_opencv(self, data, channel_reversal):
        """
        data: [H, W, 3]
        channel_reversal: True if data is RGB else False
        """
        data = copy.deepcopy(data)
        # data = np.transpose(data, [0, 2, 3, 1])

        def _cvt_one_image(image):
            if channel_reversal:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
            return self._yu12_to_yuv444(image)

        if len(data.shape) == 4:
            out = np.array([_cvt_one_image(image) for image in data])
        else:
            out = _cvt_one_image(data)
        # out = np.transpose(out, [0, 3, 1, 2])
        return out

    def _yu12_to_yuv444(self, img_yuv420sp):
        img_h = int(img_yuv420sp.shape[0] * 2 / 3)
        img_w = int(img_yuv420sp.shape[1])
        uv_start_idx = img_h * img_w
        v_size = int(img_h * img_w / 4)
        img_yuv420sp = img_yuv420sp.flatten()
        img_y = img_yuv420sp[:uv_start_idx].reshape((img_h, img_w, 1))
        uv_end_idx = uv_start_idx + v_size
        img_u = img_yuv420sp[uv_start_idx:uv_end_idx]
        img_u = img_u.reshape(
            int(math.ceil(img_h / 2.0)), int(math.ceil(img_w / 2.0)), 1
        )
        img_u = np.repeat(img_u, 2, axis=0)
        img_u = np.repeat(img_u, 2, axis=1)
        v_start_idx = uv_start_idx + v_size
        v_end_idx = uv_start_idx + 2 * v_size
        img_v = img_yuv420sp[v_start_idx:v_end_idx]
        img_v = img_v.reshape(
            int(math.ceil(img_h / 2.0)), int(math.ceil(img_w / 2.0)), 1
        )
        img_v = np.repeat(img_v, 2, axis=0)
        img_v = np.repeat(img_v, 2, axis=1)
        img_yuv444 = np.concatenate((img_y, img_u, img_v), axis=2)
        return img_yuv444

    def _bgr2nv12(self, image):
        image = copy.deepcopy(image)
        image = image.astype(np.uint8)
        height, width = image.shape[0], image.shape[1]
        frame_size = width * height
        nv12 = np.zeros(frame_size + frame_size // 2, dtype=np.int32)

        # Reshape BGR24 to height x width x 3
        bgr24_reshaped = image.reshape((height, width, 3)).astype(np.int32)
        b, g, r = bgr24_reshaped[..., 0], bgr24_reshaped[..., 1], bgr24_reshaped[..., 2]

        # Compute Y plane
        y_plane = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16
        y_plane = np.clip(y_plane, 0, 255).astype(np.uint8)
        nv12[:frame_size] = y_plane.flatten()

        uv_b = b[::2, ::2]
        uv_g = g[::2, ::2]
        uv_r = r[::2, ::2]
        # Compute UV plane for even rows
        uv_plane = np.zeros((height * width // 2), dtype=np.int32)

        u = ((-38 * uv_r - 74 * uv_g + 112 * uv_b + 128) >> 8) + 128
        v = ((112 * uv_r - 94 * uv_g - 18 * uv_b + 128) >> 8) + 128

        u = np.clip(u, 0, 255).astype(np.uint8).flatten()
        v = np.clip(v, 0, 255).astype(np.uint8).flatten()

        uv_plane[0::2] = u
        uv_plane[1::2] = v

        nv12[frame_size:] = uv_plane.flatten()

        return nv12.astype(np.uint8)

    def _nv12Toyuv444(self, data, height, width):
        data = copy.deepcopy(data)
        nv12_data = data.flatten()
        yuv444 = np.empty([height, width, 3], dtype=np.uint8)
        yuv444[:, :, 0] = nv12_data[: width * height].reshape(height, width)
        u = nv12_data[width * height :: 2].reshape(height // 2, width // 2)
        yuv444[:, :, 1] = Image.fromarray(u).resize((width, height), resample=0)
        v = nv12_data[width * height + 1 :: 2].reshape(height // 2, width // 2)
        yuv444[:, :, 2] = Image.fromarray(v).resize((width, height), resample=0)
        data = yuv444.astype(np.uint8)
        # if yuv444_output_layout == "CHW":
        #     data = np.transpose(data, (2, 0, 1))
        return data

    def _rgb2nv12(self, data):
        image = copy.deepcopy(data)
        image = image.astype(np.uint8)
        height, width = image.shape[0], image.shape[1]
        yuv420p = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420).reshape(
            (height * width * 3 // 2,)
        )
        y = yuv420p[: height * width]
        uv_planar = yuv420p[height * width :].reshape((2, height * width // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((height * width // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[: height * width] = y
        nv12[height * width :] = uv_packed
        return nv12
