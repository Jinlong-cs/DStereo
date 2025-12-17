import os
import subprocess
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from hat.utils.logger import init_logger

__all__ = ["init_test_logger"]

TMP_DIR = "/tmp"


def init_test_logger(rank):
    time_stamp = time.strftime(
        "%Y%m%d%H%M%S", time.localtime(int(time.time()))
    )
    log_file = os.path.join(TMP_DIR, f"hat_pytest_{time_stamp}.log")
    init_logger(log_file=log_file, rank=rank)


def execute_cmd(cmd):
    assert isinstance(cmd, str)
    if int(os.environ.get("USE_DCU", 0)) == 1:
        smi = "rocm-smi"
    else:
        smi = "nvidia-smi"
    exe_cmd = f"""
    {smi}
    set -e
    {cmd}
    """
    # Set shell=True if you're passing a string to subprocess.call
    subprocess.check_call(exe_cmd, shell=True)


def gen_fake_transforms_data(
    w, h, layout, fill_value=None, return_tensor=False, c=3
):
    """Generate fake data for transform test."""
    data = {}
    if layout == "hwc":
        img_shape = (h, w, c)
        flow_shape = (h, w, 2)
    elif layout == "chw":
        img_shape = (c, h, w)
        flow_shape = (2, h, w)
    elif layout == "hw":
        img_shape = (h, w)
        flow_shape = (h, w, 2)
    if fill_value is None:
        if return_tensor:
            img = torch.randint(0, 255, img_shape, dtype=torch.uint8)
            gt_seg = torch.randint(0, 10, (h, w), dtype=torch.uint8)
            gt_flow = torch.randn(flow_shape, dtype=torch.float32)
            gt_depth = (
                torch.randint(0, 2 ** 16, (h, w), dtype=torch.int32)
            ) / 255.0
            gt_disp = torch.randn((h, w), dtype=torch.float32)
        else:
            img = np.random.randint(0, 255, img_shape, dtype=np.uint8)
            gt_seg = np.random.randint(0, 10, (h, w), dtype=np.uint8)
            gt_flow = np.random.random_sample(flow_shape)
            gt_depth = (
                torch.randint(0, 2 ** 16, (h, w), dtype=torch.int32)
            ) / 255.0
            gt_disp = np.random.random_sample((h, w))
    else:
        if return_tensor:
            img = torch.full(img_shape, fill_value, dtype=torch.uint8)
            gt_seg = torch.full((h, w), fill_value, dtype=torch.uint8)
            gt_flow = torch.full(flow_shape, fill_value, dtype=torch.float32)
            gt_depth = torch.full((h, w), fill_value, dtype=torch.float32)
            gt_disp = torch.full((h, w), fill_value, dtype=torch.float32)
        else:
            img = np.full(img_shape, fill_value, dtype=np.uint8)
            gt_seg = np.full((h, w), fill_value, dtype=np.uint8)
            gt_flow = np.full(flow_shape, fill_value, dtype=np.float32)
            gt_depth = np.full((h, w), fill_value, dtype=np.float32)
            gt_disp = np.full((h, w), fill_value, dtype=np.float32)

    data["img"] = img
    data["img_height"] = h
    data["img_width"] = w
    data["img_shape"] = img.shape
    data["img_id"] = 0
    data["color_space"] = "bgr"
    data["layout"] = layout
    data["gt_seg"] = gt_seg
    data["gt_bboxes"] = np.array([[10, 10, 20, 20], [20, 20, 40, 40]])
    data["gt_classes"] = np.array([1, 0])
    data["gt_flow"] = gt_flow
    data["gt_depth"] = gt_depth
    data["gt_ldmk"] = np.random.randint(0, min(h, w), size=(5, 3))
    data["gt_ldmk"][:, 2] = 1
    data["ldmk_pairs"] = [[0, 1], [3, 4]]
    data["gt_mask"] = deepcopy(gt_seg)
    data["gt_img"] = deepcopy(data["img"])
    data["raw_pattern"] = "RGGB"
    data["cur_pattern"] = data["raw_pattern"]
    data["gt_lines"] = [
        np.array([[10, 10], [20, 30], [30, 40]]).astype(np.float32)
    ]
    data["gt_polygons"] = [
        np.array([[10, 10], [20, 30], [30, 40], [30, 10]]).astype(np.float32)
    ]
    data["gt_eye_cls_labels"] = np.random.randint(2, size=10)
    data["gt_ldmk_attr"] = np.random.randint(2, size=16)
    data["eye_status"] = np.random.randint(2, size=8)
    data["eye_vis_labels"] = np.random.randint(2, size=5)
    data["gt_disp"] = gt_disp
    data["gt_pupil_ellipse_param"] = np.array([150.0, 150.0, 20.0, 20.0, 0.0])
    return data


def gen_fake_det_label_pred_data(h, w, strides, num_classes):
    """Generate fake data for detection loss and target test."""
    cls_scores, bbox_preds, centernesses = [], [], []
    for stride in strides:
        assert (w % stride) == 0
        assert (h % stride) == 0
        stride_h = h // stride
        stride_w = w // stride
        cls_pred = torch.randn((2, num_classes, stride_h, stride_w))
        bbox_pred = torch.randn((2, 4, stride_h, stride_w))
        centerness_pred = torch.randn((2, 1, stride_h, stride_w))
        cls_scores.append(cls_pred)
        bbox_preds.append(bbox_pred)
        centernesses.append(centerness_pred)
    label = {}
    label["img_name"] = "dummy.jpg"
    label["img_id"] = 0
    label["gt_bboxes"] = [
        torch.Tensor([[w // 4, h // 4, w // 2, h // 2]]),
        torch.Tensor(
            [
                [w // 8, h // 8, w // 3, h // 3],
                [w // 2, h // 2, w * 3 // 4, h * 3 // 4],
            ]
        ),
    ]
    label["gt_classes"] = [
        torch.Tensor([0]).long(),
        torch.Tensor([1, 1]).long(),
    ]
    label["layout"] = ["chw", "chw"]
    label["pad_shape"] = [[3, h, w], [3, h, w], [3, h, w]]
    label["scale_factor"] = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    label["crop_offset"] = [[0, 0, 0, 0], [0, 0, 0, 0]]
    label["before_crop_shape"] = [[3, h, w], [3, h, w]]

    return label, (cls_scores, bbox_preds, centernesses)


def gen_fake_torch_randn_data(shape, dtype=None):
    data = torch.randn(shape, dtype=dtype)
    return data


def gen_fake_torch_randint_data(shape, low=0, high=255, dtype=None):
    data = torch.randint(low, high, shape, dtype=dtype)
    return data


def gen_fake_pil_data(shape, mode, dtype="uint8", low=0, high=255):
    """Generate fake pil data for test."""
    img = np.random.randint(low, high, shape, dtype="uint8").astype(dtype)
    img = Image.fromarray(img).convert(mode)
    return img


def gen_fake_np_data(shape, dtype="uint8", low=0, high=255):
    data = np.random.randint(low, high, shape, dtype="uint8").astype(dtype)
    return data


def gen_fake_feats(
    w, h, strides, n=1, feat_channels=64, channels_strides=None
):
    """Generate fake feature data for head and decoder test."""
    if channels_strides is not None:
        assert len(channels_strides) == len(
            strides
        ), "The length of strides and channels_strides mismatch"
    else:
        channels_strides = len(strides) * [1]
    feats = []
    stride2channels = OrderedDict()
    stride2hw = OrderedDict()
    for stride, channels_stride in zip(strides, channels_strides):
        assert (w % stride) == 0
        assert (h % stride) == 0
        assert (feat_channels % channels_stride) == 0
        feat = torch.randn(
            (n, feat_channels // channels_stride, h // stride, w // stride)
        )
        feats.append(feat)
        stride2channels[stride] = feat_channels // channels_stride
        stride2hw[stride] = (h // stride, w // stride)

    return feats, stride2channels, stride2hw


def gen_fake_seg_data(h, w, strides, num_classes, n=2, label_name="gt_seg"):
    """Generate fake data for seg test."""
    pred = []
    for stride in strides:
        assert (w % stride) == 0
        assert (h % stride) == 0
        stride_h = h // stride
        stride_w = w // stride
        pred.append(torch.randn((n, num_classes, stride_h, stride_w)))
    label = {}
    label[label_name] = torch.full(size=(n, h, w), fill_value=2)
    label["scale_factor"] = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    label["crop_offset"] = [[0, 0, 0, 0], [0, 0, 0, 0]]
    label["before_crop_shape"] = [[3, h, w], [3, h, w]]

    return pred, label


def gen_fake_depth_data(h, w, strides, label_dim, n=2, label_name="gt_depth"):
    """Generate fake data for depth test."""
    pred = []
    for stride in strides:
        assert (w % stride) == 0
        assert (h % stride) == 0
        stride_h = h // stride
        stride_w = w // stride
        pred.append(torch.randn((n, label_dim, stride_h, stride_w)))
    label = {}
    label[label_name] = (
        torch.randint(0, 2 ** 16, size=(n, label_dim, h, w)) // 256.0
    )
    label["scale_factor"] = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    label["crop_offset"] = [[0, 0, 0, 0], [0, 0, 0, 0]]
    label["before_crop_shape"] = [[3, h, w], [3, h, w]]

    return pred, label


def check(data, func, **func_kwars):
    if isinstance(data, Sequence):
        for data_i in data:
            check(data_i, func, **func_kwars)
    elif isinstance(data, Mapping):
        for _, v in data.items():
            check(v, func, **func_kwars)
    else:
        func(data, **func_kwars)


def check_shape(data, shape):
    if hasattr(data, "shape"):  # for np data
        assert data.shape == shape
    elif hasattr(data, "size"):  # for pil img
        assert data.size == (shape[1], shape[0])
    else:
        raise AssertionError()


def check_type(data, instance):
    assert isinstance(data, instance)


def check_range(data, min, max):
    assert data.min() >= min
    assert data.max() <= max
