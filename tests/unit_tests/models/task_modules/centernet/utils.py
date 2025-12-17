import numpy as np
import torch

from tests.utils import gen_fake_torch_randn_data


def gen_fake_img_meta_data_dict(height, width):
    img_meta = {
        "img_id": [1],
        "img_name": ["test.jpg"],
        "img_height": [height],
        "img_width": [width],
        "gt_bboxes": [gen_fake_torch_randn_data((8, 4))],
        "ig_bboxes": [gen_fake_torch_randn_data((1, 4))],
        "gt_classes": [torch.tensor([1, -3, -4, 1, -4, 2, -4, 1])],
        "gt_labels": [torch.tensor([1, 2, 3])],
        "color_space": ["yuv"],
        "layout": ["chw"],
        "img_shape": [np.array([3, height, width])],
    }
    return img_meta


def gen_fake_label_data_dict(height, width):
    label = {
        "img_id": [1],
        "img_name": ["test.jpg"],
        "img_height": [height],
        "img_width": [width],
        "gt_bboxes": [gen_fake_torch_randn_data((8, 4))],
        "ig_bboxes": [gen_fake_torch_randn_data((1, 4))],
        "gt_classes": [torch.tensor([1, -3, -4, 1, -4, 2, -4, 1])],
        "gt_labels": [torch.tensor([1, 2, 3])],
        "color_space": ["yuv"],
        "layout": ["chw"],
        "img_shape": [[height, width, 3]],
        "pad_shape": [[height, width, 3]],
        "keep_ratio": torch.tensor([0]),
        "scale": [torch.tensor([height]), torch.tensor([width])],
        "scale_idx": torch.tensor([0]),
    }
    return label
