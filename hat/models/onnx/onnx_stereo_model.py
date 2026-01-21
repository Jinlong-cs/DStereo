# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, List, Optional, Sequence, Tuple

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from hat.registry import OBJECT_REGISTRY


def _bgr2nv12(image):
    image = image.astype(np.uint8)
    height, width = image.shape[0], image.shape[1]
    frame_size = width * height
    nv12 = np.zeros(frame_size + frame_size // 2, dtype=np.int32)

    bgr24_reshaped = image.reshape((height, width, 3)).astype(np.int32)
    b, g, r = bgr24_reshaped[..., 0], bgr24_reshaped[..., 1], bgr24_reshaped[..., 2]

    y_plane = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16
    y_plane = np.clip(y_plane, 0, 255).astype(np.uint8)
    nv12[:frame_size] = y_plane.flatten()

    uv_b = b[::2, ::2]
    uv_g = g[::2, ::2]
    uv_r = r[::2, ::2]
    uv_plane = np.zeros((height * width // 2), dtype=np.int32)

    u = ((-38 * uv_r - 74 * uv_g + 112 * uv_b + 128) >> 8) + 128
    v = ((112 * uv_r - 94 * uv_g - 18 * uv_b + 128) >> 8) + 128

    u = np.clip(u, 0, 255).astype(np.uint8).flatten()
    v = np.clip(v, 0, 255).astype(np.uint8).flatten()

    uv_plane[0::2] = u
    uv_plane[1::2] = v

    nv12[frame_size:] = uv_plane.flatten()

    return np.ascontiguousarray(nv12.astype(np.uint8))


def _nv12_to_yuv444(data, height, width):
    nv12_data = data.flatten()
    yuv444 = np.empty([height, width, 3], dtype=np.uint8)
    yuv444[:, :, 0] = nv12_data[: width * height].reshape(height, width)
    u = nv12_data[width * height :: 2].reshape(height // 2, width // 2)
    yuv444[:, :, 1] = Image.fromarray(u).resize((width, height), resample=0)
    v = nv12_data[width * height + 1 :: 2].reshape(height // 2, width // 2)
    yuv444[:, :, 2] = Image.fromarray(v).resize((width, height), resample=0)
    return np.ascontiguousarray(yuv444.astype(np.uint8))


def _infer_preprocess(p, crop_width=640, crop_height=352):
    if not os.path.isfile(p):
        raise ValueError(p)
    img = cv2.imread(p)
    if img is None:
        raise ValueError(p)
    img = img[:, :, :3]
    h, w, _ = img.shape

    ratio = w / h
    nh = int(crop_width / ratio)
    img_resized = cv2.resize(img, (crop_width, nh))
    if img_resized.shape[1] != crop_width:
        raise ValueError(f"Unexpected resized width: {img_resized.shape}")
    if nh < crop_height:
        top_pad = (crop_height - nh) // 2
        bottom_pad = crop_height - nh - top_pad
        if top_pad > 0 or bottom_pad > 0:
            img_resized = cv2.copyMakeBorder(
                img_resized,
                top_pad,
                bottom_pad,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=0,
            )
    elif nh > crop_height:
        top = int((nh - crop_height + 1) * 0.5)
        bottom = top + crop_height
        img_resized = img_resized[top:bottom, ...]
    if img_resized.shape != (crop_height, crop_width, 3):
        raise ValueError(f"Unexpected resized shape {img_resized.shape} for {p}")

    img_nv12 = _bgr2nv12(img_resized)
    img_yuv444 = _nv12_to_yuv444(img_nv12, crop_height, crop_width)

    input = np.array(img_yuv444).astype(np.float32)
    input = input / 128.0 - 1.0
    input = np.transpose(input, (2, 0, 1))
    input = np.ascontiguousarray(input[None, ...])

    return input, np.ascontiguousarray(img_resized)


@OBJECT_REGISTRY.register
class OnnxStereoModel(nn.Module):
    """ONNXRuntime wrapper for stereo disparity inference."""

    def __init__(
        self,
        onnx_path: str,
        input_names: Optional[Sequence[str]] = None,
        output_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.onnx_path = onnx_path
        self.input_layout = "NCHW"
        self.input_dtype = "float32"
        from horizon_tc_ui import HB_ONNXRuntime

        self.sess = HB_ONNXRuntime(model_file=onnx_path)

        self.input_names = list(input_names) if input_names else [
            i.name for i in self.sess.get_inputs()
        ]
        self.input_types = [i.type for i in self.sess.get_inputs()]
        self.input_shapes = [i.shape for i in self.sess.get_inputs()]
        self.output_names = list(output_names) if output_names else [
            o.name for o in self.sess.get_outputs()
        ]
        self._preprocess_fn = _infer_preprocess
        self._crop_height = None
        self._crop_width = None
        if self.input_shapes:
            shape = self.input_shapes[0]
            if len(shape) == 4 and shape[1] == 3:
                self.input_layout = "NCHW"
            elif len(shape) == 4 and shape[-1] == 3:
                self.input_layout = "NHWC"
            if self.input_layout == "NCHW":
                self._crop_height = int(shape[2])
                self._crop_width = int(shape[3])
            else:
                self._crop_height = int(shape[1])
                self._crop_width = int(shape[2])

    def _split_inputs(self, data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        if "infra1" in data and "infra2" in data:
            if data["infra1"].shape[0] != 1 or data["infra2"].shape[0] != 1:
                raise ValueError("ONNX predictor expects batch size 1")
            return data["infra1"], data["infra2"]
        img = data["img"]
        batch = img.shape[0]
        if "gt_disp" in data:
            gt_batch = data["gt_disp"].shape[0]
            if gt_batch != 1:
                raise ValueError("ONNX predictor expects batch size 1")
            if batch != gt_batch * 2:
                raise ValueError("Expected stacked left/right images in batch")
        if batch % 2 != 0:
            raise ValueError("Expected stacked left/right images in batch")
        half = batch // 2
        return img[:half], img[half:]

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        array = tensor.detach().cpu().numpy()
        array = array.astype(np.float32)
        if self.input_layout == "NHWC":
            array = np.transpose(array, (0, 2, 3, 1))
        return np.ascontiguousarray(array)

    def _to_int8_yuv(self, yuv: np.ndarray) -> np.ndarray:
        array = yuv.astype(np.int16) - 128
        if self.input_layout == "NCHW":
            array = np.transpose(array, (0, 3, 1, 2))
        return np.ascontiguousarray(array.astype(np.int8))

    def _to_int8_from_float(self, value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        if array.ndim == 3:
            array = array[None, ...]
        array = array.astype(np.float32)
        if self.input_layout == "NHWC" and array.shape[1] == 3:
            array = np.transpose(array, (0, 2, 3, 1))
        elif self.input_layout == "NCHW" and array.shape[-1] == 3:
            array = np.transpose(array, (0, 3, 1, 2))
        array = (array + 1.0) * 128.0
        array = array.astype(np.int16) - 128
        return np.ascontiguousarray(array.astype(np.int8))

    def _to_numpy_array(self, value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return self._to_numpy(value)
        array = np.asarray(value)
        if array.ndim == 3:
            array = array[None, ...]
        array = array.astype(np.float32)
        if self.input_layout == "NHWC" and array.shape[1] == 3:
            array = np.transpose(array, (0, 2, 3, 1))
        elif self.input_layout == "NCHW" and array.shape[-1] == 3:
            array = np.transpose(array, (0, 3, 1, 2))
        return np.ascontiguousarray(array)

    def _is_int8_type(self, t) -> bool:
        if isinstance(t, int):
            return t in (2, 3)
        t_str = str(t).lower()
        return "int8" in t_str or "uint8" in t_str

    def _get_expected_int8(self) -> bool:
        for t in self.input_types:
            if self._is_int8_type(t):
                return True
        return False

    def _get_path(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                raise ValueError("Empty image path list")
            return value[0]
        return value

    def _preprocess_from_files(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        if "left_img_name" not in data or "right_img_name" not in data:
            raise ValueError("preprocess expects left_img_name/right_img_name")
        left_path = self._get_path(data["left_img_name"])
        right_path = self._get_path(data["right_img_name"])
        left_input, _ = self._preprocess_fn(
            left_path, crop_width=self._crop_width, crop_height=self._crop_height
        )
        right_input, _ = self._preprocess_fn(
            right_path, crop_width=self._crop_width, crop_height=self._crop_height
        )
        return left_input, right_input

    def _extract_yuv(self, value) -> np.ndarray:
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError("ONNX predictor expects batch size 1")
            value = value[0]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if value.ndim == 3:
            value = value[None, ...]
        return value

    def _postprocess(self, outputs: Dict[str, np.ndarray]) -> torch.Tensor:
        if "disp" in outputs and "spx" in outputs:
            disp = outputs["disp"]
            spx = outputs["spx"]
            b, _, h, w = disp.shape
            disp_t = torch.from_numpy(disp)
            spx_t = torch.from_numpy(spx)
            disp_up = F.interpolate(disp_t, (h * 4, w * 4), mode="nearest")
            disp_up = disp_up.reshape(b, 9, h * 4, w * 4)
            return torch.sum(disp_up * spx_t, dim=1)
        return torch.from_numpy(outputs[self.output_names[0]])

    def forward(self, data: Dict):  # type: ignore
        left_input, right_input = self._preprocess_from_files(data)
        if self._get_expected_int8():
            left_np = self._to_int8_from_float(left_input)
            right_np = self._to_int8_from_float(right_input)
        else:
            left_np = self._to_numpy_array(left_input)
            right_np = self._to_numpy_array(right_input)
        feed_dict = {
            self.input_names[0]: left_np,
            self.input_names[1]: right_np,
        }
        onnx_outs = self.sess.run(self.output_names, feed_dict)
        outputs = dict(zip(self.output_names, onnx_outs))
        preds = self._postprocess(outputs)
        if isinstance(preds, torch.Tensor) and "gt_disp" in data:
            preds = preds.to(data["gt_disp"].device)
        return preds, None, None
