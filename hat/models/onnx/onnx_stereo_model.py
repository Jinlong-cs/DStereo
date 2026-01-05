# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hat.registry import OBJECT_REGISTRY


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
        if self.input_shapes:
            shape = self.input_shapes[0]
            if len(shape) == 4 and shape[1] == 3:
                self.input_layout = "NCHW"
            elif len(shape) == 4 and shape[-1] == 3:
                self.input_layout = "NHWC"

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

    def _get_expected_int8(self) -> bool:
        for t in self.input_types:
            if isinstance(t, int) and t == 3:
                return True
            if "int8" in str(t).lower():
                return True
        return False

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
        if self._get_expected_int8():
            if "left_img_yuv" not in data or "right_img_yuv" not in data:
                raise ValueError("Missing YUV inputs for int8 ONNX model")
            left_yuv = self._extract_yuv(data["left_img_yuv"])
            right_yuv = self._extract_yuv(data["right_img_yuv"])
            if left_yuv.shape[0] != 1 or right_yuv.shape[0] != 1:
                raise ValueError("ONNX predictor expects batch size 1")
            left_np = self._to_int8_yuv(left_yuv)
            right_np = self._to_int8_yuv(right_yuv)
            self.input_layout = "NHWC"
        else:
            left, right = self._split_inputs(data)
            left_np = self._to_numpy(left)
            right_np = self._to_numpy(right)
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
