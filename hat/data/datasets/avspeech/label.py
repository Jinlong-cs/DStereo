# Copyright (c) Horizon Robotics, All rights reserved.

import math

import torch


class VadLabelMaskGeneratorV1(object):
    """VadLabelMaskGeneratorV1.

    多模VAD训练标签和MASK生成器.

    Args:
        fps: 视频的帧率.
    """

    def __init__(self, fps: int = 30):
        self._fps = fps

    def generate(
        self, seg_beg: float, seg_end: float, vad_beg: float, vad_end: float
    ):
        """Generate label.

        Args:
        seg_beg : 音频样本的起始时间点.
        seg_end : 音频样本的结束时间点.
        vad_beg : 音频样本的说话起始时间点.
        vad_end : 音频样本的说话结束时间点.
        """

        seg_beg_idx = math.ceil(seg_beg * self._fps)
        seg_end_idx = math.ceil(seg_end * self._fps)
        vad_beg_idx = math.ceil(vad_beg * self._fps)
        vad_end_idx = math.ceil(vad_end * self._fps)

        def label_func(idx):
            if idx >= vad_beg_idx and idx < vad_end_idx:
                return 1
            else:
                return 0

        def weight_func(idx):
            if idx >= (vad_beg_idx - 3) and idx <= (vad_beg_idx + 3):
                return ((idx - vad_beg_idx) * 0.08) ** 2 + 0.01545455
            elif idx >= (vad_end_idx - 3) and idx <= (vad_end_idx + 3):
                return ((idx - vad_end_idx) * 0.08) ** 2 + 0.01545455
            else:
                return 1

        label_mask = [
            (label_func(idx), 1) for idx in range(seg_beg_idx, seg_end_idx)
        ]
        weight_mask = [
            weight_func(idx) for idx in range(seg_beg_idx, seg_end_idx)
        ]

        label = [lm[0] for lm in label_mask]
        label = torch.Tensor(label)
        mask = [lm[1] for lm in label_mask]
        return label, mask, weight_mask
