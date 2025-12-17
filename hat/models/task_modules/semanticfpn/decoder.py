# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Any, List, Optional, Sequence

import torch
from torch.nn.functional import interpolate

from hat.core.data_struct.base_struct import Mask
from hat.registry import OBJECT_REGISTRY

__all__ = ["BMSegDecoder"]


@OBJECT_REGISTRY.register
class BMSegDecoder(torch.nn.Module):
    """Semantic Segmentation Decoder.

    Args:
        out_strides: List of output strides, represents the strides of the
            output from the segmentation head.
        do_inverse_transform: Whether do inverse transform. Default if False.
        class_mapping: An index array to convert classes.
            e.g. mapping 40 classes parsing result to 33 classes.
        to_cpu: Whether move data to cpu. Default: False.
        use_native_op: Use torch's native operators which is implemented to
            support torchdynamo when compiling models with trt and torchdynamo,
            since torch.nn.functional.interpoate is overridden by
            horizon_plugin_pytorch. Default: False.
    """

    def __init__(
        self,
        out_strides: List[int],
        do_inverse_transform: bool = False,
        class_mapping: Optional[List[int]] = None,
        use_native_op: bool = False,
        to_cpu: bool = False,
    ):
        super(BMSegDecoder, self).__init__()
        self.out_strides = out_strides
        self.do_inverse_transform = do_inverse_transform
        self.class_mapping = class_mapping
        self.to_cpu = to_cpu
        self.use_native_op = use_native_op

    def forward(
        self, pred: Sequence[torch.Tensor], label: Optional[Any] = None
    ) -> List[Mask]:
        assert len(pred) == 1
        pred = pred[0]
        assert pred.ndim == 4 and pred.shape[1] > 1, pred.shape

        # step1: upscale
        stride = self.out_strides[0]
        if self.use_native_op:
            pred = torch._C._nn.upsample_bilinear2d(
                pred,
                output_size=None,
                align_corners=False,
                scale_factors=(stride, stride),
            )
        else:
            pred = interpolate(
                pred,
                scale_factor=(stride, stride),
                mode="bilinear",
                align_corners=False,
            )

        # step2: do inverse transform if necessary
        results = []
        for idx, pred_i in enumerate(pred):
            if self.do_inverse_transform and label is not None:
                img_shape_i = label["img_shape"][idx]
                img_height_i = label["img_height"][idx]
                img_width_i = label["img_width"][idx]

                ori_shape_i = (img_height_i, img_width_i)
                if self.use_native_op:
                    pred_i = torch._C._nn.upsample_bilinear2d(
                        pred_i[
                            :, : img_shape_i[-2], : img_shape_i[-1]
                        ].unsqueeze(0),
                        output_size=ori_shape_i,
                        align_corners=False,
                        scale_factors=None,
                    ).squeeze(0)
                else:
                    pred_i = interpolate(
                        pred_i[
                            :, : img_shape_i[-2], : img_shape_i[-1]
                        ].unsqueeze(0),
                        ori_shape_i,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

            pred_i = pred_i.argmax(dim=0)
            if self.class_mapping is not None:
                if isinstance(self.class_mapping, list):
                    self.class_mapping = pred_i.new_tensor(self.class_mapping)
                pred_i = self.class_mapping[pred_i]
            results.append(pred_i)

        res = []
        for p in results:
            if self.to_cpu:
                p = p.cpu()
            res_struct = Mask(p)
            res.append(res_struct)
        return res
