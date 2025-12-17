# copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Any, Callable, List, Optional, Union

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list, to_cuda

__all__ = ["Inference"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register_module
class Inference(object):
    """Basic Inference Runner.

    Args:
        device: Device ids used for inference.
        model: Model for inference.
        march: March of model convert.
        model_convert_pipeline: Pipeline of model convert.
        pre_processors: Transforms before model inference.
        post_processors: Inverse transforms after model inference.

    """

    def __init__(
        self,
        device: Union[List[int], int],
        model: nn.Module,
        march: Optional[horizon.march.March] = None,
        model_convert_pipeline: Optional[List] = None,
        pre_processors: Optional[List[Callable]] = None,
        post_processors: Optional[List[Callable]] = None,
    ):
        if march is None:
            logging.warning(
                "march not provided, please make sure it's expected"
            )
        else:
            horizon.march.set_march(march)

        self.model = model
        if model_convert_pipeline is not None:
            self.model = model_convert_pipeline(self.model)
        self.model.eval()

        device_ids = _as_list(device)
        if len(device_ids) > 1:
            raise NotImplementedError("Current only support single device.")
        if len(device_ids) != 0:
            device_ids[0] = (
                int(device_ids[0]) if device_ids[0] is not None else None
            )
            self.device = torch.device("cuda", device_ids[0])
        else:
            self.device = torch.device("cpu")

        self.set_device(self.device)
        self.preprocessors = [] if pre_processors is None else pre_processors
        self.postprocessors = (
            [] if post_processors is None else post_processors
        )

    def set_device(self, device: torch.device) -> None:
        self.device = device
        self.model.to(device)

    def preprocess(self, data: Any) -> Any:
        for processor in self.preprocessors:
            data = processor(data)
        return data

    def postprocess(self, pred: Any, data: Any) -> Any:
        for processor in self.postprocessors:
            pred, data = processor(pred, data)
        return pred

    def forward(self, data: Any) -> Any:
        with torch.no_grad():
            data = self.preprocess(data)
            if self.device.type == "cuda":
                data = to_cuda(data, self.device, non_blocking=True)

            output = self.model(data)
            output = self.postprocess(output, data)
        return output
