import logging
from typing import Any

import torch
from aidisdk.model import DeviceMeta, InferModel

from hat.engine.inference import Inference
from hat.registry import RegistryContext, build_from_registry
from hat.utils.apply_func import to_cuda
from hat.utils.config import Config

logger = logging.getLogger(__name__)

__all__ = ["HatModel"]


class HatModel(InferModel):
    def init(self, device: DeviceMeta):
        logger.info(f"[{self.__class__.__name__}]Init model Started....")
        device_ids = [] if device.type == "cpu" else device.id
        config = Config.fromfile(self.model_file)
        assert "inference" in config, "Config must include key: inference"
        inference = config["inference"]
        inference["device"] = device_ids
        current = RegistryContext.get_current()
        if current is None:
            with RegistryContext():
                inference = build_from_registry(inference)
        else:
            inference = build_from_registry(inference)
        self.runner = inference

        # init params
        try:
            checkpoint_file = self.get_attachment("checkpoint_file")
        except Exception:
            checkpoint_file = None
        if checkpoint_file is not None:
            model = self.runner.model
            from hat.utils.checkpoint import load_state_dict

            load_state_dict(
                model,
                path_or_dict=checkpoint_file,
                map_location="cpu",
                ignore_extra=True,
            )
            self.runner.model = model
        (
            self.preprocess_runner,
            self.predict_runner,
            self.postprocess_runner,
        ) = self.split_runner_step(
            config.get("split_dataprocess_on_aidi", False)
        )

        logger.info(f"[{self.__class__.__name__}]Init model Success!")

    def split_runner_step(self, split_dataprocess_on_aidi):
        if isinstance(self.runner, Inference) and split_dataprocess_on_aidi:

            def _predict(data):
                device = self.runner.device
                if device.type == "cuda":
                    input = to_cuda(data, device, non_blocking=True)
                else:
                    input = data
                output = self.runner.model(input)
                return output

            preprocess_runner = self.runner.preprocess
            predict_runner = _predict
            postprocess_runner = self.runner.postprocess
        else:
            preprocess_runner = lambda data: data
            predict_runner = self.runner.forward
            postprocess_runner = lambda output, input_data: output

        return preprocess_runner, predict_runner, postprocess_runner

    def preprocess(self, data: Any):
        with torch.no_grad():
            return self.preprocess_runner(data)

    def predict(self, data: Any):
        with torch.no_grad():
            output = self.predict_runner(data)
            return (output, data)

    def postprocess(self, data: Any):
        output, input_data = data
        with torch.no_grad():
            output = self.postprocess_runner(output, input_data)
            return output
