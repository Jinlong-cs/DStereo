import os
from functools import partial

import singletask__base_detection_config as base_config

from hat.models.model_convert.converters import TorchCompile
from hat.models.model_convert.pipelines import ModelConvertPipeline
from hat.registry import build_from_registry

# -----------
# external
# -----------
task_names = ["traffic_light_detection"]

val_transforms = base_config.val_transforms
preprocess = base_config.preprocess
postprocess = base_config.postprocess
inference_model = base_config.get_inference_models(
    backbone_arch="base_w12",
    neck_arch="BiFPN",
    task_names=task_names,
)

inference = dict(
    type="Inference",
    device=None,
    pre_processors=[
        partial(
            preprocess,
            transforms=[build_from_registry(t) for t in val_transforms],
        )
    ],
    post_processors=[partial(postprocess, task_names=task_names)],
    model=inference_model,
    model_convert_pipeline=base_config.model_convert_pipeline,
)
