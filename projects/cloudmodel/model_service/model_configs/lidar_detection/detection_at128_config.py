import os
from functools import partial
from typing import Any, Callable, Dict, List, Union

import lidar_base_detection_config as base_config
import numpy as np
import torch
from hatbc.message import LidarFrame, PointCloud

from hat.registry import build_from_registry

task_names = ["lidar_detection_at128_7class"]

val_transforms = base_config.val_transforms
preprocess = base_config.preprocess
postprocess = base_config.postprocess

inference_model = base_config.get_inference_models(task_names[0])

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
    # model_convert_pipeline=base_config.model_convert_pipeline,
)
