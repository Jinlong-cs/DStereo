from functools import partial

import singletask__base_detection_config as base_config

from hat.registry import build_from_registry


def _update_head(config):
    config["stride2channels"] = {
        4: 256,
        8: 256,
        16: 256,
        32: 256,
        64: 256,
        128: 256,
    }
    config["feat_channels"] = 256
    return config


def _update_neck(config):
    config["out_channels"] = {s: 256 for s in [4, 8, 16, 32, 64]}
    config["stack"] = 3
    return config


# -----------
# external
# -----------
task_names = ["traffic_sign_detection"]

val_transforms = base_config.val_transforms
preprocess = base_config.preprocess
postprocess = base_config.postprocess
inference_model = base_config.get_inference_models(
    backbone_arch="small",
    neck_arch="BiFPN",
    task_names=task_names,
    updates=dict(
        head_update_fn=_update_head,
        neck_update_fn=_update_neck,
    ),
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
