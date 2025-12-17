from functools import partial

import singletask__base_segmentation_config as base_config

from hat.registry import build_from_registry

colormap = [
    {"class_name": "background", "color_map": [0, 0, 0]},
    {"class_name": "word", "color_map": [119, 11, 32]},
    {"class_name": "imprint", "color_map": [0, 0, 142]},
    {"class_name": "deceleration_lane", "color_map": [0, 0, 230]},
    {"class_name": "light_shadow", "color_map": [106, 0, 228]},
    {"class_name": "shadow", "color_map": [0, 60, 100]},
    {"class_name": "rain_trace", "color_map": [0, 80, 100]},
    {"class_name": "one2two", "color_map": [0, 0, 70]},
    {"class_name": "old_lane", "color_map": [0, 0, 192]},
    {"class_name": "reflection", "color_map": [250, 170, 30]},
    {"class_name": "seam", "color_map": [100, 170, 30]},
    {"class_name": "light", "color_map": [220, 220, 0]},
    {"class_name": "grid", "color_map": [175, 116, 175]},
]
cls_name_mapping = {idx: i["class_name"] for (idx, i) in enumerate(colormap)}


def _update_head(config):
    config["num_classes"] = 13
    return config


# external

task_names = ["badcase_parsing_13cls_segmentation"]

val_transforms = base_config.val_transforms
preprocess = base_config.preprocess
postprocess = base_config.postprocess
inference_model = base_config.get_inference_models(
    backbone_arch="swin-base",
    neck_arch="BiFPN",
    head_arch="semantic_segmentation",
    postprocess_arch="semantic_segmentation",
    task_names=task_names,
    updates=dict(head_update_fn=_update_head),
)

inference = dict(
    type="Inference",
    device=None,
    pre_processors=[
        partial(
            preprocess,
            transforms=[build_from_registry(t) for t in val_transforms],
        ),
    ],
    post_processors=[
        partial(
            postprocess,
            task_name=task_names[0],
            cls_name_mapping=cls_name_mapping,
        )
    ],
    model=inference_model,
    model_convert_pipeline=None,
)
