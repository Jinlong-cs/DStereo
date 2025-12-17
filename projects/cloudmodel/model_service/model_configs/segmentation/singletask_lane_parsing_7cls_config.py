import os
from functools import partial

import singletask__base_segmentation_config as base_config

from hat.models.model_convert.converters import TorchCompile
from hat.models.model_convert.pipelines import ModelConvertPipeline
from hat.registry import build_from_registry

desc_labels = [
    {"class_name": "background", "color_map": [0, 0, 0]},
    {"class_name": "lane", "color_map": [0, 0, 255]},
    {"class_name": "curb", "color_map": [0, 255, 0]},
    {"class_name": "double_line", "color_map": [255, 255, 0]},
    {"class_name": "wide_dashed", "color_map": [190, 153, 153]},
    {"class_name": "wide_solid", "color_map": [153, 51, 204]},
    {"class_name": "deceleration_lane", "color_map": [0, 255, 255]},
]
cls_name_mapping = {
    idx: i["class_name"] for (idx, i) in enumerate(desc_labels)
}


def _update_head(config):
    config["num_classes"] = 7
    return config


# external
val_transforms = base_config.val_transforms
task_names = ["lane_7cls_segmentation"]
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


trt_opt = os.environ.get("INFERENCE_TENSORRT_OPT", False)

if bool(int(trt_opt)):
    from hat.utils.compile_backends import tensorRT_backend

    FP16 = True
    inference.update(
        dict(
            model_convert_pipeline=ModelConvertPipeline(
                [
                    TorchCompile(
                        tensorRT_backend(fp16=FP16, dynamic_shape=True),
                        load_extensions=[
                            "group_norm",
                            "c_interpolate",
                            "dynamic_shape_pad",
                        ],
                    )
                ]
            ),
        )
    )
