import copy
import os
from functools import partial

import singletask__base_segmentation_config as base_config

from hat.core.proj_spec.descs import get_default_parsing_desc
from hat.core.proj_spec.parsing import (
    ParsingDescClassName,
    get_default_parsing_labels,
)
from hat.models.model_convert.converters import TorchCompile
from hat.models.model_convert.pipelines import ModelConvertPipeline
from hat.registry import build_from_registry

cls_name_mapping = {
    idx: item[ParsingDescClassName][0]
    for idx, item in enumerate(get_default_parsing_labels("gl_40"))
}

parsing40cls_desc = get_default_parsing_desc(desc_id="gl_40")


def _update_head(config):
    config["num_classes"] = 40
    return config


# external

task_names = ["semantic_parsing_40cls_segmentation"]

val_transforms = copy.deepcopy(base_config.val_transforms)
assert val_transforms[0]["type"] == "Resize", val_transforms
del val_transforms[0]

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
        )
    ],
    post_processors=[
        partial(
            postprocess,
            task_name=task_names[0],
            cls_name_mapping=cls_name_mapping,
            parsing_desc=parsing40cls_desc,
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
                        ],
                    )
                ]
            ),
        )
    )
