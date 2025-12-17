import copy
import os
from functools import partial
from typing import Any, Callable, Dict, List

import numpy as np
import singletask__base_segmentation_config as base_config
from hatbc.message import Attribute, Instance, Polygon2D

from hat.core.data_struct.base_struct import Mask
from hat.core.mask2polygon import Parsing2Polygon
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


def _update_head(config):
    config["num_classes"] = 40
    return config


# external

task_names = ["semantic_parsing_40cls_segmentation"]

val_transforms = copy.deepcopy(base_config.val_transforms)
assert val_transforms[0]["type"] == "Resize", val_transforms
del val_transforms[0]

preprocess = base_config.preprocess
inference_model = base_config.get_inference_models(
    backbone_arch="swin-base",
    neck_arch="BiFPN",
    head_arch="semantic_segmentation",
    postprocess_arch="semantic_segmentation",
    task_names=task_names,
    updates=dict(head_update_fn=_update_head),
)


def postprocess(
    preds: Dict[str, List[Mask]],
    data: Dict[str, Any],
    parsing2polygon_fn: Callable,
    task_name: str,
):
    pred_results = []
    for pred in preds[task_name]:
        pred_seg = pred.mask.cpu().numpy().astype(np.uint8)
        index_polygon_classids = parsing2polygon_fn(pred_seg)
        instances = []
        for index, polygon, classid in index_polygon_classids:
            polygon = Polygon2D(topic=task_name, data=polygon)
            attr_index = Attribute(topic="index", value=index)
            attr_classid = Attribute(topic="class_id", value=classid)
            cls_name = cls_name_mapping[classid]
            attr_clsname = Attribute(topic="type", value=cls_name)
            instance = Instance(
                topic=task_name,
                polygon2ds=[polygon],
                attributes=[attr_index, attr_classid, attr_clsname],
            )
            instances.append(instance)
        pred_results.append(instances)
    return pred_results, data


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
            parsing2polygon_fn=Parsing2Polygon(),
            task_name=task_names[0],
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
