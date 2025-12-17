from functools import partial
from typing import Any, Callable, Dict, List

import singletask__base_segmentation_config as base_config
from hatbc.message import Instance, Polygon2D

from hat.core.mask2polygon import Mask2Polygon
from hat.registry import build_from_registry
from hat.utils.apply_func import _as_list

# task desc
desc_attributes = dict(
    type=[
        "solid",
        "dashed",
        "wide_solid",
        "wide_dashed",
        "mixed",
        "Road_teeth",
    ],
    double_line=["no", "yes"],
    color=["white", "yellow", "green", "blue", "red", "other"],
    occlusion=[
        "full_visible",
        "occluded",
        "heavily_occluded",
        "snow_occluded",
    ],
    deceleration_lane=["no", "yes"],
    tidal_lane=["no", "yes"],
)
cls_name_mapping = {
    k: {i: v_i for i, v_i in enumerate(v)} for k, v in desc_attributes.items()
}
attr_name2num = {k: len(v) for k, v in desc_attributes.items()}
num_classes = attr_name2num.pop("type")
strides = [4]
num_grids = [100]
kernel_out_channels = 128
upsample_mask_logit = True


def postprocess(
    preds: Dict[str, List[List[Instance]]],
    data: Dict[str, Any],
    mask2polygon_fn: Callable,
    task_name,
):
    pred_results = []
    for pred in preds[task_name]:
        for instance in pred:
            mask = instance.mask2ds[0]
            polygons = mask2polygon_fn(mask.mask)
            polygons = _as_list(polygons)
            instance.polygon2ds = [
                Polygon2D(topic=task_name, data=p) for p in polygons
            ]
        pred_results.append(pred)
    return pred_results, data


def _head_update_fn(config):
    config["num_classes"] = num_classes
    config["attr_name2num"] = attr_name2num
    config["num_grids"] = num_grids
    config["upsample_mask_logit"] = upsample_mask_logit
    config["kernel_out_channels"] = kernel_out_channels
    config["mask_feature_head_updater"] = dict(
        out_channels=kernel_out_channels,
    )
    return config


def _postprocess_update_fn(config):
    config["modules"][0]["kernel_out_channels"] = kernel_out_channels
    config["modules"][0]["num_classes"] = num_classes
    config["modules"][0]["num_grids"] = num_grids
    config["modules"][0]["strides"] = strides
    config["modules"][0]["upsample_mask_logit"] = upsample_mask_logit
    config["modules"][0]["attr_names"] = list(desc_attributes.keys())
    config["modules"][0]["cls_name_mapping"] = cls_name_mapping
    return config


# ----------------------external -------------------------------
task_names = ["lane_instanceseg"]

val_transforms = base_config.val_transforms
preprocess = base_config.preprocess
inference_model = base_config.get_inference_models(
    backbone_arch="swin-small",
    neck_arch="BiFPN",
    head_arch="instance_segmentation",
    postprocess_arch="instance_segmentation",
    task_names=task_names,
    updates=dict(
        head_update_fn=_head_update_fn,
        postprocess_update_fn=_postprocess_update_fn,
    ),
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
            mask2polygon_fn=Mask2Polygon(),
            task_name=task_names[0],
        )
    ],
    model=inference_model,
    model_convert_pipeline=None,
)
