from functools import partial

import singletask__base_detection_config as base_config

from hat.registry import build_from_registry


def _head_update_fn(src):
    src.update(
        dict(
            num_classes=4,
            out_strides=[4, 8, 16, 32, 64],
            stride2channels={s: 320 for s in [4, 8, 16, 32, 64]},
            feat_channels=320,
            add_stride=False,
        )
    )
    return src


def _fcos_postprocess_update_fn(src):
    src["modules"][0]["num_classes"] = 4
    src["modules"][0]["strides"] = [4, 8, 16, 32, 64]
    src["modules"][0]["nms_sqrt"] = True
    src["modules"][1]["cls_name_mapping"] = {
        0: "head_light",
        1: "tail_light",
        2: "street_light",
        3: "reflection_point",
    }
    return src


# -----------
# external
# -----------
task_names = ["adb_light_detection"]
val_transforms = base_config.val_transforms
preprocess = base_config.preprocess
postprocess = base_config.postprocess

inference_model = base_config.get_inference_models(
    backbone_arch="base_w12",
    neck_arch="PAFPN",
    task_names=task_names,
    updates=dict(
        head_update_fn=_head_update_fn,
        postprocess_update_fn=_fcos_postprocess_update_fn,
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
    model_convert_pipeline=None,
)
