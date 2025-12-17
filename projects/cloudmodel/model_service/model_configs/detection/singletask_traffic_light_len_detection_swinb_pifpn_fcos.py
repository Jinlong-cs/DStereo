from functools import partial

import singletask__base_detection_config as base_config

from hat.registry import build_from_registry

task_names = ["traffic_light_len_detection"]

val_transforms = [
    dict(type="Resize", img_scale=(3264, 5760)),
    dict(type="ToTensor", to_yuv=False),
    dict(
        type="Normalize",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
    ),
    dict(type="Pad", size=(3264, 5760), pad_val=0),
    dict(
        type="AddKeys",
        kv={
            "img_id": 0,
            "img_name": "",
        },
    ),
]


def _update_head(config):
    config["out_strides"] = [4, 8, 16, 32, 64]
    config["add_stride"] = False
    return config


def _update_postprocess(config):
    config["modules"][0]["strides"] = [4, 8, 16, 32, 64]
    config["modules"][0]["test_cfg"] = dict(
        score_thr=0.25,
        nms_pre=1000,
        nms=dict(name="nms", iou_threshold=0.4),
        max_per_img=100,
    )
    config["modules"][0]["transforms"] = val_transforms
    return config


preprocess = base_config.preprocess
postprocess = base_config.postprocess
inference_model = base_config.get_inference_models(
    backbone_arch="base",
    neck_arch="BiFPN",
    task_names=task_names,
    updates=dict(
        head_update_fn=_update_head,
        postprocess_update_fn=_update_postprocess,
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
