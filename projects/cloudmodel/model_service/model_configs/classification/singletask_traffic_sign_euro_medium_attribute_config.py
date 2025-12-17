from collections import OrderedDict
from functools import partial

import singletask__base_classification as base_config

from hat.registry import build_from_registry


def get_converted_decoders(task_name, task_type):
    val_decoders = {
        task_name: (
            [task_name],
            dict(
                type="AttrDecoder",
                model_name="attribute_cls_v1",
                task_descs=OrderedDict([(task_name, ("pred_cls", task_type))]),
                name_dict={
                    "gl_medium_type": [
                        "speed_limit",
                        "minimum_speed",
                        "prohibit_remove",
                        "complementary",
                        "false_positive",
                        "information",
                        "other_sign",
                        "regulatory",
                        "unknown_sign",
                        "warning",
                        "SL_Other_g1",
                        "SL_Unknown_g1",
                        "MS_Other_g1",
                        "MS_Unknown_g1",
                        "PR_Other_g1",
                        "PR_Unknown_g1",
                        "W_Other_g1",
                        "W_Unknown_g1",
                        "R_Other_g1",
                        "R_Unknown_g1",
                        "C_Other_g1",
                        "C_Unknown_g1",
                        "I_Other_g1",
                        "I_Unknown_g1",
                    ],
                    "occlusion": [
                        "full_visible",
                        "occluded",
                        "heavily_occluded",
                        "invisible",
                    ],
                    "ignore": ["no", "yes"],
                    "Inside": ["no", "outside", "inside"],
                    "is_ramp": ["no", "yes"],
                    "confidence": ["High", "Middle", "Low"],
                },
                node_name=f"{object_type}_decoder",
            ),
        )
    }
    converted_decoders = OrderedDict(
        {
            (tuple(tasks), group): decoder
            for group, (tasks, decoder) in val_decoders.items()
        }
    )
    return converted_decoders


def _head_update_fn(config):
    config["num_classes"] = 38
    return config


# -----------
# external
# -----------
task_names = ["traffic_sign_gl_medium_attribute"]
object_type = "traffic_sign"  # 和检测结果的topic一致
task_type = "classification"
input_hw = (224, 224)

val_transforms = base_config.get_val_transform(input_hw, object_type)
preprocess = base_config.preprocess
postprocess = base_config.postprocess
inference_model = base_config.get_inference_models(
    backbone_arch="swin-base",
    input_hw=input_hw,
    task_names=task_names,
    converted_decoders=get_converted_decoders(
        task_name=task_names[0], task_type=task_type
    ),
    updates=dict(head_update_fn=_head_update_fn),
)

inference = dict(
    type="Inference",
    device=None,
    pre_processors=[
        partial(
            preprocess,
            transforms=[build_from_registry(t) for t in val_transforms],
            object_type=object_type,
        )
    ],
    post_processors=[
        partial(
            postprocess,
            task_name=task_names[0],
        )
    ],
    model=inference_model,
    model_convert_pipeline=base_config.model_convert_pipeline,
)
