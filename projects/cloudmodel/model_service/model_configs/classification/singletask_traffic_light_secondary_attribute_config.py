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
                    "traffic_lights_style": [
                        "L_Circle",
                        "L_Forward",
                        "L_Left",
                        "L_Right",
                        "L_Return",
                        "L_Pedestrain",
                        "L_Non_Motor",
                        "L_Time",
                        "L_Other",
                        "L_left_and_return",
                        "L_Forward_and_Left",
                        "L_Forward_and_Right",
                        "L_No_Drive_into",
                        "L_Allow_Drive_into",
                        "text_of_allow_ped",
                        "sign_of_allow_ped",
                        "text_of_forbid_ped",
                        "sign_of_forbid_ped",
                        "L_unknown",
                    ],
                    "traffic_lights_type": [
                        "red",
                        "green",
                        "yellow",
                        "white",
                        "unknow",
                        "other",
                    ],
                    "tl_color_confidence": ["High", "Middle", "Low"],
                    "tl_type_confidence": ["High", "Middle", "Low"],
                    "ignore": ["no", "yes"],
                    "occlusion": [
                        "full_visible",
                        "occluded",
                        "heavily_occluded",
                        "invisible",
                    ],
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
    config["num_classes"] = 37
    return config


# -----------
# external
# -----------
task_names = ["traffic_light_secondary_attribute"]
object_type = "traffic_light_len"  # 和检测结果的topic一致
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
