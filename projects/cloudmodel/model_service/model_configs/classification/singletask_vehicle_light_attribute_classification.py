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
                    "part": ["rear", "head", "side"],
                    "ignore": ["yes", "no"],
                    "hard_sample": ["no", "yes"],
                    "HighSideBrakeLightStatus": [
                        "off",
                        "on",
                        "Unknown",
                        "abnormal",
                        "not_exist",
                        "invisible",
                    ],
                    "left_right_side_brake_light_status": [
                        "off",
                        "on",
                        "abnormal",
                        "unknow",
                        "color_shiftt",
                        "clearance_lamp_on",
                    ],
                    "turn_signal_status": [
                        "turn_light_off",
                        "left_turn_light_on",
                        "right_turn_light_on",
                        "turn_light_all_on",
                        "turn_light_unknow",
                    ],
                    "close_veh": [
                        "not_exist",
                        "on",
                        "off",
                    ],
                    "day_night": ["night", "day"],
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
    config["num_classes"] = 29
    config["node_name"] = config["node_name"].replace(
        "_classification_prediction_head",
        "_recognition_classification_prediction_head",
    )
    return config


# -----------
# external
# -----------
task_names = ["vehicle_light"]
object_type = "rear"  # 和检测结果的topic一致
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
