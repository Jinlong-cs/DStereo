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
                    "cn_medium_type": [
                        "I_IndicationRectangle",
                        "U_Unknown",
                        "Prohibit_I",
                        "RouteID_Other",
                        "I_IndicationCircle",
                        "C_Unknown",
                        "PSL_Unknown",
                        "P_Unknown1",
                        "I_RectangleMisc",
                        "ProhibitRemove",
                        "TravelSign_BrownWhite",
                        "False_Positive",
                        "PSL_Other",
                        "I_Other_rectangle",
                        "GSGW_Other",
                        "MSL_Other",
                        "TSBW_Other",
                        "W_Other",
                        "MSL_Unknown",
                        "sign_Construction",
                        "RSBW_Other",
                        "PR_Other",
                        "RouteID_Unknown",
                        "RouteID",
                        "GuideSign_GreenWhite",
                        "Min_SpeedLim",
                        "I_Other_circle",
                        "GSWB_Unknown",
                        "TSBW_Unknown",
                        "P_ElectricalSpeedLimit",
                        "W_Unknown",
                        "GSWB_Misc",
                        "RoadSign_BlueWhite",
                        "I_Unknown_circle",
                        "P_Other1",
                        "GSWB_Other",
                        "C_Other",
                        "PR_Unknown",
                        "W_Warning",
                        "I_Unknown_rectangle",
                        "sign_back",
                        "GSGW_Misc",
                        "P_SpeedLimit",
                        "P_Unknow_SpeedLimele",
                        "RSBW_Misc",
                        "GuideSign_WhiteBlack",
                        "W_Misc",
                        "GSGW_Unknown",
                        "O_Other",
                        "P_Other_SpeedLimele",
                        "RSBW_Unknown",
                    ],
                    "occlusion": [
                        "full_visible",
                        "occluded",
                        "heavily_occluded",
                        "invisible",
                    ],
                    "ignore": ["no", "yes"],
                    "Inside": ["no", "yes", "outside", "inside"],
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
    config["num_classes"] = 61
    return config


# -----------
# external
# -----------
task_names = ["traffic_sign_medium_attribute"]
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
