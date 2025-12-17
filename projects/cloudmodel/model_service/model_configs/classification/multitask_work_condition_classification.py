from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Union

import torch
from hatbc.message import Attribute, CameraFrame, Instance
from perception_proto.frame_pb2 import CameraFrame as CameraFrameProto

from hat.core.data_struct.app_struct import DetObjects
from hat.data.collates.collates import default_collate_v2
from hat.registry import build_from_registry
from hat.utils.apply_func import _as_list

task_names = [
    "scene_classification",
    "weather_classification",
    "illumination_classification",
    "time_classification",
]
task_num_classes = [6, 6, 4, 3]
input_hw = (234, 456)
inputs = dict(img=torch.zeros((1, 3, *input_hw)))

val_transforms = [
    dict(type="YUVTurboJPEGDecoder", to_string=True),
    dict(type="ImgBufToYUV444"),
    dict(
        type="CV2AdptiveResolutionInput",
        model_input_hw=input_hw,
        scale_type="MIN",
    ),
    dict(type="ToTensor", to_yuv=False),
    dict(type="Normalize", mean=128.0, std=128.0),
]

object_type = "work_condition"
task_type = "classification"
val_decoders = {}
val_decoders[object_type] = (
    task_names,
    dict(
        type="AttrDecoder",
        model_name="wk_classification_hatv1",
        task_descs=OrderedDict(
            [
                ("scene_classification", ("pred_cls", task_type)),
                ("weather_classification", ("pred_cls", task_type)),
                ("illumination_classification", ("pred_cls", task_type)),
                ("time_classification", ("pred_cls", task_type)),
            ]
        ),
        node_name=f"{object_type}_decoder",
        name_dict={
            "scene_classification": [
                "Highway",
                "Urban",
                "Rural",
                "Tunnel",
                "Charge_Station",
                "Underground_Parking_Lot",
            ],
            "weather_classification": [
                "Sunny",
                "Cloudy",
                "Rainy",
                "Snowy",
                "Heavy_Rain",
                "Other",
            ],
            "illumination_classification": [
                "natural_light",
                "lamplight",
                "Hard_Light",
                "dark",
            ],
            "time_classification": ["Day", "Night", "Other"],
        },
        merge_result=True,
    ),
)


def build_model(backbone, num_classes, task_name):
    bn_kwargs = dict(eps=1e-5, momentum=0.1)
    model = dict(
        type="WorkConditionClassifier",
        backbone=backbone,
        backbone_extra=torch.nn.Identity(),
        prediction_head=dict(
            type="WorkConditionClsHead",
            output_dim=backbone["output_dim"],
            bn_kwargs=bn_kwargs,
            num_classes=num_classes,
            in_channel=2048,
            avg_pool_size=4,
            node_name=f"{task_name}_prediction_head",
        ),
        losses=None,
    )
    return model


backbone = dict(
    type="WorkConditionResNet",
    layers=(3, 4, 6, 3),
    output_dim=1024,
    heads=32,
    input_resolution=input_hw,
    channel_list=[32, 64, 128, 256, 512],
    bn_kwargs=dict(eps=1e-5, momentum=0.1),
    use_attnpool=False,
    node_name="backbone",
)


def get_model():
    converted_decoders = OrderedDict(
        {
            (tuple(tasks), group): decoder
            for group, (tasks, decoder) in val_decoders.items()
        }
    )
    return dict(
        type="MultitaskGraphModel",
        inputs=inputs,
        task_inputs={task: dict() for task in task_names},
        task_modules={
            task_name: build_model(backbone, num_classes, task_name)
            for num_classes, task_name in zip(task_num_classes, task_names)
        },
        funnel_modules=converted_decoders,
        flatten_outputs=False,
        lazy_forward=False,
    )


multitask_graph_deploy_model = get_model()


def camera_frame_to_hat_dict(msg: CameraFrame):
    """Convert camera frame to hat dict.

    Args:
        msg: camera frame.
    """
    img = msg.image.data
    img_meta = dict(
        img_buf=img,
        layout="hwc",
    )
    return img_meta


def preprocess(
    data: Union[
        CameraFrame,
        List[CameraFrame],
        CameraFrameProto,
        List[CameraFrameProto],
    ],
    transforms: List[Callable],
) -> Dict:
    def _transform(frame):
        img_meta = camera_frame_to_hat_dict(frame)
        for t in transforms:
            img_meta = t(img_meta)
        return img_meta

    data = _as_list(data)
    data = [
        d if isinstance(d, CameraFrame) else CameraFrame.from_proto(d)
        for d in data
    ]
    img_metas = list(map(_transform, data))
    batch_data = default_collate_v2(img_metas)
    return batch_data


def postprocess(preds: Dict[str, List[DetObjects]], data: Dict[str, Any]):
    pred_results = []
    for pred in preds[object_type]:
        instance = Instance(topic=object_type)
        for k, v in pred["attrs"].items():
            instance.attributes.append(
                Attribute(
                    topic=k,
                    value=v,
                    score=max(pred.get("cls_score_2pe", None)[k]),
                )
            )
        pred_results.append([instance])
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
    post_processors=[postprocess],
    model=multitask_graph_deploy_model,
    model_convert_pipeline=None,
)
