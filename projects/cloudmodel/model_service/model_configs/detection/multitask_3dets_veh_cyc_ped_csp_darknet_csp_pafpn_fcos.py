import os
from functools import partial
from typing import Any, Callable, Dict, List, Union

import cv2
import numpy as np
from hatbc.message import BBox2D, CameraFrame, Instance
from perception_proto.frame_pb2 import CameraFrame as CameraFrameProto

from hat.core.data_struct.app_struct import DetObjects
from hat.data.collates.collates import default_collate_v2
from hat.models.model_convert.converters import TorchCompile
from hat.models.model_convert.pipelines import ModelConvertPipeline
from hat.registry import build_from_registry
from hat.utils.apply_func import _as_list

task_names = [
    "vehicle_detection",
    "cyclist_detection",
    "person_detection",
]

mean = [123.675, 116.28, 103.53]
std = [58.395, 57.12, 57.375]
val_transforms = [
    dict(type="Resize", img_scale=(1280, 1920)),
    dict(type="ToTensor", to_yuv=False),
    dict(type="Normalize", mean=mean, std=std),
    dict(type="Pad", size=(1280, 1920), pad_val=0),
    dict(
        type="AddKeys",
        kv={
            "img_id": 0,
            "img_name": "",
        },
    ),
]

trt_opt = os.environ.get("INFERENCE_TENSORRT_OPT", False)


def build_detector_fcos(backbone, neck, task_name=""):
    return dict(
        type="BMFCOS",
        backbone=backbone,
        neck=neck,
        head=dict(
            type="FCOSHead",
            num_classes=1,
            in_strides=[4, 8, 16, 32, 64],
            out_strides=[8, 16, 32, 64, 128],
            stride2channels={
                4: 192,
                8: 192,
                16: 192,
                32: 192,
                64: 192,
                128: 192,
            },
            upscale_bbox_pred=True,
            feat_channels=192,
            stacked_convs=2,
            share_conv=True,
            use_sigmoid=True,
            share_bn=True,
            dequant_output=True,
            int8_output=False,
            use_plain_conv=True,
            use_gn=True,
            use_scale=True,
            add_stride=True,
            skip_qtensor_check=True,
            node_name=f"{task_name}_head",
        ),
        postprocess=dict(
            type="MultiInputSequential",
            modules=[
                dict(
                    type="FCOSDecoder",
                    num_classes=1,
                    strides=[8, 16, 32, 64, 128],
                    nms_use_centerness=True,
                    nms_sqrt=False,
                    filter_score_mul_centerness=True,
                    transforms=val_transforms,
                    inverse_transform_key=["scale_factor"],
                    test_cfg=dict(
                        score_thr=0.05,
                        nms_pre=1000,
                        nms=dict(name="nms", iou_threshold=0.65),
                        max_per_img=100,
                    ),
                    node_name=f"{task_name}_postprocess",
                ),
                dict(
                    type="FCOSConverter",
                    task_name=task_name,
                    cls_name_mapping={0: "_".join(task_name.split("_")[:-1])},
                    node_name=f"{task_name}_converter",
                ),
            ],
        ),
    )


backbone = dict(
    type="CSPDarknet",
    dep_mul=0.57,
    wid_mul=0.75,
    node_name="backbone",
)

neck = dict(
    type="CSPPAFPN",
    in_channels=[96, 192, 384, 768],
    num_csp_blocks=2,
    out_channels=192,
    num_outs=5,
    node_name="neck",
)

multitask_graph_deploy_model = dict(
    type="MultitaskGraphModel",
    inputs=dict(
        img=None,
        img_shape=None,
        img_height=None,
        img_width=None,
        layout=None,
        img_name=None,
        img_id=None,
        scale_factor=None,
    ),
    opt_inputs={},
    task_inputs={task_name: {} for task_name in task_names},
    task_modules={
        task_name: build_detector_fcos(backbone, neck, task_name)
        for task_name in task_names
    },
    funnel_modules={},
    flatten_outputs=False,
    lazy_forward=True,
)


def camera_frame_to_hat_dict(msg: CameraFrame):
    """Convert camera frame to hat dict.

    Args:
        msg: camera frame.
    """
    img = msg.image.data
    shape = img.shape
    layout = msg.image.layout.lower()
    assert layout == "hwc"
    color_space = msg.image.color_space
    assert color_space in ["rgb", "bgr"]
    if color_space == "bgr":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_meta = dict(
        img=img,
        layout=layout,
        img_shape=shape,
        color_space="rgb",
        img_height=shape[0],
        img_width=shape[1],
    )
    return img_meta


def preprocess(
    data: Union[
        CameraFrame,
        List[CameraFrame],
        CameraFrameProto,
        List[CameraFrameProto],
    ],
    transforms: List[Union[Dict, Callable]],
) -> Dict:
    transform_funs = [build_from_registry(t) for t in transforms]

    def _transform(frame):
        img_meta = camera_frame_to_hat_dict(frame)
        for t in transform_funs:
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
    batch_size = len(data["img"])
    for i in range(batch_size):
        img_res = list()
        for task_name in task_names:
            for pred in preds[task_name][i]:
                detbox2d = getattr(pred, task_name)
                box = detbox2d.box.cpu()
                score = detbox2d.score.cpu()
                class_name = detbox2d.cls_name
                bbox2d = BBox2D(topic=class_name, data=box, score=score)
                ins = Instance(topic=task_name, bbox2ds=[bbox2d])
                img_res.append(ins)
        pred_results.append(img_res)
    return pred_results, data


inference = dict(
    type="Inference",
    device=None,
    pre_processors=[partial(preprocess, transforms=val_transforms)],
    post_processors=[postprocess],
    model=multitask_graph_deploy_model,
    model_convert_pipeline=None,
)

if bool(int(trt_opt)):
    from hat.utils.compile_backends import tensorRT_backend

    neck.update(dict(use_native_op=True))
    inference.update(
        dict(
            model_convert_pipeline=ModelConvertPipeline(
                [
                    TorchCompile(
                        tensorRT_backend(fp16=True, dynamic_shape=True),
                        load_extensions=[
                            "group_norm",
                            "mean_replace_adp_avg_pool2d",
                            "c_interpolate",
                            "dynamic_shape_pad",
                        ],
                    )
                ]
            ),
        )
    )
