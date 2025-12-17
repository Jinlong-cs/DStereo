import os
import uuid
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
    "vehicle_wheel_detection",
    "person_head_detection",
    "person_face_detection",
    "vehicle_light_detection",
    "cyclist_wheel_detection",
    "traffic_light_len_detection",
    "cycle_detection",
    "vehicle_plate_detection",
]

mean = [123.675, 116.28, 103.53]
std = [58.395, 57.12, 57.375]
val_transforms = [
    dict(type="ToTensor", to_yuv=False),
    dict(type="Normalize", mean=mean, std=std),
    dict(type="Pad", divisor=128, pad_val=0),
    dict(
        type="AddKeys",
        kv={
            "img_id": 0,
            "img_name": "",
            "scale_factor": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        },
    ),
]

trt_opt = os.environ.get("INFERENCE_TENSORRT_OPT", False)


def build_detector_fcos(backbone, neck, task_name=""):
    in_strides = [4, 8, 16, 32, 64]
    out_strides = [8, 16, 32, 64, 128]
    stride2channels = {s: 320 for s in [4, 8, 16, 32, 64, 128]}
    feat_channels = stride2channels[in_strides[0]]
    add_stride = True
    stacked_convs = 4
    num_classes = 1
    score_thr = 0.1
    cls_name_mapping = {0: "_".join(task_name.split("_")[:-1])}

    return dict(
        type="BMFCOS",
        backbone=backbone,
        neck=neck,
        head=dict(
            type="FCOSHead",
            num_classes=num_classes,
            in_strides=in_strides,
            out_strides=out_strides,
            stride2channels=stride2channels,
            upscale_bbox_pred=True,
            feat_channels=feat_channels,
            stacked_convs=stacked_convs,
            share_conv=True,
            use_sigmoid=True,
            share_bn=True,
            dequant_output=True,
            int8_output=False,
            use_plain_conv=True,
            use_gn=True,
            use_scale=True,
            add_stride=add_stride,
            skip_qtensor_check=True,
            node_name=f"{task_name}_head",
        ),
        postprocess=dict(
            type="MultiInputSequential",
            modules=[
                dict(
                    type="FCOSDecoder",
                    num_classes=num_classes,
                    strides=out_strides,
                    nms_use_centerness=True,
                    nms_sqrt=False,
                    filter_score_mul_centerness=True,
                    transforms=val_transforms,
                    inverse_transform_key=["scale_factor"],
                    test_cfg=dict(
                        score_thr=score_thr,
                        nms_pre=1000,
                        nms=dict(name="nms", iou_threshold=0.65),
                        max_per_img=100,
                    ),
                    node_name=f"{task_name}_postprocess",
                ),
                dict(
                    type="FCOSConverter",
                    task_name=task_name,
                    cls_name_mapping=cls_name_mapping,
                    node_name=f"{task_name}_converter",
                ),
            ],
        ),
    )


backbone = dict(
    type="SwinTransformer",
    depth_list=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    embedding_dims=128,
    window_size=12,
    include_top=False,
    flat_output=False,
    pretrained_path=None,
    node_name="backbone",
    use_native_op=True if trt_opt else False,
)

neck = dict(
    type="BiFPN",
    fpn_name="bifpn_sum",
    in_strides=[4, 8, 16, 32],
    out_strides=[4, 8, 16, 32, 64],
    stride2channels={4: 128, 8: 256, 16: 512, 32: 1024},
    out_channels={s: 320 for s in [4, 8, 16, 32, 64]},
    start_level=0,
    end_level=-1,
    num_outs=5,
    stack=5,
    node_name="neck",
)
if trt_opt:
    neck.update(dict(use_fx=True, upsample_type="torch_c"))


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
                ins = Instance(
                    topic=task_name,
                    bbox2ds=[bbox2d],
                    track_id=uuid.uuid4().hex,
                )
                img_res.append(ins)
        pred_results.append(img_res)
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

if bool(int(trt_opt)):
    from hat.utils.compile_backends import tensorRT_backend

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
