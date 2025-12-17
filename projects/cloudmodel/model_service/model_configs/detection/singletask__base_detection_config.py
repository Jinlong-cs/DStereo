import os
import uuid
from typing import Any, Callable, Dict, List, Union

import cv2
import numpy as np
from hatbc.message import BBox2D, CameraFrame, Instance
from perception_proto.frame_pb2 import CameraFrame as CameraFrameProto

from hat.core.data_struct.app_struct import DetObjects
from hat.data.collates.collates import default_collate_v2
from hat.models.model_convert.converters import TorchCompile
from hat.models.model_convert.pipelines import ModelConvertPipeline
from hat.utils.apply_func import _as_list

stride2channels_dict = {
    "tiny": {4: 96, 8: 192, 16: 384, 32: 768},
    "small": {4: 96, 8: 192, 16: 384, 32: 768},
    "base": {4: 128, 8: 256, 16: 512, 32: 1024},
    "base_w12": {4: 128, 8: 256, 16: 512, 32: 1024},
    "large": {4: 192, 8: 384, 16: 768, 32: 1536},
    "large_w12": {4: 192, 8: 384, 16: 768, 32: 1536},
}
embed_dims = {
    "tiny": 96,
    "small": 96,
    "base": 128,
    "base_w12": 128,
    "large": 192,
    "large_w12": 192,
}
depths = {
    "tiny": [2, 2, 6, 2],
    "small": [2, 2, 18, 2],
    "base": [2, 2, 18, 2],
    "base_w12": [2, 2, 18, 2],
    "large": [2, 2, 18, 2],
    "large_w12": [2, 2, 18, 2],
}
num_heads = {
    "tiny": [3, 6, 12, 24],
    "small": [3, 6, 12, 24],
    "base": [4, 8, 16, 32],
    "base_w12": [4, 8, 16, 32],
    "large": [6, 12, 24, 48],
    "large_w12": [6, 12, 24, 48],
}
window_sizes = {
    "tiny": 7,
    "small": 7,
    "base": 7,
    "base_w12": 12,
    "large": 7,
    "large_w12": 12,
}
trt_opt = os.environ.get("INFERENCE_TENSORRT_OPT", False)


def get_backbone(arch, backbone_update_fn=lambda x: x):
    backbone = dict(
        type="SwinTransformer",
        depth_list=depths[arch],
        num_heads=num_heads[arch],
        embedding_dims=embed_dims[arch],
        window_size=window_sizes[arch],
        flat_output=False,
        include_top=False,
        node_name="backbone",
        use_native_op=True if trt_opt else False,
    )
    backbone = backbone_update_fn(backbone)
    return backbone


def get_neck(backbone_arch, neck_arch, neck_update_fn=lambda x: x):
    if neck_arch == "BiFPN":
        neck = dict(
            type="BiFPN",
            fpn_name="bifpn_sum",
            in_strides=[4, 8, 16, 32],
            out_strides=[4, 8, 16, 32, 64],
            stride2channels=stride2channels_dict[backbone_arch],
            out_channels={s: 320 for s in [4, 8, 16, 32, 64]},
            stack=5,
            start_level=0,
            end_level=-1,
            num_outs=5,
            node_name="neck",
        )
        if trt_opt:
            neck.update(dict(use_fx=True, upsample_type="torch_c"))

    elif neck_arch == "PAFPN":
        in_stride2channels = stride2channels_dict[backbone_arch]
        out_stride2channels = {s: 320 for s in [4, 8, 16, 32, 64]}
        in_strides = list(in_stride2channels.keys())
        start_level = in_strides.index(list(out_stride2channels.keys())[0])
        neck = dict(
            type="PAFPN",
            in_channels=list(in_stride2channels.values()),
            out_channels=out_stride2channels,
            out_strides=list(out_stride2channels.keys()),
            start_level=start_level,
            add_extra_convs="on_output",  # use P5
            num_outs=len(out_stride2channels),
            relu_before_extra_convs=True,
            norm_cfg={"norm_type": "gn", "num_groups": 32},
            node_name="neck",
        )
    else:
        raise ValueError
    neck = neck_update_fn(neck)
    return neck


def get_head(task_name, head_update_fn=lambda x: x):
    head = dict(
        type="FCOSHead",
        num_classes=1,
        in_strides=[4, 8, 16, 32, 64],
        out_strides=[8, 16, 32, 64, 128],
        stride2channels={
            4: 320,
            8: 320,
            16: 320,
            32: 320,
            64: 320,
            128: 320,
        },
        upscale_bbox_pred=True,
        feat_channels=320,
        stacked_convs=4,
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
    )
    head = head_update_fn(head)

    return head


def get_postprocess(task_name, postprocess_update_fn=lambda x: x):
    postprocess = dict(
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
                    score_thr=0.1,
                    nms_pre=1000,
                    nms=dict(name="nms", iou_threshold=0.65, max_per_img=100),
                ),
                to_cpu=False,
                node_name=f"{task_name}_postprocess",
            ),
            dict(
                type="FCOSConverter",
                task_name=task_name,
                cls_name_mapping={0: "_".join(task_name.split("_")[:-1])},
                node_name=f"{task_name}_converter",
            ),
        ],
    )
    postprocess = postprocess_update_fn(postprocess)
    return postprocess


def get_inference_models(backbone_arch, neck_arch, task_names, updates=None):
    updates = updates if updates is not None else dict()
    inputs = dict(
        img=None,
        img_shape=None,
        img_height=None,
        img_width=None,
        layout=None,
        img_name=None,
        img_id=None,
        scale_factor=None,
    )
    val_model = dict(
        type="MultitaskGraphModel",
        inputs=inputs,
        opt_inputs={},
        task_inputs={task_name: {} for task_name in task_names},
        task_modules={
            task_name: build_detector_fcos(
                backbone_arch, neck_arch, task_name, updates
            )
            for task_name in task_names
        },
        funnel_modules={},
        flatten_outputs=False,
        lazy_forward=True,
    )
    return val_model


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


# ##########################################
# external configs
# ###########################################

val_transforms = [
    dict(type="ToTensor", to_yuv=False),
    dict(
        type="Normalize",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
    ),
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


def build_detector_fcos(backbone_arch, neck_arch, task_name, updates):
    backbone = get_backbone(
        backbone_arch,
        backbone_update_fn=updates.get("backbone_update_fn", lambda x: x),
    )
    neck = get_neck(
        backbone_arch,
        neck_arch,
        neck_update_fn=updates.get("neck_update_fn", lambda x: x),
    )
    head = get_head(
        task_name=task_name,
        head_update_fn=updates.get("head_update_fn", lambda x: x),
    )
    postprocess = get_postprocess(
        task_name, updates.get("postprocess_update_fn", lambda x: x)
    )

    return dict(
        type="BMFCOS",
        backbone=backbone,
        neck=neck,
        head=head,
        target=None,
        loss=None,
        postprocess=postprocess,
    )


def postprocess(
    preds: Dict[str, List[DetObjects]],
    data: Dict[str, Any],
    task_names,
):
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


# tensorRT configs
if bool(int(trt_opt)):
    from hat.utils.compile_backends import tensorRT_backend

    model_convert_pipeline = dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="TorchCompile",
                compile_backend=tensorRT_backend(
                    fp16=True, dynamic_shape=True
                ),
                load_extensions=[
                    "group_norm",
                    "mean_replace_adp_avg_pool2d",
                    "c_interpolate",
                    "dynamic_shape_pad",
                ],
            )
        ],
    )
else:
    model_convert_pipeline = None
