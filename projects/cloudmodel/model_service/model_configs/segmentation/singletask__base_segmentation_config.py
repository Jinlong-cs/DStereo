import os
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Union

import cv2
import numpy as np
from hatbc.message import CameraFrame, DenseMap, Mask2D, MessageMeta
from hatbc.utils.compress import CompressMethod
from perception_proto.frame_pb2 import CameraFrame as CameraFrameProto

from hat.core.data_struct.base_struct import Mask
from hat.data.collates.collates import default_collate_v2
from hat.utils.apply_func import _as_list

trt_opt = os.environ.get("INFERENCE_TENSORRT_OPT", False)

stride2channels_dict = {
    "swin-tiny": {4: 96, 8: 192, 16: 384, 32: 768},
    "swin-small": {4: 96, 8: 192, 16: 384, 32: 768},
    "swin-base": {4: 128, 8: 256, 16: 512, 32: 1024},
    "swin-base_w12": {4: 128, 8: 256, 16: 512, 32: 1024},
    "swin-large": {4: 192, 8: 384, 16: 768, 32: 1536},
    "swin-large_w12": {4: 192, 8: 384, 16: 768, 32: 1536},
    "efficient-b5": {2: 24, 4: 40, 8: 64, 16: 176, 32: 512},
}
embed_dims = {
    "swin-tiny": 96,
    "swin-small": 96,
    "swin-base": 128,
    "swin-base_w12": 128,
    "swin-large": 192,
    "swin-large_w12": 192,
}
depths = {
    "swin-tiny": [2, 2, 6, 2],
    "swin-small": [2, 2, 18, 2],
    "swin-base": [2, 2, 18, 2],
    "swin-base_w12": [2, 2, 18, 2],
    "swin-large": [2, 2, 18, 2],
    "swin-large_w12": [2, 2, 18, 2],
}
num_heads = {
    "swin-tiny": [3, 6, 12, 24],
    "swin-small": [3, 6, 12, 24],
    "swin-base": [4, 8, 16, 32],
    "swin-base_w12": [4, 8, 16, 32],
    "swin-large": [6, 12, 24, 48],
    "swin-large_w12": [6, 12, 24, 48],
}
window_sizes = {
    "swin-tiny": 7,
    "swin-small": 7,
    "swin-base": 7,
    "swin-base_w12": 12,
    "swin-large": 7,
    "swin-large_w12": 12,
}


def swin_backbone(arch):
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
    return backbone


def efficient_backbone(arch):
    backbone = dict(
        type="EfficientNet",
        activation="swish",
        bn_kwargs={},
        coefficient_params=(1.6, 2.2, 456, 0.4),
        flat_output=False,
        include_top=False,
        model_type="b5",
        num_classes=1000,
        use_se_block=True,
        node_name="backbone",
    )
    return backbone


def get_backbone(backbone_arch, backbone_update_fn=lambda x: x):
    if backbone_arch.startswith("swin-"):
        backbone = swin_backbone(backbone_arch)
    elif backbone_arch.startswith("efficient-"):
        backbone = efficient_backbone(backbone_arch)
    else:
        raise ValueError()
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
            out_channels={s: 256 for s in [4, 8, 16, 32, 64]},
            stack=3,
            start_level=0,
            end_level=-1,
            num_outs=5,
            node_name="neck",
        )
    else:
        raise ValueError
    neck = neck_update_fn(neck)
    if trt_opt:
        neck.update(use_fx=True, upsample_type="torch_c")
    return neck


def get_head(head_arch, task_name, head_update_fn=lambda x: x):
    if head_arch == "semantic_segmentation":
        head = dict(
            type="SemanticFPNplusHead",
            num_classes=40,
            in_strides=[4, 8, 16],
            in_channels=[256, 256, 256],
            upscale=True,
            node_name=f"{task_name}_head",
            use_native_op=True if trt_opt else False,
        )
    elif head_arch == "instance_segmentation":
        head = dict(
            type="SOLOV2Head",
            node_name=f"{task_name}_head",
            num_classes=None,
            num_levels=1,
            attr_name2num=None,
            use_dw_conv=False,
            use_native_op=True if trt_opt else False,
        )
    else:
        raise ValueError
    head = head_update_fn(head)
    return head


def get_postprocess(
    postprocess_arch, task_name, postprocess_update_fn=lambda x: x
):
    if postprocess_arch == "semantic_segmentation":
        postprocess = dict(
            type="BMSegDecoder",
            out_strides=[2],
            do_inverse_transform=True,
            to_cpu=True,
            node_name=f"{task_name}_decoder",
        )
    elif postprocess_arch == "instance_segmentation":
        postprocess = dict(
            type="MultiInputSequential",
            modules=[
                dict(
                    type="SOLOV2Decoder",
                    __build_recursive=False,
                    node_name=f"{task_name}_decoder",
                    num_classes=None,
                    test_cfg_updater=dict(
                        nms_pre=500,
                        score_thr=0.2,
                        mask_thr=0.5,
                        filter_thr=0.2,
                        kernel="gaussian",  # gaussian/linear
                        sigma=2.0,
                        max_per_img=50,
                    ),
                    object_name="lane",
                    dict_to_hat_struct=False,
                ),
                dict(
                    type="InstanceSegToMsg",
                    node_name=f"{task_name}_convert",
                    task_name=task_name,
                    to_cpu=True,
                ),
            ],
        )
    else:
        raise ValueError
    postprocess = postprocess_update_fn(postprocess)

    return postprocess


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


def build_segmentator(
    backbone_arch, neck_arch, head_arch, postprocess_arch, task_name, updates
):
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
        head_arch=head_arch,
        task_name=task_name,
        head_update_fn=updates.get("head_update_fn", lambda x: x),
    )
    postprocess = get_postprocess(
        postprocess_arch,
        task_name,
        updates.get("postprocess_update_fn", lambda x: x),
    )

    return dict(
        type="BMSegmentor",
        backbone=backbone,
        neck=neck,
        head=head,
        postprocess=postprocess,
        target=None,
        loss=None,
    )


# ##########################################
# external configs
# ###########################################

val_transforms = [
    dict(type="Resize", img_scale=(1280, 2048)),
    dict(type="ToTensor", to_yuv=False),
    dict(
        type="Normalize",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
    ),
    dict(type="Pad", pad_val=0, divisor=64),
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


def get_inference_models(
    backbone_arch,
    neck_arch,
    head_arch,
    postprocess_arch,
    task_names,
    updates=None,
):
    updates = updates if updates is not None else dict()

    val_model = dict(
        type="MultitaskGraphModel",
        inputs=dict(img=None, img_shape=None, img_height=None, img_width=None),
        opt_inputs={},
        task_inputs={task_name: {} for task_name in task_names},
        task_modules={
            task_name: build_segmentator(
                backbone_arch,
                neck_arch,
                head_arch,
                postprocess_arch,
                task_name,
                updates,
            )
            for task_name in task_names
        },
        funnel_modules={},
        flatten_outputs=False,
        lazy_forward=True,
    )
    return val_model


def postprocess(
    preds: Dict[str, List[Mask]],
    data: Dict[str, Any],
    task_name: str,
    cls_name_mapping: Dict[int, str] = None,
    parsing_desc: str = None,
):
    pred_results = []
    for pred in preds[task_name]:
        img_res = list()
        mask = Mask2D(
            topic=task_name,
            data=pred.mask.cpu(),
            data_compress_method=CompressMethod.IMENCODE_PNG,
        )
        img_res.append(
            DenseMap(
                topic=task_name,
                mask2ds=[mask],
                class_map=cls_name_mapping,
                meta=MessageMeta(source=parsing_desc),
            )
        )
        pred_results.append(img_res)
    return pred_results, data


def postprocess_reverse_for_mask(results, data):
    task_names = results.keys()
    preds = OrderedDict()
    for task_name in task_names:
        preds[task_name] = []
        result_list = results[task_name]
        for result in result_list:
            data = result[0].mask2ds[0].data.data
            mask = Mask(mask=data.clone().detach())
            preds[task_name].append(mask)
    return preds
