import uuid
from functools import partial
from typing import Any, Callable, Dict, List, Union

import cv2
import numpy as np
import torch
from hatbc.message import BBox2D, CameraFrame, Instance
from horizon_plugin_pytorch.march import March

from hat.core.box_utils import bbox_overlaps
from hat.core.data_struct.app_struct import DetObjects
from hat.data.collates.collates import default_collate_v2
from hat.registry import build_from_registry
from hat.utils.apply_func import _as_list

task_names = [
    "person_detection",
    "cyclist_detection",
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
                        nms=dict(
                            name="nms", iou_threshold=0.65, max_per_img=100
                        ),
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

arch = "base_w12"
backbone = dict(
    type="SwinTransformer",
    depth_list=depths[arch],
    num_heads=num_heads[arch],
    embedding_dims=embed_dims[arch],
    window_size=window_sizes[arch],
    include_top=False,
    flat_output=False,
    pretrained_path=None,
    node_name="backbone",
)

in_stride2channels = stride2channels_dict[arch]
out_stride2channels = {s: 320 for s in [4, 8, 16, 32, 64]}
in_strides = list(in_stride2channels.keys())
out_strides = list(out_stride2channels.keys())
start_level = in_strides.index(list(out_stride2channels.keys())[0])
neck = dict(
    type="BiFPN",
    fpn_name="bifpn_sum",
    in_strides=in_strides,
    out_strides=out_strides,
    stride2channels=in_stride2channels,
    out_channels=out_stride2channels,
    start_level=start_level,
    num_outs=5,
    stack=5,
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
    data: Union[CameraFrame, List[CameraFrame]],
    transforms: List[Callable],
) -> Dict:
    def _transform(frame):
        img_meta = camera_frame_to_hat_dict(frame)
        for t in transforms:
            img_meta = t(img_meta)
        return img_meta

    data = _as_list(data)
    img_metas = list(map(_transform, data))
    batch_data = default_collate_v2(img_metas)
    return batch_data


# merge overlapped cyclist and person detections
# by removing one of them that has a lower detection score
def remove_duplicate(
    preds: Dict[str, List[DetObjects]],
    data: Dict[str, Any],
    iou_thr: float = 0.9,
):
    batch_size = len(data["img"])
    for i in range(batch_size):
        person_dets = preds["person_detection"][i]
        cyclist_dets = preds["cyclist_detection"][i]
        person_detboxes2d = person_dets.person_detection
        cyclist_detboxes2d = cyclist_dets.cyclist_detection
        person_bboxes = person_detboxes2d.boxes
        person_scores = person_detboxes2d.scores
        person_cls_idxs = person_detboxes2d.cls_idxs
        cyclist_bboxes = cyclist_detboxes2d.boxes
        cyclist_scores = cyclist_detboxes2d.scores
        cyclist_cls_idxs = cyclist_detboxes2d.cls_idxs
        person_det_num = person_scores.size(0)
        cyclist_det_num = cyclist_scores.size(0)

        if len(person_bboxes) > 0 and len(cyclist_bboxes) > 0:
            # find those duplicates that have an IoU > iou_thr
            ious = bbox_overlaps(person_bboxes, cyclist_bboxes)
            max_ious, max_ids = torch.max(ious, dim=1, keepdim=True)
            indexes = ((ious >= iou_thr) & (ious == max_ious)).nonzero()

            person_ids = set(range(person_det_num))
            cyclist_ids = set(range(cyclist_det_num))
            person_remove_ids = set()
            cyclist_remove_ids = set()

            # keep those dets that have a higher score
            for i in torch.arange(indexes.size(0)):
                person_id = indexes[i][0]
                cyclist_id = indexes[i][1]
                if person_scores[person_id] <= cyclist_scores[cyclist_id]:
                    person_remove_ids.add(person_id.item())
                else:
                    cyclist_remove_ids.add(cyclist_id.item())
            person_kept_ids = list(person_ids.difference(person_remove_ids))
            cyclist_kept_ids = list(cyclist_ids.difference(cyclist_remove_ids))

            # remove those duplicates
            person_detboxes2d.boxes = person_bboxes[person_kept_ids, :]
            person_detboxes2d.scores = person_scores[person_kept_ids]
            person_detboxes2d.cls_idxs = person_cls_idxs[person_kept_ids]
            cyclist_detboxes2d.boxes = cyclist_bboxes[cyclist_kept_ids, :]
            cyclist_detboxes2d.scores = cyclist_scores[cyclist_kept_ids]
            cyclist_detboxes2d.cls_idxs = cyclist_cls_idxs[cyclist_kept_ids]

    return preds


def postprocess(preds: Dict[str, List[DetObjects]], data: Dict[str, Any]):
    preds = remove_duplicate(preds, data, iou_thr=0.9)
    pred_results = []
    batch_size = len(data["img"])
    for i in range(batch_size):
        instances = []
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
                instances.append(ins)
        pred_results.append(instances)
    return pred_results, data


inference = dict(
    type="Inference",
    device=None,
    march=March.BERNOULLI2,
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
