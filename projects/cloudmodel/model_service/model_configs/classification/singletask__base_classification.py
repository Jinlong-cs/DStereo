import os
import uuid
from typing import Any, Callable, Dict, List, Union

import cv2
import torch
from hatbc.message import Attribute, CameraFrame, Instance
from perception_proto.frame_pb2 import CameraFrame as CameraFrameProto

from hat.core.data_struct.app_struct import DetObjects
from hat.data.collates.collates import default_collate_v2
from hat.models.model_convert.converters import TorchCompile
from hat.models.model_convert.pipelines import ModelConvertPipeline
from hat.utils.apply_func import _as_list

trt_opt = os.environ.get("INFERENCE_TENSORRT_OPT", False)


def collate_2d_cat(
    batch: List[Any], topic, append_roi_uuid=True
) -> Union[torch.Tensor, Dict]:
    elem = batch[0]
    # these key-value will skip default_collate
    return_data = {}

    for key in list(set(list(elem.keys()) + _as_list(topic))):
        if key == "img":
            imgs = [d["img"] for d in batch if d["img"] is not None]
            rois_valid = []
            if len(imgs) == 0:
                # add a dummy image when empty
                collate_data = torch.zeros(
                    (1, 3, elem["img_height"], elem["img_width"])
                )
                rois_valid = torch.tensor([False], dtype=torch.bool)
            else:
                collate_data = torch.cat(imgs, 0)
                rois_valid = torch.tensor(
                    [True] * len(collate_data), dtype=torch.bool
                )
            item = {"img": collate_data, "rois_valid": rois_valid}
            # items
        elif key == "obj_id":
            obj_ids = [j for i in batch for j in i["obj_id"]]
            # append a dummy object id for fake image
            if len(obj_ids) == 0:
                obj_ids.append("Dummy")
            item = {"obj_id": obj_ids}
        elif key == "ori_input":
            item = {"ori_input": [i["ori_input"] for i in batch]}
        elif key in _as_list(topic):
            topic_data = [
                d[key] for d in batch if d.get(key, None) is not None
            ]
            if len(topic_data) == 0:
                # add a dummy items
                topic_data = torch.zeros((1, 4))
            else:
                topic_data = torch.cat(topic_data, 0)
            item = {key: topic_data}
            if append_roi_uuid:
                assert "roi_uuid" not in elem
                roi_uuid = list()
                batch_uuid = list()
                for d in batch:
                    batch_uuid_i = uuid.uuid4().hex
                    batch_uuid.append(batch_uuid_i)
                    for t in _as_list(topic):
                        if t in d:
                            roi_uuid.extend([batch_uuid_i] * len(d[t]))
                if len(roi_uuid) == 0:
                    # add a dummy image uuid
                    roi_uuid = [uuid.uuid4().hex]
                assert len(roi_uuid) >= len(topic_data)

                item.update({"roi_uuid": roi_uuid, "batch_uuid": batch_uuid})
        else:
            collate_data = default_collate_v2(
                [d[key] for d in batch if key in d]
            )
            item = {key: collate_data}

        return_data.update(item)

    return return_data


def bbox2d_to_hat_dict(bbox, box_id):
    ret_label = {
        "id": box_id,
        "struct_type": "rect",
        "data": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
        "bbox_score": float(bbox.score),
        "attrs": {},
        "cls_score_2pe": {},
    }
    return ret_label


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


def get_backbone(backbone_arch, backbone_update_fn=lambda x: x):
    if backbone_arch.startswith("swin-"):
        backbone = swin_backbone(backbone_arch)
    else:
        raise ValueError()
    backbone = backbone_update_fn(backbone)
    return backbone


def get_head(backbone_arch, task_name, input_hw, head_update_fn=lambda x: x):
    head = dict(
        type="WorkConditionClsHead",
        output_dim=1024,
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
        num_classes=None,
        in_channel=list(stride2channels_dict[backbone_arch].values())[-1],
        avg_pool_size=4 if input_hw[0] == 224 else 2,
        padding_size=1,
        node_name=f"{task_name}_classification_prediction_head",
    )
    head = head_update_fn(head)
    return head


def camera_frame_to_hat_dict(msg: CameraFrame, select_bbox_topics=None):
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
        ori_input=msg,
    )

    if len(msg.perceptions) != 0:
        perceptions_dict = {}
        box_data = {}
        for perception in msg.perceptions:
            if isinstance(perception, Instance):
                assert len(perception.bbox2ds) == 1
                bbox2d = perception.bbox2ds[0]
                if (
                    select_bbox_topics is not None
                    and bbox2d.topic not in select_bbox_topics
                ):
                    continue
                box_ret = bbox2d_to_hat_dict(bbox2d, perception.track_id)
                perceptions_dict.setdefault(bbox2d.topic, []).append(box_ret)
                box_data.setdefault(bbox2d.topic, []).append(
                    box_ret["data"] + [box_ret["bbox_score"]]
                )
        box_data = {k: torch.Tensor(v) for k, v in box_data.items()}

        img_meta.update({"img_anno": perceptions_dict})
        img_meta.update(box_data)
    else:
        img_meta.update({"img_anno": dict()})

    return img_meta


def build_classifier(backbone_arch, input_hw, task_name, updates):
    backbone = get_backbone(
        backbone_arch,
        backbone_update_fn=updates.get("backbone_update_fn", lambda x: x),
    )
    head = get_head(
        backbone_arch=backbone_arch,
        task_name=task_name,
        input_hw=input_hw,
        head_update_fn=updates.get("head_update_fn", lambda x: x),
    )
    model = dict(
        type="WorkConditionClassifier",
        backbone=backbone,
        backbone_extra=torch.nn.Identity(),
        prediction_head=head,
        losses=None,
    )
    return model


# --------------- external ---------------


def get_val_transform(input_hw, object_type):
    val_transforms = [
        dict(
            type="AttrCV2CropInput",
            model_input_hw=input_hw,
            rsize_shape=512 if input_hw[0] == 224 else 256,
            detection_task_name=object_type,
            add_dummy_img_when_empty=False,
        ),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ]
    return val_transforms


def preprocess(
    data: Union[
        CameraFrame,
        List[CameraFrame],
        CameraFrameProto,
        List[CameraFrameProto],
    ],
    transforms: List[Callable],
    object_type,
) -> Dict:
    def _transform(frame):
        img_meta = camera_frame_to_hat_dict(
            frame, select_bbox_topics=_as_list(object_type)
        )
        for t in transforms:
            img_meta = t(img_meta)
        img_meta.pop("img_anno")
        return img_meta

    data = _as_list(data)
    data = [
        d if isinstance(d, CameraFrame) else CameraFrame.from_proto(d)
        for d in data
    ]
    img_metas = list(map(_transform, data))
    batch_data = collate_2d_cat(img_metas, object_type)
    return batch_data


def get_inference_models(
    backbone_arch, input_hw, task_names, converted_decoders, updates=None
):
    updates = updates if updates is not None else dict()

    val_model = dict(
        type="MultitaskGraphModel",
        inputs=dict(img=torch.zeros((1, 3, *input_hw))),
        opt_inputs={},
        task_inputs={task_name: {} for task_name in task_names},
        task_modules={
            task_name: build_classifier(
                backbone_arch, input_hw, task_name, updates
            )
            for task_name in task_names
        },
        funnel_modules=converted_decoders,
        flatten_outputs=False,
        lazy_forward=True,
    )
    return val_model


def postprocess(
    preds: Dict[str, List[DetObjects]],
    data: Dict[str, Any],
    task_name,
):
    assert (
        len(preds[task_name]) == len(data["obj_id"]) == len(data["rois_valid"])
    )
    image_id2results = {i: [] for i in data["batch_uuid"]}
    instance_id2origin = {}
    for camera_frame in data["ori_input"]:
        for perception in camera_frame.perceptions:
            if isinstance(perception, Instance):
                assert perception.track_id is not None
                instance_id2origin[perception.track_id] = perception
    assert "roi_uuid" in data
    for pred, roi_uuid_i, roi_valid, obj_id in zip(
        preds[task_name],
        data["roi_uuid"],
        data["rois_valid"],
        data["obj_id"],
    ):
        if not roi_valid:
            continue
        assert obj_id in instance_id2origin
        instance: Instance = instance_id2origin[obj_id]
        for k, v in pred["attrs"].items():
            instance.attributes.append(
                Attribute(
                    topic=k,
                    value=v,
                    score=max(pred.get("cls_score_2pe", None)[k]),
                )
            )
        image_id2results[roi_uuid_i].append(instance)
    pred_results = list(image_id2results.values())
    return pred_results, data


# tensorRT configs
if bool(int(trt_opt)):
    from hat.utils.compile_backends import tensorRT_backend

    model_convert_pipeline = ModelConvertPipeline(
        [
            TorchCompile(
                tensorRT_backend(fp16=True, dynamic_shape=True),
                load_extensions=[
                    "mean_replace_adp_avg_pool2d",
                    "dynamic_shape_pad",
                ],
            )
        ]
    )
else:
    model_convert_pipeline = None
