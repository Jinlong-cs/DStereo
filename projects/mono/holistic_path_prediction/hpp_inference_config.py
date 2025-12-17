# ------------------------
# hdflow inference config
# ------------------------
from functools import partial
from typing import Any, Callable, Dict, List, Union

import cv2
from hatbc.message import CameraFrame, Instance, KeyPoint2D
from horizon_plugin_pytorch.march import March

from hat.core.data_struct.base_struct import BaseDataList
from hat.data.collates.collates import default_collate_v2
from hat.registry import build_from_registry
from hat.utils.apply_func import _as_list

task_name = "holistic_path_prediction"

val_model = dict(
    type="HppModel",
    out_indices=2,
    backbone=dict(
        type="VargNetV2",
        input_channels=3,
        input_sequence_length=1,
        num_classes=1000,
        factor=2,
        alpha=0.5,
        bias=True,
        bn_kwargs=dict(eps=2e-5, momentum=0.1),
        group_base=8,
        include_top=False,
        head_factor=2,
    ),
    decode_head=dict(
        type="HPPDecodeHead",
        block_num=4,
        in_channels=32,
        out_channels=32,
    ),
    losses=None,
    decode=dict(
        type="HPPDecoder",
        feat_stride=8,
        img_shape=(512, 256),
        point_thresh=(0.7),
        to_hatbc_msg=True,
        task_name=task_name,
    ),
)

val_transforms = [
    dict(type="ImgBufDecoder", to_rgb=True),
    dict(type="ToTensor", to_yuv=True),
    dict(type="Normalize", mean=128.0, std=128.0),
]


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
        img_name=msg.image.name,
        img_buf=msg.image.as_bytes(),
        layout=layout,
        img_shape=shape,
        color_space="rgb",
        img_height=shape[0],
        img_width=shape[1],
    )
    return img_meta


def preprocess(
    data: Dict,
    transforms: List[Union[Dict, Callable]],
) -> Dict:
    transform_funs = [build_from_registry(t) for t in transforms]

    def _transform(frame):
        frame = camera_frame_to_hat_dict(frame)
        for t in transform_funs:
            frame = t(frame)
        return frame

    data = _as_list(data)
    data = [
        d if isinstance(d, CameraFrame) else CameraFrame.from_proto(d)
        for d in data
    ]
    img_metas = list(map(_transform, data))
    batch_data = default_collate_v2(img_metas)
    return batch_data


def postprocess(
    data: List,
    batch: Any,
):
    def _convert2instance(topic: str, data: BaseDataList):
        points = data.points.cpu().numpy()
        scores = data.scores.cpu().numpy()
        kps_list = []
        for i in range(scores.shape[0]):
            kps_list.append(
                KeyPoint2D(
                    points[i][0],
                    points[i][1],
                    score=scores[i],
                    topic=topic,
                )
            )
        instance_data = Instance(keypoint2ds=kps_list, topic=topic)
        return instance_data

    output_data = []
    topic_data = data[task_name]
    for topic_data_i in topic_data:
        output_data.append(_convert2instance(task_name, topic_data_i))

    return output_data, batch


inference = dict(
    type="Inference",
    device=None,
    model=val_model,
    pre_processors=[partial(preprocess, transforms=val_transforms)],
    post_processors=[postprocess],
    march=March.BERNOULLI2,
)
