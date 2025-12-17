import os
import warnings
from functools import partial

from hatbc.filestream.bucket.client import get_bucket_mount_root
from hpflow.eval_platform_adaptor import (
    EvalPlatformAdaptorTrafficSign,
    EvalPlatformJsonWritter,
)
from hpflow.scope import get_current_config_scope_attribute
from hpflow.structure.ptype import PTypeEnum
from hpflow.utils import TakeByKey, bbox2d_nms, parse_datapaths

mount_root = get_bucket_mount_root()
bucket_adas_root = mount_root.get("adas")
if bucket_adas_root is None:
    bucket_adas_root = "unknown"
    warnings.warn("please mount bucket `adas` first")
dataset_root = os.path.join(bucket_adas_root, "hui.xue/eval")  # noqa

datapaths = {
    PTypeEnum.kTrafficSignAssist: [
        dict(
            img_root="/jfs-public/adas/yanlei.zhang/yanlei.zhang/workspace/projects_traffic_sign/multitask_inference/data/traffic_sign/6025530",  # noqa
            anno_path="/jfs-public/adas/yanlei.zhang/yanlei.zhang/workspace/projects_traffic_sign/multitask_inference/data/traffic_sign/6025530.json",  # noqa
            dataset_id=6025530,
            save_dir="kTrafficSignAssist/6025530/predict29",
            eval_ptypes=[
                {
                    PTypeEnum.kTrafficSignAssist: [
                        PTypeEnum.kTrafficSignAssist_IR_Ramp,
                        PTypeEnum.kTrafficSignAssist_Other_GuideSign_WhiteBlack,  # noqa
                    ]
                }
            ],  # noqa
            eval_platform_result_writter=EvalPlatformJsonWritter(
                adaptor=EvalPlatformAdaptorTrafficSign(
                    take_input_func=TakeByKey(
                        [
                            PTypeEnum.kTrafficSignAssist_IR_Ramp,
                            PTypeEnum.kTrafficSignAssist_Other_GuideSign_WhiteBlack,  # noqa
                            # noqa
                        ]
                    ),
                    attrs_class_id_key="sub_type",
                    whole_image_nms_ptypes=[PTypeEnum.kTrafficSignAssist],
                    global_nms_func=partial(
                        bbox2d_nms,
                        iou_thresh=0.3,
                        get_class_id_keys=["attrs.sub_type"],
                        get_attrs_items=[
                            {
                                "key": "sub_type",
                                "index": 5,
                            }
                        ],
                    ),
                )
            ),
            skip_img_model=True,
        ),
    ],
}

global_config = get_current_config_scope_attribute()
eval_task = global_config.get("eval_task")
if eval_task is not None:
    datapaths = dict(
        list(filter(lambda k: k[0] in eval_task, datapaths.items()))
    )  # noqa

datasets = parse_datapaths(
    datapaths,
    download_from_evalplatform=True,
    bucket_first=True,
    bucket_name="auto_eval",
)
