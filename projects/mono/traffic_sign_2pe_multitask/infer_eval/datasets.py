import os
import warnings

from hatbc.filestream.bucket.client import get_bucket_mount_root
from hpflow.eval_platform_adaptor import (
    EvalPlatformAdaptorTrafficSign,
    EvalPlatformJsonWritter,
)
from hpflow.scope import get_current_config_scope_attribute
from hpflow.structure.ptype import PTypeEnum
from hpflow.utils import TakeByKey, parse_datapaths

mount_root = get_bucket_mount_root()
bucket_adas_root = mount_root.get("adas")
if bucket_adas_root is None:
    bucket_adas_root = "unknown"
    warnings.warn("please mount bucket `adas` first")
dataset_root = os.path.join(bucket_adas_root, "hui.xue/eval")  # noqa

datapaths = {
    # 2pe标志牌分类
    PTypeEnum.kTrafficSignCategory: [
        # 常规评测集
        dict(
            img_root=f"{dataset_root}/2pe_traffic_sign/6029722",
            anno_path=f"{dataset_root}/2pe_traffic_sign/6029722.json",
            dataset_id=6029722,
            save_dir="kTrafficSignCategory_kTrafficSignOcclusion/6029722",
            eval_ptypes=[
                {
                    PTypeEnum.kTrafficSignCategory: [
                        PTypeEnum.kTrafficSignCategory
                    ],
                    PTypeEnum.kTrafficSignOcclusion: [
                        PTypeEnum.kTrafficSignOcclusion
                    ],
                }
            ],  # noqa
            eval_platform_result_writter=EvalPlatformJsonWritter(
                adaptor=EvalPlatformAdaptorTrafficSign(
                    take_input_func=TakeByKey(None, as_list=True),
                )
            ),
            skip_img_model=True,
        ),
        # badcase 评测集
        dict(
            img_root=f"{dataset_root}/2pe_traffic_sign/6037748",
            anno_path=f"{dataset_root}/2pe_traffic_sign/6037748.json",
            dataset_id=6037748,
            save_dir="kTrafficSignCategory_kTrafficSignOcclusion/6037748",
            eval_ptypes=[
                {
                    PTypeEnum.kTrafficSignCategory: [
                        PTypeEnum.kTrafficSignCategory
                    ],
                    PTypeEnum.kTrafficSignOcclusion: [
                        PTypeEnum.kTrafficSignOcclusion
                    ],
                }
            ],  # noqa
            eval_platform_result_writter=EvalPlatformJsonWritter(
                adaptor=EvalPlatformAdaptorTrafficSign(
                    take_input_func=TakeByKey(None, as_list=True),
                )
            ),
            skip_img_model=True,
        ),
        # J2 评测集
        dict(
            img_root=f"{dataset_root}/2pe_traffic_sign/6038354",
            anno_path=f"{dataset_root}/2pe_traffic_sign/6038354.json",
            dataset_id=6038354,
            save_dir="kTrafficSignCategory_kTrafficSignOcclusion/6038354",
            eval_ptypes=[
                {
                    PTypeEnum.kTrafficSignCategory: [
                        PTypeEnum.kTrafficSignCategory
                    ],
                    PTypeEnum.kTrafficSignOcclusion: [
                        PTypeEnum.kTrafficSignOcclusion
                    ],
                }
            ],  # noqa
            eval_platform_result_writter=EvalPlatformJsonWritter(
                adaptor=EvalPlatformAdaptorTrafficSign(
                    take_input_func=TakeByKey(None, as_list=True),
                )
            ),
            skip_img_model=True,
        ),
    ],
    # # 2pe标志牌分类
    # PTypeEnum.kTrafficSignCategory: [
    #     dict(
    #         img_root=f"{dataset_root}/2pe_traffic_sign/6036142",
    #         anno_path=f"{dataset_root}/2pe_traffic_sign/6036142.json",
    #         dataset_id=6036142,
    #         save_dir="kTrafficSignCategory/6036142",
    #         eval_platform_result_writter=get_default_json_writter(),
    #         skip_img_model=True,
    #     )
    # ],
    # # 2pe标志牌遮挡分类
    # PTypeEnum.kTrafficSignOcclusion: [
    #     dict(
    #         img_root=f"{dataset_root}/2pe_traffic_sign/6042058",
    #         anno_path=f"{dataset_root}/2pe_traffic_sign/6042058.json",
    #         dataset_id=6042058,
    #         save_dir="kTrafficSignOcclusion/6042058",
    #         eval_platform_result_writter=get_default_json_writter(),
    #         skip_img_model=True,
    #     )
    # ],
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
