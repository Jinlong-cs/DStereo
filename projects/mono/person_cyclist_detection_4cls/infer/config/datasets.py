import os
import warnings

from hatbc.filestream.bucket.client import get_bucket_mount_root
from hpflow.scope import get_current_config_scope_attribute
from hpflow.structure.ptype import PTypeEnum
from hpflow.utils import parse_datapaths
from hpflow.utils.config_helper import get_default_json_writter

mount_root = get_bucket_mount_root()
bucket_adas_root = mount_root.get("adas")
if bucket_adas_root is None:
    bucket_adas_root = "unknown"
    warnings.warn("please mount bucket `adas` first")
dataset_root = os.path.join(bucket_adas_root, "hui.xue/eval")  # noqa

datapaths = {
    PTypeEnum.kPed2PE: [
        dict(
            img_root=f"{dataset_root}/4pe_pedestrian/6033372",
            anno_path=f"{dataset_root}/4pe_pedestrian/6033372.json",
            dataset_id=6038430,
            eval_platform_result_writter=get_default_json_writter(),
        ),
    ],
    PTypeEnum.kCyc2PE: [
        dict(
            img_root=f"{dataset_root}/4pe_pedestrian/6033428",
            anno_path=f"{dataset_root}/4pe_pedestrian/6033428.json",
            dataset_id=6038430,
            eval_platform_result_writter=get_default_json_writter(),
        ),
    ],
    PTypeEnum.kMotorCyc2PE: [
        dict(
            img_root=f"{dataset_root}/4pe_pedestrian/6033372",
            anno_path=f"{dataset_root}/4pe_pedestrian/6033372.json",
            dataset_id=6038430,
            eval_platform_result_writter=get_default_json_writter(),
        ),
    ],
    PTypeEnum.kTriCyc2PE: [
        dict(
            img_root=f"{dataset_root}/4pe_pedestrian/6033428",
            anno_path=f"{dataset_root}/4pe_pedestrian/6033428.json",
            dataset_id=6038430,
            eval_platform_result_writter=get_default_json_writter(),
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
