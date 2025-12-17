import os

from hdflow.plugins.mono.data_management_and_deploy.base.structure import (  # noqa
    DataDeployTasksCfgs,
)

__all__ = [
    "TASK_NAME_TO_TS_CFG",
]

root_dir = os.path.join(os.path.dirname(os.path.relpath(__file__)))

# eval set configs
eval_setting_config_default = {
    "cls16.2_add_freespace": "hdfs://hobot-bigdata/user/meng01.wang/setting/cls16.2_add_freespace.yaml"  # noqa
}
evalset_config_default = dict(
    task_name="Semantic Segmentation",
    dataset_name="seg-default-test",
    name_prefix="seg-default",
    eval_creator_cfg=dict(
        type="CommonParsing",
    ),
)

TASK_NAME_TO_TS_CFG = {
    # ----------------------------------------
    # Parsing
    # ----------------------------------------
    "parsing_16cls": DataDeployTasksCfgs(
        # packing
        task_ts_cfg=None,
        class_name=None,
        task_owners_email=["suyao.wang@horizon.ai"],
        parsing_label_map=os.path.join(
            root_dir,
            "parsing_labelmap_16cls_v2.py",  # noqa
        ),
        load_prelabel=True,
        is_merge=True,
        # evalset
        eval_setting_config=eval_setting_config_default,
        evalset_config=evalset_config_default,
        keep_duplicate=False,
    ),
}
