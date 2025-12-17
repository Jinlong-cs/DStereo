import os

from hdflow.plugins.mono.data_management_and_deploy.base.structure import (  # noqa
    CLASS_NAMES_DETECTION,
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
    task_name="Detection 2D",
    dataset_name="rear-default-test",
    name_prefix="rear-default",
    eval_creator_cfg=dict(
        type="CommonDet2D",
    ),
)

TASK_NAME_TO_TS_CFG = {
    # ----------------------------------------
    # vehicle rear
    # ----------------------------------------
    "traffic_sign_us_4pe": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(root_dir, "packing/anno_ts_config_v2.yaml"),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["ran.chen@horizon.ai"],
        # evalset
        eval_setting_config=eval_setting_config_default,
        evalset_config=evalset_config_default,
    ),
}
