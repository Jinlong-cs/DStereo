import os

from hdflow.plugins.mono.data_management_and_deploy.base.structure import (  # noqa
    CLASS_NAMES_DETECTION,
    DataDeployTasksCfgs,
)
from hdflow.plugins.mono.data_management_and_deploy.eval_set.eval_setting.setting import (  # noqa
    get_setting,
)

__all__ = [
    "TASK_NAME_TO_TS_CFG",
]

root_dir = os.path.join(os.path.dirname(os.path.relpath(__file__)))

settings_default = []

# eval set configs
settings_user = [
    dict(
        setting_name="rear_usrdef_tets",
        setting_path="hdfs://hobot-bigdata/user/meng01.wang/setting/cls16.2_add_freespace.yaml",  # noqa
        setting_desc="It's a test",
    )
]
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
    # ihbc
    # ----------------------------------------
    "ihbc_4pe": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/anno_ts_config.yaml",
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["kedi01.liu@horizon.ai"],
        data_pack_post_processer_config=dict(
            type="AnnotationDenseboxV2AttributeConverter",
            class_name="rear",
            attribute_name="ihbc_classification",
            base_class_id=10,
            pop_ignore=True,
        ),
        # evalset
        eval_setting_config=settings_default + settings_user,
        evalset_config=evalset_config_default,
    ),
}
