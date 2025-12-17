import os

from auto_matrix.data.densebox.anno import ClassInfo
from hdflow.plugins.mono.data_management_and_deploy.base.structure import (  # noqa
    CLASS_NAMES_DETECTION,
    DataDeployTasksCfgs,
)
from hdflow.plugins.mono.data_management_and_deploy.eval_set.eval_setting.setting_entry import (  # noqa
    get_default_settings,
)

__all__ = [
    "TASK_NAME_TO_TS_CFG",
]

root_dir = os.path.join(os.path.dirname(os.path.relpath(__file__)))


def get_traffic_cone_classinfo(
    anno_ts_fn_config,
    point_num=10,
    class_name="traffic_cone_classification",
    attribute_names=("traffic_cone_classification",),
    bbox_border_id=(0, 1, 2, 3),
):
    class_info_list = []
    class_info_list.append(
        ClassInfo(
            {
                "point_num": point_num,
                "attribute_num": len(attribute_names),
                "class_name": class_name,
                "attribute_names": attribute_names,
                "bbox_border_id": bbox_border_id,
            }
        )
    )
    return class_info_list


def anno_transform_traffic_cone_attribute(default, **kwargs):
    default["type"] = "DenseBoxTrafficSignClsAttributeAnnoTs"
    return default


# eval set configs
settings_user_traffic_cone_4pe = [
    dict(
        setting_name="0_60m-all_category-all_scene",
        setting_path="hdfs://hobot-bigdata/user/jianyu.hong/leaderboard/setting/setting-cn0820_3840x2160-merge-ALL_Category-0~60m-ALL_Scene.yaml",  # noqa
        setting_desc="",
    )
]
evalset_config_traffic_cone_4pe = dict(
    task_name="Detection 2D",
    dataset_name="traffic_cone-leaderboard-4pe-test",
    name_prefix="traffic_cone_4pe",
    eval_creator_cfg=dict(
        type="CommonDet2D",
    ),
)

settings_user_traffic_cone_2pe = [
    dict(
        setting_name="0_60m-6cls-all_scene",
        setting_path="hdfs://hobot-bigdata/user/jianyu.hong/leaderboard/setting/setting-cn0820_3840x2160-6cls-0~60m-ALL_Scene.yaml",  # noqa
        setting_desc="",
    )
]
evalset_config_traffic_cone_2pe = dict(
    task_name="ADAS Classification V2",
    dataset_name="traffic_cone-leaderboard-2pe-6cls-test",
    name_prefix="traffic_cone_2pe",
    eval_creator_cfg=dict(
        type="CommonClassification",
    ),
)

TASK_NAME_TO_TS_CFG = {
    # ----------------------------------------
    # traffic_cone
    # ----------------------------------------
    "traffic_cone_4pe": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/4pe_traffic_cone_config.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["jianyu.hong@horizon.ai"],
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="TrafficCone_4pe_det",
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
            custom_settings=settings_user_traffic_cone_4pe,
        ),
        evalset_config=evalset_config_traffic_cone_4pe,
    ),
    "traffic_cone_2pe": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/cone_bollard_classification_6cls.yaml"
        ),
        class_name=["traffic_cone_classification"],
        task_owners_email=["jianyu.hong@horizon.ai"],
        classinfo_update_fn=get_traffic_cone_classinfo,
        anno_transformer_update_fn=anno_transform_traffic_cone_attribute,
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="TrafficCone_2pe_cls",
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
            custom_settings=settings_user_traffic_cone_2pe,
        ),
        evalset_config=evalset_config_traffic_cone_2pe,
    ),
}
