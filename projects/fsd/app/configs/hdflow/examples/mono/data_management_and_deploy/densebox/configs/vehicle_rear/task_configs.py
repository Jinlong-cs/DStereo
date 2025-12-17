import os

from hdflow.plugins.mono.data_management_and_deploy.base.structure import (  # noqa
    CLASS_NAMES_DETECTION,
    DataDeployTasksCfgs,
)
from hdflow.plugins.mono.data_management_and_deploy.eval_set.eval_setting.setting_entry import (  # noqa
    get_default_settings,
)

from auto_matrix.data.densebox.anno.anno import ClassInfo  # isort:skip

__all__ = [
    "TASK_NAME_TO_TS_CFG",
]

root_dir = os.path.join(os.path.dirname(os.path.relpath(__file__)))


def update_type_name(default, new_type, **kwargs):
    default["type"] = new_type
    return default


def update_2pe_prelabel(default, **kwargs):
    default["type"] = "DenseBoxSubboxClsAnnoTs"
    update_g_tags_configs = kwargs["update_g_tags_configs"]
    configs = dict(
        type="Compose",
        transformer=[
            dict(
                type="ExtendInputFromPrelabel",
                update_field=update_g_tags_configs["update_field"],
                model_name=update_g_tags_configs["prelabel_model_name"],
                take_keys=("vehicle_rear",),
                allow_empty_prelabel=False,
                perception_extension_fn=None,
            ),
            default,
        ],
    )
    return configs


def get_vehicle_light_classinfo(
    anno_ts_fn_config,
    point_num=10,
    class_name="vehicle_light_classification",
    attribute_names=("vehicle_light_classification",),
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


CLASS_NAMES_2PE_VEHICLE_REAR_FP = [
    "full_visible",
    "occluded",
    "heavily_occluded",
    "invisible",
    "unknown",
]

evalset_config_rear_fp = dict(
    task_name="Detection 2D FP",
    dataset_name="vehicle-rear-fp-tunnel",
    name_prefix="vehicle-rear-fp-tunnel",
    eval_creator_cfg=dict(type="CommonFP", class_names=["vehicle"]),
)


def anno_transform_vehicle_light_attribute(default, **kwargs):
    default["type"] = "DenseBoxDetAnnoTs"
    return default


settings_user_vehicle_light_classification = [
    dict(
        setting_name="0_60m-6cls-all_scene",
        # 7 cls
        # setting_path="hdfs://hobot-bigdata-aliyun/user/xu.chen/evaluate_configs/vehicle_light_classification/setting_7cls-CN0820-0~100m.yaml",  # noqa
        # 9 cls, ./settings文件夹中有
        setting_path="hdfs://hobot-bigdata/user/sha.huang/evaluate_configs/vehicle_light_classification/setting_9cls-X8B-0~60m.yaml",  # noqa
        setting_desc="",
    )
]

evalset_config_vehicle_light_classification = dict(
    task_name="ADAS Classification V2",
    dataset_name="vehicle_light_classification-7cls-test",
    name_prefix="vehicle_light_classification",
    eval_creator_cfg=dict(
        type="CommonClassification",
    ),
)

# eval set configs
settings_user = []
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
    # vehicle rear 4pe
    # ----------------------------------------
    "vehicle_rear_4pe": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/det_vehicle_rear_v13.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["ni.jiang@horizon.ai"],
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="VehicleRear_4pe_det",
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
            custom_settings=settings_user,
        ),
        evalset_config=evalset_config_default,
    ),
    "vehicle_rear_4pe_select_special": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/det_vehicle_rear_special_v1.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["ni.jiang@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: update_type_name(
            default, "SpecialVehicle", **kwargs
        ),
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="VehicleRear_4pe_det",
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
            custom_settings=settings_user,
        ),
        evalset_config=evalset_config_default,
    ),
    # ----------------------------------------
    # vehicle rear 2pe
    # ----------------------------------------
    "vehicle_rear_occlusion_2pe": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/cls_rear_occlusion_2pe.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        prelabel_for_dataset=True,
        anno_transformer_update_fn=lambda default, **kwargs: update_2pe_prelabel(  # noqa
            default, **kwargs
        ),
        task_owners_email=["ni.jiang@horizon.ai"],
        # evalset
        eval_setting_config=None,
        evalset_config=None,
    ),
    "vehicle_rear_occlusion_2pe_no_prelabel": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/cls_subbox_rear_occlusion_v3.yaml",
        ),
        class_name=CLASS_NAMES_2PE_VEHICLE_REAR_FP,
        task_owners_email=["yue01.shi@horizon.ai"],
        # evalset
        eval_setting_config=None,
        evalset_config=None,
    ),
    "2pe_vehicle_rear_fp": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            # "packing/cls_rear_neg_side_tunnel.yaml",
            "packing/cls_rear_neg_v5.yaml",
        ),
        class_name=CLASS_NAMES_2PE_VEHICLE_REAR_FP,
        task_owners_email=["yue01.shi@horizon.ai"],
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="VehicleRearType3_2pe_cls",
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
            custom_settings=settings_user,
        ),
        evalset_config=evalset_config_rear_fp,
    ),
    # ----------------------------------------
    # vehicle light classification
    # ----------------------------------------
    "vehicle_light_classification": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/vehicle_rear_light_9_classification_config.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["sha.huang@horizon.ai"],
        classinfo_update_fn=get_vehicle_light_classinfo,
        anno_transformer_update_fn=anno_transform_vehicle_light_attribute,
        # evalset
        eval_setting_config=settings_user_vehicle_light_classification,
        evalset_config=evalset_config_vehicle_light_classification,
    ),
}
