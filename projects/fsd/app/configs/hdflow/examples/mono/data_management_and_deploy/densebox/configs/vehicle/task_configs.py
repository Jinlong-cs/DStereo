import os

from hdflow.plugins.mono.data_management_and_deploy.base.structure import (  # noqa
    CLASS_NAMES_DETECTION,
    DataDeployTasksCfgs,
)
from hdflow.plugins.mono.data_management_and_deploy.eval_set.eval_setting.setting_entry import (  # noqa
    get_default_settings,
)

from auto_matrix.data.anno_transformer.anno import (  # isort:skip
    get_default_classinfo,
)

__all__ = [
    "TASK_NAME_TO_TS_CFG",
]

root_dir = os.path.join(os.path.dirname(os.path.relpath(__file__)))


def update_2pe_prelabel(default, **kwargs):
    default["type"] = "DenseBoxSubboxDetAnnoTs"
    update_g_tags_configs = kwargs["update_g_tags_configs"]
    configs = dict(
        type="Compose",
        transformer=[
            dict(
                type="VehicleFullExtendInputFromPrelabel",
                update_field=update_g_tags_configs["update_field"],
                model_name=update_g_tags_configs["prelabel_model_name"],
                take_keys=("vehicle",),
                push_keys=("vehicle_full",),
                allow_empty_prelabel=False,
                perception_extension_fn="merge_data",
            ),
            default,
        ],
    )
    return configs


def update_2pe_det_classinfo_fn(config, point_num=20):

    return get_default_classinfo(config, point_num)


def update_type_name(default, new_type, **kwargs):
    default["type"] = new_type
    return default


# eval set configs
settings_user = []
evalset_config_default = dict(
    task_name="Detection 2D",
    dataset_name="vehicle-default-test",
    name_prefix="vehicle-default",
    eval_creator_cfg=dict(
        type="CommonDet2D",
    ),
)
evalset_config_vehicle_category_2pe = dict(
    task_name="ADAS Classification V2",
    dataset_name="vehicle_category_eval",
    name_prefix="vehicle_category_2pe",
    eval_creator_cfg=dict(
        type="CommonClassification",
    ),
)
evalset_config_vehicle_left_door_2pe = dict(
    task_name="ADAS Classification V2",
    dataset_name="vehicle_left_door_eval",
    name_prefix="vehicle_left_door_2pe",
    eval_creator_cfg=dict(
        type="CommonClassification",
        filter_configs=[dict(type="VehicleLeftDoor2PEAttrsConvert")],
    ),
)
evalset_config_vehicle_right_door_2pe = dict(
    task_name="ADAS Classification V2",
    dataset_name="vehicle_right_door_eval",
    name_prefix="vehicle_right_door_2pe",
    eval_creator_cfg=dict(
        type="CommonClassification",
        filter_configs=[dict(type="VehicleRightDoor2PEAttrsConvert")],
    ),
)
evalset_config_vehicle_trunk_door_2pe = dict(
    task_name="ADAS Classification V2",
    dataset_name="vehicle_trunk_door_eval",
    name_prefix="vehicle_trunk_door_2pe",
    eval_creator_cfg=dict(
        type="CommonClassification",
        filter_configs=[dict(type="VehicleTrunkDoor2PEAttrsConvert")],
    ),
)
CLASS_NAMES_2PE_VEHICLE_FP = [
    "full_visible",
    "occluded",
    "heavily_occluded",
    "invisible",
    "unknown",
]
evalset_config_vehicle_fp = dict(
    task_name="Detection 2D FP",
    dataset_name="vehicle-fp-tunnel",
    name_prefix="vehicle-fp-tunnel",
    eval_creator_cfg=dict(type="CommonFP", class_names=["vehicle"]),
)

TASK_NAME_TO_TS_CFG = {
    # ----------------------------------------
    # vehicle
    # ----------------------------------------
    "vehicle_4pe": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/legacy_densebox_vehicle_det_train_anno_ts_config_v4.yaml",
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["yonggang.yang@horizon.ai"],
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="VehicleFull_4pe_det",
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
            custom_settings=settings_user,
        ),
        evalset_config=evalset_config_default,
    ),
    "vehicle_2pe_detection_matched": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/det_subbox_vehicle_full_v2_iou.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        prelabel_for_dataset=True,
        anno_transformer_update_fn=lambda default, **kwargs: update_2pe_prelabel(  # noqa
            default, **kwargs
        ),
        classinfo_update_fn=lambda default: update_2pe_det_classinfo_fn(
            default
        ),
        task_owners_email=["chunhong01.chen@horizon.ai"],
        # evalset
        eval_setting_config=None,
        evalset_config=None,
    ),
    "vehicle_2pe_detection": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/det_subbox_vehicle_full_v2_iou.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        prelabel_for_dataset=False,
        anno_transformer_update_fn=lambda default, **kwargs: update_type_name(
            default, "DenseBoxSubboxDetAnnoTs", **kwargs
        ),
        classinfo_update_fn=lambda default: update_2pe_det_classinfo_fn(
            default
        ),
        task_owners_email=["chunhong01.chen@horizon.ai"],
        # evalset
        eval_setting_config=None,
        evalset_config=None,
    ),
    "vehicle_category_2pe": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/vehicle_category_cls.yaml",
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["guowei.hao@horizon.ai"],
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="VehicleType_2pe_cls",
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
            custom_settings=settings_user,
        ),
        evalset_config=evalset_config_vehicle_category_2pe,
    ),
    "2pe_vehicle_fp": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/cls_vehicle_full_fp_v8.yaml",
        ),
        class_name=CLASS_NAMES_2PE_VEHICLE_FP,
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
        evalset_config=evalset_config_vehicle_fp,
    ),
    "vehicle_occlusion_2pe_no_prelabel": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/2pe_vehicle_occlusion_config_no_prelabel.yaml",
        ),
        class_name=CLASS_NAMES_2PE_VEHICLE_FP,
        task_owners_email=["yue01.shi@horizon.ai"],
        # evalset
        eval_setting_config=None,
        evalset_config=None,
    ),
    "vehicle_left_door_cls_2pe": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/vehicle_left_door_cls.yaml",
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["sha.huang@horizon.ai"],
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="VehicleLeftDoorStatus_2pe_cls",
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
            custom_settings=settings_user,
        ),
        evalset_config=evalset_config_vehicle_left_door_2pe,
    ),
    "vehicle_right_door_cls_2pe": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/vehicle_right_door_cls.yaml",
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["sha.huang@horizon.ai"],
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="VehicleRightDoorStatus_2pe_cls",
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
            custom_settings=settings_user,
        ),
        evalset_config=evalset_config_vehicle_right_door_2pe,
    ),
    "vehicle_trunk_door_cls_2pe": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/vehicle_trunk_door_cls.yaml",
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["sha.huang@horizon.ai"],
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="VehicleTrunkDoorStatus_2pe_cls",
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
            custom_settings=settings_user,
        ),
        evalset_config=evalset_config_vehicle_trunk_door_2pe,
    ),
}
