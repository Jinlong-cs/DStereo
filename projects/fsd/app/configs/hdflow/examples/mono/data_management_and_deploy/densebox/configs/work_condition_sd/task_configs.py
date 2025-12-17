import os

from hdflow.plugins.mono.data_management_and_deploy.base.structure import (  # noqa
    DataDeployTasksCfgs,
)
from hdflow.plugins.mono.data_management_and_deploy.eval_set.eval_setting.setting_entry import (  # noqa
    get_workcondition_settings,
)

__all__ = ["TASK_NAME_TO_TS_CFG"]


root_dir = os.path.join(os.path.dirname(os.path.relpath(__file__)))


def anno_transform_list(default, **kwargs):
    sensor_map = {
        "10635": [(1280, 720)],
        "820": [(3840, 2160)],
        "390": [(1920, 1080)],
        "0323": [(2280, 1080), (2048, 1080), (1920, 908), (1920, 910)],  # noqa
        "0220": [(1824, 940)],
    }
    sensor_crop = {  # define by [up, bottom, left, right]
        (1920, 1080): [0, 144, 48, 48],  # 390
        (3840, 2160): [0, 288, 96, 96],  # 820 (96, 0), (3744, 1872)
        (2048, 1080): [0, 144, 112, 112],  # 323 (112, 0), (1936, 936)
        (2280, 1080): [0, 144, 228, 228],  # 323
        (1920, 908): [0, 121, 193, 193],  # 323
        (1920, 910): [0, 121, 192, 192],  # 323
        (1824, 940): [0, 0, 0, 0],  # 220
        (1280, 720): [0, 0, 0, 0],  # 10635
    }
    keep_sensor = ["0323", "10635", "390", "0220", "820"]
    keep_shape = []
    for item in keep_sensor:
        keep_shape += sensor_map[item]

    anno_transformer = dict(
        type="Compose",
        transformer=[
            dict(
                type="DenseBoxWorkConditionAnnoTs",
                one_class=True,
                verbose=False,
                keep_shape=tuple(keep_shape),
                sensor_crop=sensor_crop,
            ),
            default,
        ],
    )
    return anno_transformer


evalset_config_work_condition_cls = dict(
    task_name="ADAS Classification V2",
    dataset_name="work_condition_cls_eval",
    name_prefix="work_condition_2pe",
    eval_creator_cfg=dict(
        type="CommonClassification",
    ),
)

TASK_NAME_TO_TS_CFG = {
    # ----------------------------------------
    # sd work condition classification data pack
    # ----------------------------------------
    # ------------------------------ scene ------------------------------------
    "sd_scene_classification": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/scene_classification_config.yaml",
        ),
        class_name=[
            "Highway",
            "National_road",
            "Urban",
            "Rural",
            "Tunnel",
            "Tunnel_Entry",
            "Charge_Station",
            "Bridge",
            "Indoor_Parking_Lot_Entrance",
            "Indoor_Parking_Lot",
            "Parking_Lot_Ramp",
            "Openair_Parking_Lot_Entrance",
            "Openair_Parking_Lot",
        ],
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=anno_transform_list,
        eval_setting_config=dict(
            eval_setting_genenrator=get_workcondition_settings,
            classname=[
                "SDWorkConditionScene14_2pe_cls",
            ],
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
            custom_settings=None,
        ),
        evalset_config=evalset_config_work_condition_cls,
    ),
    # ------------------------------ weather ----------------------------------
    "sd_weather_classification": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/weather_classification_config.yaml",
        ),
        class_name=[
            "medium_snow",
            "Sunny",
            "Cloudy",
            "Rainy",
            "Snowy",
            "medium_rain",
            "Heavy_Rain",
            "flurry",
            "Snowy",
            "Haze",
            "medium_fog",
            "Foggy",
            "Heavy_Rain",
            "Other",
        ],
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=anno_transform_list,
        eval_setting_config=dict(
            eval_setting_genenrator=get_workcondition_settings,
            classname=[
                "SDWorkConditionWeathe12_2pe_cls",
                "SDWorkConditionWeatheFuzzy12_2pe_cls",
            ],
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
            custom_settings=None,
        ),
        evalset_config=evalset_config_work_condition_cls,
    ),
    # ------------------------------- light -----------------------------------
    "sd_lightstatus_classification": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/lightstatus_classification_config.yaml",
        ),
        class_name=["natural_light", "lamplight", "Hard_Light", "dark"],
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=anno_transform_list,
        eval_setting_config=dict(
            eval_setting_genenrator=get_workcondition_settings,
            classname=[
                "SDWorkConditionLight4_2pe_cls",
            ],
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
            custom_settings=None,
        ),
        evalset_config=evalset_config_work_condition_cls,
    ),
    # --------------------------------- time ----------------------------------
    "sd_time_classification": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/time_classification_config.yaml",
        ),
        class_name=["Day", "Night", "Other"],
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=anno_transform_list,
        eval_setting_config=dict(
            eval_setting_genenrator=get_workcondition_settings,
            classname=[
                "SDWorkConditionTime3_2pe_cls",
            ],
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
            custom_settings=None,
        ),
        evalset_config=evalset_config_work_condition_cls,
    ),
    # ---------------------- mono old wk data weather -------------------------
    "mono_old_wk_weather_classification": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/weather_classification_mono_wk_config.yaml",
        ),
        class_name=[
            "Sunny",
            "Cloudy",
            "Rainy",
            "Snowy",
            "medium_rain",
            "Heavy_Rain",
            "flurry",
            "medium_snow",
            "Snowy",
            "Haze",
            "medium_fog",
            "Foggy",
            "Other",
        ],
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=anno_transform_list,
        eval_setting_config=dict(
            eval_setting_genenrator=get_workcondition_settings,
            classname=[
                "SDWorkConditionWeathe12_2pe_cls",
                "SDWorkConditionWeatheFuzzy12_2pe_cls",
            ],
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
            custom_settings=None,
        ),
        evalset_config=evalset_config_work_condition_cls,
    ),
    "sd_wk_classification_eval": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/weather_classification_mono_wk_config.yaml",
        ),
        class_name=[],
        task_owners_email=["fan.lv@horizon.ai"],
        eval_setting_config=dict(
            eval_setting_genenrator=get_workcondition_settings,
            classname=[
                "SDWorkConditionWeathe12_2pe_cls",
                "SDWorkConditionWeatheFuzzy12_2pe_cls",
                "SDWorkConditionScene14_2pe_cls",
                "SDWorkConditionTime3_2pe_cls",
                "SDWorkConditionLight4_2pe_cls",
            ],
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
            custom_settings=None,
        ),
        evalset_config=evalset_config_work_condition_cls,
    ),
}
