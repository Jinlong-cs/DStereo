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


def get_traffic_light_classinfo(anno_ts_fn_config):
    class_info_list = []
    for class_id in range(1, anno_ts_fn_config["num_classes"] + 1):
        if class_id == anno_ts_fn_config["current_class_id"]:
            class_info_list.append(
                ClassInfo(
                    {
                        "point_num": 20,
                        "attribute_num": 2,
                        "class_name": "CLASS_%02d" % class_id,
                        "attribute_names": [
                            "traffic_lights_style",
                            "traffic_lights_type",
                        ],
                        "bbox_border_id": [0, 1, 2, 3],
                    }
                )
            )
        else:
            class_info_list.append(
                ClassInfo(
                    {
                        "point_num": 20,
                        "attribute_num": 0,
                        "class_name": "CLASS_%02d" % class_id,
                        "attribute_names": [],
                        "bbox_border_id": [0, 1, 2, 3],
                    }
                )
            )
    return class_info_list


def insert_transform_cfg(default, new_type, **kwargs):
    res = dict(
        type="Compose",
        transformer=[
            dict(
                type=new_type,
                one_class=True,
                verbose=False,
            ),
            default,
        ],
    )
    return res


def anno_transform_roi_list(default, new_type, **kwargs):
    default["type"] = new_type
    res = default
    if kwargs["task_configs"].use_roilist:
        res = dict(
            anno_transformer=dict(
                type="Compose",
                transformer=[
                    default,
                    dict(
                        type="DenseBoxRoilistAnnoTsOnline",
                        iou_threshold=0.1,
                        online_roilist=True,
                        model_name=kwargs["update_g_tags_configs"][
                            "prelabel_model_name"
                        ],
                    ),
                ],
            ),
        )
    return res


CLASS_NAMES_TRAFFIC_LENS_TPYE = [
    "L_Circle",
    "L_Forward",
    "L_Left",
    "L_Right",
    "L_Return",
    "L_Pedestrain",
    "L_Non_Motor",
    "L_Time",
    "L_left_and_return",
    "L_Forward_and_Left",
    "L_Forward_and_Right",
    "L_No_Drive_into",
    "L_Allow_Drive_into",
]

CLASS_NAMES_TRAFFIC_LENS_COLOR = ["green", "yellow", "red"]

# eval set configs
settings_user_traffic_light_4pe = [
    dict(
        setting_name="tl_0-170m_max-10pixels_crop",
        setting_path="hdfs://hobot-bigdata/user/jinsong.liu/leaderboard/setting/4pe/tl_170m_max-10pixels_crop_day.yaml",  # noqa
        setting_desc="",
    )
]

evalset_config_traffic_light_4pe = dict(
    task_name="Detection 2D",
    dataset_name="traffic_light_leaderboard_4pe",
    name_prefix="traffic_light_4pe",
    eval_creator_cfg=dict(
        type="CommonDet2D",
        filter_configs=[
            dict(type="TrafficLightAttrsConvert", one_class=True),
        ],
    ),
)
eval_setting_config_traffic_light_2pe = {
    "color_3_cls": "hdfs://hobot-bigdata/user/jinsong.liu/leaderboard/setting/2pe/color_3_cls.yaml"  # noqa
}
evalset_config_traffic_light_cls = dict(
    task_name="ADAS Classification V2",
    dataset_name="traffic_light_cls_eval",
    name_prefix="traffic_light_2pe",
    eval_creator_cfg=dict(
        type="CommonClassification",
        filter_configs=[
            dict(type="TrafficLightAttrsConvert", one_class=True),
        ],
    ),
)
evalset_config_traffic_lens_det = dict(
    task_name="Detection 2D",
    dataset_name="traffic_lens_det_eval",
    name_prefix="traffic_light_2pe",
    eval_creator_cfg=dict(
        type="CommonClassification",
        filter_configs=[
            dict(type="TrafficLightAttrsConvert", one_class=True),
        ],
    ),
)

TASK_NAME_TO_TS_CFG = {
    # ----------------------------------------
    # traffic_light
    # ----------------------------------------
    # ------------------------------- day -------------------------------------
    # traffic_light 4pe
    "traffic_light_det": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(root_dir, "packing/traffic_light_det.yaml"),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["fan.lv@horizon.ai"],
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="TrafficLight_4pe_det",
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
            custom_settings=settings_user_traffic_light_4pe,
        ),
        evalset_config=evalset_config_traffic_light_4pe,
    ),
    "traffic_light_det_roilist": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(root_dir, "packing/traffic_light_det.yaml"),
        class_name=CLASS_NAMES_DETECTION,
        prelabel_for_dataset=True,
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: anno_transform_roi_list(  # noqa
            default, "DenseBoxDetAnnoTs", **kwargs
        ),
        use_roilist=True,
        keep_duplicate=False,
        # evalset
        eval_setting_config=settings_user_traffic_light_4pe,
        evalset_config=evalset_config_traffic_light_4pe,
    ),
    # traffic_light 2pe
    "traffic_lens_det_roilist": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(root_dir, "packing/traffic_lens_det.yaml"),
        class_name=CLASS_NAMES_TRAFFIC_LENS_TPYE,
        prelabel_for_dataset=True,
        task_owners_email=["fan.lv@horizon.ai"],
        # data_pack_post_processer_config=dict(
        #     type="AnnotationDenseboxV2RoiList2Pbrec",
        #     class_name="Traffic_light_shell",
        #     attribute_name="traffic_light_color_cls",
        # ),
        anno_transformer_update_fn=lambda default, **kwargs: anno_transform_roi_list(  # noqa
            default, "DenseBoxTrafficLightLensDetAnnoTs", **kwargs
        ),
        use_roilist=True,
        keep_duplicate=False,
        classinfo_update_fn=get_traffic_light_classinfo,
        # TODO use new visualizer
        # viz_fn_type="VizRoiDenseBoxDetAnno",
        # viz_fn_ext_attr=dict(
        #     sub_class_name=CLASS_NAMES_TRAFFIC_LENS_COLOR,
        #     viz_sub_class_id=list(
        #         range(1, len(CLASS_NAMES_TRAFFIC_LENS_COLOR) + 1)
        #     ),
        #     roi_lt_point_id=10,
        #     roi_rb_point_id=12,
        #     attr_class_index=0,
        #     attr_sub_class_index=1,
        # ),
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname=[
                "TrafficLightType4_2pe_det",
                "TrafficLightType9_2pe_det",
                "TrafficLightColor3Joint_2pe_det",
                "TrafficLightType4Joint_2pe_det",
                "TrafficLightType9Joint_2pe_det",
            ],
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
            custom_settings=None,
        ),
        evalset_config=evalset_config_traffic_lens_det,
    ),
    "traffic_light_color_cls": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/traffic_light_color_cls.yaml",
        ),
        class_name=[
            "green",
            "yellow",
            "red",
            "unknow",
            "other",
            "white",
            "off",
            "back",
            "side",
        ],
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: insert_transform_cfg(  # noqa
            default, "DenseBoxTrafficLightAnnoTs", **kwargs
        ),
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname=[
                "TrafficLightColor3_2pe_cls",
                "TrafficLightColor8_2pe_cls",
            ],
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
            custom_settings=None,
        ),
        evalset_config=evalset_config_traffic_light_cls,
    ),
    "traffic_light_color_cls_badcase": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/traffic_light_color_cls_badcase.yaml",
        ),
        class_name=[
            "green",
            "yellow",
            "red",
            "unknow",
            "other",
            "white",
            "off",
            "back",
            "side",
        ],
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: insert_transform_cfg(  # noqa
            default, "DenseBoxTrafficLightAnnoTs", **kwargs
        ),
    ),
    "traffic_light_type_cls": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/traffic_light_type_cls.yaml",
        ),
        class_name=[
            "L_Circle",
            "L_Forward",
            "L_Left",
            "L_Right",
            "L_Return",
            "L_Pedestrain",
            "L_Non_Motor",
            "L_Time",
            "L_Other",
            "L_left_and_return",
            "L_Forward_and_Left",
            "L_Forward_and_Right",
            "L_No_Drive_into",
            "L_Allow_Drive_into",
            "L_unknown",
            "text_of_allow_ped",
            "sign_of_allow_ped",
            "text_of_forbid_ped",
            "sign_of_forbid_ped",
        ],
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: insert_transform_cfg(  # noqa
            default, "DenseBoxTrafficLightAnnoTs", **kwargs
        ),
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname=[
                "TrafficLightType4_2pe_cls",
                "TrafficLightType9_2pe_cls",
                "TrafficLightType18_2pe_cls",
            ],
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
            custom_settings=None,
        ),
        evalset_config=evalset_config_traffic_light_cls,
    ),
    "traffic_light_type_cls_badcase": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/traffic_light_type_cls_badcase.yaml",
        ),
        class_name=[
            "L_Circle",
            "L_Forward",
            "L_Left",
            "L_Right",
            "L_Return",
            "L_Pedestrain",
            "L_Non_Motor",
            "L_Time",
            "L_Other",
            "L_left_and_return",
            "L_Forward_and_Left",
            "L_Forward_and_Right",
            "L_No_Drive_into",
            "L_Allow_Drive_into",
            "L_unknown",
            "text_of_allow_ped",
            "sign_of_allow_ped",
            "text_of_forbid_ped",
            "sign_of_forbid_ped",
        ],
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: insert_transform_cfg(  # noqa
            default, "DenseBoxTrafficLightAnnoTs", **kwargs
        ),
    ),
    "traffic_light_fp_cls": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/traffic_light_fp_cls.yaml",
        ),
        class_name=["Positive", "Negative"],
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: insert_transform_cfg(  # noqa
            default, "DenseBoxTrafficLightAnnoTs", **kwargs
        ),
        # evalset
        eval_setting_config=eval_setting_config_traffic_light_2pe,
        evalset_config=evalset_config_traffic_light_cls,
    ),
    "traffic_light_fp_cls_badcase": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/traffic_light_fp_cls_badcase.yaml",
        ),
        class_name=["Positive", "Negative"],
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: insert_transform_cfg(  # noqa
            default, "DenseBoxTrafficLightAnnoTs", **kwargs
        ),
    ),
    "traffic_light_time_cls": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/traffic_light_time_cls.yaml",
        ),
        class_name=[f"{i}" for i in range(1000)],
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: insert_transform_cfg(  # noqa
            default, "DenseBoxTrafficLightAnnoTs", **kwargs
        ),
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname=[
                "TrafficLightTiming_2pe_cls",
            ],
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
            custom_settings=None,
        ),
        evalset_config=evalset_config_traffic_light_cls,
    ),
    # ----------------------------- night -------------------------------------
    "traffic_light_night_det_roilist": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/traffic_light_night_det.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        prelabel_for_dataset=True,
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: anno_transform_roi_list(  # noqa
            default, "DenseBoxDetAnnoTs", **kwargs
        ),
        use_roilist=True,
        keep_duplicate=False,
        # evalset
        eval_setting_config=settings_user_traffic_light_4pe,
        evalset_config=evalset_config_traffic_light_4pe,
    ),
    "traffic_lens_night_det_roilist": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/traffic_lens_night_det.yaml"
        ),
        class_name=CLASS_NAMES_TRAFFIC_LENS_TPYE,
        prelabel_for_dataset=True,
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: anno_transform_roi_list(  # noqa
            default, "DenseBoxTrafficLightLensDetAnnoTs", **kwargs
        ),
        use_roilist=True,
        keep_duplicate=False,
        classinfo_update_fn=get_traffic_light_classinfo,
        # viz_fn_type="VizRoiDenseBoxDetAnno",
        # viz_fn_ext_attr=dict(
        #     sub_class_name=CLASS_NAMES_TRAFFIC_LENS_COLOR,
        #     viz_sub_class_id=list(
        #         range(1, len(CLASS_NAMES_TRAFFIC_LENS_COLOR) + 1)
        #     ),
        #     roi_lt_point_id=10,
        #     roi_rb_point_id=12,
        #     attr_class_index=0,
        #     attr_sub_class_index=1,
        # ),
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname=[
                "TrafficLightType4_2pe_det",
                "TrafficLightType9_2pe_det",
                "TrafficLightColor3Joint_2pe_det",
                "TrafficLightType4Joint_2pe_det",
                "TrafficLightType9Joint_2pe_det",
            ],
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
            custom_settings=None,
        ),
        evalset_config=evalset_config_traffic_lens_det,
    ),
    "traffic_light_night_fp_cls": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/traffic_light_night_fp_cls.yaml"
        ),
        class_name=["Positive", "Negative"],
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: insert_transform_cfg(  # noqa
            default, "DenseBoxTrafficLightAnnoTs", **kwargs
        ),
        # evalset
        eval_setting_config=eval_setting_config_traffic_light_2pe,
        evalset_config=evalset_config_traffic_light_cls,
    ),
    "traffic_light_night_type_cls": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/traffic_light_night_type_cls.yaml",
        ),
        class_name=[
            "L_Circle",
            "L_Forward",
            "L_Left",
            "L_Right",
            "L_Return",
            "L_Pedestrain",
            "L_Non_Motor",
            "L_Time",
            "L_Other",
            "L_left_and_return",
            "L_Forward_and_Left",
            "L_Forward_and_Right",
            "L_No_Drive_into",
            "L_Allow_Drive_into",
            "L_unknown",
            "text_of_allow_ped",
            "sign_of_allow_ped",
            "text_of_forbid_ped",
            "sign_of_forbid_ped",
        ],
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: insert_transform_cfg(  # noqa
            default, "DenseBoxTrafficLightAnnoTs", **kwargs
        ),
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname=[
                "TrafficLightType4_2pe_cls",
                "TrafficLightType9_2pe_cls",
                "TrafficLightType18_2pe_cls",
            ],
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
            custom_settings=None,
        ),
        evalset_config=evalset_config_traffic_light_cls,
    ),
    "traffic_light_night_type_cls_badcase": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/traffic_light_night_type_cls_badcase.yaml",
        ),
        class_name=[
            "L_Circle",
            "L_Forward",
            "L_Left",
            "L_Right",
            "L_Return",
            "L_Pedestrain",
            "L_Non_Motor",
            "L_Time",
            "L_Other",
            "L_left_and_return",
            "L_Forward_and_Left",
            "L_Forward_and_Right",
            "L_No_Drive_into",
            "L_Allow_Drive_into",
            "L_unknown",
            "text_of_allow_ped",
            "sign_of_allow_ped",
            "text_of_forbid_ped",
            "sign_of_forbid_ped",
        ],
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: insert_transform_cfg(  # noqa
            default, "DenseBoxTrafficLightAnnoTs", **kwargs
        ),
    ),
    "traffic_light_night_color_cls": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/traffic_light_night_color_cls.yaml",
            # "packing/traffic_light_night_color_cls_badcase.yaml"
        ),
        class_name=[
            "green",
            "yellow",
            "red",
            "unknow",
            "other",
            "white",
            "off",
            "back",
            "side",
        ],
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: insert_transform_cfg(  # noqa
            default, "DenseBoxTrafficLightAnnoTs", **kwargs
        ),
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname=[
                "TrafficLightColor3_2pe_cls",
                "TrafficLightColor8_2pe_cls",
            ],
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
            custom_settings=None,
        ),
        evalset_config=evalset_config_traffic_light_cls,
    ),
    "traffic_light_night_color_cls_badcase": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/traffic_light_night_color_cls_badcase.yaml",
        ),
        class_name=[
            "green",
            "yellow",
            "red",
            "unknow",
            "other",
            "white",
            "off",
            "back",
            "side",
        ],
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: insert_transform_cfg(  # noqa
            default, "DenseBoxTrafficLightAnnoTs", **kwargs
        ),
    ),
    "traffic_light_night_time_cls": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/traffic_light_time_cls.yaml",
        ),
        class_name=[f"{i}" for i in range(1000)],
        task_owners_email=["fan.lv@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: insert_transform_cfg(  # noqa
            default, "DenseBoxTrafficLightAnnoTs", **kwargs
        ),
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname=[
                "TrafficLightTiming_2pe_cls",
            ],
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
            custom_settings=None,
        ),
        evalset_config=evalset_config_traffic_light_cls,
    ),
}
