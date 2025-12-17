import copy
import os

from hdflow.plugins.mono.data_management_and_deploy.base.structure import (  # noqa
    CLASS_NAMES_DETECTION,
    DataDeployTasksCfgs,
)
from hdflow.plugins.mono.data_management_and_deploy.eval_set.eval_setting.setting_entry import (  # noqa
    get_default_settings,
    get_person_2pe_settings,
)

__all__ = [
    "TASK_NAME_TO_TS_CFG",
]

root_dir = os.path.join(os.path.dirname(os.path.relpath(__file__)))


# eval set configs
settings_user = []


x8b_2pe_default_setting = lambda classname: dict(  # noqa
    eval_setting_genenrator=get_person_2pe_settings,
    classname=classname,
    sensor="X8B",  # ov10652
    model_type="2pe",
    custom_tags=None,
    custom_settings=settings_user,
)


def update_type_name(default, new_type, **kwargs):
    default["type"] = new_type
    return default


def anno_transform_roi_list(default, **kwargs):
    res = default
    out_dir = kwargs["output_dir"]
    if kwargs["task_configs"].use_roilist:
        res = dict(
            roilist_generator=dict(
                type="PersonRoiListCreator",
                class_ids=(0, 1, 7, 9, 49),
                selected_ids=(1, -1, 7, -7, 9, -9),
                class_ids_map={49: 0},
                target_roi_list_path=os.path.join(
                    out_dir, "cache_roi_list.json"
                ),
                basic_transformer=default,
                num_worker=8,
            ),
            anno_transformer=dict(
                dict(
                    type="PersonRoiListTransformer",
                    roi_file_lst=[
                        os.path.join(out_dir, "cache_roi_list.json")
                    ],
                    verbose=True,
                ),
            ),
        )
    return res


# 解决 CLASS_NAMES_DETECTION 中 person_head 和 traffic_cone 冲突
CLASS_NAMES_DETECTION_PERSON = copy.deepcopy(CLASS_NAMES_DETECTION)
CLASS_NAMES_DETECTION_PERSON[8] = "person_head"

TASK_NAME_TO_TS_CFG = {
    # ----------------------------------------
    # person
    # ----------------------------------------
    # person 4PE
    "person_det_multitask": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/det_person_multitask_v7.3.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION_PERSON,
        task_owners_email=["kaijun01.zhang@horizon.ai"],
        anno_transformer_update_fn=anno_transform_roi_list,  # noqa
        use_roilist=True,
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="Person_4pe_det",
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
            custom_settings=settings_user,
        ),
        evalset_config=dict(
            task_name="Detection 2D",
            dataset_name="vru_detection-default-test",
            name_prefix="vru_detection",
            eval_creator_cfg=dict(
                type="CommonDet2D",
                filter_configs=[
                    dict(
                        type="Person4PEAttrsConvert",
                        remove_empty=False,
                        base_type="person",
                        extra_types=["cyclist"],
                        subtypes_map={
                            "PersonRideBicycle": "Bicyclist",
                            "PersonRideTricycle": "Tricyclist",
                            "PersonRideMotorcycle": "Motorcyclist",
                            "PersonPushBicycle": "Bicyclist",
                            "PersnPushTricycle": "Tricyclist",
                            "PersonPushMotorcycle": "Motorcyclist",
                            "AEB_PersonRideBicycle": "AEB_Bicyclist",
                            "AEB_PersonRideMotorcycle": "AEB_Motorcyclist",
                            "PersonRideBicycle_AEB": "AEB_Bicyclist",
                            "PersonRideMotorcycle_AEB": "AEB_Motorcyclist",
                            "PersonRideMotorcycleWithShed": "MotorcyclistWithShed",  # noqa
                        },
                    )
                ],
            ),
        ),
    ),
    "person_det_multicls": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/det_person_multicls_v7.3.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION_PERSON,
        anno_transformer_update_fn=anno_transform_roi_list,  # noqa
        task_owners_email=["kaijun01.zhang@horizon.ai"],
        use_roilist=True,
        # evalset
        eval_setting_config=None,
        evalset_config=None,
    ),
    # person 2pe
    "person_age_classification": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/person_age_classification_train_anno_ts_config.yaml",
        ),
        class_name=["Adult", "Child"],
        task_owners_email=["hao.chen@horizon.ai"],
        data_pack_update_fn=lambda default, **kwargs: update_type_name(
            default, "LegacyDenseboxDataPackerBasePython", **kwargs
        ),
        eval_setting_config=x8b_2pe_default_setting("PersonAge_2pe_cls"),
        evalset_config=dict(
            task_name="ADAS Classification V2",
            dataset_name="person_age_classification-default-test",
            name_prefix="person_age_classification",
            eval_creator_cfg=dict(
                type="CommonClassification",
                filter_configs=[dict(type="InvalidImageFilter")],
            ),
        ),
    ),
    "person_head_detection": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/person_head_detection_train_anno_ts_config.yaml",
        ),
        class_name=CLASS_NAMES_DETECTION_PERSON,
        task_owners_email=["hao.chen@horizon.ai"],
        data_pack_update_fn=lambda default, **kwargs: update_type_name(
            default, "LegacyDenseboxDataPackerBasePython", **kwargs
        ),
        anno_transformer_update_fn=lambda default, **kwargs: update_type_name(
            default, "DenseBoxSubboxDetAnnoTs", **kwargs
        ),
        eval_setting_config=None,
        evalset_config=dict(
            task_name="Detection 2D",
            dataset_name="person_head_detection-default-test",
            name_prefix="person_head_detection",
            eval_creator_cfg=dict(
                type="Common2PEDet2D",
                filter_configs=[
                    # prediction name is `person_head` but `head` in label
                    # platform. we convert the gt rather than prediction to
                    # simplify prediction.
                    dict(type="LabelTransform"),
                    dict(
                        type="HeadAttrsConvert",
                        remove_empty=True,
                        keep_person_without_head=False,
                    ),
                    dict(type="InvalidImageFilter"),
                ],
            ),
        ),
    ),
    "person_orientation_classification": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/person_orientation_classification_train_anno_ts_config.yaml",  # noqa
        ),
        class_name=[
            "back",
            "front",
            "left",
            "left_anterior",
            "left_back",
            "right",
            "right_back",
            "right_front",
        ],
        task_owners_email=["hao.chen@horizon.ai"],
        data_pack_update_fn=lambda default, **kwargs: update_type_name(
            default, "LegacyDenseboxDataPackerBasePython", **kwargs
        ),
        eval_setting_config=x8b_2pe_default_setting("Person_orientation_cls"),
        evalset_config=dict(
            task_name="ADAS Classification V2",
            dataset_name="person_orientation_classification-default-test",
            name_prefix="person_orientation_classification",
            eval_creator_cfg=dict(
                type="CommonClassification",
                filter_configs=[
                    dict(type="InvalidImageFilter"),
                    dict(type="BBoxFilter", keep_name_in_attrs="orientation"),
                ],
            ),
        ),
    ),
    "person_pose_classification": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/person_pose_classification_train_anno_ts_config.yaml",
        ),
        class_name=["Bended", "Cyclist", "Lier", "Pedestrian", "Sitter"],
        task_owners_email=["hao.chen@horizon.ai"],
        data_pack_update_fn=lambda default, **kwargs: update_type_name(
            default, "LegacyDenseboxDataPackerBasePython", **kwargs
        ),
        eval_setting_config=x8b_2pe_default_setting("PersonPose_2pe_cls"),
        evalset_config=dict(
            task_name="ADAS Classification V2",
            dataset_name="person_pose_classification-default-test",
            name_prefix="person_pose_classification",
            eval_creator_cfg=dict(
                type="CommonClassification",
                filter_configs=[dict(type="InvalidImageFilter")],
            ),
        ),
    ),
    # packing only
    "person_posneg_occ_classification": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/person_posneg_occ_classification_train_anno_ts_config_v3.yaml",  # noqa
        ),
        class_name=[
            "Negative",
            "full_visible",
            "occluded",
            "heavily_occluded",
            "invisible",
        ],
        task_owners_email=["hao.chen@horizon.ai"],
        data_pack_update_fn=lambda default, **kwargs: update_type_name(
            default, "LegacyDenseboxDataPackerBasePython", **kwargs
        ),
        anno_transformer_update_fn=lambda default, **kwargs: update_type_name(
            default, "DenseBoxDetAnnoTsPedPosNegOccCls", **kwargs
        ),
        eval_setting_config=None,
        evalset_config=None,
    ),
    # eval only
    "person_occlusion_classification": DataDeployTasksCfgs(
        task_ts_cfg=None,
        class_name=[
            "full_visible",
            "occluded",
            "heavily_occluded",
            "invisible",
        ],
        task_owners_email=["hao.chen@horizon.ai"],
        eval_setting_config=x8b_2pe_default_setting("PersonOcclusion_2pe_cls"),
        evalset_config=dict(
            task_name="ADAS Classification V2",
            dataset_name="person_occlusion_classification-default-test",
            name_prefix="person_occlusion_classification",
            eval_creator_cfg=dict(
                type="CommonClassification",
                filter_configs=[dict(type="InvalidImageFilter")],
            ),
        ),
    ),
    "person_posneg_classification": DataDeployTasksCfgs(
        task_ts_cfg=None,
        class_name=["Negative", "Positive"],
        task_owners_email=["hao.chen@horizon.ai"],
        eval_setting_config=None,
        evalset_config=dict(
            task_name="Detection 2D FP",
            dataset_name="person_posneg_classification-default-test",
            name_prefix="person_posneg_classification",
            eval_creator_cfg=dict(
                type="CommonFP",
                class_names=["person"],
                filter_configs=[dict(type="InvalidImageFilter")],
            ),
        ),
    ),
    # cyclist
    "cyclist_det_multitask": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/det_person_multitask_v7.3.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION_PERSON,
        task_owners_email=["kaijun01.zhang@horizon.ai"],
        anno_transformer_update_fn=anno_transform_roi_list,  # noqa
        use_roilist=True,
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="Cyclist_4pe_det",
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
            custom_settings=settings_user,
        ),
        evalset_config=dict(
            task_name="Detection 2D",
            dataset_name="vru_cyclist_detection-default-test",
            name_prefix="vru_cyclist_detection",
            eval_creator_cfg=dict(
                type="CommonDet2D",
                filter_configs=[
                    dict(
                        type="Person4PEAttrsConvert",
                        remove_empty=False,
                        base_type="person",
                        extra_types=["cyclist"],
                        subtypes_map={
                            "PersonRideBicycle": "Bicyclist",
                            "PersonRideTricycle": "Tricyclist",
                            "PersonRideMotorcycle": "Motorcyclist",
                            "PersonPushBicycle": "Bicyclist",
                            "PersnPushTricycle": "Tricyclist",
                            "PersonPushMotorcycle": "Motorcyclist",
                            "AEB_PersonRideBicycle": "AEB_Bicyclist",
                            "AEB_PersonRideMotorcycle": "AEB_Motorcyclist",
                            "PersonRideBicycle_AEB": "AEB_Bicyclist",
                            "PersonRideMotorcycle_AEB": "AEB_Motorcyclist",
                            "PersonRideMotorcycleWithShed": "MotorcyclistWithShed",  # noqa
                        },
                    )
                ],
            ),
        ),
    ),
}
