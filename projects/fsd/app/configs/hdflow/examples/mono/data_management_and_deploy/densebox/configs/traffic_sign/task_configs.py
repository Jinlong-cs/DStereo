import copy
import datetime
import os
from functools import partial

import yaml
from auto_matrix.data.densebox.anno import ClassInfo
from eval_setting import task_name2setting_info
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
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# eval set configs
settings_user_traffic_sign_4pe = [
    dict(
        setting_name="0-50m-det-ALL",
        setting_path="hdfs://hobot-bigdata/user/yanlei.zhang/eval_profiles/0820_eval_data_det_allclass_fix/0-50m-det-ALL.yaml",  # noqa
        setting_desc="",
    )
]

evalset_config_traffic_sign_4pe = dict(
    task_name="Detection 2D",
    dataset_name="traffic_sign_4pe_detection",
    name_prefix="traffic_sign_4pe",
    eval_cpu=6,
    eval_cpu_mem_ratio=6,
    eval_creator_cfg=dict(
        type="CommonDet2D",
        filter_configs=[
            dict(
                type="MergeCrowdedAbreastObjectAnnoTs",
                verbose=False,
                config=yaml.load(
                    open(
                        os.path.join(
                            root_dir,
                            "packing/merge_vertical_bbox.yaml",
                        )
                    ),
                    Loader=yaml.FullLoader,
                ),
                root_dir="./",
                visualization=False,
                visualization_path=f"dmpv2://auto_tmp_2/yanlei.zhang/data_packing/merge_bbox_{current_time}",  # noqa
            ),
            dict(
                type="FixNestedAnnoTs",
                verbose=False,
                config=yaml.load(
                    open(
                        os.path.join(root_dir, "packing/fix_nested.yaml"),
                    ),
                    Loader=yaml.FullLoader,
                ),
                root_dir="./",
                visualization=False,
                visualization_path=f"dmpv2://auto_tmp_2/yanlei.zhang/data_packing/fix_nested_{current_time}",  # noqa
            ),
            dict(
                type="UpdateAttrsAnnoTs",
                config=yaml.load(
                    open(
                        os.path.join(
                            root_dir,
                            "packing/update_attrs.yaml",
                        )
                    ),
                    Loader=yaml.FullLoader,
                ),
                root_dir="./",
                verbose=False,
            ),
        ],
    ),
)

evalset_config_traffic_sign_4pe_det_cls_eval = copy.deepcopy(
    evalset_config_traffic_sign_4pe
)
evalset_config_traffic_sign_4pe_det_cls_eval[
    "dataset_name"
] = "traffic_sign_4pe_det_cls"

evalset_config_traffic_sign_gl_4pe_det_cls_eval = copy.deepcopy(
    evalset_config_traffic_sign_4pe
)
evalset_config_traffic_sign_gl_4pe_det_cls_eval[
    "dataset_name"
] = "traffic_sign_gl_4pe_det_cls"

eval_setting_config_traffic_sign_2pe_type_cls = [
    dict(
        setting_name="0-50m-cls241-ALL",
        setting_path="hdfs://hobot-bigdata/user/yanlei.zhang/eval_profiles/cnts-0820-cls241-0820_cls241/0-50m-cls241-ALL.yaml",  # noqa
        setting_desc="",
    )
]
evalset_config_traffic_sign_2pe_type_cls = dict(
    task_name="ADAS Classification V2",
    dataset_name="traffic_sign_2pe_type_cls",
    name_prefix="traffic_sign_2pe",
    eval_creator_cfg=dict(
        type="CommonClassification",
        filter_configs=[
            dict(
                type="MergeCrowdedAbreastObjectAnnoTs",
                verbose=False,
                config=yaml.load(
                    open(
                        os.path.join(
                            root_dir,
                            "packing/merge_vertical_bbox.yaml",
                        )
                    ),
                    Loader=yaml.FullLoader,
                ),
                root_dir="./",
                visualization=False,
                visualization_path=f"dmpv2://auto_tmp_2/yanlei.zhang/data_packing/merge_bbox_{current_time}",  # noqa
            ),
            dict(
                type="FixNestedAnnoTs",
                verbose=False,
                config=yaml.load(
                    open(
                        os.path.join(root_dir, "packing/fix_nested.yaml"),
                    ),
                    Loader=yaml.FullLoader,
                ),
                root_dir="./",
                visualization=False,
                visualization_path=f"dmpv2://auto_tmp_2/yanlei.zhang/data_packing/fix_nested_{current_time}",  # noqa
            ),
            dict(
                type="UpdateAttrsAnnoTs",
                config=yaml.load(
                    open(
                        os.path.join(
                            root_dir,
                            "packing/update_attrs.yaml",
                        )
                    ),
                    Loader=yaml.FullLoader,
                ),
                root_dir="./",
                verbose=False,
            ),
        ],
    ),
)

eval_setting_config_traffic_sign_2pe_occlusion_cls = [
    dict(
        setting_name="occ-0-50m-cls4-ALL",
        setting_path="hdfs://hobot-bigdata/user/yanlei.zhang/eval_profiles/cnts_occ4_cls4_profiles/occ-0-50m-cls4-ALL.yaml",  # noqa
        setting_desc="",
    )
]
evalset_config_traffic_sign_2pe_occlusion_cls = dict(
    task_name="ADAS Classification V2",
    dataset_name="traffic_sign_2pe_occlusion_cls",
    name_prefix="traffic_sign_2pe",
    eval_creator_cfg=dict(
        type="CommonClassification",
        filter_configs=[
            dict(
                type="UpdateAttrsAnnoTs",
                config=yaml.load(
                    open(
                        os.path.join(
                            root_dir,
                            "packing/update_attrs.yaml",
                        )
                    ),
                    Loader=yaml.FullLoader,
                ),
                root_dir="./",
                verbose=False,
            ),
        ],
    ),
)

eval_setting_config_traffic_sign_2pe_assist_detection = {
    "0-50m-det-cls2-ALL": "hdfs://hobot-bigdata/user/yanlei.zhang/eval_profiles/cnts_10652_assist_profiles/0-50m-det-cls2-ALL.yaml"  # noqa
}
evalset_config_traffic_sign_2pe_assist_detection = dict(
    task_name="ADAS Classification V2",
    dataset_name="traffic_sign_2pe_assist_detection",
    name_prefix="traffic_sign_2pe",
    eval_creator_cfg=dict(
        type="CommonDet2D",
        filter_configs=[
            dict(
                type="UpdateAttrsAnnoTs",
                config=yaml.load(
                    open(
                        os.path.join(
                            root_dir,
                            "packing/update_attrs.yaml",
                        )
                    ),
                    Loader=yaml.FullLoader,
                ),
                root_dir="./",
                verbose=False,
            ),
        ],
    ),
)


def update_type_name(default, new_type, **kwargs):
    default["type"] = new_type
    return default


def get_traffic_sign_classinfo(
    config,
    point_num,
    attribute_num=0,
    attribute_names=(),
    bbox_border_id=(0, 1, 2, 3),
    class_mapper_key="class_mappers",
):
    class_info_list = []
    for class_id in range(1, config["num_classes"] + 1):
        cur_class_mapper_ = list(
            filter(
                lambda class_mapper: class_mapper["id"] == class_id,
                config[class_mapper_key],
            )
        )
        if len(cur_class_mapper_):
            class_name = cur_class_mapper_[0].get(
                "name", "CLASS_%02d" % class_id
            )  # noqa
        else:
            class_name = "CLASS_%02d" % class_id
        class_info_list.append(
            ClassInfo(
                {
                    "point_num": point_num,
                    "attribute_num": attribute_num,
                    "class_name": class_name,
                    "attribute_names": attribute_names,
                    "bbox_border_id": bbox_border_id,
                }
            )
        )
    return class_info_list


get_traffic_sign_assist_detection_classinfo = partial(
    get_traffic_sign_classinfo,
    point_num=20,
    attribute_num=0,
    attribute_names=(),
    bbox_border_id=(10, 11, 12, 13),
    class_mapper_key="children_class_mappers",
)


def insert_transform_cfg(default, **kwargs):
    res = dict(
        type="Compose",
        transformer=[
            dict(
                type="MergeCrowdedAbreastObjectAnnoTs",
                verbose=False,
                config=yaml.load(
                    open(
                        os.path.join(
                            root_dir,
                            "packing/merge_vertical_bbox.yaml",
                        )
                    ),
                    Loader=yaml.FullLoader,
                ),
                root_dir="./",
                visualization=False,
                visualization_path=f"dmpv2://auto_tmp_2/yanlei.zhang/data_packing/merge_bbox_{current_time}",  # noqa
            ),
            dict(
                type="FixNestedAnnoTs",
                verbose=False,
                config=yaml.load(
                    open(
                        os.path.join(root_dir, "packing/fix_nested.yaml"),
                    ),
                    Loader=yaml.FullLoader,
                ),
                root_dir="./",
                visualization=False,
                visualization_path=f"dmpv2://auto_tmp_2/yanlei.zhang/data_packing/fix_nested_{current_time}",  # noqa
            ),
            default,
        ],
    )
    return res


def insert_assist_detection_transform_cfg(default, **kwargs):
    res = dict(
        type="Compose",
        transformer=[
            dict(
                type="FixNestedAnnoTs",
                verbose=False,
                config=yaml.load(
                    open(
                        os.path.join(root_dir, "packing/fix_nested.yaml"),
                    ),
                    Loader=yaml.FullLoader,
                ),
                root_dir="./",
                visualization=False,
                visualization_path=f"dmpv2://auto_tmp_2/yanlei.zhang/data_packing/fix_nested_{current_time}",  # noqa
            ),
            dict(
                type="DenseBoxSubboxTrafficSignAssistDetAnnoTs",
                verbose=False,
                config=yaml.load(
                    open(
                        os.path.join(
                            root_dir,
                            "packing/2pe_traffic_sign_assist_detection_anno_ts.yaml",  # noqa
                        ),  # noqa
                    ),
                    Loader=yaml.FullLoader,
                ),
                root_dir="./",
            ),
        ],
    )
    return res


TASK_NAME_TO_TS_CFG = {
    # ----------------------------------------
    # traffic sign
    # ----------------------------------------
    "4pe_traffic_sign_j3mono": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            # root_dir, "packing/4pe_traffic_sign_j3mono_anno_ts.yaml"
            root_dir,
            "packing/4pe_traffic_sign_j3mono_withunknown_anno_ts.yaml",
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["yanlei.zhang@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: (
            insert_transform_cfg(default, **kwargs)
        ),
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname=[
                "TrafficSign_4pe_det",
                # 可以在检测评测集上创建联合评测的setting, 但会导致setting 太多评测太慢
                # 因此可以分成两个评测集
                # "TrafficSign_4pe_det_cls",
            ],
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
            custom_settings=settings_user_traffic_sign_4pe,
        ),
        evalset_config=evalset_config_traffic_sign_4pe,
    ),
    "4pe_traffic_sign_j3mono_det_cls_eval": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/4pe_traffic_sign_j3mono_withunknown_anno_ts.yaml",
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["yanlei.zhang@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: (
            insert_transform_cfg(default, **kwargs)
        ),
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname=[
                "TrafficSign_4pe_det_cls",
            ],
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
            custom_settings=None,
        ),
        evalset_config=evalset_config_traffic_sign_4pe_det_cls_eval,
    ),
    "4pe_traffic_sign_j2mono": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/4pe_traffic_sign_j2mono_anno_ts.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["yanlei.zhang@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: (
            insert_transform_cfg(default, **kwargs)
        ),
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="TrafficSignJ2_4pe_det",
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
            custom_settings=settings_user_traffic_sign_4pe,
            task_name2setting_info=task_name2setting_info,
        ),
        evalset_config=evalset_config_traffic_sign_4pe,
    ),
    "4pe_traffic_sign_glts": DataDeployTasksCfgs(
        # @ran.chen 补充这个yaml 文件
        task_ts_cfg=os.path.join(
            root_dir, "packing/4pe_traffic_sign_glts_anno_ts.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["ran.chen@horizon.ai", "yanlei.zhang@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: (
            insert_transform_cfg(default, **kwargs)
        ),
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname=[
                "TrafficSignGL_4pe_det",
                # "TrafficSignGL_4pe_det_cls",
            ],
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
            custom_settings=None,
            task_name2setting_info=task_name2setting_info,
        ),
        evalset_config=evalset_config_traffic_sign_4pe,
    ),
    "4pe_traffic_sign_glts_det_cls_eval": DataDeployTasksCfgs(
        # @ran.chen 补充这个yaml 文件
        task_ts_cfg=os.path.join(
            root_dir, "packing/4pe_traffic_sign_glts_anno_ts.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["ran.chen@horizon.ai", "yanlei.zhang@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: (
            insert_transform_cfg(default, **kwargs)
        ),
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname=[
                # "TrafficSignGL_4pe_det",
                "TrafficSignGL_4pe_det_cls",
            ],
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
            custom_settings=None,
            task_name2setting_info=task_name2setting_info,
        ),
        evalset_config=evalset_config_traffic_sign_gl_4pe_det_cls_eval,
    ),
    "2pe_traffic_sign_assist_detection": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/2pe_traffic_sign_assist_detection_anno_ts.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["yanlei.zhang@horizon.ai"],
        anno_transformer_update_fn=lambda default, **kwargs: (
            insert_assist_detection_transform_cfg(default, **kwargs)
        ),
        classinfo_update_fn=get_traffic_sign_assist_detection_classinfo,
        # evalset
        eval_setting_config=eval_setting_config_traffic_sign_2pe_assist_detection,  # noqa
        evalset_config=evalset_config_traffic_sign_2pe_assist_detection,
    ),
    "2pe_traffic_sign_cls_j3mono": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/2pe_traffic_sign_cls_j3mono_anno_ts.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["yanlei.zhang@horizon.ai"],
        data_pack_post_processer_config=dict(
            type="AnnotationDenseboxV2AttributeConverter",
            class_name="traffic_sign",
            attribute_name="traffic_sign_cn_category_classification",
        ),
        anno_transformer_update_fn=lambda default, **kwargs: (
            insert_transform_cfg(default, **kwargs)
        ),
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname=[
                "TrafficSignType_2pe_cls",
                "TrafficSignTypeSpeedLimit_2pe_cls",
            ],
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
            custom_settings=None,
            task_name2setting_info=task_name2setting_info,
        ),
        evalset_config=evalset_config_traffic_sign_2pe_type_cls,
    ),
    "2pe_traffic_sign_cls_j3mono_types258": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/2pe_traffic_sign_cls_j3mono_types258_anno_ts.yaml",
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["yanlei.zhang@horizon.ai"],
        data_pack_post_processer_config=dict(
            type="AnnotationDenseboxV2AttributeConverter",
            class_name="traffic_sign",
            attribute_name="traffic_sign_cn_category_classification",
        ),
        anno_transformer_update_fn=lambda default, **kwargs: (
            insert_transform_cfg(default, **kwargs)
        ),
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname=[
                "TrafficSignType_2pe_cls",
                "TrafficSignTypeSpeedLimit_2pe_cls",
            ],
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
            custom_settings=None,
            task_name2setting_info=task_name2setting_info,
        ),
        evalset_config=evalset_config_traffic_sign_2pe_type_cls,
    ),
    "2pe_traffic_sign_occlusion_cls": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/2pe_traffic_sign_occlusion_cls_anno_ts.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["yanlei.zhang@horizon.ai"],
        data_pack_post_processer_config=dict(
            type="AnnotationDenseboxV2AttributeConverter",
            class_name="traffic_sign",
            attribute_name="traffic_sign_occlusion_classification",
        ),
        # evalset
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="TrafficSignOcclusion_2pe_cls",
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
            custom_settings=None,
            task_name2setting_info=task_name2setting_info,
        ),
        evalset_config=evalset_config_traffic_sign_2pe_occlusion_cls,
    ),
    "2pe_traffic_sign_cls_global_types714": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/2pe_traffic_sign_cls_global_types714_anno_ts.yaml",
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["ran.chen@horizon.ai", "yanlei.zhang@horizon.ai"],
        data_pack_post_processer_config=dict(
            type="AnnotationDenseboxV2AttributeConverter",
            class_name="traffic_sign",
            attribute_name="traffic_sign_gl_category_classification",
        ),
        anno_transformer_update_fn=lambda default, **kwargs: (
            insert_transform_cfg(default, **kwargs)
        ),
        # evalset
        eval_setting_config=None,
        evalset_config=None,
    ),
}
