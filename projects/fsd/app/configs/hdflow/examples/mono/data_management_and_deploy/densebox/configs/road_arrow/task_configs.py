import os
from functools import partial
from typing import Dict

from easydict import EasyDict
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

root_dir = os.path.join(os.path.dirname(__file__))

settings_user = ["xiufeng.zhou"]


def prelabel_config_update_fn(default: Dict, **kwargs):
    return default


def anno_transform_roi_list(default, new_type, roilist_file_list, **kwargs):
    assert len(roilist_file_list) > 0
    default["type"] = new_type
    dmp_client = kwargs["dmp_client"]
    res = default
    if kwargs["task_configs"].use_roilist:
        res = dict(
            anno_transformer=dict(
                type="Compose",
                transformer=[
                    default,
                    dict(
                        type="RoadArrowDenseBoxRoilistAnnoTsOnline",
                        dmp_client=dmp_client,
                        roi_file_lst=roilist_file_list,
                        anno_data_key="road_arrow",
                        pidx_1=4,
                        pidx_2=7,
                        pidx_center=4,
                        roi_score_thred=0.1,
                        verbose=True,
                    ),
                ],
            ),
        )
    return res


def anno_transform_2pe(
    default,
    *args,
    new_type="DenseBoxDetAnnoTs",
    perception_extension_fn=None,
    **kwargs
):
    if new_type:
        default["type"] = new_type
    if perception_extension_fn:
        update_g_tags_configs = kwargs["update_g_tags_configs"]
        update_g_tags_configs.update(
            dict(
                update_field="g_tags",
                info_keys=[
                    [
                        "road_arrow",
                    ]
                ],
            )
        )
        configs = dict(
            type="Compose",
            transformer=[
                dict(
                    type="RoadArrowExtendInputFromPrelabel",
                    update_field=update_g_tags_configs["update_field"],
                    model_name=update_g_tags_configs["prelabel_model_name"],
                    take_keys=("road_arrow",),
                    allow_empty_prelabel=False,
                    perception_extension_fn=perception_extension_fn,
                ),
                default,
            ],
        )
    else:
        configs = default
    return configs


def data_packer_crop_2pe_roi(default, *args, **kwargs):
    crop_2pe_roi_cfg = EasyDict(
        norm_len=200,
        norm_method="max_width_height",
        output_wh=[256, 256],
        min_crop_scale=0.95,
        max_jitter_xy=0.1,
        img_min_scale=0.01,
        img_max_scale=100,
        padd_val=0,
        boundary_keep_ioa_ratio=0.3,
        instance_point_id0=0,
        instance_point_id1=2,
        cache_dir=os.path.expanduser("~/tmp"),
        # 2pe bbox idx
        # det_point_id0=0,
        # det_point_id1=2,
    )
    if kwargs:
        crop_2pe_roi_cfg.update(kwargs.get("crop_2pe_roi_cfg", {}))
    default.update(
        dict(
            type="LegacyDenseboxROIDataPacker",
            crop_cfg=crop_2pe_roi_cfg,
            pass_through=False,
            encoding=".jpg",  # '.jpg' or '.png'
        )
    )
    return default


# 训练的时候学习的类，顺序不能乱。
# 同打包时候的顺序 也是模型输出时候的顺序
CLASS_NAMES_ROAD_ARROW = {
    "fp": [
        "pos",
        "fp",
    ],
    "cn_fp": [
        "pos",
        "fp",
    ],
    "us_dt4": [
        "forward",
        "backward",
        "leftward",
        "rightward",
    ],
    "cn_dt": [
        "forward",
        "backward",
        "leftward",
        "rightward",
    ],
    "us_dt3": [
        "forward",
        "backward",
        "leftorright",
    ],
    "us_cls19": [
        "Straight",
        "LeftTurn",
        "RightTurn",
        "LeftTurnOrStraight",
        "RightTurnOrStraight",
        "StraightOrLeftTurnOrRightTurn",
        "U_turn",
        "LeftTurnOrRightTurn",
        "LaneReductionToLeft",
        "LaneReductionToRight",
        "LeftMerge",
        "RightMerge",
        "RailwayAhead",
        "Bike",
        "Diamond",
        "SlowDownTriangle",
        "SpeedHump",
        "No",
        "Other",
    ],
    "us_cls22": [
        "Straight",
        "LeftTurn",
        "RightTurn",
        "LeftTurnOrStraight",
        "RightTurnOrStraight",
        "StraightOrLeftTurnOrRightTurn",
        "U_turn",
        "LeftTurnOrRightTurn",
        "LaneReductionToLeft",
        "LaneReductionToRight",
        "LeftMerge",
        "RightMerge",
        "RailwayAhead",
        "Bike",
        "Diamond",
        "SlowDownTriangle",
        "SpeedHump",
        "ManholeCover",
        "No",
        "Other",
        "Unknown" "FP",
    ],
    "cn_cls17": [
        "Straight",
        "LeftTurn",
        "RightTurn",
        "LeftTurnOrStraight",
        "RightTurnOrStraight",
        "U_turn",
        "StraightOrU_turn",
        "LeftTurnOrU_turn",
        "LeftTurnOrRightTurn",
        "LeftMerge",
        "RightMerge",
        "StraightOrLeftTurnOrRightTurn",
        "No",
        "Diamond",
        "SlowDownTriangle",
        "DistanceConfirmationLine",
        "Other",
    ],
    "cn_cls19": [
        "Straight",
        "LeftTurn",
        "RightTurn",
        "LeftTurnOrStraight",
        "RightTurnOrStraight",
        "U_turn",
        "StraightOrU_turn",
        "LeftTurnOrU_turn",
        "LeftTurnOrRightTurn",
        "LeftMerge",
        "RightMerge",
        "StraightOrLeftTurnOrRightTurn",
        "No",
        "Diamond",
        "SlowDownTriangle",
        "DistanceConfirmationLine",
        "ManholeCover",
        "Other",
        "FP",
    ],
}

_dir_of_this_file = os.path.dirname(__file__)

TASK_NAME_TO_TS_CFG = {
    # ----------------------------------------
    # road arrow: the keys is the workflow_task_name
    # ----------------------------------------
    # cn road arrow 4pe and roilist
    "cn_road_arrow_4pe": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            _dir_of_this_file, "packing", "cn", "anno_ts_4pe.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["xiufeng.zhou@horizon.ai"],
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="RoadArrow_4pe_det",
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
        ),
        evalset_config=dict(
            # leaderboard评测任务
            task_name="Detection 2D",
            # leaderboard上评测集的名字
            dataset_name="cn_road_arrow_4pe_eval",
            # 暂时没用到
            name_prefix="cn_road_arrow_4pe_eval",
            # eval creator的配置。配置的定义上使用了registry，整体的格式与HAT类似，
            # type 定义了eval_creator的名字，eval_creator的参数通过字典的key-value传入。
            eval_creator_cfg=dict(
                # 用户自定义的格式转换函数。
                type="CommonDet2D",
            ),
        ),
    ),
    "cn_road_arrow_4pe_joint": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            _dir_of_this_file, "packing", "cn", "anno_ts_4pe.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["xiufeng.zhou@horizon.ai"],
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="RoadArrowJoint_4pe_det",
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
        ),
        evalset_config=dict(
            # leaderboard评测任务
            task_name="Detection 2D",
            # leaderboard上评测集的名字
            dataset_name="cn_road_arrow_4pe_joint_eval",
            # 暂时没用到
            name_prefix="cn_road_arrow_4pe_joint_eval",
            # eval creator的配置。配置的定义上使用了registry，整体的格式与HAT类似，
            # type 定义了eval_creator的名字，eval_creator的参数通过字典的key-value传入。
            eval_creator_cfg=dict(
                # 用户自定义的格式转换函数。
                type="CommonDet2D",
            ),
        ),
    ),
    "cn_road_arrow_4pe_joint_narrow": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            _dir_of_this_file, "packing", "cn", "anno_ts_4pe.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["xiufeng.zhou@horizon.ai"],
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="RoadArrowJoint_4pe_det",
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
        ),
        evalset_config=dict(
            # leaderboard评测任务
            task_name="Detection 2D",
            # leaderboard上评测集的名字
            dataset_name="cn_road_arrow_4pe_joint_narrow_eval",
            # 暂时没用到
            name_prefix="cn_road_arrow_4pe_joint_narrow_eval",
            # eval creator的配置。配置的定义上使用了registry，整体的格式与HAT类似，
            # type 定义了eval_creator的名字，eval_creator的参数通过字典的key-value传入。
            eval_creator_cfg=dict(
                # 用户自定义的格式转换函数。
                type="CommonDet2D",
            ),
        ),
    ),
    "cn_road_arrow_4pe_joint_fp": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            _dir_of_this_file, "packing", "cn", "anno_ts_4pe.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["xiufeng.zhou@horizon.ai"],
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="RoadArrowJointFp_4pe_det",
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
        ),
        evalset_config=dict(
            # leaderboard评测任务
            task_name="Detection 2D",
            # leaderboard上评测集的名字
            dataset_name="cn_road_arrow_4pe_joint_fp_eval",
            # 暂时没用到
            name_prefix="cn_road_arrow_4pe_joint_fp_eval",
            # eval creator的配置。配置的定义上使用了registry，整体的格式与HAT类似，
            # type 定义了eval_creator的名字，eval_creator的参数通过字典的key-value传入。
            eval_creator_cfg=dict(
                # 用户自定义的格式转换函数。
                type="CommonDet2D",
            ),
        ),
    ),
    "cn_road_arrow_4pe_underpack": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            _dir_of_this_file, "packing", "cn", "anno_ts_4pe.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["xiufeng.zhou@horizon.ai"],
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="RoadArrow_4pe_det",
            sensor="X8B",
            model_type="Merge",
            custom_tags=None,
        ),
        evalset_config=dict(
            # leaderboard评测任务
            task_name="Detection 2D",
            # leaderboard上评测集的名字
            dataset_name="cn_road_arrow_4pe_underpack_eval",
            # 暂时没用到
            name_prefix="cn_road_arrow_4pe_underpack_eval",
            # eval creator的配置。配置的定义上使用了registry，整体的格式与HAT类似，
            # type 定义了eval_creator的名字，eval_creator的参数通过字典的key-value传入。
            eval_creator_cfg=dict(
                # 用户自定义的格式转换函数。
                type="CommonDet2D",
            ),
        ),
    ),
    "cn_road_arrow_4pe_CNX8B_roilist": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            _dir_of_this_file, "packing", "cn", "anno_ts_4pe.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["xiufeng.zhou@horizon.ai"],
        use_roilist=True,
        anno_transformer_update_fn=lambda default, **kwargs: anno_transform_roi_list(  # noqa
            default,
            "DenseBoxDetAnnoTs",
            [
                "dmpv2://mono/xiufeng.zhou/"
                + "Datasets/road_arrow/roilists/CNX8Btrain.json"
            ],
            **kwargs
        ),
    ),
    "cn_road_arrow_4pe_CN0820_roilist": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            _dir_of_this_file, "packing", "cn", "anno_ts_4pe.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["xiufeng.zhou@horizon.ai"],
        use_roilist=True,
        anno_transformer_update_fn=lambda default, **kwargs: anno_transform_roi_list(  # noqa
            default,
            "DenseBoxDetAnnoTs",
            [
                "dmpv2://mono/xiufeng.zhou/"
                + "Datasets/road_arrow/roilists/CN0820train.json"
            ],
            **kwargs
        ),
    ),
    "cn_road_arrow_4pe_Badcase_roilist": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            _dir_of_this_file, "packing", "cn", "anno_ts_4pe_badcase.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["xiufeng.zhou@horizon.ai"],
        use_roilist=True,
        anno_transformer_update_fn=lambda default, **kwargs: anno_transform_roi_list(  # noqa
            default,
            "DenseBoxDetAnnoTs",
            [
                "dmpv2://mono/xiufeng.zhou/"
                + "Datasets/road_arrow/roilists/Badcase.json"
            ],
            **kwargs
        ),
    ),
    # cn road arrow 2pe cls and fp and dt
    "cn_road_arrow_2pe_cls": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            _dir_of_this_file, "packing", "cn", "anno_ts_2pe_cls.yaml"
        ),
        class_name=CLASS_NAMES_ROAD_ARROW["cn_cls17"],
        task_owners_email=["xiufeng.zhou@horizon.ai"],
        data_pack_update_fn=data_packer_crop_2pe_roi,
        anno_transformer_update_fn=anno_transform_2pe,
        # eval
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="RoadArrow_2pe_cls",
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
        ),
        evalset_config=dict(
            # leaderboard评测任务
            task_name="ADAS Classification V2",
            # leaderboard上评测集的名字
            dataset_name="cn_road_arrow_2pe_cls_eval",
            # 暂时没用到
            name_prefix="cn_road_arrow_2pe_cls_eval",
            # eval creator的配置。配置的定义上使用了registry，整体的格式与HAT类似，
            # type 定义了eval_creator的名字，eval_creator的参数通过字典的key-value传入。
            eval_creator_cfg=dict(
                # 用户自定义的格式转换函数。
                type="CommonClassification",
            ),
        ),
    ),
    "cn_road_arrow_2pe_fp": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            _dir_of_this_file, "packing", "cn", "anno_ts_2pe_fp.yaml"
        ),
        class_name=CLASS_NAMES_ROAD_ARROW["cn_fp"],
        task_owners_email=["xiufeng.zhou@horizon.ai"],
        data_pack_update_fn=data_packer_crop_2pe_roi,
        anno_transformer_update_fn=anno_transform_2pe,
        # eval
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="RoadArrowFp_2pe_cls",
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
        ),
        evalset_config=dict(
            # leaderboard评测任务
            task_name="ADAS Classification V2",
            # leaderboard上评测集的名字
            dataset_name="cn_road_arrow_2pe_fp_eval",
            # 暂时没用到
            name_prefix="cn_road_arrow_2pe_fp_eval",
            # eval creator的配置。配置的定义上使用了registry，整体的格式与HAT类似，
            # type 定义了eval_creator的名字，eval_creator的参数通过字典的key-value传入。
            eval_creator_cfg=dict(
                # 用户自定义的格式转换函数。
                type="CommonClassification",
            ),
        ),
    ),
    "cn_road_arrow_2pe_dt": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            _dir_of_this_file, "packing", "cn", "anno_ts_2pe_dt.yaml"
        ),
        class_name=CLASS_NAMES_ROAD_ARROW["cn_dt"],
        task_owners_email=["xiufeng.zhou@horizon.ai"],
        data_pack_update_fn=data_packer_crop_2pe_roi,
        anno_transformer_update_fn=anno_transform_2pe,
        # eval
        eval_setting_config=dict(
            eval_setting_genenrator=get_default_settings,
            classname="RoadArrowDt_2pe_cls",
            sensor="X8B",
            model_type="2pe",
            custom_tags=None,
        ),
        evalset_config=dict(
            # leaderboard评测任务
            task_name="ADAS Classification V2",
            # leaderboard上评测集的名字
            dataset_name="cn_road_arrow_2pe_dt_eval",
            # 暂时没用到
            name_prefix="cn_road_arrow_2pe_dt_eval",
            # eval creator的配置。配置的定义上使用了registry，整体的格式与HAT类似，
            # type 定义了eval_creator的名字，eval_creator的参数通过字典的key-value传入。
            eval_creator_cfg=dict(
                # 用户自定义的格式转换函数。
                type="CommonClassification",
            ),
        ),
    ),
    "cn_road_arrow_2pe_fp_prelabel": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            _dir_of_this_file, "packing", "cn", "anno_ts_2pe_fp.yaml"
        ),
        class_name=CLASS_NAMES_ROAD_ARROW["cn_fp"],
        task_owners_email=["xiufeng.zhou@horizon.ai"],
        data_pack_update_fn=data_packer_crop_2pe_roi,
        prelabel_for_dataset=True,
        prelabel_config_update_fn=prelabel_config_update_fn,
        anno_transformer_update_fn=partial(
            anno_transform_2pe, perception_extension_fn="add_fp_merge"
        ),
    ),
}
