import copy
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
    "0-50m": "hdfs://hobot-bigdata/user/lele.liu/setting/veh_2kps_0to50m.yaml"  # noqa
}

evalset_config_default = dict(
    task_name="Keypoints Detection",
    dataset_name="vehicle wheel 2kps",
    name_prefix="vehicle-wheel-kps",
    eval_creator_cfg=dict(
        type="Common2PEKpsDet2D",  # need changing
        filter_configs=[
            dict(
                type="WheelKeypointsAttrConvert",
                class_name="vehicle",
                attr_name="p_WheelKeyPoints_2",
            )
        ],
    ),
)

num_kps = 2
kps_occ_id_dict = {
    "full_visible": 2,
    "occluded": 1,
    "outside": 0,
    "ignore": 3,
    "": 4,
}
annotation_ts_config = dict(
    type="DenseBoxKpsDetAnnoTs",
    verbose=False,
    kps_occ_id_dict=kps_occ_id_dict,
    num_kps=num_kps,
    cls_name="p_WheelKeyPoints_",
    root_dir="./",
)


def update_anno_params(default, **kwargs):
    config = copy.deepcopy(default["config"])
    [default.pop(name) for name in list(default.keys())]
    default["config"] = config
    default.update(annotation_ts_config)

    return default


def update_type_name(default, new_type, **kwargs):
    default["type"] = new_type
    return default


TASK_NAME_TO_TS_CFG = {
    # ----------------------------------------
    # vehicle wheel
    # ----------------------------------------
    "vehicle_wheel_keypoints": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir, "packing/vehicle_wheel_kps_2pe.yaml"
        ),
        class_name=CLASS_NAMES_DETECTION,
        task_owners_email=["lele.liu@horizon.ai"],
        data_pack_update_fn=lambda default, **kwargs: update_type_name(
            default, "LegacyDenseboxDataPackerBasePython", **kwargs
        ),
        anno_transformer_update_fn=lambda default, **kwargs: update_anno_params(  # noqa
            default, **kwargs
        ),
        # evalset
        eval_setting_config=eval_setting_config_default,
        evalset_config=evalset_config_default,
    ),
}
