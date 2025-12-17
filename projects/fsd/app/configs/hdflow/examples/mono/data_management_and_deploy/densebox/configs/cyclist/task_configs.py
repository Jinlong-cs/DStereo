import os

from hdflow.plugins.mono.data_management_and_deploy.base.structure import (  # noqa
    CLASS_NAMES_DETECTION,
    DataDeployTasksCfgs,
)

__all__ = [
    "TASK_NAME_TO_TS_CFG",
]

root_dir = os.path.join(os.path.dirname(os.path.relpath(__file__)))

eval_setting_config_default = {
    "0-20m": "hdfs://hobot-bigdata/user/lele.liu/setting/cyc_2kps_0to20m.yaml",  # noqa
    "20-40m": "hdfs://hobot-bigdata/user/lele.liu/setting/cyc_2kps_20to40m.yaml",  # noqa
    "40-60m": "hdfs://hobot-bigdata/user/lele.liu/setting/cyc_2kps_40to60m.yaml",  # noqa
    "0-60m": "hdfs://hobot-bigdata/user/lele.liu/setting/cyc_2kps_0to60m.yaml",  # noqa
}


evalset_config_default = dict(
    task_name="Keypoints Detection",
    dataset_name="cyclist wheel 2kps",
    name_prefix="cyclist-wheel-kps",
    eval_creator_cfg=dict(
        type="Common2PEKpsDet2D",  # need changing
        filter_configs=[
            dict(
                type="WheelKeypointsAttrConvert",
                class_name="person",
                attr_name="p_WheelKeyPoints_2",
            )
        ],
    ),
)


def update_type_name(default, new_type, **kwargs):
    default["type"] = new_type
    return default


num_kps = 2

# [
#     bbox_type,
#     bbox_clur, bbox_occ, bbox_ignore,
#     bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax,
#     kp_back_x, kp_back_y, kp_front_x, kp_front_y,
#     occ_back, occ_front
# ]
obj_ele_num_2kps = 1 + 3 + 4 + 4 + 2

# [
#     bbox_type,
#     bbox_clur, bbox_occ, bbox_ignore,
#     bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax,
#     kp_back_right_x, kp_back_right_y, kp_back_left_x, kp_back_left_y,
#     kp_front_x, kp_front_y, occ_back_right, occ_back_left, occ_front
# ]
obj_ele_num_3kps = 1 + 3 + 4 + 6 + 3

# [
#     bbox_type,
#     bbox_clur, bbox_occ, bbox_ignore,
#     bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax,
#     (kps_i_x, kps_i_y) * num_kps,
#     occ_i * num_kps
# ]
obj_ele_num_xkps = 1 + 3 + 4 + num_kps * 2 + num_kps

# valid_types = [
#     'Sedan_Car', 'Car', 'SUV', 'MiniVan', 'Van', 'Tram', 'Bus', 'Misc',
#     'BigTruck', 'Truck', 'Lorry', 'MotorcycleWithPerson', 'BikeWithPerson',
#     'other', 'unknown', 'Trucks', 'Small_Medium_Car'
# ]

valid_types = ["PersonRideBicycle", "PersonRideMotorcycle"]


def get_kps_ann_ts_config(default, **kwargs):
    args = kwargs.get("args", None)
    skip_invalid = args.workflow_pass_invalid if args is not None else False
    ann_ts_kps = dict(
        type="Compose",
        transformer=[
            dict(
                type="KeyPointAnnoJsonTs",
                num_kps=2,
                conf_ignore_list=[],
                class_name="person",
                kps_prefix="p_WheelKeyPoints_",
            ),
            dict(
                type="KeyPointAnnoTs",
                num_kps=2,
                valid_types=valid_types,
                min_width=30,
                min_height=30,
                kps_occ_id_dict={
                    "full_visible": 2,
                    "occluded": 1,
                    "outside": 0,
                    "ignore": 3,
                },
                occ_ignore_list=["invisible"],
                obj_ele_num_2kps=obj_ele_num_2kps,
                obj_ele_num_3kps=obj_ele_num_3kps,
                obj_ele_num_xkps=obj_ele_num_xkps,
                skip_invalid=skip_invalid,
            ),
        ],
    )
    return ann_ts_kps


def get_kps_packing_configs(default, **kwargs):
    args = kwargs.get("args", None)
    skip_invalid = args.workflow_pass_invalid if args is not None else False
    num_workers = kwargs["num_workers"]
    pack_configs = dict(
        # pack image data
        image_packer=dict(
            type="HorizonImageDataPacker",
            uri=kwargs["output_rec_path"],
            num_worker=num_workers,
            skip_invalid=skip_invalid,
        ),
        # dataset for read packed image data
        image_dataset=dict(
            type="HorizonImageDataDataset",
            rec_path=kwargs["output_rec_path"],
            rec_idx_file=kwargs["output_rec_path"] + ".idx",
            to_rgb=True,
            decode_img=False,
        ),
        # packer for pack annotation data
        anno_packer=dict(
            type="KPSAnnoPacker",
            uri=kwargs["output_anno_path"],
            num_worker=num_workers,
        ),
    )
    return pack_configs


TASK_NAME_TO_TS_CFG = {
    # ----------------------------------------
    # cyclist
    # ----------------------------------------
    "cyclist_kps": DataDeployTasksCfgs(
        # no yaml type
        task_ts_cfg=None,
        packing_task_mode="default",
        task_owners_email=["lele.liu@horizon.ai"],
        data_pack_update_fn=lambda default, **kwargs: get_kps_packing_configs(
            default, **kwargs
        ),
        anno_transformer_update_fn=lambda default, **kwargs: get_kps_ann_ts_config(  # noqa
            default, **kwargs
        ),
        class_name=[],
        eval_setting_config=eval_setting_config_default,
        evalset_config=evalset_config_default,
    ),
}
