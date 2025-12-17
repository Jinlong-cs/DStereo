import os

from easydict import EasyDict

NAME2DATASETID = {
    # sod
    "parking_column": (
        6038980,  # V1.9.3VAL
        # 6038039,  # V1.8.0
        # 6037309, # V1.7.0
        # 6036670, # V1.5.4
        "parking_column",
    ),
    "traffic_bollard": (
        6038981,  # V1.9.3VAL
        # 6038040,  # V1.8.0
        # 6037304, # V1.7.0
        # 6037271, # V1.5.4
        "traffic_common_bc",
    ),
    "traffic_cone": (
        6038982,  # V1.9.3VAL
        # 6038041,  # V1.8.0
        # 6037305, # V1.7.0
        # 6037272, # V1.5.4
        "traffic_common_bc",
    ),
    "aframe_sign": (
        6039139,  # V1.9.3VAL
        # 6038042,  # V1.8.0
        # 6037306, # V1.7.0
        # 6037273, # V1.5.4
        "aframe_sign",
    ),
    "parking_lock_open": (
        6038984,  # V1.9.3VAL
        # 6038066,  # V1.8.0 same source
        # 6037307, # V1.7.0
        # 6036677, # V1.5.4
        "parking_lock_open",
    ),
    "parking_lock_close": (
        6038977,  # V1.9.3VAL
        # 6038067,  # V1.8.0 same source
        # 6037308, # V1.7.0
        # 6036678, # V1.5.4
        "parking_lock_close",
    ),
    # 2d vd/vru
    "cyclist": (6037203, "cyclist"),
    "person": (6037159, "person"),
    "face": (6037152, "face"),
    "person_occlusion_classification": (
        6036786,
        "person_occlusion_classification",
    ),
    "person_orientation_classification": (
        6037156,
        "person_orientation_classification",
    ),
    "person_pose_classification": (6036788, "person_pose_classification"),
    "rear": (6036810, "rear"),
    "rear_occlusion_classification": (
        6036804,
        "rear_occlusion_classification",
    ),
    "rear_part_classification": (6037150, "rear_part_classification"),
    "plate": (6037151, "plate"),
    "vehicle": (6036754, "vehicle"),
    "vehicle_category_classification": (
        6036753,
        "vehicle_category_classification",
    ),
    "vehicle_occlusion_classification": (
        6036751,
        "vehicle_occlusion_classification",
    ),
    "vehicle_truncation_classification": (
        6036752,
        "vehicle_truncation_classification",
    ),
    "vehicle_wheel_kps": (6036781, "vehicle_wheel_kps"),
}
NAME2EMPTYDATASETID = {
    "cyclist": (6037251, "cyclist"),
    "person": (6037255, "person"),
}
NAME2VALDATASETID = {
    "cyclist": (6036583, "cyclist"),
    "person": (6036584, "person"),
    "vehicle": (6036593, "vehicle"),
    "rear": (6036588, "rear"),
}

is_local = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local else "/bucket/input"

sp_root = bucket_root + "/SuperParking"
sp_last_version_root = sp_root + "/LastVersion/Dataset"
sp_vw_root = sp_root + "/VWVersion/Dataset"
sod_root_v1_9_0_new = os.path.join(
    bucket_root,
    "SuperParking/LastVersion/Dataset/SOD/V1.9.0/",
)
sod_root_v1_9_3_new = os.path.join(
    bucket_root,
    "SuperParking/LastVersion/Dataset/SOD/V1.9.3/",
)
root_mini = os.path.join(
    bucket_root,
    "HDLTAlgorithm/data/sp_ci_data/mini_trainsets/",
)
# --------------------------- SOD --------------------------------
parsing_root = bucket_root + "/SuperParking"
laneparsing_root = bucket_root + "/SuperParking"
mono_root = bucket_root + "/mono"
# ----- Required -----
datapaths = dict(
    vehicle=dict(
        train_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VD/vehicle/example_pack_densebox_vehicle_data_2022-03-17_2022-05-25/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VD/vehicle/example_pack_densebox_vehicle_data_2022-03-17_2022-05-25/train/train.anno.pb_rec",  # noqa
                sample_weight=118189,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/vehicle/pack_vehicle_data_2022-05-26_2022-07-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/vehicle/pack_vehicle_data_2022-05-26_2022-07-20/train/train.anno.pb_rec",  # noqa
                sample_weight=123866,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.2/VD/vehicle/pack_vehicle_data_2022-07-28_2022-10-07/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.2/VD/vehicle/pack_vehicle_data_2022-07-28_2022-10-07/train/train.anno.pb_rec",  # noqa
                sample_weight=68903,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.3/VD/vehicle/pack_vehicle_data_2022-10-08_2022-10-31/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.3/VD/vehicle/pack_vehicle_data_2022-10-08_2022-10-31/train/train.anno.pb_rec",  # noqa
                sample_weight=7544,
            ),
            dict(
                rec_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VD/vehicle/pack_vehicle_data_2023-04-10_2023-04-10/train/train.rec",  # noqa
                anno_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VD/vehicle/pack_vehicle_data_2023-04-10_2023-04-10/train/train.anno.pb_rec",  # noqa
                sample_weight=449,  # 449
            ),
        ],
        val_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VD/vehicle/example_pack_densebox_vehicle_data_2022-03-17_2022-05-25/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VD/vehicle/example_pack_densebox_vehicle_data_2022-03-17_2022-05-25/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/vehicle/pack_vehicle_data_2022-05-26_2022-07-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/vehicle/pack_vehicle_data_2022-05-26_2022-07-20/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/VD_VRU2D/VD/vehicle/train.rec",  # noqa
                anno_path=f"{root_mini}/VD_VRU2D/VD/vehicle/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        aidiEval_data_paths=[
            dict(
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/aidi_test_datatsetV1.0/output_veh_full/upload_file/data.json",  # noqa
                rec_path=None,
                sample_weight=1,
            ),
        ],
    ),
    vehicle_wheel_kps=dict(
        train_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VD/vehicle_kps2/example_pack_densebox_vehicle_data_20220531_232736/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VD/vehicle_kps2/example_pack_densebox_vehicle_data_20220531_232736/train/train.pb_rec",  # noqa
                sample_weight=60361,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/vehicle_wheel_kps/pack_vehicle_wheel_kps_data_2022-06-01_2022-07-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/vehicle_wheel_kps/pack_vehicle_wheel_kps_data_2022-06-01_2022-07-20/train/train.pb_rec",  # noqa
                sample_weight=91950,
            ),
            dict(
                rec_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VD/vehicle_wheel_kps/pack_vehicle_wheel_kps_data_2023-04-06_2023-04-14/train/train.rec",  # noqa
                anno_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VD/vehicle_wheel_kps/pack_vehicle_wheel_kps_data_2023-04-06_2023-04-14/train/train.pb_rec",  # noqa
                sample_weight=408,  # 408
            ),
        ],
        val_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VD/vehicle_kps2/example_pack_densebox_vehicle_data_20220531_232736/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VD/vehicle_kps2/example_pack_densebox_vehicle_data_20220531_232736/val/val.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/vehicle_wheel_kps/pack_vehicle_wheel_kps_data_2022-06-01_2022-07-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/vehicle_wheel_kps/pack_vehicle_wheel_kps_data_2022-06-01_2022-07-20/val/val.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/VD_VRU2D/VD/vehicle_wheel_kps/train.rec",  # noqa
                anno_path=f"{root_mini}/VD_VRU2D/VD/vehicle_wheel_kps/train.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        aidiEval_data_paths=[
            dict(
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/aidi_test_datatsetV1.0/output_veh_kps_2/upload_file/data.json",  # noqa
                rec_path=None,
                sample_weight=1,
            ),
        ],
    ),
    person=dict(
        train_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VRU/person/example_pack_densebox_person_data_2022-04-06_2022-05-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VRU/person/example_pack_densebox_person_data_2022-04-06_2022-05-20/train/train.anno.pb_rec",  # noqa
                sample_weight=56647,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person/pack_person_data_2022-05-21_2022-07-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person/pack_person_data_2022-05-21_2022-07-20/train/train.anno.pb_rec",  # noqa
                sample_weight=43250,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.2/VRU/person/pack_person_data_2022-07-28_2022-10-08/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.2/VRU/person/pack_person_data_2022-07-28_2022-10-08/train/train.anno.pb_rec",  # noqa
                sample_weight=26068,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.3/VRU/person/pack_person_data_2022-10-09_2022-10-31/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.3/VRU/person/pack_person_data_2022-10-09_2022-10-31/train/train.anno.pb_rec",  # noqa
                sample_weight=13231,
            ),
            dict(
                rec_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VRU/person/pack_person_data_2023-04-06_2023-04-06/train/train.rec",  # noqa
                anno_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VRU/person/pack_person_data_2023-04-06_2023-04-06/train/train.anno.pb_rec",  # noqa
                sample_weight=78,  # 78
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VRU/person/example_pack_densebox_person_data_2022-04-06_2022-05-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VRU/person/example_pack_densebox_person_data_2022-04-06_2022-05-20/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person/pack_person_data_2022-05-21_2022-07-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person/pack_person_data_2022-05-21_2022-07-20/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/VD_VRU2D/VRU/person/train.rec",  # noqa
                anno_path=f"{root_mini}/VD_VRU2D/VRU/person/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        aidiEval_data_paths=[
            dict(
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/aidi_test_datatsetV1.0/output_ped_fix/upload_file/data.json",  # noqa
                rec_path=None,
                sample_weight=1,
            ),
        ],
        aidiEval_empty_data_paths=[
            dict(
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/aidi_test_datatsetV1.0/output_ped_empty/upload_file/data.json",  # noqa
                rec_path=None,
                sample_weight=1,
            ),
        ],
    ),
    cyclist=dict(
        train_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/cyclist/pack_cyclist_data_2022-04-06_2022-07-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/cyclist/pack_cyclist_data_2022-04-06_2022-07-20/train/train.anno.pb_rec",  # noqa
                sample_weight=32319,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.2/VRU/cyclist/pack_cyclist_data_2022-07-28_2022-10-08/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.2/VRU/cyclist/pack_cyclist_data_2022-07-28_2022-10-08/train/train.anno.pb_rec",  # noqa
                sample_weight=3160,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.3/VRU/cyclist/pack_cyclist_data_2022-10-09_2022-10-31/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.3/VRU/cyclist/pack_cyclist_data_2022-10-09_2022-10-31/train/train.anno.pb_rec",  # noqa
                sample_weight=4409,
            ),
        ],
        val_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/cyclist/pack_cyclist_data_2022-04-06_2022-07-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/cyclist/pack_cyclist_data_2022-04-06_2022-07-20/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/VD_VRU2D/VRU/cyclist/train.rec",  # noqa
                anno_path=f"{root_mini}/VD_VRU2D/VRU/cyclist/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        aidiEval_data_paths=[
            dict(
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/aidi_test_datatsetV1.0/output_cyclist_fix_2/upload_file/data.json",  # noqa
                rec_path=None,
                sample_weight=1,
            ),
        ],
        aidiEval_empty_data_paths=[
            dict(
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/aidi_test_datatsetV1.0/output_cyclist_empty/upload_file/data.json",  # noqa
                rec_path=None,
                sample_weight=1,
            ),
        ],
    ),
    person_head_detection=dict(
        train_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VRU/person_head/pack_person_head_data_2022-06-01_2022-06-28/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VRU/person_head/pack_person_head_data_2022-06-01_2022-06-28/train/train.anno.pb_rec",  # noqa
                sample_weight=23946,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_head/pack_person_head_data_2022-06-29_2022-07-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_head/pack_person_head_data_2022-06-29_2022-07-20/train/train.anno.pb_rec",  # noqa
                sample_weight=14703,
            ),
            dict(
                rec_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VRU/person_head/pack_person_head_data_2023-04-06_2023-04-14/train/train.rec",  # noqa
                anno_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VRU/person_head/pack_person_head_data_2023-04-06_2023-04-14/train/train.anno.pb_rec",  # noqa
                sample_weight=27,  # 27
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VRU/person_head/pack_person_head_data_2022-06-01_2022-06-28/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VRU/person_head/pack_person_head_data_2022-06-01_2022-06-28/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_head/pack_person_head_data_2022-06-29_2022-07-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_head/pack_person_head_data_2022-06-29_2022-07-20/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        aidiEval_data_paths=[
            dict(
                rec_path=None,
                anno_path=None,
                sample_weight=1,
            ),
        ],
    ),
    person_age_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VRU/person_age/pack_ped_age_data_2022-05-24_2022-06-27/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VRU/person_age/pack_ped_age_data_2022-05-24_2022-06-27/train/train.anno.pb_rec",  # noqa
                sample_weight=26668,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_age/pack_person_age_data_2022-06-28_2022-07-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_age/pack_person_age_data_2022-06-28_2022-07-20/train/train.anno.pb_rec",  # noqa
                sample_weight=5641,
            ),
            dict(
                rec_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VRU/person_age/pack_person_age_data_2023-04-06_2023-04-14/train/train.rec",  # noqa
                anno_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VRU/person_age/pack_person_age_data_2023-04-06_2023-04-14/train/train.anno.pb_rec",  # noqa
                sample_weight=37,  # 37
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VRU/person_age/pack_ped_age_data_2022-05-24_2022-06-27/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VRU/person_age/pack_ped_age_data_2022-05-24_2022-06-27/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_age/pack_person_age_data_2022-06-28_2022-07-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_age/pack_person_age_data_2022-06-28_2022-07-20/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        aidiEval_data_paths=[
            dict(
                rec_path=None,
                anno_path=None,
                sample_weight=1,
            )
        ],
    ),
    person_occlusion_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VRU/person_occlusion/pack_ped_occlusion_data_2022-05-24_2022-06-27/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VRU/person_occlusion/pack_ped_occlusion_data_2022-05-24_2022-06-27/train/train.anno.pb_rec",  # noqa
                sample_weight=30227,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_occlusion/pack_person_occlusion_data_2022-06-28_2022-07-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_occlusion/pack_person_occlusion_data_2022-06-28_2022-07-20/train/train.anno.pb_rec",  # noqa
                sample_weight=6034,
            ),
            dict(
                rec_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VRU/person_occlusion/pack_person_occlusion_data_2023-04-06_2023-04-14/train/train.rec",  # noqa
                anno_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VRU/person_occlusion/pack_person_occlusion_data_2023-04-06_2023-04-14/train/train.anno.pb_rec",  # noqa
                sample_weight=37,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VRU/person_occlusion/pack_ped_occlusion_data_2022-05-24_2022-06-27/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VRU/person_occlusion/pack_ped_occlusion_data_2022-05-24_2022-06-27/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_occlusion/pack_person_occlusion_data_2022-06-28_2022-07-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_occlusion/pack_person_occlusion_data_2022-06-28_2022-07-20/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/VD_VRU2D/VRU/person_occlusion/train.rec",  # noqa
                anno_path=f"{root_mini}/VD_VRU2D/VRU/person_occlusion/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        aidiEval_data_paths=[
            dict(
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/aidi_test_datatsetV1.0/output_ped_occ/upload_file/data.json",  # noqa
                rec_path=None,
                sample_weight=1,
            ),
        ],
    ),
    person_orientation_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VRU/person_orientation/pack_ped_orientation_data_2022-05-24_2022-06-27/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VRU/person_orientation/pack_ped_orientation_data_2022-05-24_2022-06-27/train/train.anno.pb_rec",  # noqa
                sample_weight=28241,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_orientation/pack_person_orientation_data_2022-06-28_2022-07-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_orientation/pack_person_orientation_data_2022-06-28_2022-07-20/train/train.anno.pb_rec",  # noqa
                sample_weight=3189,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VRU/person_orientation/pack_ped_orientation_data_2022-05-24_2022-06-27/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VRU/person_orientation/pack_ped_orientation_data_2022-05-24_2022-06-27/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_orientation/pack_person_orientation_data_2022-06-28_2022-07-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_orientation/pack_person_orientation_data_2022-06-28_2022-07-20/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/VD_VRU2D/VRU/person_orientation/train.rec",  # noqa
                anno_path=f"{root_mini}/VD_VRU2D/VRU/person_orientation/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        aidiEval_data_paths=[
            dict(
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/aidi_test_datatsetV1.0/output_ped_orien/upload_file/data.json",  # noqa
                rec_path=None,
                sample_weight=1,
            ),
        ],
    ),
    person_pose_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VRU/person_pose/pack_ped_pose_data_2022-05-24_2022-06-27/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VRU/person_pose/pack_ped_pose_data_2022-05-24_2022-06-27/train/train.anno.pb_rec",  # noqa
                sample_weight=28292,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_pose/pack_person_pose_data_2022-06-28_2022-07-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_pose/pack_person_pose_data_2022-06-28_2022-07-20/train/train.anno.pb_rec",  # noqa
                sample_weight=5700,
            ),
            dict(
                rec_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VRU/person_pose/pack_person_pose_data_2023-04-06_2023-04-14/train/train.rec",  # noqa
                anno_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VRU/person_pose/pack_person_pose_data_2023-04-06_2023-04-14/train/train.anno.pb_rec",  # noqa
                sample_weight=34,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VRU/person_pose/pack_ped_pose_data_2022-05-24_2022-06-27/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VRU/person_pose/pack_ped_pose_data_2022-05-24_2022-06-27/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_pose/pack_person_pose_data_2022-06-28_2022-07-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_pose/pack_person_pose_data_2022-06-28_2022-07-20/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/VD_VRU2D/VRU/person_pose/train.rec",  # noqa
                anno_path=f"{root_mini}/VD_VRU2D/VRU/person_pose/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        aidiEval_data_paths=[
            dict(
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/aidi_test_datatsetV1.0/output_ped_pose/upload_file/data.json",  # noqa
                rec_path=None,
                sample_weight=1,
            ),
        ],
    ),
    cyclist_kps=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VRU/cyclist_kps2/cyclist_kps_data_20220608_203735/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VRU/cyclist_kps2/cyclist_kps_data_20220608_203735/train/train.pb_rec",  # noqa
                sample_weight=28170,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/cyclist_kps/pack_cyclist_kps_data_2022-06-09_2022-07-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/cyclist_kps/pack_cyclist_kps_data_2022-06-09_2022-07-20/train/train.pb_rec",  # noqa
                sample_weight=27724,
            ),
        ],
        val_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VRU/cyclist_kps2/cyclist_kps_data_20220608_203735/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VRU/cyclist_kps2/cyclist_kps_data_20220608_203735/val/val.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/cyclist_kps/pack_cyclist_kps_data_2022-06-09_2022-07-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/cyclist_kps/pack_cyclist_kps_data_2022-06-09_2022-07-20/val/val.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        aidiEval_data_paths=[
            dict(
                rec_path=None,
                anno_path=None,
                sample_weight=1,
            )
        ],
    ),
    vehicle_category=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.2/VD/vehicle_category/pack_vehicle_category_data_2022-05-24_2022-07-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.2/VD/vehicle_category/pack_vehicle_category_data_2022-05-24_2022-07-20/train/train.anno.pb_rec",  # noqa
                sample_weight=132762,
            ),
            dict(
                rec_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VD/vehicle_category/pack_vehicle_category_data_2023-04-10_2023-04-10/train/train.rec",  # noqa
                anno_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VD/vehicle_category/pack_vehicle_category_data_2023-04-10_2023-04-10/train/train.anno.pb_rec",  # noqa
                sample_weight=440,  # 440
            ),
        ],
        val_batch_size_per_ctx=16,
        val_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.2/VD/vehicle_category/pack_vehicle_category_data_2022-05-24_2022-07-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.2/VD/vehicle_category/pack_vehicle_category_data_2022-05-24_2022-07-20/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/VD_VRU2D/VD/vehicle_category/train.rec",  # noqa
                anno_path=f"{root_mini}/VD_VRU2D/VD/vehicle_category/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        aidiEval_data_paths=[
            dict(
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/aidi_test_datatsetV1.0/output_veh_type/upload_file/data.json",  # noqa
                rec_path=None,
                sample_weight=1,
            ),
        ],
    ),
    vehicle_truncation_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/vehicle_truncation/pack_vehicle_truncation_data_2022-05-24_2022-07-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/vehicle_truncation/pack_vehicle_truncation_data_2022-05-24_2022-07-20/train/train.anno.pb_rec",  # noqa
                sample_weight=141416,
            ),
            dict(
                rec_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VD/vehicle_truncation/pack_vehicle_truncation_data_2023-04-10_2023-04-10/train/train.rec",  # noqa
                anno_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VD/vehicle_truncation/pack_vehicle_truncation_data_2023-04-10_2023-04-10/train/train.anno.pb_rec",  # noqa
                sample_weight=449,
            ),
        ],
        val_batch_size_per_ctx=16,
        val_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/vehicle_truncation/pack_vehicle_truncation_data_2022-05-24_2022-07-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/vehicle_truncation/pack_vehicle_truncation_data_2022-05-24_2022-07-20//val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/VD_VRU2D/VD/vehicle_truncation/train.rec",  # noqa
                anno_path=f"{root_mini}/VD_VRU2D/VD/vehicle_truncation/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        aidiEval_data_paths=[
            dict(
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/aidi_test_datatsetV1.0/output_veh_trunc/upload_file/data.json",  # noqa
                rec_path=None,
                sample_weight=1,
            ),
        ],
    ),
    vehicle_occlusion_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/vehicle_occlusion/pack_vehicle_occlusion_data_2022-05-24_2022-07-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/vehicle_occlusion/pack_vehicle_occlusion_data_2022-05-24_2022-07-20/train/train.anno.pb_rec",  # noqa
                sample_weight=142706,
            ),
            dict(
                rec_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VD/vehicle_occlusion/pack_vehicle_occlusion_data_2023-04-06_2023-04-14/train/train.rec",  # noqa
                anno_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VD/vehicle_occlusion/pack_vehicle_occlusion_data_2023-04-06_2023-04-14/train/train.anno.pb_rec",  # noqa
                sample_weight=449,  # 449
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/vehicle_occlusion/pack_vehicle_occlusion_data_2022-05-24_2022-07-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/vehicle_occlusion/pack_vehicle_occlusion_data_2022-05-24_2022-07-20/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/VD_VRU2D/VD/vehicle_occlusion/train.rec",  # noqa
                anno_path=f"{root_mini}/VD_VRU2D/VD/vehicle_occlusion/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        aidiEval_data_paths=[
            dict(
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/aidi_test_datatsetV1.0/output_veh_occ/upload_file/data.json",  # noqa
                rec_path=None,
                sample_weight=1,
            ),
        ],
    ),
    rear=dict(
        train_batch_size_per_ctx=24,
        train_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VD/rear/example_pack_densebox_rear_data_2022-03-17_2022-05-25/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VD/rear/example_pack_densebox_rear_data_2022-03-17_2022-05-25/train/train.anno.pb_rec",  # noqa
                sample_weight=81042,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/rear/pack_rear_data_2022-05-26_2022-07-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/rear/pack_rear_data_2022-05-26_2022-07-20/train/train.anno.pb_rec",  # noqa
                sample_weight=109182,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.2/VD/rear/pack_rear_data_2022-07-28_2022-10-07/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.2/VD/rear/pack_rear_data_2022-07-28_2022-10-07/train/train.anno.pb_rec",  # noqa
                sample_weight=63688,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.3/VD/rear/pack_rear_data_2022-10-08_2022-10-31/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.3/VD/rear/pack_rear_data_2022-10-08_2022-10-31/train/train.anno.pb_rec",  # noqa
                sample_weight=15145,
            ),
            dict(
                rec_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VD/rear/pack_rear_data_2023-04-06_2023-04-14/train/train.rec",  # noqa
                anno_path=f"{sp_vw_root}/2DVDVRU/demo.202305/VD/rear/pack_rear_data_2023-04-06_2023-04-14/train/train.anno.pb_rec",  # noqa
                sample_weight=469,  # 469
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VD/rear/example_pack_densebox_rear_data_2022-03-17_2022-05-25/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VD/rear/example_pack_densebox_rear_data_2022-03-17_2022-05-25/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/rear/pack_rear_data_2022-05-26_2022-07-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/rear/pack_rear_data_2022-05-26_2022-07-20/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/VD_VRU2D/VD/rear/train.rec",  # noqa
                anno_path=f"{root_mini}/VD_VRU2D/VD/rear/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        aidiEval_data_paths=[
            dict(
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/aidi_test_datatsetV1.0/output_veh_rear_2/upload_file/data.json",  # noqa
                rec_path=None,
                sample_weight=1,
            ),
        ],
    ),
    rear_occlusion_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VRU/rear_occlusion/pack_rear_occlusion_data_2022-05-24_2022-06-27/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VRU/rear_occlusion/pack_rear_occlusion_data_2022-05-24_2022-06-27/train/train.anno.pb_rec",  # noqa
                sample_weight=85341,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/rear_occlusion/pack_rear_occlusion_data_2022-06-28_2022-07-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/rear_occlusion/pack_rear_occlusion_data_2022-06-28_2022-07-20/train/train.anno.pb_rec",  # noqa
                sample_weight=52686,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VRU/rear_occlusion/pack_rear_occlusion_data_2022-05-24_2022-06-27/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/xiang.hu/Detection2D/datasetV1.0/VRU/rear_occlusion/pack_rear_occlusion_data_2022-05-24_2022-06-27/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/rear_occlusion/pack_rear_occlusion_data_2022-06-28_2022-07-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/rear_occlusion/pack_rear_occlusion_data_2022-06-28_2022-07-20/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/VD_VRU2D/VD/rear_occlusion/train.rec",  # noqa
                anno_path=f"{root_mini}/VD_VRU2D/VD/rear_occlusion/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        aidiEval_data_paths=[
            dict(
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/aidi_test_datatsetV1.0/output_rear_occ_2/upload_file/data.json",  # noqa
                rec_path=None,
                sample_weight=1,
            ),
        ],
    ),
    rear_part_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VD/rear_part/pack_rear_part_data_2022-05-24_2022-06-27/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VD/rear_part/pack_rear_part_data_2022-05-24_2022-06-27/train/train.anno.pb_rec",  # noqa
                sample_weight=68994,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/rear_part/pack_rear_part_data_2022-06-28_2022-07-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/rear_part/pack_rear_part_data_2022-06-28_2022-07-20/train/train.anno.pb_rec",  # noqa
                sample_weight=37255,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VD/rear_part/pack_rear_part_data_2022-05-24_2022-06-27/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.0/VD/rear_part/pack_rear_part_data_2022-05-24_2022-06-27/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/rear_part/pack_rear_part_data_2022-06-28_2022-07-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/rear_part/pack_rear_part_data_2022-06-28_2022-07-20/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/VD_VRU2D/VD/rear_part/train.rec",  # noqa
                anno_path=f"{root_mini}/VD_VRU2D/VD/rear_part/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        aidiEval_data_paths=[
            dict(
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/aidi_test_datatsetV1.0/output_rear_part_2/upload_file/data.json",  # noqa
                rec_path=None,
                sample_weight=1,
            ),
        ],
    ),
    rear_plate_detection=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/rear_plate/pack_rear_plate_data_2022-05-24_2022-07-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/rear_plate/pack_rear_plate_data_2022-05-24_2022-07-20/train/train.anno.pb_rec",  # noqa
                sample_weight=69873,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/rear_plate/pack_rear_plate_data_2022-05-24_2022-07-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VD/rear_plate/pack_rear_plate_data_2022-05-24_2022-07-20/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/VD_VRU2D/VD/rear_plate/train.rec",  # noqa
                anno_path=f"{root_mini}/VD_VRU2D/VD/rear_plate/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        aidiEval_data_paths=[
            dict(
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/aidi_test_datatsetV1.0/output_rear_plate/upload_file/data.json",  # noqa
                rec_path=None,
                sample_weight=1,
            ),
        ],
    ),
    person_face_detection=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_face/pack_person_face_data_2022-05-24_2022-07-20/train/train.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_face/pack_person_face_data_2022-05-24_2022-07-20/train/train.anno.pb_rec",  # noqa
                sample_weight=5103,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_face/pack_person_face_data_2022-05-24_2022-07-20/val/val.rec",  # noqa
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/datasetV1.1/VRU/person_face/pack_person_face_data_2022-05-24_2022-07-20/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/VD_VRU2D/VRU/person_face/train.rec",  # noqa
                anno_path=f"{root_mini}/VD_VRU2D/VRU/person_face/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        aidiEval_data_paths=[
            dict(
                anno_path=f"{sp_last_version_root}/2DVDVRU/fabing.tan/Detection2D/aidi_test_datatsetV1.0/output_ped_face/upload_file/data.json",  # noqa
                rec_path=None,
                sample_weight=1,
            ),
        ],
    ),
    parking_column=dict(
        train_data_paths=[
            dict(
                rec_path=f"{sod_root_v1_9_3_new}/column/pack_densebox_column_data_2022-03-28_2022-09-30/train/train.rec",  # noqa
                anno_path=f"{sod_root_v1_9_3_new}/column/pack_densebox_column_data_2022-03-28_2022-09-30/train/train.anno.pb_rec",  # noqa
                sample_weight=44099,
            ),
            dict(
                rec_path=f"{sod_root_v1_9_0_new}/column/pack_densebox_column_data_2022-10-15_2022-10-31/train/train.rec",  # noqa
                anno_path=f"{sod_root_v1_9_0_new}/column/pack_densebox_column_data_2022-10-15_2022-10-31/train/train.anno.pb_rec",  # noqa
                sample_weight=3983,
            ),
            dict(
                rec_path=f"{sod_root_v1_9_3_new}/column/pack_densebox_column_data_2022-10-01_2022-10-09/train/train.rec",  # noqa
                anno_path=f"{sod_root_v1_9_3_new}/column/pack_densebox_column_data_2022-10-01_2022-10-09/train/train.anno.pb_rec",  # noqa
                sample_weight=5781,
            ),
            dict(
                rec_path=f"{sp_vw_root}/SOD/demo.202305/column/pack_column_2023-04-06_2023-04-14/train/train.rec",  # noqa
                anno_path=f"{sp_vw_root}/SOD/demo.202305/column/pack_column_2023-04-06_2023-04-14/train/train.anno.pb_rec",  # noqa
                sample_weight=58,  # 58
            ),
        ],
        val_data_paths=[
            dict(
                rec_path=f"{sod_root_v1_9_3_new}/column/pack_densebox_column_data_2022-03-28_2022-09-30/val/val.rec",  # noqa
                anno_path=f"{sod_root_v1_9_3_new}/column/pack_densebox_column_data_2022-03-28_2022-09-30/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{sod_root_v1_9_0_new}/column/pack_densebox_column_data_2022-10-15_2022-10-31/val/val.rec",  # noqa
                anno_path=f"{sod_root_v1_9_0_new}/column/pack_densebox_column_data_2022-10-15_2022-10-31/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/SOD/column/train/train.rec",  # noqa
                anno_path=f"{root_mini}/SOD/column/train/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
    ),
    parking_lock_open=dict(
        train_data_paths=[
            dict(
                rec_path=f"{sod_root_v1_9_3_new}/parking_lock_open/pack_densebox_parking_lock_data_2022-03-22_2022-09-30/train/train.rec",  # noqa
                anno_path=f"{sod_root_v1_9_3_new}/parking_lock_open/pack_densebox_parking_lock_data_2022-03-22_2022-09-30/train/train.anno.pb_rec",  # noqa
                sample_weight=10269,
            ),
            dict(
                rec_path=f"{sod_root_v1_9_0_new}/parking_lock_open/pack_densebox_parking_lock_data_2022-10-15_2022-10-31/train/train.rec",  # noqa
                anno_path=f"{sod_root_v1_9_0_new}/parking_lock_open/pack_densebox_parking_lock_data_2022-10-15_2022-10-31/train/train.anno.pb_rec",  # noqa
                sample_weight=1770,
            ),
            dict(
                rec_path=f"{sod_root_v1_9_3_new}/parking_lock_open/pack_densebox_parking_lock_data_2022-10-01_2022-10-09/train/train.rec",  # noqa
                anno_path=f"{sod_root_v1_9_3_new}/parking_lock_open/pack_densebox_parking_lock_data_2022-10-01_2022-10-09/train/train.anno.pb_rec",  # noqa
                sample_weight=977,
            ),
        ],
        val_data_paths=[
            dict(
                rec_path=f"{sod_root_v1_9_3_new}/parking_lock_open/pack_densebox_parking_lock_data_2022-03-22_2022-09-30/val/val.rec",  # noqa
                anno_path=f"{sod_root_v1_9_3_new}/parking_lock_open/pack_densebox_parking_lock_data_2022-03-22_2022-09-30/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{sod_root_v1_9_0_new}/parking_lock_open/pack_densebox_parking_lock_data_2022-10-15_2022-10-31/val/val.rec",  # noqa
                anno_path=f"{sod_root_v1_9_0_new}/parking_lock_open/pack_densebox_parking_lock_data_2022-10-15_2022-10-31/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/SOD/lock_open/train/train.rec",  # noqa
                anno_path=f"{root_mini}/SOD/lock_open/train/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
    ),
    parking_lock_close=dict(
        train_data_paths=[
            dict(
                rec_path=f"{sod_root_v1_9_3_new}/parking_lock_close/pack_densebox_parking_lock_data_2022-03-22_2022-09-30/train/train.rec",  # noqa
                anno_path=f"{sod_root_v1_9_3_new}/parking_lock_close/pack_densebox_parking_lock_data_2022-03-22_2022-09-30/train/train.anno.pb_rec",  # noqa
                sample_weight=22706,
            ),
            dict(
                rec_path=f"{sod_root_v1_9_0_new}/parking_lock_close/pack_densebox_parking_lock_data_2022-10-15_2022-10-31/train/train.rec",  # noqa
                anno_path=f"{sod_root_v1_9_0_new}/parking_lock_close/pack_densebox_parking_lock_data_2022-10-15_2022-10-31/train/train.anno.pb_rec",  # noqa
                sample_weight=2594,
            ),
            dict(
                rec_path=f"{sod_root_v1_9_3_new}/parking_lock_close/pack_densebox_parking_lock_data_2022-10-01_2022-10-09/train/train.rec",  # noqa
                anno_path=f"{sod_root_v1_9_3_new}/parking_lock_close/pack_densebox_parking_lock_data_2022-10-01_2022-10-09/train/train.anno.pb_rec",  # noqa
                sample_weight=2056,
            ),
        ],
        val_data_paths=[
            dict(
                rec_path=f"{sod_root_v1_9_3_new}/parking_lock_close/pack_densebox_parking_lock_data_2022-03-22_2022-09-30/val/val.rec",  # noqa
                anno_path=f"{sod_root_v1_9_3_new}/parking_lock_close/pack_densebox_parking_lock_data_2022-03-22_2022-09-30/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{sod_root_v1_9_0_new}/parking_lock_close/pack_densebox_parking_lock_data_2022-10-15_2022-10-31/val/val.rec",  # noqa
                anno_path=f"{sod_root_v1_9_0_new}/parking_lock_close/pack_densebox_parking_lock_data_2022-10-15_2022-10-31/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/SOD/lock_close/train/train.rec",  # noqa
                anno_path=f"{root_mini}/SOD/lock_close/train/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
    ),
    traffic_bollard=dict(
        train_data_paths=[
            dict(
                rec_path=f"{sod_root_v1_9_0_new}/traffic_bollard/pack_densebox_traffic_bollard_data_2022-10-15_2022-10-31/train/train.rec",  # noqa
                anno_path=f"{sod_root_v1_9_0_new}/traffic_bollard/pack_densebox_traffic_bollard_data_2022-10-15_2022-10-31/train/train.anno.pb_rec",  # noqa
                sample_weight=3342,
            ),
            dict(
                rec_path=f"{sod_root_v1_9_3_new}/traffic_bollard/pack_densebox_traffic_bollard_data_2022-03-22_2022-09-30/train/train.rec",  # noqa
                anno_path=f"{sod_root_v1_9_3_new}/traffic_bollard/pack_densebox_traffic_bollard_data_2022-03-22_2022-09-30/train/train.anno.pb_rec",  # noqa
                sample_weight=57690,
            ),
            dict(
                rec_path=f"{sod_root_v1_9_3_new}/traffic_bollard/pack_densebox_traffic_bollard_data_2022-10-01_2022-10-08/train/train.rec",  # noqa
                anno_path=f"{sod_root_v1_9_3_new}/traffic_bollard/pack_densebox_traffic_bollard_data_2022-10-01_2022-10-08/train/train.anno.pb_rec",  # noqa
                sample_weight=4469,
            ),
            dict(
                rec_path=f"{sp_vw_root}/SOD/demo.202305/traffic_bollard/pack_traffic_bollard_2023-04-06_2023-04-14/train/train.rec",  # noqa
                anno_path=f"{sp_vw_root}/SOD/demo.202305/traffic_bollard/pack_traffic_bollard_2023-04-06_2023-04-14/train/train.anno.pb_rec",  # noqa
                sample_weight=133,  # 133
            ),
        ],
        val_data_paths=[
            dict(
                rec_path=f"{sod_root_v1_9_0_new}/traffic_bollard/pack_densebox_traffic_bollard_data_2022-10-15_2022-10-31/val/val.rec",  # noqa
                anno_path=f"{sod_root_v1_9_0_new}/traffic_bollard/pack_densebox_traffic_bollard_data_2022-10-15_2022-10-31/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{sod_root_v1_9_3_new}/traffic_bollard/pack_densebox_traffic_bollard_data_2022-03-22_2022-09-30/val/val.rec",  # noqa
                anno_path=f"{sod_root_v1_9_3_new}/traffic_bollard/pack_densebox_traffic_bollard_data_2022-03-22_2022-09-30/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/SOD/bollard/train/train.rec",  # noqa
                anno_path=f"{root_mini}/SOD/bollard/train/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
    ),
    traffic_cone=dict(
        train_data_paths=[
            dict(
                rec_path=f"{sod_root_v1_9_0_new}/traffic_cone/pack_densebox_traffic_cone_data_2022-10-15_2022-10-31/train/train.rec",  # noqa
                anno_path=f"{sod_root_v1_9_0_new}/traffic_cone/pack_densebox_traffic_cone_data_2022-10-15_2022-10-31/train/train.anno.pb_rec",  # noqa
                sample_weight=11812,
            ),
            dict(
                rec_path=f"{sod_root_v1_9_3_new}/traffic_cone/pack_densebox_traffic_cone_data_2022-03-22_2022-09-30/train/train.rec",  # noqa
                anno_path=f"{sod_root_v1_9_3_new}/traffic_cone/pack_densebox_traffic_cone_data_2022-03-22_2022-09-30/train/train.anno.pb_rec",  # noqa
                sample_weight=80868,
            ),
            dict(
                rec_path=f"{sod_root_v1_9_3_new}/traffic_cone/pack_densebox_traffic_cone_data_2022-10-01_2022-10-08/train/train.rec",  # noqa
                anno_path=f"{sod_root_v1_9_3_new}/traffic_cone/pack_densebox_traffic_cone_data_2022-10-01_2022-10-08/train/train.anno.pb_rec",  # noqa
                sample_weight=12854,
            ),
            dict(
                rec_path=f"{sp_vw_root}/SOD/demo.202305/traffic_cone/pack_traffic_cone_2023-04-06_2023-04-14/train/train.rec",  # noqa
                anno_path=f"{sp_vw_root}/SOD/demo.202305/traffic_cone/pack_traffic_cone_2023-04-06_2023-04-14/train/train.anno.pb_rec",  # noqa
                sample_weight=69,  # 69
            ),
        ],
        val_data_paths=[
            dict(
                rec_path=f"{sod_root_v1_9_0_new}/traffic_cone/pack_densebox_traffic_cone_data_2022-10-15_2022-10-31/val/val.rec",  # noqa
                anno_path=f"{sod_root_v1_9_0_new}/traffic_cone/pack_densebox_traffic_cone_data_2022-10-15_2022-10-31/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{sod_root_v1_9_3_new}/traffic_cone/pack_densebox_traffic_cone_data_2022-03-22_2022-09-30/val/val.rec",  # noqa
                anno_path=f"{sod_root_v1_9_3_new}/traffic_cone/pack_densebox_traffic_cone_data_2022-03-22_2022-09-30/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/SOD/cone/train/train.rec",  # noqa
                anno_path=f"{root_mini}/SOD/cone/train/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
    ),
    aframe_sign=dict(
        train_data_paths=[
            dict(
                rec_path=f"{sod_root_v1_9_0_new}/AFrame/pack_densebox_aframe_data_2022-10-15_2022-10-31/train/train.rec",  # noqa
                anno_path=f"{sod_root_v1_9_0_new}/AFrame/pack_densebox_aframe_data_2022-10-15_2022-10-31/train/train.anno.pb_rec",  # noqa
                sample_weight=710,
            ),
            dict(
                rec_path=f"{sod_root_v1_9_3_new}/AFrame/pack_densebox_aframe_data_2022-03-22_2022-09-30/train/train.rec",  # noqa
                anno_path=f"{sod_root_v1_9_3_new}/AFrame/pack_densebox_aframe_data_2022-03-22_2022-09-30/train/train.anno.pb_rec",  # noqa
                sample_weight=4240,
            ),
            dict(
                rec_path=f"{sod_root_v1_9_3_new}/AFrame/pack_densebox_aframe_data_2022-10-01_2022-10-08/train/train.rec",  # noqa
                anno_path=f"{sod_root_v1_9_3_new}/AFrame/pack_densebox_aframe_data_2022-10-01_2022-10-08/train/train.anno.pb_rec",  # noqa
                sample_weight=596,
            ),
        ],
        val_data_paths=[
            dict(
                rec_path=f"{sod_root_v1_9_0_new}/AFrame/pack_densebox_aframe_data_2022-10-15_2022-10-31/val/val.rec",  # noqa
                anno_path=f"{sod_root_v1_9_0_new}/AFrame/pack_densebox_aframe_data_2022-10-15_2022-10-31/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{sod_root_v1_9_3_new}/AFrame/pack_densebox_aframe_data_2022-03-22_2022-09-30/val/val.rec",  # noqa
                anno_path=f"{sod_root_v1_9_3_new}/AFrame/pack_densebox_aframe_data_2022-03-22_2022-09-30/val/val.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        mini_data_paths=[
            dict(
                rec_path=f"{root_mini}/SOD/aframe/train/train.rec",  # noqa
                anno_path=f"{root_mini}/SOD/aframe/train/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
    ),
)
datapaths = EasyDict(datapaths)
