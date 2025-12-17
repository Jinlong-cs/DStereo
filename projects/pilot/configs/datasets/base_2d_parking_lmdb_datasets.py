from importlib import import_module


def update_data_path(base_data_paths, q_data_path):
    if not bool(q_data_path):
        return base_data_paths
    is_matched = False
    for idx in range(len(base_data_paths)):
        if q_data_path["data_path"] == base_data_paths[idx]["data_path"]:
            # if q_data_path["anno_path"] == base_data_paths[idx]["anno_path"]:
            # same dataset
            is_matched = True
            base_data_paths[idx]["sample_weight"] = q_data_path[
                "sample_weight"
            ]

    if not is_matched:
        base_data_paths.append(q_data_path)
    return base_data_paths


def merge_datapaths(base_datapaths, query_datapaths):
    for key in query_datapaths:
        # first level of dict in datapaths: task
        if key in base_datapaths:
            if base_datapaths[key] is None or query_datapaths[key] is None:
                pass
            else:
                assert isinstance(base_datapaths[key], dict) and isinstance(
                    query_datapaths[key], dict
                )
                # second level of dict in datapaths:
                # train/val_batch_size_per_ctx: int
                # train/val_data_paths: list/None
                for sub_key in query_datapaths[key]:
                    if isinstance(query_datapaths[key][sub_key], int):
                        base_datapaths[key].update(
                            {sub_key: query_datapaths[key][sub_key]}
                        )
                    elif isinstance(query_datapaths[key][sub_key], list):
                        if base_datapaths[key][sub_key] is None:
                            base_datapaths[key][sub_key] = query_datapaths[
                                key
                            ][sub_key]
                        for data_dict in query_datapaths[key][sub_key]:
                            base_datapaths[key][sub_key] = update_data_path(
                                base_datapaths[key][sub_key], data_dict
                            )
        else:
            base_datapaths.update({key: query_datapaths[key]})
    return base_datapaths


datasets = import_module("base_2d_day_lmdb_datasets")
datapaths = datasets.datapaths
datasets_night = import_module("base_2d_night_lmdb_datasets")
datapaths = merge_datapaths(datapaths, datasets_night.datapaths)

root = "dmpv2://matrix2"


# ----- Required -----

parking_datapaths = {
    "rear_plate_detection": dict(),
    "person_face_detection": dict(),
    "person_occlusion_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/C385_V1/Pilot_person_occ_5v_CAMERA_CO_OX3GB_D100_L_all/20220621-003707/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/C385_v05/Pilot_person_occ_5v_CAMERA_CO_OX3GB_D100_L_all/20220523-233433/data",
                "sample_weight": 16,
            },
        ]
    },
    "vehicle_occlusion_classification": dict(),
    "cyclist_kps": dict(),
    "person_head_detection": dict(),
    "person_pose_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/C385_V1/Pilot_person_pose_5v_CAMERA_CO_OX3GB_D100_L_all/20220621-010354/data",
                "sample_weight": 40,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/C385_v05/Pilot_person_pose_5v_CAMERA_CO_OX3GB_D100_L_all/20220523-234335/data",
                "sample_weight": 40,
            },
        ]
    },
    "vehicle_light_classification": dict(),
    "vehicle_category": dict(),
    "person_age_classification": dict(),
    "person_orientation_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/rare_ori_cls/Pilot_person_orientation_5v_CAMERA_CO_OX3GB_D100_L_all/20220620-234645/data",
                "sample_weight": 32,
            },
        ]
    },
    "cyclist": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/majiajia.data/C385_V1/Pilot_cyclist_5v_CAMERA_CO_OX3GB_D100_L_all/20220621-171148/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/jianhang.he.data/C385_V2/Pilot_cyclist_5v_CO_OX3GB_D100_L_all/20220805-210426/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_underground_park_Cam_5V/20221014-155654/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_underground_park_Cam_5V/20220908-135342/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_parking_Jira_Cam_5V/20221014-214402/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_underground_park_Cam_5V/20221124-224155/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_parking_Jira_Cam_5V/20221127-195043/data",
                "sample_weight": 2,
            },
        ]
    },
    "person": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/C385_V1/Pilot_person_5v_CAMERA_CO_OX3GB_D100_L_all/20220621-003109/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/C385_v05/Pilot_person_5v_CAMERA_CO_OX3GB_D100_L_day/20220523-232136/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/C385_v05/Pilot_person_5v_CAMERA_CO_OX3GB_D100_L_night/20220523-231054/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/jianhang.he.data/C385_V2/Pilot_person_5v_CO_OX3GB_D100_L_all/20220805-205614/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_underground_park_Cam_5V/20221014-155712/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_underground_park_Cam_5V/20220908-135320/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/Pilot_person_5v_all_all_Child/20221012-170449/data",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_parking_Jira_Cam_5V/20221014-213958/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_underground_park_Cam_5V/20221124-220020/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_underground_park_Cam_5V/20221124-222923/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_parking_Jira_Cam_5V/20221127-194944/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_parking_Jira_Cam_5V/20230126-151952/data",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_parking_Cam_5V/20230126-154426/data",
                "sample_weight": 10,
            },
        ]
    },
    "vehicle": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_350703_CAMERA_CO_OX3GB_D100_L_TIME_ALL_all/20220818-202734/data",
                "sample_weight": 32,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_common_20220905_CAMERA_CO_OX3GB_D100_L_TIME_ALL_all/20220905-164533/data",
                "sample_weight": 32,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_350703_CAMERA_CO_OX3GB_D100_L_TIME_ALL_all/20220831-181342/data",
                "sample_weight": 0.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_common_20221008_CAMERA_CO_OX3GB_D100_L_TIME_ALL_all/20221008-161126/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_common_20221114_CAMERA_CO_OX3GB_D100_L_TIME_ALL_all/20221114-204456/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_wall_FP_20221125_CAMERA_CO_OX3GB_D100_L_TIME_ALL_all/20221125-085017/data",
                "sample_weight": 0.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_common_20221008_CAMERA_CO_OX3GB_D100_L_TIME_ALL_all/20230120-155231/data",
                "sample_weight": 16,
            },
        ]
    },
    "vehicle_truncation_classification": dict(),
    "semantic_parsing": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/train-1080p-id20220519-underground_29248",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_5017_20220527_anno16.2_c385",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4620_anno16.2_c385/20220617-204617/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_937_20220621_anno16.2_c385_parking",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1271_20220621_anno16.2_c385_parking",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/SP_parsing_data_to_pilot/train_rec/fishe_parsing_v0.8.0/recs/sp_train_2_pilot",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_666_anno16.2_c385_parking_20220721/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2562_anno16.2_c385_parking/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_3435_anno16.2_c385_parking_20220802/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2606_anno16.2_c385_parking_83708_20220907/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_5584_anno16.2_c385_parking_68699_20220907_2/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2611_anno16.2_c385_parking_84973/20221011-112415/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_1cam_roundview_num_268_anno16.2_c385_parking_20221120_jira/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_549_anno16.2_c385_parking/20230131-171222/data",
                "sample_weight": 4,
            },
        ]
    },
    "lane_parsing": dict(),
    "vehicle_wheel_kps": dict(),
    "vehicle_ground_line": dict(),
    "vehicle_wheel_detection": dict(),
    "rear": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_C385_5V_PARKING_all/20220524-122038/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_day/20220621-011046/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CO-OX3GB-D100-L_day/20220812-004728/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CO-OX3GB-D100-L_all/20220908-190201/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CO-OX3GB-D100-L_all/20221122-121515/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CO-OX3GB-D100-L_all/20230127-003401/data",
                "sample_weight": 1,
            },
        ]
    },
    "rear_occlusion_classification": dict(),
    "rear_part_classification": dict(),
}

datapaths = merge_datapaths(datapaths, parking_datapaths)
# default parsing only use underground parking train datasets
datapaths["semantic_parsing"] = parking_datapaths["semantic_parsing"]
