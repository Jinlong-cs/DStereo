from easydict import EasyDict

root = "dmpv2://matrix2"
datapaths = {
    "rear_plate_detection": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/data/pilot/plate/train_220319/0233/train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/data/pilot/plate/train_220319/x3c/train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/data/pilot/plate/train_220319/unknow/train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CW-A23GB-A114-L_day_2pe/20220504-102122/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CO-OX3GB-A100-L_night_2pe/20220504-103634/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CW-A23GB-A100-L_night_2pe/20220504-103906/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CO-OX3GB-A100-L_day_2pe/20220504-103937/data",
                "sample_weight": 16,
            },
        ]
    },
    "person_face_detection": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_2pe/20220720-214115/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_2pe/20220720-205236/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_2pe/20220720-174226/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_2pe/20220720-174425/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_as33-5v_all_2pe/20220720-174221/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_cc02-5v_all_2pe/20220720-174714/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_maxfa-5v_all_2pe/20220720-180427/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_2pe/20220721-202711/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_as33-5v_all_2pe/20220809-212818/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_cc02-5v_all_2pe_2022-07-20_2022-08-09/20220810-114649/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_maxfa-5v_all_2pe_2022-07-20_2022-08-09/20220810-114506/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_pilot-pdt-5v_all_2pe_2022-07-20_2022-08-09/20220810-115032/data",
                "sample_weight": 16,
            },
        ]
    },
    "plate": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/jinchuan01.xiao/data/plate/0820/v220305/4pe/train",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/jinchuan01.xiao/data/plate/unknow/v220305/4pe/train",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/jinchuan01.xiao/data/plate/wissen/v220305/4pe/train",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/jinchuan01.xiao/data/plate/x3c/v220305/4pe/train",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/jinchuan01.xiao/data/plate/plate_v211108/train/train",
                "sample_weight": 1,
            },
        ]
    },
    "face": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_4pe/20220715-201117/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_4pe/20220715-184010/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_4pe/20220715-162048/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_4pe/20220715-181100/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_4pe/20220726-155303/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_4pe/20220726-143048/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_as33-5v_all_4pe/20220726-143247/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_cc02-5v_all_4pe/20220726-144014/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_maxfa-5v_all_4pe/20220726-151851/data",
                "sample_weight": 1,
            },
        ]
    },
    "person_occlusion_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_occlusion_classification/cls_person_occlusion_v1-20190810-train/data",
                "sample_weight": 16,
            },
        ]
    },
    "vehicle_occlusion_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_subbox_vehicle_full_occlusion_v2-20200825-train_old/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_subbox_vehicle_full_occlusion_v2-20201117-vehicle_full_0323/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_subbox_vehicle_full_occlusion_v2-20200621-vehicle_full_0220/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_vehicle_full_fp_v3-train_other/data",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_vehicle_full_fp-vehicle_full_fp-data_3900/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_vehicle_full_fp_v3-20201120-train_0323/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_occ_classification/pilot3.0_0233_data_before_0406_vehicle_occ_cls/data",
                "sample_weight": 16,
            },
        ]
    },
    "cyclist_classification": {"train_data_paths": []},
    "cyclist_kps": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_kps/gluon_rec_width12/lf_new/data.train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/2pe_cyclist_kps/cyc_kps_zhengwei/20220121-132406/data",
                "sample_weight": 4,
            },
        ]
    },
    "person_head_detection": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_head_detection/white_changan_20190212_min_head_height12/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_head_detection/id1-4_head_min_height_22/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_head_detection/id5-8_head_min_height_22/data",
                "sample_weight": 1.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_head_detection/id24-27_head_min_height_22/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/2pe_person_head_detection/example_pack_densebox_ped_head_data_20220121_124814_0233_all/20220121-141027/data",
                "sample_weight": 4,
            },
        ]
    },
    "person_pose_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_pose_classification/cls_person_pose_v2_upsample-20190810-train/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/2pe_person_pose_classification/example_pack_densebox_ped_pose_data_20220121_122405_0233_all/20220121-140616/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/2pe_person_pose_classification/Pilot_person_pose_5v_CAMERA_CW_A23GB_A114_L_day/20220623-163511/data",
                "sample_weight": 10,
            },
        ]
    },
    "vehicle_light_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_light/full_1025/vehicle_light_train_1025_full",
                "sample_weight": 32,
            },
        ]
    },
    "vehicle_category": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_senyun_day/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_weissen_day/data",
                "sample_weight": 1,
            },
        ]
    },
    "person_age_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_age_classification/cls_person_age_haokai_v5_clean_Adult-20191012-train/data",
                "sample_weight": 30,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_age_classification/cls_person_age_haokai_v5_clean_Child-20191012-train/data",
                "sample_weight": 2,
            },
        ]
    },
    "person_orientation_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_orientation_classification/cls_person_orientation_v11_upsample-20191204_1-train/data",
                "sample_weight": 32,
            },
        ]
    },
    "cyclist": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/majiajia.data/CC02_V5.0/Pilot_cyclist_5v_CO-OX3GB-A100-L_day/20220611-151613/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/han.shen.data/cyc_data/20190522-remove-truck/20190522-remove-truck",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/han.shen.data/cyc_data/cyc_20190918",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/zhengwei.data/cyc_data/cyc_20200613_cn_390_person_tain_history",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/kaijun01.zhang/data/data_2020/0323_10635_quad_cyc_20200723/0323_10635_quad_cyc_20200723",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/kaijun01.zhang/data/data_2020/0323_0220_cyc_20201119/0323_0220_cyc_20201119",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/20210915_sensor0233_train_id_4218-9655_part6_day_upsample3_gt63_remove_empty_up300pix_30_heavyocc_normal/train",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/20220121-190855-x3c-all/data",
                "sample_weight": 30,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220510/cyclist/day/20220511-142409/data",
                "sample_weight": 30,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/Pilot_cyclist_5v_CAMERA_CW_A23GB_A114_L_day_5w/20220804-134412/data",
                "sample_weight": 24,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/majiajia.data/NonTarget_exp/exp5/Pilot_cyclist_5v__all/20220712-043858/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20220907-200731/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/fp/Veh_Head/Pilot_cyclist_5v_CW-A23GB-A114-L_day_fp/20220926-150141/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221011-224648/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221124-230709/data",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221127-200817/data",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221014-200602/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221127-194742/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20230126-144639/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20230126-161858/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20230126-161135/data",
                "sample_weight": 10,
            },
            # ev52 v4
            {
                "data_path": f"{root}/multicam_pilot/data/train/cyclist_detection/CP-OX3GB-D100-L_train_day_Cam_5V/20230317_195457",
                "sample_weight": 1,
            },
            # ev52 v5 jira
            {
                "data_path": f"{root}/multicam_pilot/data/train/cyclist_detection/kCycBBox2D_project_ev52_CO-OX3GB-D100-L_train_day_Cam_5V_Jira/20230501-133148",
                "sample_weight": 1,
            },
            # ev52 v5 normal
            {
                "data_path": f"{root}/multicam_pilot/data/train/cyclist_detection/kCycBBox2D_project_ev52_CO-OX3GB-D100-L_train_day_Cam_5V_Small/20230502-041036",
                "sample_weight": 1,
            },
        ]
    },
    "person": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/rare_pose_exp/Pilot_person_5v_CW-A23GB-A100-L_day/20220602-180009/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/CC02_V5/Pilot_person_5v_CO-OX3GB-A100-L_day/20220611-170110/data",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_20181210_train_12cam_update",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_20190605_train_part_2_update",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_20190918",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/zhengwei.data/ped_data/ped_20200613_cn_390_person_tain_history",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_street_people_data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20210915_sensor0233_train_id_4218-9655_part6_day_upsample3_gt63_remove_empty/train",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20211119-153000-big-person/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A100_L_day_big_person/20220620-201707/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220214-031030-0233-toll-gate/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220329-151248-x3c-tunnel/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/tunnel_ped_by_zhongpeng/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A114_L_day_5w/20220804-122458/data",
                "sample_weight": 24,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220121-192622-x3c-all/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220510/person/day/20220511-142910/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/NonTarget_exp/exp5/Pilot_person_5v__all/20220708-063813/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20220907-200310/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/fp/Veh_Head/Pilot_person_5v_CW-A23GB-A114-L_day_fp/20220926-152912/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221011-223223/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221014-201823/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/Pilot_person_5v_all_all_Child/20221015-081347/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221124-233317/data",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221127-201047/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221127-194559/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20230126-144504/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20230126-145220/data",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20230126-161618/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20230126-160449/data",
                "sample_weight": 10,
            },
            # ev52 v4
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/CP-OX3GB-D100-L_train_day_Cam_5V/20230317_195129",
                "sample_weight": 1,
            },
            # rare pose
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/project_maxfa_5v_CP-OX3GB-D100-L_train_day_Cam_5V/20230313_142610",
                "sample_weight": 1,
            },
            # rare pose
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/project_cc02_5v_train_day_Cam_5V/20230313_142957",
                "sample_weight": 3,
            },
            # rare pose
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/project_as33_5v_train_day_Cam_5V/20230313_143511",
                "sample_weight": 4,
            },
            # worker
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/pilot_unknown_driving_day_unknown_worker_unknown_data_mining_manual_label_right_rear_person_detection/20230411_192938",
                "sample_weight": 2,
            },
            # worker
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/pilot_ev52_driving_day_unknown_worker_unknown_data_mining_manual_label_left_rear_person_detection/20230411_193244",
                "sample_weight": 1,
            },
            # worker
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/pilot_ev52_driving_day_unknown_worker_unknown_data_mining_manual_label_right_rear_person_detection/20230411_193528",
                "sample_weight": 1,
            },
            # worker
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/pilot_ev52_driving_day_unknown_worker_unknown_data_mining_manual_label_right_front_person_detection/20230411_194915",
                "sample_weight": 3,
            },
            # worker
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/pilot_ev52_driving_day_unknown_worker_unknown_data_mining_manual_label_left_front_person_detection/20230411_195311",
                "sample_weight": 1,
            },
            # ev52 v5 normal
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_ev52_CO-OX3GB-D100-L_train_day_Cam_5V_Big/20230501-121210",
                "sample_weight": 1,
            },
            # ev52 v6 jira
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_train_day_Cam_5V_Jira_FN/20230613-162508",
                "sample_weight": 1,
            },
            # ev52 v6 Vehicle to Person
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_train_day_Cam_5V_Vehicle2Person/20230613-161711",
                "sample_weight": 1,
            },
            # rare pose
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_ev52_train_day_Cam_5V_Pose/20230614-123959",
                "sample_weight": 1,
            },
        ]
    },
    "vehicle": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.train.basenewv1",
                "sample_weight": 10,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.train.12camnewv1",
                "sample_weight": 10,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.train.newdatav2",
                "sample_weight": 2,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/vehicle_full_data_1080_cam3900/det_vehicle_full_v1_0-20200527_v1/train.mayv9",
                "sample_weight": 5,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/vehicle_full_data_1080/det_vehicle_full_v1_0-20200102_v1/train",
                "sample_weight": 5,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/yonggang.yang/data/vehicle_full_data_0323/det_vehicle_full_v1_0-20201118_v1/train",
                "sample_weight": 5,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/yonggang.yang/data/vehicle_full_data_1080/det_vehicle_full_v1_0-20200706_v1/train",
                "sample_weight": 5,
                "partition": "base",
            },
            # mcp全量白天全车数据 约19w张
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot3.0_0233_data_before_0714_day/train",
                "sample_weight": 20,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection_tmp/pilot3.0_0233_data_before_0406_bigtruck_new/data",
                "sample_weight": 5,
                "partition": "base",
            },
            # 白天 韦森 10w图
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_local/20210827-195610/data",
                "sample_weight": 10,
                "partition": "base",
            },
            # 白天 韦森  8.7w图
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_local_local/20210928-120519/data",
                "sample_weight": 10,
                "partition": "base",
            },
            # full_vehicle_badcase_self_camera_local
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/badcase/full_vehicle_badcase_self_camera_local/20210827-212337/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # tunnel badcase 1.5w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_badcase_local/20211019-194438/data",
                "sample_weight": 6,
                "partition": "difficult_scene",
            },
            # 韦森 day 3.3w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_trace/20211124-163642/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # X3C 3.3w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_trace/20211129-163640/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # X3C白天32246
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_x3c/20211222-124627/data",
                "sample_weight": 4,
                "partition": "base",
            },
            # 0233badcase白天9859
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_badcase/20211222-121533/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # 0233badcase白天6w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_night_x3c/20220121-114854/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # cc02 白天 21745
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_data_5v_32393_2.1w_CAMERA_CO_OX3GB_A100_L_TIME_DAY_day/20220406-125044/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # x3c 白天 45265
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_DG4W_data_5v_CAMERA_CO_OX3GB_D100_L_TIME_DAY_day/20220401-184015/data",
                "sample_weight": 4,
                "partition": "base",
            },
            # 逆光 7304
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_LIGHT07_data_5v_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220401-164602/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # x3c 白天 10763
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_29384_10763_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220426-101418/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # CW_A23GB_A114_L day 2.2W
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_data_5v_as33_25469_2.2w_CAMERA_CW_A23GB_A114_L_TIME_DAY_day/20220413-111602/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # 0233WISSEN day 2.2W
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/polit3.0_v9.0_0315_0233_day/20220316-142817/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # mono  白天夜间 大车异型 86868
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/det_vehicle_full_v1_0-20220407_v1/train",
                "sample_weight": 2,
                "partition": "special_vehicle",
            },
            # CW-A23GB-A100-L day 5.6k
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/pilot_vehicle_1v_CW-A23GB-A100-L_day/20220518-142747/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # CW-A23GB-A100-L 大车 35940
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_BIG_CAR_DAY_day/20220621-202410/data",
                "sample_weight": 4,
                "partition": "special_vehicle",
            },
            # CO_OX3GB_A100_L 大车 3219
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CO_OX3GB_A100_L_TIME_BIG_CAR_DAY_day/20220621-192421/data",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            # CW-A23GB-A100-L day 快递车数据
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_day/20220617-131419/data",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            # CW_A23GB_A114_L 常规 白天 10842
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day_normal/20220706-134742/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # CW_A23GB_A114_L 逆光 白天 1900
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day/20220621-161421/data",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # 各LTC小车挡大车
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_day_overlap_vehicle_Cam_5V/20220826-161108/data",
                "sample_weight": 0.5,
                "partition": "special_vehicle",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_day_overlap_vehicle_Cam_5V/20220826-163619/data",
                "sample_weight": 0.5,
                "partition": "special_vehicle",
            },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_pilot_pdt_5v_CW-A23GB-A100-L_train_day_overlap_vehicle_Cam_5V/20220826-163951/data",
            #     "sample_weight": 0,
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_day_overlap_vehicle_Cam_5V/20220826-163258/data",
            #     "sample_weight": 0,
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_cc02_5v_CO-OX3GB-A100-L_train_day_overlap_vehicle_Cam_5V/20220826-162631/data",
            #     "sample_weight": 0,
            # },
            # 逆光 5694
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_pilot_pdt_5v_CW-A23GB-A100-L_train_day_glare_Cam_5V/20220729-121524/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # maxfa day special car 10259
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_54665_10259_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220707-160146/data",
                "sample_weight": 2,
                "partition": "special_vehicle",
            },
            # 全量数据快递车 7w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-121816/data",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-121723/data",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-134219/data",
                "sample_weight": 4,
                "partition": "special_vehicle",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-160123/data",
                "sample_weight": 4,
                "partition": "special_vehicle",
            },
            # mono 三轮车2M
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_1v_OV10652_TIME_DAY_day/20220713-053741/data",
                "sample_weight": 7,
                "partition": "special_vehicle",
            },
            # mono 三轮车8M
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_1v_OV10652_TIME_DAY_day/20220713-061904/data",
                "sample_weight": 11,
                "partition": "special_vehicle",
            },
            # c385 6802
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20220902-200336/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # 环境误检 AS33 白天 1912
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_SCENE-FP_day/20221115-182150/data",
                "sample_weight": 4,
                "partition": "difficult_scene",
            },
            # 白天异型车 8010
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_day_special_vehicle_Cam_5V/20220926-103720/data",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            # c385 白天 11114
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221012-122031/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # c385 EV52 JIRA-badcase 948
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO-OX3GB-D100-L_JIRA-badcase_all/20230616-092700",
                "sample_weight": 5,
                "partition": "jira",
            },
            # GALAXY 环境误检 白天 1512
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221115-190526/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # AS33 jira-badcase 白天/夜晚 369
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_JIRA-badcase_all/20221110-192849/data",
                "sample_weight": 1,
                "partition": "jira",
            },
            # GALAXY 环境误检 白天 531
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221219-120427/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # maxfa 白天异型车 17352
            {
                "data_path": f"{root}/multicam_pilot/data/train/vehicle_detection/project_maxfa_5v_CP-OX3GB-D100-L_train_day_special_vehicle_Cam_5V/20230320_111109",
                "sample_weight": 2,
                "partition": "special_vehicle",
            },
            # c385 白天 7166
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO_OX3GB_D100_L_day_all/20230118-115653/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # c385 白天jirabadcase 714
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO_OX3GB_D100_L_jira_badcase_day_all/20230118-114938/data",
                "sample_weight": 2,
                "partition": "jira",
            },
            # ev52 白天 15582
            {
                "data_path": f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CP_OX3GB_D100_L_day_15582_all/20230605-125926",
                "sample_weight": 1.5,
            },
            # ev52 白天 异型车 3481
            {
                "data_path": f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CP_OX3GB_D100_L_day_3481_special_car_all/20230605-124741",
                "sample_weight": 0.4,
            },
            # c385 白天 异型车 3847
            {
                "data_path": f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO_OX3GB_D100_L_day_3847_special_car_all/20230605-133334",
                "sample_weight": 0.4,
            },
            # ev52 白天 环境误检 1075
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_EV52_SCENE-FP_day/20230608-124930",
                "sample_weight": 0.5,
            },
            # galaxy 白天 异型车 3863
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SPECIAL_VEHICLE_day/20230616-123700",
                "sample_weight": 1,
            },
        ]
    },
    "vehicle_truncation_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_truncation_classification/pilot3.0_0233_data_before_0406_vehicle_truncation_cls/data",
                "sample_weight": 32,
            },
        ]
    },
    "semantic_parsing": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/data/default_parsing/train_data/mono3.0/16.2c/data_all_0220model_2020-06-22/parsing-720p10635-train-anno16.2_59070",
                "sample_weight": 7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/data/default_parsing/train_data/mono3.0/16.2c/train-1080p-id20220324_62910",
                "sample_weight": 8,
            },
            # sensing day (special for large person)
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_2872_20210804_anno16.2",
                "sample_weight": 1,
            },
            # sensing day (rain)
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_950_20220527_anno16.2_day",
                "sample_weight": 1,
            },
            # wissen day&night (special for merge truck bottom)
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_2912_20220111_anno16.2",
                "sample_weight": 1,
            },
            # as33 day&night (snow)
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_4876_20220415_anno16.2",
                "sample_weight": 2,
            },
            # sensing day
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_15836_20220111_anno16.2_sy_day",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_4671_20220111_anno16.2_sy_day",
                "sample_weight": 2.5,
            },
            # wissen day
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_13566_20220111_anno16.2_ws_day",
                "sample_weight": 8.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_3457_20220111_anno16.2_ws_day",
                "sample_weight": 2.5,
            },
            # wissen night
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_16089_20220111_anno16.2_ws_night",
                "sample_weight": 9,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_3504_20220111_anno16.2_ws_night",
                "sample_weight": 2,
            },
            # x3c night
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4987_20220111_anno16.2_x3c_night",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4300_20220214_anno16.2_x3c_night",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4060_20220428_anno16.2_x3c_night",
                "sample_weight": 4,
            },
            # x3c day
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4500_20220111_anno16.2_x3c_day",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_10228_20220214_anno16.2_x3c_day",
                "sample_weight": 9,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1010_20220428_anno16.2_x3c_day",
                "sample_weight": 1,
            },
            # cc02 day
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2671_20220513_anno16.2_cc02_day",
                "sample_weight": 4,
            },
            # as33 day
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_1202_20220602_anno16.2_as33_day",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_1660_20220825_anno16.2_as33_day/20220825-140857/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_4247_anno16.2_as33_day/data",  # 20220923
                "sample_weight": 4,
            },
            # c385 x3c 176, ID 86907
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_176_anno16.2_c385_day_86907/20221011-011836/data",
                "sample_weight": 4,
            },
            # c385 x3c 318, ID 86913
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_318_anno16.2_c385_day_86913/20221011-012023/data",
                "sample_weight": 4,
            },
            # c385 x3c 604, ID 88317
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_604_anno16.2_c385_day_88317/20221011-010935/data",
                "sample_weight": 4,
            },
            # c385 x3c 618, ID 88356
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_618_anno16.2_c385_day_88356/20221011-010444/data",
                "sample_weight": 4,
            },
            # c385 x3c 2770, ID 90719
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2770_anno16.2_c385_day_90719/20221011-022443/data",
                "sample_weight": 4,
            },
            # c385 x3c 1357, ID 198347,94273
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1357_anno16.2_c385_day_20221123/data",
                "sample_weight": 4,
            },
            # c385 x3c 2665：ID 137848,133554,133427,144173
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2665_anno16.2_c385_day/20230126-042950/data",
                "sample_weight": 4,
            },
            # c385 jira 数据，路中人行道，高速围栏，ID 144701，144682
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_145_anno16.2_c385_day/20230126-005237/data",
                "sample_weight": 2,
            },
            # ev52 day
            {
                "data_path": f"{root}/multicam_pilot/data/train/default_segmentation/default_segmentation_CP-OX3GB-D100-L_train_day/20230318_121905",
                "sample_weight": 2,  # num 1301
            },
            # ev52 night
            {
                "data_path": f"{root}/multicam_pilot/data/train/default_segmentation/default_segmentation_CP-OX3GB-D100-L_train_night/20230318_125541",
                "sample_weight": 1.5,  # num 937
            },
        ]
    },
    "lane_parsing": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_18772_20210419_5_2classes",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_8571_20210524_5_2classes",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_24305_20210711_5_2classes",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_17311_20210711_5_2classes",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_28019_20210803_5_2classes",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_21736_20210803_5_2classes",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_71524_20210825_5_2classes",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_13711_20210825_5_2classes",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_55922_20211018_5_2classes",
                "sample_weight": 15,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_36387_20211018_5_2classes",
                "sample_weight": 15,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_160_20211129_5_2classes",
                "sample_weight": 0.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_wissen_num_78742_20211128_5_2classes",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_22101_20211129_5_2classes",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_wissen_num_4549_20211129_5_2classes",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_47972_20211220_5_2classes",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_16073_20211220_5_2classes",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_64275_20220118_5_2classes",
                "sample_weight": 30,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_27923_20220118_5_2classes",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_14440_20220214_5_2classes",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_2924_20220214_5_2classes",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_num_24851_20220328_5_2classes",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_num_5356_20220328_5_2classes",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_D100L_num_17734_20220424_5_2classes",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_CO_OX3GB_A100L_num_7589_20220505_5_2classes",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_CO_OX3GB_A100L_num_745_20220505_5_2classes",
                "sample_weight": 0.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_4481_20220525_5_2classes",
                "sample_weight": 2.3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_0233_CW_A23GB_A114L_num_5460_20220525_5_2classes",
                "sample_weight": 2.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_6271_20220525_5_2classes",
                "sample_weight": 3.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_6551_20220608_5_2classes",
                "sample_weight": 3.3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_0233_CW_A23GB_A114L_num_1995_20220621_5_2classes",
                "sample_weight": 0.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_220829/lane_parsing_train_day_badcase_as33/20220727-223306/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_221141/lane_parsing_train_day_badcase_as33/20220727-222539/data",
                "sample_weight": 0.4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_221458/lane_parsing_train_day_badcase_as33/20220727-222511/data",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_221812/lane_parsing_train_day_badcase_as33/20220727-222648/data",
                "sample_weight": 0.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_222305/lane_parsing_train_day_special_as33/20220727-223226/data",
                "sample_weight": 0.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_222623/lane_parsing_train_day_special_as33/20220727-223356/data",
                "sample_weight": 0.06,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_222930/lane_parsing_train_day_special_as33/20220727-223812/data",
                "sample_weight": 0.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_223832/lane_parsing_train_day_normal_as33/20220727-224554/data",
                "sample_weight": 0.09,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_224147/lane_parsing_train_day_normal_as33/20220727-224901/data",
                "sample_weight": 0.06,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_224502/lane_parsing_train_day_normal_as33/20220727-225403/data",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_000716/pilot_lane_parsing_annoset/20221012-010546/data",
                "sample_weight": 2.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_005334/pilot_lane_parsing_annoset/20221012-012546/data",
                "sample_weight": 2.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_012136/pilot_lane_parsing_annoset/20221012-015617/data",
                "sample_weight": 1.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_014836/pilot_lane_parsing_annoset/20221012-022211/data",
                "sample_weight": 1.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_123750/pilot_lane_parsing_annoset/20221126-132031/data",
                "sample_weight": 2.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_125221/pilot_lane_parsing_annoset/20221126-132842/data",
                "sample_weight": 2.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_130644/pilot_lane_parsing_annoset/20221126-133734/data",
                "sample_weight": 1.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20230128_224418_local/pilot_lane_parsing_annoset/20230128-224930/data",
                "sample_weight": 2.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20230128_234420_local/pilot_lane_parsing_annoset/20230128-234745/data",
                "sample_weight": 0.5,
            },
            # ev52 v4.0 day
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/lane_parsing/lane_parsing_packing_data_publish_annoset_20230320_234145/pilot_lane_parsing_annoset/20230320_234564/data",
                "sample_weight": 2,
            },
            # ev52 v4.0 night
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/lane_parsing/lane_parsing_packing_data_publish_annoset_20230320_234235/pilot_lane_parsing_annoset/20230320_234381/data",
                "sample_weight": 0.5,
            },
            # ev52 v5.0 day side
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/parsing/lane_parsing/lane_parsing_packing_data_publish_annoset_20230504_115236/pilot_lane_parsing_annoset/20230504-115730",
                "sample_weight": 2.0,
            },
            # ev52 v5.0 day rear
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/parsing/lane_parsing/lane_parsing_packing_data_publish_annoset_20230504_120321/pilot_lane_parsing_annoset/20230504-120426",
                "sample_weight": 2.0,
            },
            # ev52 v5.0 night side
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/parsing/lane_parsing/lane_parsing_packing_data_publish_annoset_20230504_120658/pilot_lane_parsing_annoset/20230504-121007",
                "sample_weight": 0.5,
            },
            # ev52 v5.0 night rear
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/parsing/lane_parsing/lane_parsing_packing_data_publish_annoset_20230504_121434/pilot_lane_parsing_annoset/20230504-121503",
                "sample_weight": 0.5,
            },
        ]
    },
    "vehicle_wheel_kps": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_kps/vehicle_2_kps/vehicle_2_kps_all_20201021.train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_kps_detection/release_rec/vehicle_2_kps_add0323cutin_20201125",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_0233/20220119-212808/data",
                "sample_weight": 50,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_X3C/20220119-122143/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_X3C/20220224-131626/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220315-163131/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220425-161851/data",
                "sample_weight": 1,
            },
        ]
    },
    "vehicle_ground_line": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_0233/20220119-221935/data",
                "sample_weight": 50,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_X3C/20220119-125418/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_X3C/20220224-133404/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220315-163208/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220425-161825/data",
                "sample_weight": 1,
            },
        ]
    },
    "vehicle_flank": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_Cam_5V/20220706-213158/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_ar0233-RGGB_train_Cam_5V/20220706-205154/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_ar0233-WISSEN_train_Cam_5V/20220706-165341/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_ar0233-WISSEN_train_day_big_vehicle_Cam_5V/20220706-123556/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_x3c_train_day_Cam_5V/20220706-112001/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_project_cc02_5v_CO-OX3GB-A100-L_train_day_big_vehicle_Cam_5V/20220706-102652/data",
                "sample_weight": 1,
            },
        ]
    },
    "vehicle_wheel_detection": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_wheel_detection/det_subbox_vehicle_wheel_belongto_clean-veh_wheel-20210407_train_clean_323_all_trans_pb_rec/train",
                "sample_weight": 171,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_wheel_detection/MCP_wheel_detection_day_0927/train",
                "sample_weight": 85,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehWheelBBox2D/kVehWheelBBox2D_train_day_big_vehicle_Cam_5V_Cam_5V/20220617-193224/data",
                "sample_weight": 2,
            },
        ]
    },
    "rear": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v3-20190726-train_base/data",
                "sample_weight": 8,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v3-20190726-train_gray/data",
                "sample_weight": 6,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v3-20190726-train_night/data",
                "sample_weight": 3,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v6-20200624-train_390_5200_day_FrontRear/data",
                "sample_weight": 2,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v6-20200624-train_390_5200_day_NonFrontRear/data",
                "sample_weight": 3,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v6-20200624-train_390_5200_night/data",
                "sample_weight": 2,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/ni.jiang/data/vehicle_rear_data/det_vehicle_rear_v8-20201119-train_part_4_0323_0220_exclude_20200601_exclude_part33/data",
                "sample_weight": 12,
                "partition": "base",
            },
            # 维森 白天 330660
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_0233/20220119-012241/data",
                "sample_weight": 50,
                "partition": "base",
            },
            # X3C 白天 12906
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_X3C/20220224-113552/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # X3C 白天 15688
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_day/20220315-163248/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # X3C 白天 13505
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_day/20220401-140046/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # AS33 day 24985
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_rear_5v_AS33_day/20220412-220556/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # PDT 大车 day 3205
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_rear_5v_CW_A23GB_A100_L_big_car_day/20220412-212012/data",
                "sample_weight": 1,
                "partition": "badcase",
            },
            # PROJECT_CC02_5V 白天 4910
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_day/20220407-195106/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_CC02_5V 白天 8526 annoset(16130)
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_day/20220510-214104/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_AS33_5V 白天 2113
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW-A23GB-A114-L_day/20220510-103100/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_c385_5V 白天 4879
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CO-OX3GB-D100-L_day/20221127-140301/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # ev52 白天 8448
            {
                "data_path": f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_CP_OX3GB_D100_L_day_8448_all/20230609-172935",
                "sample_weight": 1,
                "partition": "base",
            },
        ]
    },
    "rear_occlusion_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_subbox_rear_occlusion_v8-20190719-train_base_night_gray-cn93-cn750-model-score0.18/data",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_subbox_rear_occlusion_v8-20200903-train_part_4_0323_ispDefault_2280x1080_exclude_part33-cn750-cn759/data",
                "sample_weight": 18,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_rear_neg_v4-20200901-fp_train/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_0233/20220119-041133/data",
                "sample_weight": 50,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_X3C/20220119-120507/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_X3C/20220224-123226/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_CAMERA_X3C_day/20220316-102102/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_CAMERA_X3C_day/20220401-160904/data",
                "sample_weight": 2,
            },
        ]
    },
    "rear_part_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-old/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-hard/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-res/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-0323/data",
                "sample_weight": 9,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_0233/20220119-210054/data",
                "sample_weight": 50,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_X3C/20220119-120509/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_X3C/20220224-130525/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_CAMERA_X3C_day/20220315-163229/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear_part_classification/pilot_rear_part_5v_CAMERA_CW_A23GB_A114_L_day_bigcar/20220822-093413/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear_part_classification/pilot_rear_part_5v_CO-OX3GB-D100-L_day/20221127-145644/data",
                "sample_weight": 1,
            },
            # ev52 白天 4245
            {
                "data_path": f"{root}/multicam_pilot/data/train/vehicle_rear_part_classification/pilot_rear_part_5v_CAMERA_CP_OX3GB_D100_L_day_4254_all/20230605-141629",
                "sample_weight": 0.4,
            },
        ]
    },
}
datapaths = EasyDict(datapaths)
