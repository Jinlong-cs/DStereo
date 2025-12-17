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
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/2pe_person_pose_classification/Pilot_person_pose_5v_CAMERA_CW_A23GB_A114_L_night/20220706-113735/data",
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
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_senyun_night/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_weissen_night/data",
                "sample_weight": 3,
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
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/majiajia.data/CC02_V5/Pilot_cyclist_5v_CO-OX3GB-A100-L_night/20220611-163932/data",
                "sample_weight": 40,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/majiajia.data/MaxFaV10.0/Cyclist_lighted_night/Pilot_cyclist_5v_MaxFaV10_Cyclist_Lighted_Night_night/20220425-182351/data",
                "sample_weight": 40,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/majiajia.data/cyc_light_CW_A23GB_A100_L/pilot_cyclist_packing_data_publish/20220330-211259/data",
                "sample_weight": 40,
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
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220510/cyclist/night/20220511-124542/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/Pilot_cyclist_5v_CAMERA_CW_A23GB_A114_L_night/20220624-203955/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/majiajia.data/NonTarget_exp/exp6/Pilot_cyclist_5v_all_night/20220713-225739/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/glare/lighted_cyclist/debug/Pilot_cyclist_5v_all_night_glare/20220923-222554/data",
                "sample_weight": 60,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/glare/Pilot_cyclist_5v_CAMERA_CW_A23GB_A100_L_Night_night/20220929-145754/data",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20221128-104916/data",
                "sample_weight": 3,
            },
            # AS33 vehicle glare annoset(89252) 4274
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_AS33_glare_fp_train_night/20230131-114216/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Jira_Cam_5V/20230126-152414/data",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20230126-163331/data",
                "sample_weight": 5,
            },
            # ev52 v4
            {
                "data_path": f"{root}/multicam_pilot/data/train/cyclist_detection/CP-OX3GB-D100-L_train_night_Cam_5V/20230317_195347",
                "sample_weight": 1,
            },
            # ev52 v6 close cyclist < 2m
            {
                "data_path": f"{root}/multicam_pilot/data/train/cyclist_detection/kCycBBox2D_project_pilot3_train_night_Cam_5V_Close/20230613-201258",
                "sample_weight": 1,
            },
            # ev52 v6 close cyclist < 10m
            {
                "data_path": f"{root}/multicam_pilot/data/train/cyclist_detection/kCycBBox2D_project_pilot3_train_night_Cam_5V_Close/20230615-123408",
                "sample_weight": 5,
            },
        ]
    },
    "person": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/CC02_V5/Pilot_person_5v_CO-OX3GB-A100-L_night/20220611-164225/data",
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
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_20180814_train_gray_update",
                "sample_weight": 3,
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
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/kaijun01.zhang/data/data_2020/0323_0220_ped_20201119/0323_0220_ped_20201119",
                "sample_weight": 3,
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
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220121-192622-x3c-all/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/tunnel_ped_by_zhongpeng/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220329-151248-x3c-tunnel/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220510/person/night/20220511-124631/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A114_L_day_5w/20220826-121734/data",
                "sample_weight": 24,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A100_L_night_big_person/20220620-201624/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/NonTarget_exp/exp6/Pilot_person_5v_all_night/20220713-233017/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20221128-104636/data",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Jira_Cam_5V/20230126-152323/data",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20230126-162729/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_AS33_glare_fp_train_night/20230130-121601/data",
                "sample_weight": 4,
            },
            # ev52 v4
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/CP-OX3GB-D100-L_train_night_Cam_5V/20230317_195254",
                "sample_weight": 1,
            },
            # ev52 v5 normal
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_ev52_CO-OX3GB-D100-L_train_night_Cam_5V_Big/20230501-123308",
                "sample_weight": 1,
            },
            # ev52 v6 pillar fp
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_ev52_train_night_Cam_5V_Pillar/20230614-121204",
                "sample_weight": 3,
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
            },
            # tunnel badcase 1.5w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_badcase_local/20211019-194438/data",
                "sample_weight": 6,
            },
            # 韦森 day 3.3w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_trace/20211124-163642/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # 0233 夜晚 全量数据 222100
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_night_0233/20220119-002848/data",
                "sample_weight": 60,
                "partition": "base",
            },
            # X3C 3.3w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_trace/20211129-163640/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # X3C夜间19185
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_night_x3c/20211222-132550/data",
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
            # cc02 夜晚 0.68w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_data_5v_32394_0.68w_CAMERA_CO_OX3GB_A100_L_TIME_NIGHT_night/20220406-124214/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # x3c 白天 45265
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_DG4W_data_5v_CAMERA_CO_OX3GB_D100_L_TIME_DAY_day/20220401-184015/data",
                "sample_weight": 4,
                "partition": "base",
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
            # CW-A23GB-A100-L dark night 31121
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kVehBBox2D/kVehBBox2D_CW-A23GB-A100-L_train_night_dark_night/20220610-160750/data",
                "sample_weight": 4,
                "partition": "difficult_scene",
            },
            # CP-OX3GB-D100-L dark night 2763
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_night_dark_night/20220610-144900/data",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # CW-A23GB-A100-L dark night 2243
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_dark_night/20220610-144130/data",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # CW-A23GB-A100-L night 标识牌
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_day/20220620-181555/data",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # CCO2 night 6661
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_33333_6661_CAMERA_CO_OX3GB_A100-L_TIME_NIGHT_night/20220527-145352/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # CCO2 night 9964
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_32658_9964_CAMERA_CO_OX3GB_A100-L_TIME_NIGHT_night/20220527-145938/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # CCO2 night 12284
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_34231_12284_CAMERA_CO_OX3GB_A100-L_TIME_NIGHT_night/20220527-150019/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # CW-A23GB-A100-L 大车 白天 35940
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_BIG_CAR_DAY_day/20220621-202410/data",
                "sample_weight": 4,
                "partition": "special_vehicle",
            },
            # CO_OX3GB_A100_L 大车 白天 3219
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CO_OX3GB_A100_L_TIME_BIG_CAR_DAY_day/20220621-192421/data",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            # CW-A23GB-A100-L 大车 夜晚 11187
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_BIG_CAR_night/20220629-121819/data",
                "sample_weight": 16,
                "partition": "special_vehicle",
            },
            # CW-A23GB-A100-L 大车 白天 8902
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day_bigcar/20220704-185847/data",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_night_badcase_dark_night_Cam_5V/20220709-175425/data",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_badcase_dark_night_Cam_5V/20220709-173429/data",
                "sample_weight": 3,
                "partition": "difficult_scene",
            },
            # CO-OX3GB-A100-L 大车 夜晚 2416
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO-OX3GB-A100-L_TIME_BIG_CAR_night/20220725-143019/data",
                "sample_weight": 2,
                "partition": "special_vehicle",
            },
            # CW_A23GB_A114_L 常规 白天 10842
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day_normal/20220706-134742/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # CW_A23GB_A114_L 低照 夜间 2243
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_night_dark_0736/20220726-165515/data",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # CW-A23GB-A100-L 大车 夜晚 3751
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_night_bigcar_0715/20220715-134225/data",
                "sample_weight": 1,
                "partition": "special_vehicle",
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
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_night_overlap_vehicle_Cam_5V/20220721-123722/data",
            #     "sample_weight": 0,
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_overlap_vehicle_Cam_5V/20220721-124308/data",
            #     "sample_weight": 0,
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_cc02_5v_CO-OX3GB-A100-L_train_night_overlap_vehicle_Cam_5V/20220721-124751/data",
            #     "sample_weight": 0,
            # },
            # maxfa day special car 10259
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_54665_10259_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220707-160146/data",
                "sample_weight": 2,
                "partition": "special_vehicle",
            },
            # as33 night special car 7858
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_73388_73389_73390_7858_CAMERA_CW_A23GB_A114_L_night/20220726-181251/data",
                "sample_weight": 2,
                "partition": "special_vehicle",
            },
            # AS33 night glare 9091
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_64258_64255_65416_9091_CAMERA_CW_A23GB_A114_L_TIME_NIGHT_night/20220707-143748/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
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
            # CW-A23GB-A100-L 标识牌 夜晚 2248
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DARK_NIGHT_night/20220731-090659/data",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # 环境误检 AS33 白天 1912
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_SCENE-FP_day/20221115-182150/data",
                "sample_weight": 4,
                "partition": "difficult_scene",
            },
            # 环境误检 AS33 夜晚 2973
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_SCENE-FP_night/20221115-182152/data",
                "sample_weight": 4,
                "partition": "difficult_scene",
            },
            # c385 夜晚 6881
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20221012-114409/data",
                "sample_weight": 1,
                "partition": "base",
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
            # AS33夜晚眩光jiracase 528
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_glare_Cam_5V/20221009-193824/data",
                "sample_weight": 1,
                "partition": "jira",
            },
            # GALAXY 环境误检 白天 1512
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221115-190526/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # GALAXY 环境误检 夜晚 682
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_night/20221115-190527/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # AS33 jira-badcase 白天/夜晚 369
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_JIRA-badcase_all/20221110-192849/data",
                "sample_weight": 1,
                "partition": "jira",
            },
            # c385 夜晚眩光 373
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_glare_Cam_5V/20221103-204258/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # as33 夜晚眩光 夜晚异型车 共871
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_special_vehicle_Cam_5V/20221103-204024/data",
                "sample_weight": 2,
                "partition": "special_vehicle",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_glare_Cam_5V/20221103-205303/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # as33 眩光 4528
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_glare_Cam_5V/20221114-194733/data",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # as33 眩光trigger-score 2272
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_night-glareV4_all/20221205-170615/data",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # GALAXY 环境误检 白天 531
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221219-120427/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # GALAXY 环境误检 夜晚 692
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_night/20221219-120553/data",
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
            # galaxy 环境误检 夜晚 1201
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_night/20230221-104721",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # ev52 白天 15582
            {
                "data_path": f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CP_OX3GB_D100_L_day_15582_all/20230605-125926",
                "sample_weight": 1.5,
            },
            # ev52 夜晚 10131
            {
                "data_path": f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CP_OX3GB_D100_L_night_10131_all/20230605-125403",
                "sample_weight": 0.5,
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
            # ev52 夜晚 异型车 1286
            {
                "data_path": f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CP_OX3GB_D100_L_night_1286_special_car_all/20230605-124352",
                "sample_weight": 0.5,
            },
            # c385 夜晚 异型车 1260
            {
                "data_path": f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO_OX3GB_D100_L_night_1260_special_car_all/20230605-132847",
                "sample_weight": 0.5,
            },
            # 夜晚 眩光 jira 180
            {
                "data_path": f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO_OX3GB_D100_L_night_180_jira_light_all/20230605-132905",
                "sample_weight": 0.05,
            },
            # ev52 夜晚 眩光 1699
            {
                "data_path": f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CP_OX3GB_D100_L_night_1699_light_all/20230605-124442",
                "sample_weight": 0.5,
            },
            # ev52 夜晚 环境误检 1376
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_EV52_SCENE-FP_night/20230608-125331",
                "sample_weight": 0.5,
            },
            # ev52 夜晚 近处大车 1029
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_EV52_BIGCAR_NEARBY_night/20230426-110226",
                "sample_weight": 0.5,
            },
            # AS33 夜晚眩光 8159
            {
                "data_path": f"{root}/multicam_pilot/data/train/vehicle_detection/project_as33_5v_CW-A23GB-A114-L_train_night_glare_Cam_5V/20230317_202437",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # galaxy 白天 异型车 3863
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SPECIAL_VEHICLE_day/20230616-123700",
                "sample_weight": 1,
            },
            # galaxy 夜晚 异型车 1842
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SPECIAL_VEHICLE_night/20230616-123154",
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
                "sample_weight": 13,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_3504_20220111_anno16.2_ws_night",
                "sample_weight": 3,
            },
            # x3c night
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4987_20220111_anno16.2_x3c_night",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4300_20220214_anno16.2_x3c_night",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4060_20220428_anno16.2_x3c_night",
                "sample_weight": 6,
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
            # c385 x3c 1358, ID 91711
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1358_anno16.2_c385_night_91711/20221011-011708/data",
                "sample_weight": 4,
            },
            # c385 x3c 543, ID 133425,133419,144172
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_543_anno16.2_c385_night/20230119-194419/data",
                "sample_weight": 2,
            },
            # c385 jira 数据，暗光夜晚围栏，ID 144713，144712
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_488_anno16.2_c385_night/20230119-203402/data",
                "sample_weight": 4,
            },
            # ev52 day
            {
                "data_path": f"{root}/multicam_pilot/data/train/default_segmentation/default_segmentation_CP-OX3GB-D100-L_train_day/20230318_121905",
                "sample_weight": 2,  # num 1301
            },
            # ev52 night
            {
                "data_path": f"{root}/multicam_pilot/data/train/default_segmentation/default_segmentation_CP-OX3GB-D100-L_train_night/20230318_125541",
                "sample_weight": 3,  # num 937
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
                "sample_weight": 0.4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_num_24851_20220328_5_2classes",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_num_5356_20220328_5_2classes",
                "sample_weight": 0.4,
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
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_4481_20220525_5_2classes",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_0233_CW_A23GB_A114L_num_5460_20220525_5_2classes",
                "sample_weight": 5.0,
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
                "sample_weight": 2.0,
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
                "sample_weight": 1.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_005334/pilot_lane_parsing_annoset/20221012-012546/data",
                "sample_weight": 1.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_012136/pilot_lane_parsing_annoset/20221012-015617/data",
                "sample_weight": 2.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_014836/pilot_lane_parsing_annoset/20221012-022211/data",
                "sample_weight": 2.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_123750/pilot_lane_parsing_annoset/20221126-132031/data",
                "sample_weight": 1.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_125221/pilot_lane_parsing_annoset/20221126-132842/data",
                "sample_weight": 1.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_130644/pilot_lane_parsing_annoset/20221126-133734/data",
                "sample_weight": 2.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20230128_224418_local/pilot_lane_parsing_annoset/20230128-224930/data",
                "sample_weight": 1.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20230128_234420_local/pilot_lane_parsing_annoset/20230128-234745/data",
                "sample_weight": 2.0,
            },
            # ev52 v4.0 day
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/lane_parsing/lane_parsing_packing_data_publish_annoset_20230320_234145/pilot_lane_parsing_annoset/20230320_234564/data",
                "sample_weight": 1.0,
            },
            # ev52 v4.0 night
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/lane_parsing/lane_parsing_packing_data_publish_annoset_20230320_234235/pilot_lane_parsing_annoset/20230320_234381/data",
                "sample_weight": 2.0,
            },
            # ev52 v5.0 day side
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/parsing/lane_parsing/lane_parsing_packing_data_publish_annoset_20230504_115236/pilot_lane_parsing_annoset/20230504-115730",
                "sample_weight": 1.0,
            },
            # ev52 v5.0 day rear
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/parsing/lane_parsing/lane_parsing_packing_data_publish_annoset_20230504_120321/pilot_lane_parsing_annoset/20230504-120426",
                "sample_weight": 1.0,
            },
            # ev52 v5.0 night side
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/parsing/lane_parsing/lane_parsing_packing_data_publish_annoset_20230504_120658/pilot_lane_parsing_annoset/20230504-121007",
                "sample_weight": 1.0,
            },
            # ev52 v5.0 night rear
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/parsing/lane_parsing/lane_parsing_packing_data_publish_annoset_20230504_121434/pilot_lane_parsing_annoset/20230504-121503",
                "sample_weight": 1.0,
            },
            # ev52 v6.0 side
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/lane_parsing/lane_parsing_packing_data_publish_annoset_20230612_163320_local/pilot_lane_parsing_annoset/20230612-164121",
                "sample_weight": 1.0,
            },
            # ev52 v6.0 rear
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/lane_parsing/lane_parsing_packing_data_publish_annoset_20230612_165737_local/pilot_lane_parsing_annoset/20230612-165914",
                "sample_weight": 1.0,
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
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_night_0233/20220120-114207/data",
                "sample_weight": 40,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_X3C/20220119-122143/data",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_night_X3C/20220119-123010/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_X3C/20220224-131626/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220315-163131/data",
                "sample_weight": 3,
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
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_night_0233/20220120-115549/data",
                "sample_weight": 40,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_X3C/20220119-125418/data",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_night_X3C/20220119-141510/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_X3C/20220224-133404/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220315-163208/data",
                "sample_weight": 3,
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
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_x3c_train_night_Cam_5V/20220706-163624/data",
                "sample_weight": 6,
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
            # 0233 白天 330660
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_0233/20220119-012241/data",
                "sample_weight": 10,
                "partition": "base",
            },
            # 0233 夜晚 81783
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_night_0233/20220118-233709/data",
                "sample_weight": 40,
                "partition": "base",
            },
            # X3C 白天 84312
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_X3C/20220119-111048/data",
                "sample_weight": 5,
                "partition": "base",
            },
            # X3C 夜晚 24160
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_night_X3C/20220119-105348/data",
                "sample_weight": 8,
                "partition": "base",
            },
            # X3C 白天 12906
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_X3C/20220224-113552/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # X3C 夜晚 22000
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_night_X3C/20220224-115939/data",
                "sample_weight": 8,
                "partition": "base",
            },
            # X3C 白天 15688
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_day/20220315-163248/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # X3C 夜晚 7612
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_night/20220315-164230/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # X3C 白天 13505
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_day/20220401-140046/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # PROJECT_CC02_5V 白天 4910
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_day/20220407-195106/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_CC02_5V 夜晚 4758
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_night/20220407-195117/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # CAMERA_CW_A23GB_A100_L 夜晚 9739 annoset(12296)
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_CW_A23GB_A100_L_night/20220425-140822/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_AS33_5V 夜晚 8872
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_AS33_night/20220412-213815/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_PDT 夜晚 大车 1306
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW_A23GB_A100_L_big_car_night/20220412-212057/data",
                "sample_weight": 1,
                "partition": "badcase",
            },
            # PROJECT_CC02_5V 白天 8526 annoset(16130)
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_day/20220510-214104/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_CC02_5V 夜晚 2159 annoset(16131)
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_night/20220510-213158/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_AS33_5V 白天 2113
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW-A23GB-A114-L_day/20220510-103100/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_AS33_5V 夜晚 1014
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW-A23GB-A114-L_night/20220510-115328/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # CAMERA_X3C_5V 夜晚 1.9K
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_rear_5v_CAMERA_X3C_night/20220621-162154/data",
                "sample_weight": 1,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear/det_vehicle_rear_v10-20211229-10652-split_special_and_BigTrucks/data",
                "sample_weight": 10,
                "partition": "badcase",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear/det_vehicle_rear_v10-20220129-jira_and_close_vehicle_data/data",
                "sample_weight": 10,
                "partition": "badcase",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear/det_vehicle_rear_v8-0220_0323_10635_bigcar-20210917/data",
                "sample_weight": 10,
                "partition": "badcase",
            },
            # CW_A23GB_A114_L 夜晚 眩光 2015
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CAMERA_CW_A23GB_A114_L_night_glare_0831/20220831-181141/data",
                "sample_weight": 1,
                "partition": "badcase",
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
            # ev52 夜晚 4384
            {
                "data_path": f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_CP_OX3GB_D100_L_night_4384_all/20230609-172503",
                "sample_weight": 0.5,
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
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_night_0233/20220119-020318/data",
                "sample_weight": 40,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_X3C/20220119-120507/data",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_night_X3C/20220119-113231/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_X3C/20220224-123226/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_night_X3C/20220224-125801/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_CAMERA_X3C_day/20220316-102102/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_CAMERA_X3C_night/20220315-162936/data",
                "sample_weight": 1,
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
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_night_0233/20220119-203134/data",
                "sample_weight": 40,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_X3C/20220119-120509/data",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_night_X3C/20220119-115855/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_X3C/20220224-130525/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_CAMERA_X3C_day/20220315-163229/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_CAMERA_X3C_night/20220315-163054/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_CAMERA_X3C_night/20220425-142431/data",
                "sample_weight": 1,
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
            # ev52 夜晚 1936
            {
                "data_path": f"{root}/multicam_pilot/data/train/vehicle_rear_part_classification/pilot_rear_part_5v_CAMERA_CP_OX3GB_D100_L_night_1936_all/20230605-141013",
                "sample_weight": 0.1,
            },
        ]
    },
}
datapaths = EasyDict(datapaths)
