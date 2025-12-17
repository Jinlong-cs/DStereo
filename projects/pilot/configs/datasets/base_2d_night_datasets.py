import os

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
root = os.path.join(bucket_root, "matrix2")


# ----- Required -----

datapaths = dict(
    rear_plate_detection=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/data/pilot/plate/train_220319/0233/train.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/data/pilot/plate/train_220319/0233/train.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/data/pilot/plate/train_220319/x3c/train.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/data/pilot/plate/train_220319/x3c/train.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/data/pilot/plate/train_220319/unknow/train.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/data/pilot/plate/train_220319/unknow/train.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CW-A23GB-A114-L_day_2pe/20220504-102122/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CW-A23GB-A114-L_day_2pe/20220504-102122/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CO-OX3GB-A100-L_night_2pe/20220504-103634/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CO-OX3GB-A100-L_night_2pe/20220504-103634/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CW-A23GB-A100-L_night_2pe/20220504-103906/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CW-A23GB-A100-L_night_2pe/20220504-103906/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CO-OX3GB-A100-L_day_2pe/20220504-103937/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CO-OX3GB-A100-L_day_2pe/20220504-103937/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    person_face_detection=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                # 11132
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_2pe/20220720-214115/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_2pe/20220720-214115/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                # 4499
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_2pe/20220720-205236/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_2pe/20220720-205236/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                # 3914
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_2pe/20220720-174226/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_2pe/20220720-174226/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                # 4637
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_2pe/20220720-174425/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_2pe/20220720-174425/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                # 148
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_as33-5v_all_2pe/20220720-174221/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_as33-5v_all_2pe/20220720-174221/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                # 1185
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_cc02-5v_all_2pe/20220720-174714/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_cc02-5v_all_2pe/20220720-174714/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                # 1688
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_maxfa-5v_all_2pe/20220720-180427/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_maxfa-5v_all_2pe/20220720-180427/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                # 2648
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_2pe/20220721-202711/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_2pe/20220721-202711/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                # 8843
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_as33-5v_all_2pe/20220809-212818/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_as33-5v_all_2pe/20220809-212818/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                # 6347
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_cc02-5v_all_2pe_2022-07-20_2022-08-09/20220810-114649/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_cc02-5v_all_2pe_2022-07-20_2022-08-09/20220810-114649/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                # 279
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_maxfa-5v_all_2pe_2022-07-20_2022-08-09/20220810-114506/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_maxfa-5v_all_2pe_2022-07-20_2022-08-09/20220810-114506/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                # 3469
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_pilot-pdt-5v_all_2pe_2022-07-20_2022-08-09/20220810-115032/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_pilot-pdt-5v_all_2pe_2022-07-20_2022-08-09/20220810-115032/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    person_occlusion_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/person_occlusion_classification/cls_person_occlusion_v1-20190810-train/data.rec",  # noqa
                anno_path=f"{root}/data/person_occlusion_classification/cls_person_occlusion_v1-20190810-train/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{root}/data/person_occlusion_classification/person_occlusion_v1-20190628-val/data.rec",  # noqa
                anno_path=f"{root}/data/person_occlusion_classification/person_occlusion_v1-20190628-val/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
        ],
    ),
    vehicle_occlusion_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/vehicle_full_cls_data/cls_subbox_vehicle_full_occlusion_v2-20200825-train_old/data.rec",  # noqa
                anno_path=f"{root}/data/vehicle_full_cls_data/cls_subbox_vehicle_full_occlusion_v2-20200825-train_old/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_full_cls_data/cls_subbox_vehicle_full_occlusion_v2-20201117-vehicle_full_0323/data.rec",  # noqa
                anno_path=f"{root}/data/vehicle_full_cls_data/cls_subbox_vehicle_full_occlusion_v2-20201117-vehicle_full_0323/data.anno.pb_rec",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_full_cls_data/cls_subbox_vehicle_full_occlusion_v2-20200621-vehicle_full_0220/data.rec",  # noqa
                anno_path=f"{root}/data/vehicle_full_cls_data/cls_subbox_vehicle_full_occlusion_v2-20200621-vehicle_full_0220/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_full_cls_data/cls_vehicle_full_fp_v3-train_other/data.rec",  # noqa
                anno_path=f"{root}/data/vehicle_full_cls_data/cls_vehicle_full_fp_v3-train_other/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_full_cls_data/cls_vehicle_full_fp-vehicle_full_fp-data_3900/data.rec",  # noqa
                anno_path=f"{root}/data/vehicle_full_cls_data/cls_vehicle_full_fp-vehicle_full_fp-data_3900/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_full_cls_data/cls_vehicle_full_fp_v3-20201120-train_0323/data.rec",  # noqa
                anno_path=f"{root}/data/vehicle_full_cls_data/cls_vehicle_full_fp_v3-20201120-train_0323/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_occ_classification/pilot3.0_0233_data_before_0406_vehicle_occ_cls/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_occ_classification/pilot3.0_0233_data_before_0406_vehicle_occ_cls/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    cyclist_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    cyclist_kps=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            # dict(
            #     rec_path=f'{root}/data/cyclist_kps/gluon_rec/20190802_new/data.train.pb_rec',  # noqa
            #     anno_path=f'{root}/data/cyclist_kps/gluon_rec/20190802_new/label.train.pb_rec',  # noqa
            #     sample_weight=16
            # ),
            #  dict(
            #     rec_path=f'{root}/data/cyclist_kps/gluon_rec/cyclist_2_kps_390_20201207/data.train.pb_rec',  # noqa
            #     anno_path=f'{root}/data/cyclist_kps/gluon_rec/cyclist_2_kps_390_20201207/label.train.pb_rec',  # noqa
            #     sample_weight=16
            # ),
            # min_width = 12
            # dict(
            #     rec_path=f'{root}/data/cyclist_kps/gluon_rec_width12/old/data_old/data.train.pb_rec',  # noqa
            #     anno_path=f'{root}/data/cyclist_kps/gluon_rec_width12/old/data_old/data.train.anno.pb_rec',  # noqa
            #     sample_weight=16
            # ),
            dict(
                rec_path=f"{root}/data/cyclist_kps/gluon_rec_width12/lf_new/data.train.pb_rec",  # noqa
                anno_path=f"{root}/data/cyclist_kps/gluon_rec_width12/lf_new/data.train.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/2pe_cyclist_kps/cyc_kps_zhengwei/20220121-132406/data.pb_rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/2pe_cyclist_kps/cyc_kps_zhengwei/20220121-132406/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    person_head_detection=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/person_head_detection/white_changan_20190212_min_head_height12/data.rec",  # noqa
                anno_path=f"{root}/data/person_head_detection/white_changan_20190212_min_head_height12/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/data/person_head_detection/id1-4_head_min_height_22/data.rec",  # noqa
                anno_path=f"{root}/data/person_head_detection/id1-4_head_min_height_22/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/data/person_head_detection/id5-8_head_min_height_22/data.rec",  # noqa
                anno_path=f"{root}/data/person_head_detection/id5-8_head_min_height_22/data.anno.pb_rec",  # noqa
                sample_weight=1.5,
            ),
            dict(
                rec_path=f"{root}/data/person_head_detection/id24-27_head_min_height_22/data.rec",  # noqa
                anno_path=f"{root}/data/person_head_detection/id24-27_head_min_height_22/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/2pe_person_head_detection/example_pack_densebox_ped_head_data_20220121_124814_0233_all/20220121-141027/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/2pe_person_head_detection/example_pack_densebox_ped_head_data_20220121_124814_0233_all/20220121-141027/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    person_pose_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/person_pose_classification/cls_person_pose_v2_upsample-20190810-train/data.rec",  # noqa
                anno_path=f"{root}/data/person_pose_classification/cls_person_pose_v2_upsample-20190810-train/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/2pe_person_pose_classification/example_pack_densebox_ped_pose_data_20220121_122405_0233_all/20220121-140616/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/2pe_person_pose_classification/example_pack_densebox_ped_pose_data_20220121_122405_0233_all/20220121-140616/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # as33 night
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/2pe_person_pose_classification/Pilot_person_pose_5v_CAMERA_CW_A23GB_A114_L_night/20220706-113735/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/2pe_person_pose_classification/Pilot_person_pose_5v_CAMERA_CW_A23GB_A114_L_night/20220706-113735/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{root}/data/person_pose_classification/person_pose_v1-20190628-val/data.rec",  # noqa
                anno_path=f"{root}/data/person_pose_classification/person_pose_v1-20190628-val/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
        ],
    ),
    vehicle_light_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/vehicle_light/full_1025/vehicle_light_train_1025_full.rec",  # noqa
                anno_path=f"{root}/data/vehicle_light/full_1025/vehicle_light_train_1025_full.anno.pb_rec",  # noqa
                sample_weight=32,
            ),
        ],
        val_batch_size_per_ctx=10,
        val_data_paths=None,
    ),
    vehicle_category=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_senyun_night/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_senyun_night/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_weissen_night/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_weissen_night/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
        ],
        val_batch_size_per_ctx=16,
        val_data_paths=[
            dict(
                rec_path=f"{root}/data/vehicle_category/vehicle_category_7cls_haokai_v6_cut_in_new_rular_add_fisheyedata-vehicle_category-val/data.rec",  # noqa
                anno_path=f"{root}/data/vehicle_category/vehicle_category_7cls_haokai_v6_cut_in_new_rular_add_fisheyedata-vehicle_category-val/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
        ],
    ),
    person_age_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/person_age_classification/cls_person_age_haokai_v5_clean_Adult-20191012-train/data.rec",  # noqa
                rec_path_md5sum="9135089844ae3c3f78789d0503eb7504",
                anno_path=f"{root}/data/person_age_classification/cls_person_age_haokai_v5_clean_Adult-20191012-train/data.anno.pb_rec",  # noqa
                anno_path_md5sum="c368a38a133a11d4aa9f09311a49703b",
                sample_weight=30,
            ),
            dict(
                rec_path=f"{root}/data/person_age_classification/cls_person_age_haokai_v5_clean_Child-20191012-train/data.rec",  # noqa
                rec_path_md5sum="cb30528844206a4eed3c84586fc67573",
                anno_path=f"{root}/data/person_age_classification/cls_person_age_haokai_v5_clean_Child-20191012-train/data.anno.pb_rec",  # noqa
                anno_path_md5sum="0ce65d28cc2ad37814aaa952fb4648d4",
                sample_weight=2,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{root}/data/person_age_classification/cls_person_age_haokai_v1_Adult_val-merge_val_test-merge_val/data.rec",  # noqa
                rec_path_md5sum="c4310897dc59948024b4a15f17af54bf",
                anno_path=f"{root}/data/person_age_classification/cls_person_age_haokai_v1_Adult_val-merge_val_test-merge_val/data.anno.pb_rec",  # noqa
                anno_path_md5sum="8c25fe4dbe986f8ee3fa1c74f305731f",
                sample_weight=1,
            ),
        ],
    ),
    person_orientation_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/person_orientation_classification/cls_person_orientation_v11_upsample-20191204_1-train/data.rec",  # noqa
                rec_path_md5sum="a3abe9896dae77cef5cf403e422a893d",
                anno_path=f"{root}/data/person_orientation_classification/cls_person_orientation_v11_upsample-20191204_1-train/data.anno.pb_rec",  # noqa
                anno_path_md5sum="b9e23dc4268afad2eb41e12d28f119ef",
                sample_weight=32,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    cyclist=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/cyclist_detection/majiajia.data/CC02_V5/Pilot_cyclist_5v_CO-OX3GB-A100-L_night/20220611-163932/data.rec",  # noqa
                anno_path=f"{root}/data/cyclist_detection/majiajia.data/CC02_V5/Pilot_cyclist_5v_CO-OX3GB-A100-L_night/20220611-163932/data.anno.pb_rec",  # noqa
                sample_weight=40,
            ),
            # cyclist lighted badcase night
            dict(
                rec_path=f"{root}/data/cyclist_detection/majiajia.data/MaxFaV10.0/Cyclist_lighted_night/Pilot_cyclist_5v_MaxFaV10_Cyclist_Lighted_Night_night/20220425-182351/data.rec",  # noqa
                anno_path=f"{root}/data/cyclist_detection/majiajia.data/MaxFaV10.0/Cyclist_lighted_night/Pilot_cyclist_5v_MaxFaV10_Cyclist_Lighted_Night_night/20220425-182351/data.anno.pb_rec",  # noqa
                sample_weight=40,
            ),
            # cyclist lighted badcase night
            dict(
                rec_path=f"{root}/data/cyclist_detection/majiajia.data/cyc_light_CW_A23GB_A100_L/pilot_cyclist_packing_data_publish/20220330-211259/data.rec",  # noqa
                anno_path=f"{root}/data/cyclist_detection/majiajia.data/cyc_light_CW_A23GB_A100_L/pilot_cyclist_packing_data_publish/20220330-211259/data.anno.pb_rec",  # noqa
                sample_weight=40,
            ),
            dict(
                rec_path=f"{root}/data/cyclist_detection/han.shen.data/cyc_data/20190522-remove-truck/20190522-remove-truck.rec",  # noqa
                rec_path_md5sum="40f538974a8f0656f1fe1df8e555b079",
                anno_path=f"{root}/data/cyclist_detection/han.shen.data/cyc_data/20190522-remove-truck/20190522-remove-truck.anno.pb_rec",  # noqa
                anno_path_md5sum="1d3aa84bb2fe54abeacbd53e94cc140d",
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/data/cyclist_detection/han.shen.data/cyc_data/cyc_20190918.rec",  # noqa
                rec_path_md5sum="7e3261963c2a8d56a4318c030b0ee823",
                anno_path=f"{root}/data/cyclist_detection/han.shen.data/cyc_data/cyc_20190918.anno.pb_rec",  # noqa
                anno_path_md5sum="6012511274785ac5630539ae6f3c575a",
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/data/cyclist_detection/zhengwei.data/cyc_data/cyc_20200613_cn_390_person_tain_history.rec",  # noqa
                rec_path_md5sum="da1348e8e0835850d7a0352a49f2083c",
                anno_path=f"{root}/data/cyclist_detection/zhengwei.data/cyc_data/cyc_20200613_cn_390_person_tain_history.anno.pb_rec",  # noqa
                anno_path_md5sum="1fe1a4b5a9424fd42f7b3ebfdcce1cb7",
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/mono_0323_data/user/kaijun01.zhang/data/data_2020/0323_10635_quad_cyc_20200723/0323_10635_quad_cyc_20200723.rec",  # noqa
                anno_path=f"{root}/mono_0323_data/user/kaijun01.zhang/data/data_2020/0323_10635_quad_cyc_20200723/0323_10635_quad_cyc_20200723.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/mono_0323_data/user/kaijun01.zhang/data/data_2020/0323_0220_cyc_20201119/0323_0220_cyc_20201119.rec",  # noqa
                anno_path=f"{root}/mono_0323_data/user/kaijun01.zhang/data/data_2020/0323_0220_cyc_20201119/0323_0220_cyc_20201119.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # mcp day: 18.6w
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/cyclist_detection/20210915_sensor0233_train_id_4218-9655_part6_day_upsample3_gt63_remove_empty_up300pix_30_heavyocc_normal/train.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/cyclist_detection/20210915_sensor0233_train_id_4218-9655_part6_day_upsample3_gt63_remove_empty_up300pix_30_heavyocc_normal/train.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            # x3c day&night
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/cyclist_detection/20220121-190855-x3c-all/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/cyclist_detection/20220121-190855-x3c-all/data.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            # cc02 night
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20220510/cyclist/night/20220511-124542/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20220510/cyclist/night/20220511-124542/data.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            # as33 night all 18.6k
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/cyclist_detection/Pilot_cyclist_5v_CAMERA_CW_A23GB_A114_L_night/20220624-203955/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/person/cyclist_detection/Pilot_cyclist_5v_CAMERA_CW_A23GB_A114_L_night/20220624-203955/data.anno.pb_rec",
                sample_weight=20,
            ),
            # history nontarget
            dict(
                rec_path=f"{root}/data/cyclist_detection/majiajia.data/NonTarget_exp/exp6/Pilot_cyclist_5v_all_night/20220713-225739/data.rec",  # noqa
                anno_path=f"{root}/data/cyclist_detection/majiajia.data/NonTarget_exp/exp6/Pilot_cyclist_5v_all_night/20220713-225739/data.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            # cyclist glare
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/cyclist_detection/glare/lighted_cyclist/debug/Pilot_cyclist_5v_all_night_glare/20220923-222554/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/cyclist_detection/glare/lighted_cyclist/debug/Pilot_cyclist_5v_all_night_glare/20220923-222554/data.anno.pb_rec",  # noqa
                sample_weight=60,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/cyclist_detection/glare/Pilot_cyclist_5v_CAMERA_CW_A23GB_A100_L_Night_night/20220929-145754/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/cyclist_detection/glare/Pilot_cyclist_5v_CAMERA_CW_A23GB_A100_L_Night_night/20220929-145754/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # c385 v5 4K
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20221128-104916/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20221128-104916/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # AS33 vehicle glare annoset(89252) 4274
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_AS33_glare_fp_train_night/20230131-114216/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_AS33_glare_fp_train_night/20230131-114216/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # c385 v6 night Jira 900
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Jira_Cam_5V/20230126-152414/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Jira_Cam_5V/20230126-152414/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # c385 v6 night normal 1w
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20230126-163331/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20230126-163331/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{root}/data/cyclist_detection/han.shen.data/cyc_data/cyclist_h37_w13_classid7_20190522/val.rec",  # noqa
                rec_path_md5sum="ce17a462fb3a9c3ea2c3821f1e2fcf4d",
                anno_path=f"{root}/data/cyclist_detection/han.shen.data/cyc_data/cyclist_h37_w13_classid7_20190522/val.anno.pb_rec",  # noqa
                anno_path_md5sum="0cb44df1c6b5735d5b948f515d28400c",
                sample_weight=1,
            ),
        ],
    ),
    person=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/pedestrain_detection/majiajia.data/CC02_V5/Pilot_person_5v_CO-OX3GB-A100-L_night/20220611-164225/data.rec",  # noqa
                anno_path=f"{root}/data/pedestrain_detection/majiajia.data/CC02_V5/Pilot_person_5v_CO-OX3GB-A100-L_night/20220611-164225/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/data/pedestrain_detection/han.shen.data/ped_data/ped_20181210_train_12cam_update.rec",  # noqa
                rec_path_md5sum="40069dd07439f59f568d2eeb47e74c3e",
                anno_path=f"{root}/data/pedestrain_detection/han.shen.data/ped_data/ped_20181210_train_12cam_update.anno.pb_rec",  # noqa
                anno_path_md5sum="6758b401e7cf56687da9e10b5a766584",
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/data/pedestrain_detection/han.shen.data/ped_data/ped_20190605_train_part_2_update.rec",  # noqa
                rec_path_md5sum="c7e02716451d8eafcbe3c7c3f09734f1",
                anno_path=f"{root}/data/pedestrain_detection/han.shen.data/ped_data/ped_20190605_train_part_2_update.anno.pb_rec",  # noqa
                anno_path_md5sum="a93da3d509520ca07cff13239dbd06f0",
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/data/pedestrain_detection/han.shen.data/ped_data/ped_20180814_train_gray_update.rec",  # noqa
                rec_path_md5sum="471d89edbd37267d3d2d751c28a8092a",
                anno_path=f"{root}/data/pedestrain_detection/han.shen.data/ped_data/ped_20180814_train_gray_update.anno.pb_rec",  # noqa
                anno_path_md5sum="bc0aaf8137b35a0eac72d0488be04774",
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/data/pedestrain_detection/han.shen.data/ped_data/ped_20190918.rec",  # noqa
                rec_path_md5sum="0949a859e6d2cffccfe05bca9d7e9949",
                anno_path=f"{root}/data/pedestrain_detection/han.shen.data/ped_data/ped_20190918.anno.pb_rec",  # noqa
                anno_path_md5sum="2e9b33b70946e04c43a81dabf5d8e28d",
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/data/pedestrain_detection/zhengwei.data/ped_data/ped_20200613_cn_390_person_tain_history.rec",  # noqa
                rec_path_md5sum="892da9d5aa979de9961c6aa2eeb0e2b0",
                anno_path=f"{root}/data/pedestrain_detection/zhengwei.data/ped_data/ped_20200613_cn_390_person_tain_history.anno.pb_rec",  # noqa
                anno_path_md5sum="56de27a02d5581aa1c37b84087ff3c52",
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/data/pedestrain_detection/han.shen.data/ped_data/ped_street_people_data.rec",  # noqa
                rec_path_md5sum="639080e860e562737626d7b8b6e4a0f7",
                anno_path=f"{root}/data/pedestrain_detection/han.shen.data/ped_data/ped_street_people_data.anno.pb_rec",  # noqa
                anno_path_md5sum="e0bf4b912cff5464b2e78fecefa391ec",
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/mono_0323_data/user/kaijun01.zhang/data/data_2020/0323_0220_ped_20201119/0323_0220_ped_20201119.rec",  # noqa
                anno_path=f"{root}/mono_0323_data/user/kaijun01.zhang/data/data_2020/0323_0220_ped_20201119/0323_0220_ped_20201119.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # mcp day: 33w
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20210915_sensor0233_train_id_4218-9655_part6_day_upsample3_gt63_remove_empty/train.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20210915_sensor0233_train_id_4218-9655_part6_day_upsample3_gt63_remove_empty/train.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            # weisen big person
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20211119-153000-big-person/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20211119-153000-big-person/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # x3c day&night
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20220121-192622-x3c-all/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20220121-192622-x3c-all/data.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            # tunnel 7k
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/tunnel_ped_by_zhongpeng/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/tunnel_ped_by_zhongpeng/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # tunnel 2012
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20220329-151248-x3c-tunnel/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20220329-151248-x3c-tunnel/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # cc02 person night
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20220510/person/night/20220511-124631/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20220510/person/night/20220511-124631/data.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            # as33 night all 41k
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A114_L_day_5w/20220826-121734/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A114_L_day_5w/20220826-121734/data.anno.pb_rec",  # noqa
                sample_weight=24,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A100_L_night_big_person/20220620-201624/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A100_L_night_big_person/20220620-201624/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # history nontarget
            dict(
                rec_path=f"{root}/data/pedestrain_detection/majiajia.data/NonTarget_exp/exp6/Pilot_person_5v_all_night/20220713-233017/data.rec",  # noqa
                anno_path=f"{root}/data/pedestrain_detection/majiajia.data/NonTarget_exp/exp6/Pilot_person_5v_all_night/20220713-233017/data.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            # c385 v5 4K
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20221128-104636/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20221128-104636/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # c385 v6 night Jira 900
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Jira_Cam_5V/20230126-152323/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Jira_Cam_5V/20230126-152323/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # c385 v6 night normal 1w
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20230126-162729/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20230126-162729/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # AS33 vehicle glare annoset(88986) 4274
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_AS33_glare_fp_train_night/20230130-121601/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_AS33_glare_fp_train_night/20230130-121601/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    vehicle=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.train.basenewv1.rec",  # noqa
                rec_path_md5sum="f09a02bed0f2dd9cc9b7b5c148322d56",
                anno_path=f"{root}/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.train.basenewv1.anno.pb_rec",  # noqa
                anno_path_md5sum="04b69be03cf43c6dd387e6da5e453f51",
                sample_weight=10,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.train.12camnewv1.rec",  # noqa
                rec_path_md5sum="0318cf6462a18242139667c6f1be8f43",
                anno_path=f"{root}/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.train.12camnewv1.anno.pb_rec",  # noqa
                anno_path_md5sum="a0bcb854879e0be133c1def934ef6737",
                sample_weight=10,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.train.newdatav2.rec",  # noqa
                rec_path_md5sum="b21cbfb1b6175463c03f8439a785cfd7",
                anno_path=f"{root}/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.train.newdatav2.anno.pb_rec",  # noqa
                anno_path_md5sum="49f78c12a98aea40a72bcb199a955e0b",
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/vehicle_full_data_1080_cam3900/det_vehicle_full_v1_0-20200527_v1/train.mayv9.rec",  # noqa
                rec_path_md5sum="636288fd86f7f02696dfb8d3579c63ca",
                anno_path=f"{root}/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/vehicle_full_data_1080_cam3900/det_vehicle_full_v1_0-20200527_v1/train.mayv9.anno.pb_rec",  # noqa
                anno_path_md5sum="1e667d9db77059ce96a35d888b889ce6",
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/vehicle_full_data_1080/det_vehicle_full_v1_0-20200102_v1/train.rec",  # noqa
                rec_path_md5sum="27f75e393c04abd93db21f28492e7954",
                anno_path=f"{root}/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/vehicle_full_data_1080/det_vehicle_full_v1_0-20200102_v1/train.anno.pb_rec",  # noqa
                anno_path_md5sum="cc245db028e3850a67d5e418ef528255",
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/mono_0323_data/user/yonggang.yang/data/vehicle_full_data_0323/det_vehicle_full_v1_0-20201118_v1/train.rec",  # noqa
                anno_path=f"{root}/mono_0323_data/user/yonggang.yang/data/vehicle_full_data_0323/det_vehicle_full_v1_0-20201118_v1/train.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/mono_0323_data/user/yonggang.yang/data/vehicle_full_data_1080/det_vehicle_full_v1_0-20200706_v1/train.rec",  # noqa
                anno_path=f"{root}/mono_0323_data/user/yonggang.yang/data/vehicle_full_data_1080/det_vehicle_full_v1_0-20200706_v1/train.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # mcp全量白天全车数据 约19w张
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot3.0_0233_data_before_0714_day/train.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot3.0_0233_data_before_0714_day/train.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            # 白天 韦森 10w图
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_local/20210827-195610/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_local/20210827-195610/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # 白天 韦森  8.7w图
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_local_local/20210928-120519/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_local_local/20210928-120519/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # full_vehicle_badcase_self_camera_local
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/badcase/full_vehicle_badcase_self_camera_local/20210827-212337/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/badcase/full_vehicle_badcase_self_camera_local/20210827-212337/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # tunnel badcase 1.5w
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_badcase_local/20211019-194438/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_badcase_local/20211019-194438/data.anno.pb_rec",  # noqa
                sample_weight=6,
            ),
            # 韦森 day 3.3w
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_trace/20211124-163642/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_trace/20211124-163642/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # 0233 夜晚 全量数据 222100
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_night_0233/20220119-002848/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_night_0233/20220119-002848/data.anno.pb_rec",  # noqa
                sample_weight=60,
            ),
            # X3C 3.3w
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_trace/20211129-163640/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_trace/20211129-163640/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # X3C夜间19185
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_night_x3c/20211222-132550/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_night_x3c/20211222-132550/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # X3C白天32246
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_x3c/20211222-124627/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_x3c/20211222-124627/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # 0233badcase白天9859
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_badcase/20211222-121533/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_badcase/20211222-121533/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # 0233badcase白天6w
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_night_x3c/20220121-114854/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_night_x3c/20220121-114854/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # cc02 白天 21745
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_data_5v_32393_2.1w_CAMERA_CO_OX3GB_A100_L_TIME_DAY_day/20220406-125044/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_data_5v_32393_2.1w_CAMERA_CO_OX3GB_A100_L_TIME_DAY_day/20220406-125044/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # cc02 夜晚 0.68w
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_data_5v_32394_0.68w_CAMERA_CO_OX3GB_A100_L_TIME_NIGHT_night/20220406-124214/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_data_5v_32394_0.68w_CAMERA_CO_OX3GB_A100_L_TIME_NIGHT_night/20220406-124214/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # x3c 白天 45265
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_DG4W_data_5v_CAMERA_CO_OX3GB_D100_L_TIME_DAY_day/20220401-184015/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_DG4W_data_5v_CAMERA_CO_OX3GB_D100_L_TIME_DAY_day/20220401-184015/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # x3c 白天 10763
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_29384_10763_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220426-101418/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_29384_10763_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220426-101418/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # CW_A23GB_A114_L day 2.2W
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_data_5v_as33_25469_2.2w_CAMERA_CW_A23GB_A114_L_TIME_DAY_day/20220413-111602/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_data_5v_as33_25469_2.2w_CAMERA_CW_A23GB_A114_L_TIME_DAY_day/20220413-111602/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # 0233WISSEN day 2.2W
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/polit3.0_v9.0_0315_0233_day/20220316-142817/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/polit3.0_v9.0_0315_0233_day/20220316-142817/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # mono  白天夜间 大车异型 86868
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/det_vehicle_full_v1_0-20220407_v1/train.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/det_vehicle_full_v1_0-20220407_v1/train.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # CW-A23GB-A100-L dark night 31121
            dict(
                rec_path=f"{root}/multicam_pilot/data/kVehBBox2D/kVehBBox2D_CW-A23GB-A100-L_train_night_dark_night/20220610-160750/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/kVehBBox2D/kVehBBox2D_CW-A23GB-A100-L_train_night_dark_night/20220610-160750/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # CP-OX3GB-D100-L dark night 2763
            dict(
                rec_path=f"{root}/multicam_pilot/data/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_night_dark_night/20220610-144900/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_night_dark_night/20220610-144900/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # CW-A23GB-A100-L dark night 2243
            dict(
                rec_path=f"{root}/multicam_pilot/data/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_dark_night/20220610-144130/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_dark_night/20220610-144130/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # CW-A23GB-A100-L night 标识牌
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_day/20220620-181555/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_day/20220620-181555/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # CCO2 night 6661
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_33333_6661_CAMERA_CO_OX3GB_A100-L_TIME_NIGHT_night/20220527-145352/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_33333_6661_CAMERA_CO_OX3GB_A100-L_TIME_NIGHT_night/20220527-145352/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # CCO2 night 9964
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_32658_9964_CAMERA_CO_OX3GB_A100-L_TIME_NIGHT_night/20220527-145938/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_32658_9964_CAMERA_CO_OX3GB_A100-L_TIME_NIGHT_night/20220527-145938/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # CCO2 night 12284
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_34231_12284_CAMERA_CO_OX3GB_A100-L_TIME_NIGHT_night/20220527-150019/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_34231_12284_CAMERA_CO_OX3GB_A100-L_TIME_NIGHT_night/20220527-150019/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # CW-A23GB-A100-L 大车 白天 35940
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_BIG_CAR_DAY_day/20220621-202410/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_BIG_CAR_DAY_day/20220621-202410/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # CO_OX3GB_A100_L 大车 白天 3219
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CO_OX3GB_A100_L_TIME_BIG_CAR_DAY_day/20220621-192421/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CO_OX3GB_A100_L_TIME_BIG_CAR_DAY_day/20220621-192421/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # CW-A23GB-A100-L 大车 夜晚 11187
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_BIG_CAR_night/20220629-121819/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_BIG_CAR_night/20220629-121819/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            # CW-A23GB-A100-L 大车 白天 8902
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day_bigcar/20220704-185847/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day_bigcar/20220704-185847/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_night_badcase_dark_night_Cam_5V/20220709-175425/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_night_badcase_dark_night_Cam_5V/20220709-175425/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_badcase_dark_night_Cam_5V/20220709-173429/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_badcase_dark_night_Cam_5V/20220709-173429/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # CO-OX3GB-A100-L 大车 夜晚 2416
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO-OX3GB-A100-L_TIME_BIG_CAR_night/20220725-143019/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO-OX3GB-A100-L_TIME_BIG_CAR_night/20220725-143019/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # CW_A23GB_A114_L 常规 白天 10842
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day_normal/20220706-134742/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day_normal/20220706-134742/data.anno.pb_rec",
                sample_weight=2,
            ),
            # CW_A23GB_A114_L 低照 夜间 2243
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_night_dark_0736/20220726-165515/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_night_dark_0736/20220726-165515/data.anno.pb_rec",
                sample_weight=1,
            ),
            # CW-A23GB-A100-L 大车 夜晚 3751
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_night_bigcar_0715/20220715-134225/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_night_bigcar_0715/20220715-134225/data.anno.pb_rec",
                sample_weight=1,
            ),
            # 各LTC小车挡大车
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_day_overlap_vehicle_Cam_5V/20220826-161108/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_day_overlap_vehicle_Cam_5V/20220826-161108/data.anno.pb_rec",  # noqa
                sample_weight=0.5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_day_overlap_vehicle_Cam_5V/20220826-163619/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_day_overlap_vehicle_Cam_5V/20220826-163619/data.anno.pb_rec",  # noqa
                sample_weight=0.5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_pilot_pdt_5v_CW-A23GB-A100-L_train_day_overlap_vehicle_Cam_5V/20220826-163951/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_pilot_pdt_5v_CW-A23GB-A100-L_train_day_overlap_vehicle_Cam_5V/20220826-163951/data.anno.pb_rec",  # noqa
                sample_weight=0,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_day_overlap_vehicle_Cam_5V/20220826-163258/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_day_overlap_vehicle_Cam_5V/20220826-163258/data.anno.pb_rec",  # noqa
                sample_weight=0,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_cc02_5v_CO-OX3GB-A100-L_train_day_overlap_vehicle_Cam_5V/20220826-162631/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_cc02_5v_CO-OX3GB-A100-L_train_day_overlap_vehicle_Cam_5V/20220826-162631/data.anno.pb_rec",  # noqa
                sample_weight=0,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_night_overlap_vehicle_Cam_5V/20220721-123722/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_night_overlap_vehicle_Cam_5V/20220721-123722/data.anno.pb_rec",
                sample_weight=0,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_overlap_vehicle_Cam_5V/20220721-124308/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_overlap_vehicle_Cam_5V/20220721-124308/data.anno.pb_rec",
                sample_weight=0,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_cc02_5v_CO-OX3GB-A100-L_train_night_overlap_vehicle_Cam_5V/20220721-124751/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_cc02_5v_CO-OX3GB-A100-L_train_night_overlap_vehicle_Cam_5V/20220721-124751/data.anno.pb_rec",
                sample_weight=0,
            ),
            # maxfa day special car 10259
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_54665_10259_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220707-160146/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_54665_10259_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220707-160146/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # as33 night special car 7858
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_73388_73389_73390_7858_CAMERA_CW_A23GB_A114_L_night/20220726-181251/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_73388_73389_73390_7858_CAMERA_CW_A23GB_A114_L_night/20220726-181251/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # AS33 night glare 9091
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_64258_64255_65416_9091_CAMERA_CW_A23GB_A114_L_TIME_NIGHT_night/20220707-143748/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_64258_64255_65416_9091_CAMERA_CW_A23GB_A114_L_TIME_NIGHT_night/20220707-143748/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # 全量数据快递车 7w
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-121816/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-121816/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-121723/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-121723/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-134219/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-134219/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-160123/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-160123/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # mono 三轮车2M
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_1v_OV10652_TIME_DAY_day/20220713-053741/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_1v_OV10652_TIME_DAY_day/20220713-053741/data.anno.pb_rec",  # noqa
                sample_weight=7,
            ),
            # mono 三轮车8M
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_1v_OV10652_TIME_DAY_day/20220713-061904/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_1v_OV10652_TIME_DAY_day/20220713-061904/data.anno.pb_rec",  # noqa
                sample_weight=11,
            ),
            # CW-A23GB-A100-L 标识牌 夜晚 2248
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DARK_NIGHT_night/20220731-090659/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DARK_NIGHT_night/20220731-090659/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # 环境误检 AS33 白天 1912
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_SCENE-FP_day/20221115-182150/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_SCENE-FP_day/20221115-182150/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # 环境误检 AS33 夜晚 2973
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_SCENE-FP_night/20221115-182152/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_SCENE-FP_night/20221115-182152/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # c385 夜晚 6881
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20221012-114409/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20221012-114409/data.anno.pb_rec",
                sample_weight=1,
            ),
            # c385 白天 11114
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221012-122031/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221012-122031/data.anno.pb_rec",
                sample_weight=1,
            ),
            # c385 JIRA-badcase 700
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO-OX3GB-D100-L_JIRA-badcase_all/20221229-173836/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO-OX3GB-D100-L_JIRA-badcase_all/20221229-173836/data.anno.pb_rec",
                sample_weight=5,
            ),
            # AS33夜晚眩光jiracase 528
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_glare_Cam_5V/20221009-193824/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_glare_Cam_5V/20221009-193824/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # GALAXY 环境误检 白天 1512
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221115-190526/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221115-190526/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # GALAXY 环境误检 夜晚 682
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_night/20221115-190527/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_night/20221115-190527/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # AS33 jira-badcase 白天/夜晚 369
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_JIRA-badcase_all/20221110-192849/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_JIRA-badcase_all/20221110-192849/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # c385 夜晚眩光 373
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_glare_Cam_5V/20221103-204258/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_glare_Cam_5V/20221103-204258/data.anno.pb_rec",
                sample_weight=2,
            ),
            # as33 夜晚眩光 夜晚异型车 共871
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_special_vehicle_Cam_5V/20221103-204024/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_special_vehicle_Cam_5V/20221103-204024/data.anno.pb_rec",
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_glare_Cam_5V/20221103-205303/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_glare_Cam_5V/20221103-205303/data.anno.pb_rec",
                sample_weight=2,
            ),
            # as33 眩光 4528
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_glare_Cam_5V/20221114-194733/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_glare_Cam_5V/20221114-194733/data.anno.pb_rec",
                sample_weight=1,
            ),
            # as33 眩光trigger-score 2272
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_night-glareV4_all/20221205-170615/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_night-glareV4_all/20221205-170615/data.anno.pb_rec",
                sample_weight=1,
            ),
            # GALAXY 环境误检 白天 531
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221219-120427/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221219-120427/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # GALAXY 环境误检 夜晚 692
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_night/20221219-120553/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_night/20221219-120553/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # maxfa 白天异型车 17352
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CP_OX3GB_D100_L_day_special_car_all/20230315-213329/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CP_OX3GB_D100_L_day_special_car_all/20230315-213329/data.anno.pb_rec",
                sample_weight=2,
            ),
            # c385 白天 7166
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO_OX3GB_D100_L_day_all/20230118-115653/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO_OX3GB_D100_L_day_all/20230118-115653/data.anno.pb_rec",
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{root}/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.val.basenewv1.rec",  # noqa
                rec_path_md5sum="be3819cf5deec623136df54740fec203",
                anno_path=f"{root}/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.val.basenewv1.anno.pb_rec",  # noqa
                anno_path_md5sum="4406b156ea6046faffe69621232e89e6",
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.val.12camnewv1.rec",  # noqa
                rec_path_md5sum="9645bef98a0451db1af96cd21a1525eb",
                anno_path=f"{root}/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.val.12camnewv1.anno.pb_rec",  # noqa
                anno_path_md5sum="a591e1a6df99e8778b0e31bd9ef45979",
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.val.newdatav2.rec",  # noqa
                rec_path_md5sum="897b6e5cd4a8df77999dce35825becce",
                anno_path=f"{root}/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.val.newdatav2.anno.pb_rec",  # noqa
                anno_path_md5sum="c00af83c366aa4ef89dce82f1caa4d0b",
                sample_weight=1,
            ),
        ],
    ),
    vehicle_truncation_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_truncation_classification/pilot3.0_0233_data_before_0406_vehicle_truncation_cls/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_truncation_classification/pilot3.0_0233_data_before_0406_vehicle_truncation_cls/data.json",  # noqa
                pb_anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_truncation_classification/pilot3.0_0233_data_before_0406_vehicle_truncation_cls/data.anno.pb_rec",  # noqa
                sample_weight=32,
            ),
        ],
        val_batch_size_per_ctx=16,
        val_data_paths=None,
    ),
    semantic_parsing=dict(
        train_batch_size_per_ctx=16,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/default_parsing/train_data/mono3.0/16.2c/data_all_0220model_2020-06-22/parsing-720p10635-train-anno16.2_59070.rec",  # noqa
                anno_path=f"{root}/data/default_parsing/train_data/mono3.0/16.2c/data_all_0220model_2020-06-22/parsing-720p10635-train-anno16.2_59070.anno.pb_rec",  # noqa
                sample_weight=7,
            ),
            dict(
                rec_path=f"{root}/data/default_parsing/train_data/mono3.0/16.2c/train-1080p-id20220324_62910.rec",  # noqa
                anno_path=f"{root}/data/default_parsing/train_data/mono3.0/16.2c/train-1080p-id20220324_62910.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # sensing day (special for large person)
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_2872_20210804_anno16.2.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_2872_20210804_anno16.2.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # sensing day (rain)
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_950_20220527_anno16.2_day.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_950_20220527_anno16.2_day.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # wissen day&night (special for merge truck bottom)
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_2912_20220111_anno16.2.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_2912_20220111_anno16.2.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # as33 day&night (snow)
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_4876_20220415_anno16.2.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_4876_20220415_anno16.2.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # sensing day
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_15836_20220111_anno16.2_sy_day.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_15836_20220111_anno16.2_sy_day.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_4671_20220111_anno16.2_sy_day.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_4671_20220111_anno16.2_sy_day.anno.pb_rec",  # noqa
                sample_weight=2.5,
            ),
            # wissen day
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_13566_20220111_anno16.2_ws_day.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_13566_20220111_anno16.2_ws_day.anno.pb_rec",  # noqa
                sample_weight=8.5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_3457_20220111_anno16.2_ws_day.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_3457_20220111_anno16.2_ws_day.anno.pb_rec",  # noqa
                sample_weight=2.5,
            ),
            # wissen night
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_16089_20220111_anno16.2_ws_night.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_16089_20220111_anno16.2_ws_night.anno.pb_rec",  # noqa
                sample_weight=13,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_3504_20220111_anno16.2_ws_night.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_3504_20220111_anno16.2_ws_night.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # x3c night
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4987_20220111_anno16.2_x3c_night.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4987_20220111_anno16.2_x3c_night.anno.pb_rec",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4300_20220214_anno16.2_x3c_night.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4300_20220214_anno16.2_x3c_night.anno.pb_rec",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4060_20220428_anno16.2_x3c_night.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4060_20220428_anno16.2_x3c_night.anno.pb_rec",  # noqa
                sample_weight=6,
            ),
            # x3c day
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4500_20220111_anno16.2_x3c_day.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4500_20220111_anno16.2_x3c_day.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_10228_20220214_anno16.2_x3c_day.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_10228_20220214_anno16.2_x3c_day.anno.pb_rec",  # noqa
                sample_weight=9,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1010_20220428_anno16.2_x3c_day.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1010_20220428_anno16.2_x3c_day.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # cc02 day
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2671_20220513_anno16.2_cc02_day.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2671_20220513_anno16.2_cc02_day.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # as33 day
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_1202_20220602_anno16.2_as33_day.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_1202_20220602_anno16.2_as33_day.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_1660_20220825_anno16.2_as33_day/20220825-140857/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_1660_20220825_anno16.2_as33_day/20220825-140857/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # 20220923
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_4247_anno16.2_as33_day/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_4247_anno16.2_as33_day/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # c385 x3c 1358, ID 91711
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1358_anno16.2_c385_night_91711/20221011-011708/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1358_anno16.2_c385_night_91711/20221011-011708/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # 133425，133419，144172
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_543_anno16.2_c385_night/20230119-194419/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_543_anno16.2_c385_night/20230119-194419/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # jira 数据，暗光夜晚围栏，144713，144712
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_488_anno16.2_c385_night/20230119-203402/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_488_anno16.2_c385_night/20230119-203402/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{root}/data/default_parsing/train_data/matrix2.1-V3.0/CN/val-720p-id20200204_4183.rec",  # noqa
                rec_path_md5sum="366b4d5346f8608fe336d66a8ce6ab50",
                anno_path=f"{root}/data/default_parsing/train_data/matrix2.1-V3.0/CN/val-720p-id20200204_4183.anno.pb_rec",  # noqa
                anno_path_md5sum="056eaa903fa99123cf21ca5ad0b5bc83",
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/data/default_parsing/train_data/matrix2.1-V3.0/CN/val-1080p-id20200204_1682.rec",  # noqa
                rec_path_md5sum="db373f1d0c058172522a3d88725311db",
                anno_path=f"{root}/data/default_parsing/train_data/matrix2.1-V3.0/CN/val-1080p-id20200204_1682.anno.pb_rec",  # noqa
                anno_path_md5sum="97d2ee5bc04d546fe115c31e37d744f1",
                sample_weight=1,
            ),
        ],
    ),
    lane_parsing=dict(
        train_batch_size_per_ctx=16,
        train_data_paths=[
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_18772_20210419_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_18772_20210419_5_2classes.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_8571_20210524_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_8571_20210524_5_2classes.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_24305_20210711_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_24305_20210711_5_2classes.anno.pb_rec",  # noqa
                sample_weight=12,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_17311_20210711_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_17311_20210711_5_2classes.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_28019_20210803_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_28019_20210803_5_2classes.anno.pb_rec",  # noqa
                sample_weight=12,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_21736_20210803_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_21736_20210803_5_2classes.anno.pb_rec",  # noqa
                sample_weight=12,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_71524_20210825_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_71524_20210825_5_2classes.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_13711_20210825_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_13711_20210825_5_2classes.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_55922_20211018_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_55922_20211018_5_2classes.anno.pb_rec",  # noqa
                sample_weight=15,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_36387_20211018_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_36387_20211018_5_2classes.anno.pb_rec",  # noqa
                sample_weight=15,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_160_20211129_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_160_20211129_5_2classes.anno.pb_rec",  # noqa
                sample_weight=0.1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_wissen_num_78742_20211128_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_wissen_num_78742_20211128_5_2classes.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_22101_20211129_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_22101_20211129_5_2classes.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_wissen_num_4549_20211129_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_wissen_num_4549_20211129_5_2classes.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_47972_20211220_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_47972_20211220_5_2classes.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_16073_20211220_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_16073_20211220_5_2classes.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_64275_20220118_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_64275_20220118_5_2classes.anno.pb_rec",  # noqa
                sample_weight=30,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_27923_20220118_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_27923_20220118_5_2classes.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_14440_20220214_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_14440_20220214_5_2classes.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_2924_20220214_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_2924_20220214_5_2classes.anno.pb_rec",  # noqa
                sample_weight=0.4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_num_24851_20220328_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_num_24851_20220328_5_2classes.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_num_5356_20220328_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_num_5356_20220328_5_2classes.anno.pb_rec",  # noqa
                sample_weight=0.4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_D100L_num_17734_20220424_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_D100L_num_17734_20220424_5_2classes.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_CO_OX3GB_A100L_num_7589_20220505_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_CO_OX3GB_A100L_num_7589_20220505_5_2classes.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_CO_OX3GB_A100L_num_745_20220505_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_CO_OX3GB_A100L_num_745_20220505_5_2classes.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_4481_20220525_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_4481_20220525_5_2classes.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_0233_CW_A23GB_A114L_num_5460_20220525_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_0233_CW_A23GB_A114L_num_5460_20220525_5_2classes.anno.pb_rec",  # noqa
                sample_weight=5.0,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_6271_20220525_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_6271_20220525_5_2classes.anno.pb_rec",  # noqa
                sample_weight=3.1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_6551_20220608_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_6551_20220608_5_2classes.anno.pb_rec",  # noqa
                sample_weight=3.3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_0233_CW_A23GB_A114L_num_1995_20220621_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_0233_CW_A23GB_A114L_num_1995_20220621_5_2classes.anno.pb_rec",  # noqa
                sample_weight=2.0,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_220829/lane_parsing_train_day_badcase_as33/20220727-223306/data.rec",
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_220829/lane_parsing_train_day_badcase_as33/20220727-223306/data.anno.pb_rec",
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_221141/lane_parsing_train_day_badcase_as33/20220727-222539/data.rec",
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_221141/lane_parsing_train_day_badcase_as33/20220727-222539/data.anno.pb_rec",
                sample_weight=0.4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_221458/lane_parsing_train_day_badcase_as33/20220727-222511/data.rec",
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_221458/lane_parsing_train_day_badcase_as33/20220727-222511/data.anno.pb_rec",
                sample_weight=0.2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_221812/lane_parsing_train_day_badcase_as33/20220727-222648/data.rec",
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_221812/lane_parsing_train_day_badcase_as33/20220727-222648/data.anno.pb_rec",
                sample_weight=0.1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_222305/lane_parsing_train_day_special_as33/20220727-223226/data.rec",
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_222305/lane_parsing_train_day_special_as33/20220727-223226/data.anno.pb_rec",
                sample_weight=0.1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_222623/lane_parsing_train_day_special_as33/20220727-223356/data.rec",
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_222623/lane_parsing_train_day_special_as33/20220727-223356/data.anno.pb_rec",
                sample_weight=0.06,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_222930/lane_parsing_train_day_special_as33/20220727-223812/data.rec",
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_222930/lane_parsing_train_day_special_as33/20220727-223812/data.anno.pb_rec",
                sample_weight=0.1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_223832/lane_parsing_train_day_normal_as33/20220727-224554/data.rec",
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_223832/lane_parsing_train_day_normal_as33/20220727-224554/data.anno.pb_rec",
                sample_weight=0.09,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_224147/lane_parsing_train_day_normal_as33/20220727-224901/data.rec",
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_224147/lane_parsing_train_day_normal_as33/20220727-224901/data.anno.pb_rec",
                sample_weight=0.06,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_224502/lane_parsing_train_day_normal_as33/20220727-225403/data.rec",
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_224502/lane_parsing_train_day_normal_as33/20220727-225403/data.anno.pb_rec",
                sample_weight=0.2,
            ),
            # day side 1861
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_000716/pilot_lane_parsing_annoset/20221012-010546/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_000716/pilot_lane_parsing_annoset/20221012-010546/data.anno.pb_rec",  # noqa
                sample_weight=1.0,
            ),
            # day rear 650
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_005334/pilot_lane_parsing_annoset/20221012-012546/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_005334/pilot_lane_parsing_annoset/20221012-012546/data.anno.pb_rec",  # noqa
                sample_weight=1.0,
            ),
            # night side 1427
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_012136/pilot_lane_parsing_annoset/20221012-015617/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_012136/pilot_lane_parsing_annoset/20221012-015617/data.anno.pb_rec",  # noqa
                sample_weight=2.0,
            ),
            # night rear 874
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_014836/pilot_lane_parsing_annoset/20221012-022211/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_014836/pilot_lane_parsing_annoset/20221012-022211/data.anno.pb_rec",  # noqa
                sample_weight=2.0,
            ),
            # day side 4325
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_123750/pilot_lane_parsing_annoset/20221126-132031/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_123750/pilot_lane_parsing_annoset/20221126-132031/data.anno.pb_rec",  # noqa
                sample_weight=1.0,
            ),
            # day rear 2737
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_125221/pilot_lane_parsing_annoset/20221126-132842/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_125221/pilot_lane_parsing_annoset/20221126-132842/data.anno.pb_rec",  # noqa
                sample_weight=1.0,
            ),
            # night side 2013
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_130644/pilot_lane_parsing_annoset/20221126-133734/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_130644/pilot_lane_parsing_annoset/20221126-133734/data.anno.pb_rec",  # noqa
                sample_weight=2.0,
            ),
            # v6.0  day
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20230128_224418_local/pilot_lane_parsing_annoset/20230128-224930/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20230128_224418_local/pilot_lane_parsing_annoset/20230128-224930/data.anno.pb_rec",  # noqa
                sample_weight=1.0,
            ),
            # v6.0  night
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20230128_234420_local/pilot_lane_parsing_annoset/20230128-234745/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20230128_234420_local/pilot_lane_parsing_annoset/20230128-234745/data.anno.pb_rec",  # noqa
                sample_weight=2.0,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{root}/data/lane_parsing/store/cn/matrix_2.x_cn_1080p_val_20200625_4classes/val.rec",  # noqa
                rec_path_md5sum="c468b5bac7e64d32beebbd19688d1b8b",
                anno_path=f"{root}/data/lane_parsing/store/cn/matrix_2.x_cn_1080p_val_20200625_4classes/val.anno.pb_rec",  # noqa
                anno_path_md5sum="4948a1123fcf39a60b7fbc9344d8f6e3",
                sample_weight=1,
            ),
        ],
    ),
    vehicle_wheel_kps=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/vehicle_kps/vehicle_2_kps/vehicle_2_kps_all_20201021.train.rec",  # noqa
                anno_path=f"{root}/data/vehicle_kps/vehicle_2_kps/aptiv_vehicle_2_kps_all_20201021_min30_consistent.train.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_kps_detection/release_rec/vehicle_2_kps_add0323cutin_20201125.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_kps_detection/release_rec/vehicle_2_kps_add0323cutin_20201125.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # 0233 白天 137993
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_0233/20220119-212808/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_0233/20220119-212808/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # 0233 夜晚 62143
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_night_0233/20220120-114207/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_night_0233/20220120-114207/data.anno.pb_rec",  # noqa
                sample_weight=40,
            ),
            # X3C 白天 34092
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_X3C/20220119-122143/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_X3C/20220119-122143/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # X3C 夜晚 54023
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_night_X3C/20220119-123010/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_night_X3C/20220119-123010/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # X3C 白天 16632
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_X3C/20220224-131626/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_X3C/20220224-131626/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 白天 51208
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220315-163131/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220315-163131/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # CAMERA_X3C 白天 9776 annoset(7184)
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220425-161851/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220425-161851/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=16,
        val_data_paths=None,
    ),
    vehicle_ground_line=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            # 0233 白天 137993
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_0233/20220119-221935/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_0233/20220119-221935/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # 0233 夜晚 62143
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_night_0233/20220120-115549/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_night_0233/20220120-115549/data.anno.pb_rec",  # noqa
                sample_weight=40,
            ),
            # X3C 白天 34092
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_X3C/20220119-125418/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_X3C/20220119-125418/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # X3C 夜晚 54023
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_night_X3C/20220119-141510/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_night_X3C/20220119-141510/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # X3C 白天 16632
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_X3C/20220224-133404/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_X3C/20220224-133404/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 白天 51208
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220315-163208/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220315-163208/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # CAMERA_X3C 白天 9776 annoset(12336)
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220425-161825/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220425-161825/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=16,
        val_data_paths=None,
    ),
    vehicle_flank=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            # 白天 kVehFlank_Cam_5V annoset(30652) 6572
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehFlank/kVehFlank_Cam_5V/20220706-213158/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehFlank/kVehFlank_Cam_5V/20220706-213158/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # 白天 kVehFlank_ar0233-RGGB_train_Cam_5V annoset(30651) 52561
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehFlank/kVehFlank_ar0233-RGGB_train_Cam_5V/20220706-205154/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehFlank/kVehFlank_ar0233-RGGB_train_Cam_5V/20220706-205154/data.anno.pb_rec",  # noqa
                sample_weight=6,
            ),
            # 白天 kVehFlank_ar0233-WISSEN_train_Cam_5V annoset(30650) 3010
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehFlank/kVehFlank_ar0233-WISSEN_train_Cam_5V/20220706-165341/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehFlank/kVehFlank_ar0233-WISSEN_train_Cam_5V/20220706-165341/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # 白天 kVehFlank_ar0233-WISSEN_train_day_big_vehicle_Cam_5V annoset(30648) 1772
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehFlank/kVehFlank_ar0233-WISSEN_train_day_big_vehicle_Cam_5V/20220706-123556/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehFlank/kVehFlank_ar0233-WISSEN_train_day_big_vehicle_Cam_5V/20220706-123556/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # 白天 kVehFlank_x3c_train_day_Cam_5V annoset(30647) 94622
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehFlank/kVehFlank_x3c_train_day_Cam_5V/20220706-112001/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehFlank/kVehFlank_x3c_train_day_Cam_5V/20220706-112001/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # 白天 kVehFlank_project_cc02_5v_CO-OX3GB-A100-L_train_day_big_vehicle_Cam_5V annoset(30646) 1258
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehFlank/kVehFlank_project_cc02_5v_CO-OX3GB-A100-L_train_day_big_vehicle_Cam_5V/20220706-102652/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehFlank/kVehFlank_project_cc02_5v_CO-OX3GB-A100-L_train_day_big_vehicle_Cam_5V/20220706-102652/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # 夜晚 kVehFlank_x3c_train_night_Cam_5V annoset(30649) 54023
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehFlank/kVehFlank_x3c_train_night_Cam_5V/20220706-163624/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehFlank/kVehFlank_x3c_train_night_Cam_5V/20220706-163624/data.anno.pb_rec",  # noqa
                sample_weight=6,
            ),
        ],
        val_batch_size_per_ctx=16,
        val_data_paths=None,
    ),
    vehicle_wheel_detection=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/vehicle_wheel_detection/det_subbox_vehicle_wheel_belongto_clean-veh_wheel-20210407_train_clean_323_all_trans_pb_rec/train.rec",  # noqa
                anno_path=f"{root}/data/vehicle_wheel_detection/det_subbox_vehicle_wheel_belongto_clean-veh_wheel-20210407_train_clean_323_all_trans_pb_rec/train.anno.pb_rec",  # noqa
                sample_weight=171,
            ),
            # MCP day
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_wheel_detection/MCP_wheel_detection_day_0927/train.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_wheel_detection/MCP_wheel_detection_day_0927/train.anno.pb_rec",  # noqa
                sample_weight=85,
            ),
            # kVehWheelBBox2D_train_day_big_vehicle_Cam_5V_Cam_5V
            # 4332 annoset(24608)
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehWheelBBox2D/kVehWheelBBox2D_train_day_big_vehicle_Cam_5V_Cam_5V/20220617-193224/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehWheelBBox2D/kVehWheelBBox2D_train_day_big_vehicle_Cam_5V_Cam_5V/20220617-193224/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
        ],
        val_batch_size_per_ctx=16,
        val_data_paths=None,
    ),
    rear=dict(
        train_batch_size_per_ctx=24,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v3-20190726-train_base/data.rec",  # noqa
                rec_path_md5sum="99538fca93e75e72ea62acb3e0b51c8c",
                anno_path=f"{root}/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v3-20190726-train_base/data.anno.pb_rec",  # noqa
                anno_path_md5sum="bf7bdb8d0c9e80db1eedbc9a5ef888c5",
                sample_weight=8,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v3-20190726-train_gray/data.rec",  # noqa
                rec_path_md5sum="495ce769847fa704671acf7cd5786a23",
                anno_path=f"{root}/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v3-20190726-train_gray/data.anno.pb_rec",  # noqa
                anno_path_md5sum="63401a30528664d812d8354923b8af43",
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v3-20190726-train_night/data.rec",  # noqa
                rec_path_md5sum="cd04a592e2033a423dc3ef9c7355dbc9",
                anno_path=f"{root}/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v3-20190726-train_night/data.anno.pb_rec",  # noqa
                anno_path_md5sum="a1a2b09555124a06cb5e567eff838834",
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v6-20200624-train_390_5200_day_FrontRear/data.rec",  # noqa
                rec_path_md5sum="98f5d8915da016a4957e673ac3030d69",
                anno_path=f"{root}/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v6-20200624-train_390_5200_day_FrontRear/data.anno.pb_rec",  # noqa
                anno_path_md5sum="1368e5e717a6c02847d71eb7d0c5bd6f",
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v6-20200624-train_390_5200_day_NonFrontRear/data.rec",  # noqa
                rec_path_md5sum="33499ea612d9579ace348d349a697a3a",
                anno_path=f"{root}/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v6-20200624-train_390_5200_day_NonFrontRear/data.anno.pb_rec",  # noqa
                anno_path_md5sum="a4c51fcf563e66c646b2052852a57b83",
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v6-20200624-train_390_5200_night/data.rec",  # noqa
                rec_path_md5sum="e39cddd2572a54044083c9f025a0aee7",
                anno_path=f"{root}/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v6-20200624-train_390_5200_night/data.anno.pb_rec",  # noqa
                anno_path_md5sum="f6112957cb11d4841d8f3f4dc6266f13",
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/mono_0323_data/user/ni.jiang/data/vehicle_rear_data/det_vehicle_rear_v8-20201119-train_part_4_0323_0220_exclude_20200601_exclude_part33/data.rec",  # noqa
                anno_path=f"{root}/mono_0323_data/user/ni.jiang/data/vehicle_rear_data/det_vehicle_rear_v8-20201119-train_part_4_0323_0220_exclude_20200601_exclude_part33/data.anno.pb_rec",  # noqa
                sample_weight=12,
            ),
            # 0233 白天 330660
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_0233/20220119-012241/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_0233/20220119-012241/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # 0233 夜晚 81783
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_night_0233/20220118-233709/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_night_0233/20220118-233709/data.anno.pb_rec",  # noqa
                sample_weight=40,
            ),
            # X3C 白天 84312
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_X3C/20220119-111048/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_X3C/20220119-111048/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # X3C 夜晚 24160
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_night_X3C/20220119-105348/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_night_X3C/20220119-105348/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # X3C 白天 12906
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_X3C/20220224-113552/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_X3C/20220224-113552/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 夜晚 22000
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_night_X3C/20220224-115939/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_night_X3C/20220224-115939/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # X3C 白天 15688
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_day/20220315-163248/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_day/20220315-163248/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 夜晚 7612
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_night/20220315-164230/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_night/20220315-164230/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 白天 13505
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_day/20220401-140046/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_day/20220401-140046/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # PROJECT_CC02_5V 白天 4910
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_day/20220407-195106/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_day/20220407-195106/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # PROJECT_CC02_5V 夜晚 4758
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_night/20220407-195117/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_night/20220407-195117/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # CAMERA_CW_A23GB_A100_L 夜晚 9739 annoset(12296)
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_CW_A23GB_A100_L_night/20220425-140822/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_CW_A23GB_A100_L_night/20220425-140822/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # PROJECT_AS33_5V 夜晚 8872
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_AS33_night/20220412-213815/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_AS33_night/20220412-213815/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # PROJECT_PDT 夜晚 大车 1306
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW_A23GB_A100_L_big_car_night/20220412-212057/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW_A23GB_A100_L_big_car_night/20220412-212057/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # PROJECT_CC02_5V 白天 8526 annoset(16130)
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_day/20220510-214104/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_day/20220510-214104/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # PROJECT_CC02_5V 夜晚 2159 annoset(16131)
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_night/20220510-213158/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_night/20220510-213158/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # PROJECT_AS33_5V 白天 2113
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW-A23GB-A114-L_day/20220510-103100/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW-A23GB-A114-L_day/20220510-103100/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # PROJECT_AS33_5V 夜晚 1014
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW-A23GB-A114-L_night/20220510-115328/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW-A23GB-A114-L_night/20220510-115328/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # CAMERA_X3C_5V 夜晚 1.9K
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_CAMERA_X3C_night/20220621-162154/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_CAMERA_X3C_night/20220621-162154/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear/det_vehicle_rear_v10-20211229-10652-split_special_and_BigTrucks/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear/det_vehicle_rear_v10-20211229-10652-split_special_and_BigTrucks/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear/det_vehicle_rear_v10-20220129-jira_and_close_vehicle_data/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear/det_vehicle_rear_v10-20220129-jira_and_close_vehicle_data/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear/det_vehicle_rear_v8-0220_0323_10635_bigcar-20210917/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear/det_vehicle_rear_v8-0220_0323_10635_bigcar-20210917/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # CW_A23GB_A114_L 夜晚 眩光 2015
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CAMERA_CW_A23GB_A114_L_night_glare_0831/20220831-181141/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CAMERA_CW_A23GB_A114_L_night_glare_0831/20220831-181141/data.anno.pb_rec",
                sample_weight=1,
            ),
            # PROJECT_c385_5V 白天 4879
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CO-OX3GB-D100-L_day/20221127-140301/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CO-OX3GB-D100-L_day/20221127-140301/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{root}/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v1-20190726-val_base/data.rec",  # noqa
                rec_path_md5sum="1537d9e9acce893f00642a58bce0f2f2",
                anno_path=f"{root}/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v1-20190726-val_base/data.anno.pb_rec",  # noqa
                anno_path_md5sum="348b4468836157b280beda6a0c011da8",
                sample_weight=1,
            ),
        ],
    ),
    rear_occlusion_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_subbox_rear_occlusion_v8-20190719-train_base_night_gray-cn93-cn750-model-score0.18/data.rec",  # noqa
                anno_path=f"{root}/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_subbox_rear_occlusion_v8-20190719-train_base_night_gray-cn93-cn750-model-score0.18/data.anno.pb_rec",  # noqa
                sample_weight=12,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_subbox_rear_occlusion_v8-20200903-train_part_4_0323_ispDefault_2280x1080_exclude_part33-cn750-cn759/data.rec",  # noqa
                anno_path=f"{root}/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_subbox_rear_occlusion_v8-20200903-train_part_4_0323_ispDefault_2280x1080_exclude_part33-cn750-cn759/data.anno.pb_rec",  # noqa
                sample_weight=18,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_rear_neg_v4-20200901-fp_train/data.rec",  # noqa
                anno_path=f"{root}/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_rear_neg_v4-20200901-fp_train/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # 0233 白天 330660
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_0233/20220119-041133/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_0233/20220119-041133/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # 0233 夜晚 81783
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_night_0233/20220119-020318/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_night_0233/20220119-020318/data.anno.pb_rec",  # noqa
                sample_weight=40,
            ),
            # X3C 白天 84312
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_X3C/20220119-120507/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_X3C/20220119-120507/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # X3C 夜晚 24160
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_night_X3C/20220119-113231/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_night_X3C/20220119-113231/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # X3C 白天 12657
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_X3C/20220224-123226/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_X3C/20220224-123226/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 夜晚 20497
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_night_X3C/20220224-125801/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_night_X3C/20220224-125801/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # X3C 白天 15083
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_CAMERA_X3C_day/20220316-102102/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_CAMERA_X3C_day/20220316-102102/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 夜晚 7191
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_CAMERA_X3C_night/20220315-162936/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_CAMERA_X3C_night/20220315-162936/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 白天 13505
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_CAMERA_X3C_day/20220401-160904/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_CAMERA_X3C_day/20220401-160904/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{root}/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_subbox_rear_occlusion_v2-20190719-val/data.rec",  # noqa
                anno_path=f"{root}/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_subbox_rear_occlusion_v2-20190719-val/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
        ],
    ),
    rear_part_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-old/data.rec",  # noqa
                anno_path=f"{root}/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-old/data.anno.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-hard/data.rec",  # noqa
                anno_path=f"{root}/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-hard/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-res/data.rec",  # noqa
                anno_path=f"{root}/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-res/data.anno.pb_rec",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-0323/data.rec",  # noqa
                anno_path=f"{root}/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-0323/data.anno.pb_rec",  # noqa
                sample_weight=9,
            ),
            # 0233 白天 127529
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_0233/20220119-210054/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_0233/20220119-210054/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # 0233 夜晚 57252
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_night_0233/20220119-203134/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_night_0233/20220119-203134/data.anno.pb_rec",  # noqa
                sample_weight=40,
            ),
            # X3C 白天 67690
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_X3C/20220119-120509/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_X3C/20220119-120509/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # X3C 夜晚 24160
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_night_X3C/20220119-115855/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_night_X3C/20220119-115855/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # X3C 白天 15788
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_X3C/20220224-130525/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_X3C/20220224-130525/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 白天 12323
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_CAMERA_X3C_day/20220315-163229/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_CAMERA_X3C_day/20220315-163229/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 夜晚 9351
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_CAMERA_X3C_night/20220315-163054/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_CAMERA_X3C_night/20220315-163054/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # CAMERA_X3C 夜晚 9086 annoset(12297)
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_CAMERA_X3C_night/20220425-142431/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_CAMERA_X3C_night/20220425-142431/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # as33 白天 大车
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_rear_part_classification/pilot_rear_part_5v_CAMERA_CW_A23GB_A114_L_day_bigcar/20220822-093413/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_rear_part_classification/pilot_rear_part_5v_CAMERA_CW_A23GB_A114_L_day_bigcar/20220822-093413/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # c385 day 8495
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_rear_part_classification/pilot_rear_part_5v_CO-OX3GB-D100-L_day/20221127-145644/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_rear_part_classification/pilot_rear_part_5v_CO-OX3GB-D100-L_day/20221127-145644/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=f"{root}/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-rear_cls_v2-imgs/data.rec",  # noqa
                anno_path=f"{root}/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-rear_cls_v2-imgs/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
        ],
    ),
)
