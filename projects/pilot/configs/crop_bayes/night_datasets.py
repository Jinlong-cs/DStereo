import os

from easydict import EasyDict

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
root = os.path.join(bucket_root, "matrix2")


# ----- Required -----

datapaths = dict(
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
                rec_path=f"{root}/multicam_pilot/data/train/person/2pe_cyclist_kps/20210507_sensor0233_train_id/train.pb_rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/2pe_cyclist_kps/20210507_sensor0233_train_id/train.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
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
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_day_0233/20220119-041133/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_day_0233/20220119-041133/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # 0233 夜晚 81783
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_night_0233/20220119-020318/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_night_0233/20220119-020318/data.anno.pb_rec",  # noqa
                sample_weight=40,
            ),
            # X3C 白天 84312
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_day_X3C/20220119-120507/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_day_X3C/20220119-120507/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # X3C 夜晚 24160
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_night_X3C/20220119-113231/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_night_X3C/20220119-113231/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # X3C 白天 12657
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_day_X3C/20220224-123226/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_day_X3C/20220224-123226/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 夜晚 20497
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_night_X3C/20220224-125801/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_night_X3C/20220224-125801/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # X3C 白天 15083
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_CAMERA_X3C_day/20220316-102102/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_CAMERA_X3C_day/20220316-102102/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 夜晚 7191
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_CAMERA_X3C_night/20220315-162936/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_CAMERA_X3C_night/20220315-162936/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 白天 13505
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_CAMERA_X3C_day/20220401-160904/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_CAMERA_X3C_day/20220401-160904/data.anno.pb_rec",  # noqa
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
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_day_0233/20220119-210054/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_day_0233/20220119-210054/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # 0233 夜晚 57252
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_night_0233/20220119-203134/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_night_0233/20220119-203134/data.anno.pb_rec",  # noqa
                sample_weight=40,
            ),
            # X3C 白天 67690
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_day_X3C/20220119-120509/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_day_X3C/20220119-120509/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # X3C 夜晚 24160
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_night_X3C/20220119-115855/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_night_X3C/20220119-115855/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # X3C 白天 15788
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_day_X3C/20220224-130525/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_day_X3C/20220224-130525/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 白天 12323
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_CAMERA_X3C_day/20220315-163229/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_CAMERA_X3C_day/20220315-163229/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 夜晚 9351
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_CAMERA_X3C_night/20220315-163054/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_CAMERA_X3C_night/20220315-163054/data.anno.pb_rec",  # noqa
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
    person_head_detection=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/person_head_detection/Aptiv_Las_Vegas_20190422_700-1630_train_min_head_height12/data.rec",  # noqa
                anno_path=f"{root}/data/person_head_detection/Aptiv_Las_Vegas_20190422_700-1630_train_min_head_height12/data.anno.pb_rec",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/data/person_head_detection/a_las_vegas_1903_min_head_height12/data.rec",  # noqa
                anno_path=f"{root}/data/person_head_detection/a_las_vegas_1903_min_head_height12/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/data/person_head_detection/white_changan_20190212_min_head_height12/data.rec",  # noqa
                anno_path=f"{root}/data/person_head_detection/white_changan_20190212_min_head_height12/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/data/person_head_detection/id1-4_head_min_height_22/data.rec",  # noqa
                anno_path=f"{root}/data/person_head_detection/id1-4_head_min_height_22/data.anno.pb_rec",  # noqa
                sample_weight=4,
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
                rec_path=f"{root}/multicam_pilot/data/train/person/2pe_person_head_detection/20210507_sensor0233_train_id/train.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/2pe_person_head_detection/20210507_sensor0233_train_id/train.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # dict(
            #     rec_path=f'/mnt/cephfs-adas-boschhwy/adas/zhengwei.hu/project_space/gluon_horizon_internal/scripts/auto_matrix_pack_tools/data/rec/2pe_person_head_detection_test/train.rec',  # noqa
            #     anno_path=f'/mnt/cephfs-adas-boschhwy/adas/zhengwei.hu/project_space/gluon_horizon_internal/scripts/auto_matrix_pack_tools/data/rec/2pe_person_head_detection_test/train.anno.pb_rec',  # noqa
            #     sample_weight=2
            # )
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
                sample_weight=32,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/2pe_person_pose_classification/20210507_sensor0233_train_id/train.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/2pe_person_pose_classification/20210507_sensor0233_train_id/train.anno.pb_rec",  # noqa
                sample_weight=4,
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
    vehicle_3d_detection=dict(
        train_batch_size_per_ctx=12,
        train_data_paths=[
            dict(
                # 65U3D
                # 1057926
                rec_path=[
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210418_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210420_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210421_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210422_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210423_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210424_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210425_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210426_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210427_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210428_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210429_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210430_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210502_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210506_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210507_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210508_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210509_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210510_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210511_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210512_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210513_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210514_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210517_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210518_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210519_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210520_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210521_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210523_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210527_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210528_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210529_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210530_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210531_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210601_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210602_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210603_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210604_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210605_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210606_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210607_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210611_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210612_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210613_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210614_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210615_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210616_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210621_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210622_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210623_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210624_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210625_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/65U3D/data_65U3D_20210626_v05__no_front__.rec",  # noqa
                ],
                sample_weight=8,
            ),
            dict(
                # 30C1G
                # 731340
                rec_path=[
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210423_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210425_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210426_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210427_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210428_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210429_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210430_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210501_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210506_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210507_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210508_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210509_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210510_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210511_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210512_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210513_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210514_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210515_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210517_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210518_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210519_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210520_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210521_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210522_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210523_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210524_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210525_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210527_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210528_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210531_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210601_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210602_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210603_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210604_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210605_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210609_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210610_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210611_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210612_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210613_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210614_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210616_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210617_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210618_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210619_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210620_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210621_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210624_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210625_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210626_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210627_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v05/30C1G/data_30C1G_20210628_v05__no_front__.rec",  # noqa
                ],
                sample_weight=7.5,
            ),
            dict(
                # hard case
                rec_path=[
                    f"{root}/multicam_pilot/data/train/vehicle/vehicle_3D_detection/hard_case/65U3D_filtered_10m_contain_bus_truck_specialcar_weisen_v04.rec",  # noqa
                    f"{root}/multicam_pilot/data/train/vehicle/vehicle_3D_detection/hard_case/30C1G_filtered_10m_contain_bus_truck_specialcar_weisen_v04.rec",  # noqa
                ],
                sample_weight=0.5,
            ),
        ],
        val_batch_size_per_ctx=10,
        val_data_paths=[
            dict(
                rec_path=[
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210503_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210504_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210505_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210515_v04__no_front__.rec",  # noqa
                ],
                sample_weight=5,
            ),
            dict(
                rec_path=[
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210504_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210505_v04__no_front__.rec",  # noqa
                ],
                sample_weight=5,
            ),
        ],
    ),
    person_3d_detection=dict(
        train_batch_size_per_ctx=16,
        train_data_paths=[
            dict(
                # 65U3D
                # 1057926
                rec_path=[
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210418_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210420_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210421_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210422_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210423_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210424_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210425_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210426_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210427_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210428_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210429_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210430_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210502_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210506_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210507_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210508_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210509_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210510_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210511_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210512_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210513_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210514_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210517_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210518_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210519_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210520_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210521_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210523_v04__no_front__.rec",  # noqa
                ],
                anno_path=[
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210418_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210420_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210421_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210422_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210423_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210424_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210425_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210426_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210427_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210428_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210429_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210430_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210502_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210506_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210507_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210508_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210509_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210510_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210511_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210512_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210513_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210514_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210517_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210518_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210519_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210520_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210521_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210523_v04__no_front__.anno.pb_rec",  # noqa
                ],
                sample_weight=9,
            ),
            dict(
                # 30C1G
                # 731340
                rec_path=[
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210423_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210425_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210426_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210427_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210428_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210429_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210430_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210501_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210506_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210507_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210508_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210509_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210510_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210511_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210512_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210513_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210514_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210515_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210517_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210518_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210519_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210520_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210521_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210522_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210523_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210524_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210525_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210527_v04__no_front__.rec",  # noqa
                ],
                anno_path=[
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210423_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210425_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210426_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210427_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210428_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210429_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210430_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210501_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210506_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210507_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210508_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210509_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210510_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210511_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210512_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210513_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210514_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210515_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210517_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210518_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210519_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210520_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210521_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210522_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210523_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210524_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210525_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210527_v04__no_front__.anno.pb_rec",  # noqa
                ],
                sample_weight=7,
            ),
        ],
        val_batch_size_per_ctx=10,
        val_data_paths=[
            dict(
                rec_path=[
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210503_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210504_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210505_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210515_v04__no_front__.rec",  # noqa
                ],
                anno_path=[
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210503_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210504_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210505_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210515_v04__no_front__.anno.pb_rec",  # noqa
                ],
                sample_weight=5,
            ),
            dict(
                rec_path=[
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210504_v04__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210505_v04__no_front__.rec",  # noqa
                ],
                anno_path=[
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210504_v04__no_front__.anno.pb_rec",  # noqa
                    f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210505_v04__no_front__.anno.pb_rec",  # noqa
                ],
                sample_weight=5,
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
    vehicle_wheel_kps=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/vehicle_kps/vehicle_2_kps/vehicle_2_kps_all_20201021.train.pb_rec",  # noqa
                anno_path=f"{root}/data/vehicle_kps/vehicle_2_kps/aptiv_vehicle_2_kps_all_20201021_min30_consistent.train.pb_rec",  # noqa
                sample_weight=16,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_kps_detection/release_rec/vehicle_2_kps_add0323cutin_20201125.pb_rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_kps_detection/release_rec/vehicle_2_kps_add0323cutin_20201125.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # 0233 白天 137993
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_day_0233/20220119-212808/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_day_0233/20220119-212808/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # 0233 夜晚 62143
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_night_0233/20220120-114207/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_night_0233/20220120-114207/data.anno.pb_rec",  # noqa
                sample_weight=40,
            ),
            # X3C 白天 34092
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_day_X3C/20220119-122143/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_day_X3C/20220119-122143/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # X3C 夜晚 54023
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_night_X3C/20220119-123010/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_night_X3C/20220119-123010/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # X3C 白天 16632
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_day_X3C/20220224-131626/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_day_X3C/20220224-131626/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 白天 51208
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220315-163131/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220315-163131/data.anno.pb_rec",  # noqa
                sample_weight=3,
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
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_day_0233/20220119-221935/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_day_0233/20220119-221935/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # 0233 夜晚 62143
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_night_0233/20220120-115549/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_night_0233/20220120-115549/data.anno.pb_rec",  # noqa
                sample_weight=40,
            ),
            # X3C 白天 34092
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_day_X3C/20220119-125418/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_day_X3C/20220119-125418/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # X3C 夜晚 54023
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_night_X3C/20220119-141510/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_night_X3C/20220119-141510/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # X3C 白天 16632
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_day_X3C/20220224-133404/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_day_X3C/20220224-133404/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 白天 51208
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220315-163208/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220315-163208/data.anno.pb_rec",  # noqa
                sample_weight=3,
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
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_wheel_detection/rear_view_data/MCP_wheel_detection_0528/train.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_wheel_detection/rear_view_data/MCP_wheel_detection_0528/train.anno.pb_rec",  # noqa
                sample_weight=85,
            ),
        ],
        val_batch_size_per_ctx=16,
        val_data_paths=None,
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
            # CW_A23GB_A100_L cyclist lighted badcase night
            dict(
                rec_path=f"{root}/data/cyclist_detection/majiajia.data/cyc_light_CW_A23GB_A100_L/pilot_cyclist_packing_data_publish/20220330-211259/data.rec",  # noqa
                anno_path=f"{root}/data/cyclist_detection/majiajia.data/cyc_light_CW_A23GB_A100_L/pilot_cyclist_packing_data_publish/20220330-211259/data.anno.pb_rec",  # noqa
                sample_weight=40,
            ),
            # mcp night weisen:
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/example_pack_densebox_cyclist_data_20220318_023053_0233_night/20220318-041101/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/example_pack_densebox_cyclist_data_20220318_023053_0233_night/20220318-041101/data.anno.pb_rec",  # noqa
                sample_weight=40,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/example_cyclist_pack_train/20220414-181959/data.rec",
                anno_path=f"{root}/multicam_pilot/data/tmp/example_cyclist_pack_train/20220414-181959/data.anno.pb_rec",
                sample_weight=8,
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
    traffic_light=dict(
        train_batch_size_per_ctx=8,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/traffic_light_detection/cn/densebox_traffic_light_color.train.lx.tl.quad1080.cn.2.0.rec",  # noqa
                rec_path_md5sum="4eb126e52ff9a3897d39b92f7dfa874f",
                anno_path=f"{root}/data/traffic_light_detection/cn/densebox_traffic_light_color.train.lx.tl.quad1080.cn.2.0.anno.pb_rec",  # noqa
                anno_path_md5sum="ef6b9462a78a6d526bd267c296a3668c",
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_detection/cn/densebox_traffic_light_color.train.lx.tl.quad1080.cn.2.1.rec",  # noqa
                rec_path_md5sum="968811464fdfbcecc6e5bffca4e24759",
                anno_path=f"{root}/data/traffic_light_detection/cn/densebox_traffic_light_color.train.lx.tl.quad1080.cn.2.1.anno.pb_rec",  # noqa
                anno_path_md5sum="9a96db1f911cf95b52f3bf3549a42176",
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/mono_0323_data/user/zhaoxu01.li/tl/densebox_traffic_light_color.train.zx.tl.cn.0303.addid400.withus1080p.horizonshell.filter.all.black.rec",  # noqa
                anno_path=f"{root}/mono_0323_data/user/zhaoxu01.li/tl/densebox_traffic_light_color.train.zx.tl.cn.0303.addid400.withus1080p.horizonshell.filter.all.black.anno.pb_rec",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/mono_0323_data/user/zhaoxu01.li/tl/densebox_traffic_light_color.train.zx.tl.cn.1112.1080p.add.Geely0323.nopart9.2280x1080.shortlen8.rec",  # noqa
                anno_path=f"{root}/mono_0323_data/user/zhaoxu01.li/tl/densebox_traffic_light_color.train.zx.tl.cn.1112.1080p.add.Geely0323.nopart9.2280x1080.shortlen8.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/mono_0323_data/user/zhaoxu01.li/tl/densebox_traffic_light_color.train.zx.tl.cn.0810.cn.1080p.qud.filter.all.black.rec",  # noqa
                anno_path=f"{root}/mono_0323_data/user/zhaoxu01.li/tl/densebox_traffic_light_color.train.zx.tl.cn.0810.cn.1080p.qud.filter.all.black.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    traffic_cone=dict(
        train_batch_size_per_ctx=2,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/traffic_cone_detection/matrix2.1_V3.0/CN/det_cone_v1-CN-qx1-train/data.rec",  # noqa
                rec_path_md5sum="c6ca567f3dec7760109b0f41d8fe22a1",
                anno_path=f"{root}/data/traffic_cone_detection/matrix2.1_V3.0/CN/det_cone_v1-CN-qx1-train/data.anno.pb_rec",  # noqa
                anno_path_md5sum="78744d779992bd937de7dc94fe3f07ff",
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/data/traffic_cone_detection/matrix2.1_V3.0/CN/det_cone_v1-us_merge_data_311-420-train/data.rec",  # noqa
                rec_path_md5sum="2be4f48a965ee6cd2708d61d61b93988",
                anno_path=f"{root}/data/traffic_cone_detection/matrix2.1_V3.0/CN/det_cone_v1-us_merge_data_311-420-train/data.anno.pb_rec",  # noqa
                anno_path_md5sum="78a38af9befc4eca33ec31b0175f53c1",
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/data/traffic_cone_detection/matrix2.1_V3.0/CN/det_cone_v2-CN_390_wcg-train/data.rec",  # noqa
                rec_path_md5sum="2fe7e474a73976ae32a11af03f550e5d",
                anno_path=f"{root}/data/traffic_cone_detection/matrix2.1_V3.0/CN/det_cone_v2-CN_390_wcg-train/data.anno.pb_rec",  # noqa
                anno_path_md5sum="13816b7a33baa0723cfd2b101eb2ff6e",
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/data/traffic_cone_detection/matrix2.1_V3.0/CN/det_cone_v2-CN_390_wcg_6_29-train/data.rec",  # noqa
                rec_path_md5sum="38a71a1467c62f1a14c0fdeb45ad4ec9",
                anno_path=f"{root}/data/traffic_cone_detection/matrix2.1_V3.0/CN/det_cone_v2-CN_390_wcg_6_29-train/data.anno.pb_rec",  # noqa
                anno_path_md5sum="f90242483ce70a1eff6f5b0bffec7984",
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=2,
        val_data_paths=[
            dict(
                rec_path=f"{root}/data/traffic_cone_detection/matrix2.1_V3.0/CN/det_cone_v1-CN-qx1-val/data.rec",  # noqa
                rec_path_md5sum="c3850c0aaabc8e9a0bdd97317994eb7c",
                anno_path=f"{root}/data/traffic_cone_detection/matrix2.1_V3.0/CN/det_cone_v1-CN-qx1-val/data.anno.pb_rec",  # noqa
                anno_path_md5sum="a9d52c82d6d3864659b16392acdfcd57",
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/data/traffic_cone_detection/matrix2.1_V3.0/CN/det_cone_v1-us_merge_data_311-420-val/data.rec",  # noqa
                rec_path_md5sum="4d8db14b779e8b653881bf23baa4eec6",
                anno_path=f"{root}/data/traffic_cone_detection/matrix2.1_V3.0/CN/det_cone_v1-us_merge_data_311-420-val/data.anno.pb_rec",  # noqa
                anno_path_md5sum="64ac285bb59ab559b1c63fcb4ca24ae8",
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/data/traffic_cone_detection/matrix2.1_V3.0/CN/det_cone_v2-CN_390_wcg-val/data.rec",  # noqa
                rec_path_md5sum="b53bce30842245535a50469188eeb17f",
                anno_path=f"{root}/data/traffic_cone_detection/matrix2.1_V3.0/CN/det_cone_v2-CN_390_wcg-val/data.anno.pb_rec",  # noqa
                anno_path_md5sum="048c265bea62cfdd7942bddc078d2c05",
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/data/traffic_cone_detection/matrix2.1_V3.0/CN/det_cone_v2-CN_390_wcg_6_29-val/data.rec",  # noqa
                rec_path_md5sum="dbe11599418de98a36ce35591f0898f1",
                anno_path=f"{root}/data/traffic_cone_detection/matrix2.1_V3.0/CN/det_cone_v2-CN_390_wcg_6_29-val/data.anno.pb_rec",  # noqa
                anno_path_md5sum="a06e414acf1d6c14ad8a834907ae8806",
                sample_weight=1,
            ),
        ],
    ),
    person=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            # mcp night weisen:
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/example_pack_densebox_person_data_20220318_023028_v2.2_0233_night/20220318-043106/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/example_pack_densebox_person_data_20220318_023028_v2.2_0233_night/20220318-043106/data.anno.pb_rec",  # noqa
                sample_weight=40,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/example_person_pack_train/20220414-182707/data.rec",
                anno_path=f"{root}/multicam_pilot/data/tmp/example_person_pack_train/20220414-182707/data.anno.pb_rec",
                sample_weight=8,
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
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection_tmp/pilot3.0_0233_data_before_0406_bigtruck_new/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection_tmp/pilot3.0_0233_data_before_0406_bigtruck_new/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_rear_local/20210827-205314/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_rear_local/20210827-205314/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # 韦森 9800 badcase
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_badcase/20211222-121533/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_badcase/20211222-121533/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # 0233 夜晚 全量数据 222100
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_5v_night_0233/20220119-002848/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_5v_night_0233/20220119-002848/data.anno.pb_rec",  # noqa
                sample_weight=60,
            ),
            # CW_A23GB_A100_L DarkNight 18802
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DARK_NIGHT_night/20220316-115110/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DARK_NIGHT_night/20220316-115110/data.anno.pb_rec",  # noqa
                sample_weight=2,
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
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_day_0233/20220119-012241/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_day_0233/20220119-012241/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # 0233 夜晚 81783
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_night_0233/20220118-233709/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_night_0233/20220118-233709/data.anno.pb_rec",  # noqa
                sample_weight=40,
            ),
            # X3C 白天 84312
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_day_X3C/20220119-111048/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_day_X3C/20220119-111048/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # X3C 夜晚 24160
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_night_X3C/20220119-105348/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_night_X3C/20220119-105348/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # X3C 白天 12906
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_day_X3C/20220224-113552/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_day_X3C/20220224-113552/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 夜晚 22000
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_night_X3C/20220224-115939/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_night_X3C/20220224-115939/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # X3C 白天 15688
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_CAMERA_X3C_day/20220315-163248/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_CAMERA_X3C_day/20220315-163248/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 夜晚 7612
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_CAMERA_X3C_night/20220315-164230/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_CAMERA_X3C_night/20220315-164230/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # X3C 白天 13505
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_CAMERA_X3C_day/20220401-140046/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_CAMERA_X3C_day/20220401-140046/data.anno.pb_rec",  # noqa
                sample_weight=2,
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
    traffic_light_len=dict(
        train_batch_size_per_ctx=4,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/traffic_light_lens_detection/cn/densebox_traffic_light_color.train.us.tl.day.lens.3.0.rec",  # noqa
                rec_path_md5sum="b540a3de4b46ed9d77483435f4c3276b",
                anno_path=f"{root}/data/traffic_light_lens_detection/cn/densebox_traffic_light_color.train.us.tl.day.lens.3.0.anno.pb_rec",  # noqa
                anno_path_md5sum="45dfd0bc6f1c9dabbf57a3dbf0c67993",
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_lens_detection/cn/densebox_traffic_light_color.train.us.tl.night.lens.2.0.rec",  # noqa
                rec_path_md5sum="cf45a8aa6458617b045c000aaed371f6",
                anno_path=f"{root}/data/traffic_light_lens_detection/cn/densebox_traffic_light_color.train.us.tl.night.lens.2.0.anno.pb_rec",  # noqa
                anno_path_md5sum="91ad1cc3aecbb3fbebbe02f136c7a4c4",
                sample_weight=2,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    traffic_sign=dict(
        train_batch_size_per_ctx=4,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/traffic_sign_detection/cn/quad_720p_pw_indicationcircle_pos_addotherunknown_20200626/train_cnts_720p.rec",  # noqa
                rec_path_md5sum="c8469ba94c4211f4f6a1217877862848",
                anno_path=f"{root}/data/traffic_sign_detection/cn/quad_720p_pw_indicationcircle_pos_addotherunknown_20200626/train_cnts_720p.anno.pb_rec",  # noqa
                anno_path_md5sum="05459ddafb6f0b03b90ceca29afe6046",
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/data/traffic_sign_detection/cn/quad_1080p_pw_indicationcircle_pos_addotherunknown_20200626/train_cnts_1080p.rec",  # noqa
                rec_path_md5sum="28f45e3a7885318f547e2a9d82a5cdf9",
                anno_path=f"{root}/data/traffic_sign_detection/cn/quad_1080p_pw_indicationcircle_pos_addotherunknown_20200626/train_cnts_1080p.anno.pb_rec",  # noqa
                anno_path_md5sum="0c246b494542c6a816000f6af74a39cb",
                sample_weight=2,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    vehicle_truncation_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_truncation_classification/pilot3.0_0233_data_before_0406_vehicle_truncation_cls/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_truncation_classification/pilot3.0_0233_data_before_0406_vehicle_truncation_cls/data.anno.pb_rec",  # noqa
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
                rec_path=f"{root}/data/default_parsing/train_data/mono3.0/16.2c/data_all_0220model_2020-09-30/parsing-1080p-train-anno16.2_26550.rec",  # noqa
                anno_path=f"{root}/data/default_parsing/train_data/mono3.0/16.2c/data_all_0220model_2020-09-30/parsing-1080p-train-anno16.2_26550.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_8298_20210610_anno16.2.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_8298_20210610_anno16.2.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_2446_20210610_anno16.2.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_2446_20210610_anno16.2.anno.pb_rec",  # noqa
                sample_weight=1.5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_7538_20210528_anno16.2.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_7538_20210528_anno16.2.anno.pb_rec",  # noqa
                sample_weight=3.5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_2225_20210528_anno16.2.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_2225_20210528_anno16.2.anno.pb_rec",  # noqa
                sample_weight=1.5,
            ),
            # night
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_1767_20210704_anno16.2.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_1767_20210704_anno16.2.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_1898_20210710_anno16.2.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_1898_20210710_anno16.2.anno.pb_rec",  # noqa
                sample_weight=1,
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
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/matrix2.x_1080x1920_train_pinhole_num_54968_0920_5classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/matrix2.x_1080x1920_train_pinhole_num_54968_0920_5classes.anno.pb_rec",  # noqa
                sample_weight=12,
            ),
            dict(
                rec_path=f"{root}/mono_0323_data/user/ling.yao/anno_mono_5cls_onlywide_0323_20210218/train.rec",  # noqa
                anno_path=f"{root}/mono_0323_data/user/ling.yao/anno_mono_5cls_onlywide_0323_20210218/train.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/mono_0323_data/user/ling.yao/anno_mono_5cls_wide_0323_20210218/train.rec",  # noqa
                anno_path=f"{root}/mono_0323_data/user/ling.yao/anno_mono_5cls_wide_0323_20210218/train.anno.pb_rec",  # noqa
                sample_weight=7,
            ),
            dict(
                rec_path=f"{root}/mono_0323_data/user/ling.yao/anno_mono_5cls_wide_720p_201223/train.rec",  # noqa
                anno_path=f"{root}/mono_0323_data/user/ling.yao/anno_mono_5cls_wide_720p_201223/train.anno.pb_rec",  # noqa
                sample_weight=7,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_18772_20210420_5classes_new.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_18772_20210420_5classes_new.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_3500_20210420_5classes_new.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_3500_20210420_5classes_new.anno.pb_rec",  # noqa
                sample_weight=1.5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_8571_20210524_5classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_8571_20210524_5classes.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_3902_20210524_5classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_3902_20210524_5classes.anno.pb_rec",  # noqa
                sample_weight=1.5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_24305_20210711_5classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_24305_20210711_5classes.anno.pb_rec",  # noqa
                sample_weight=12,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_6291_20210711_5classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_6291_20210711_5classes.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_1899_20210711_5classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_1899_20210711_5classes.anno.pb_rec",  # noqa
                sample_weight=0.5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_17311_20210711_5classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_17311_20210711_5classes.anno.pb_rec",  # noqa
                sample_weight=8,
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
)

datapaths = EasyDict(datapaths)
buckets = ["matrix2"]
