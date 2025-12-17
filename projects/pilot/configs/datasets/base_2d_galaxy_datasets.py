import os

from easydict import EasyDict

from hat.utils.logger import rank_zero_info

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
root = os.path.join(bucket_root, "matrix2")


def append_data_not_exists_in_galaxy(
    input_datapaths, output_datapaths, weight_append_ratio
):
    """
    1. Only process 2d data.
    2. Append data exists in input_datapaths and not exists in output_dataths to
    output_datapths;
    3. weight_append_ratio: if value of key <=0, the data of the key will not be appended.
    4. Format:
    weight_append_ratio: dict(
        class: ratio
    )
    """
    for key in output_datapaths:
        assert key in input_datapaths, f"{key} not in input_datapaths"
        assert key in weight_append_ratio
        ratio = weight_append_ratio[key]
        if ratio <= 0:
            rank_zero_info(f"skip all class {key} data")
            continue
        input_train_paths = input_datapaths[key].train_data_paths
        output_train_paths = output_datapaths[key].train_data_paths
        for data_path in input_train_paths:
            anno_path = data_path.anno_path
            if anno_path not in [
                data_path.anno_path for data_path in output_train_paths
            ]:
                data_path.sample_weight *= ratio
                rank_zero_info(
                    f"append and reset weight to {data_path.sample_weight}: {anno_path}"
                )
                output_datapaths[key].train_data_paths.append(data_path)


# pilot3.0 base 2d data
pilot3_day_datapaths = dict(
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
    plate=dict(
        train_batch_size_per_ctx=24,
        train_data_paths=[
            # pack cond
            # ignore condition: ignore points >= 1
            # hard condition: w <= 8, h <= 6
            # 0820 1.4W
            dict(
                rec_path=f"{root}/users/jinchuan01.xiao/data/plate/0820/v220305/4pe/train.rec",  # noqa
                anno_path=f"{root}/users/jinchuan01.xiao/data/plate/0820/v220305/4pe/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # unknow 3.8W
            dict(
                rec_path=f"{root}/users/jinchuan01.xiao/data/plate/unknow/v220305/4pe/train.rec",  # noqa
                anno_path=f"{root}/users/jinchuan01.xiao/data/plate/unknow/v220305/4pe/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # wissen 2.8W
            dict(
                rec_path=f"{root}/users/jinchuan01.xiao/data/plate/wissen/v220305/4pe/train.rec",  # noqa
                anno_path=f"{root}/users/jinchuan01.xiao/data/plate/wissen/v220305/4pe/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # x3c 3.7W
            dict(
                rec_path=f"{root}/users/jinchuan01.xiao/data/plate/x3c/v220305/4pe/train.rec",  # noqa
                anno_path=f"{root}/users/jinchuan01.xiao/data/plate/x3c/v220305/4pe/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/users/jinchuan01.xiao/data/plate/plate_v211108/train/train.rec",  # noqa
                anno_path=f"{root}/users/jinchuan01.xiao/data/plate/plate_v211108/train/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
    ),
    face=dict(
        train_batch_size_per_ctx=24,
        train_data_paths=[
            # 上传路径有误，实际上是face
            # dict(
            #     rec_path=f"{root}/users/yuanzhuo.peng/data/plate/x3c/v220505/4pe/train.rec",
            #     anno_path=f"{root}/users/yuanzhuo.peng/data/plate/x3c/v220505/4pe/train.json",
            #     sample_weight=1,
            # ),
            # dict(
            #     rec_path=f"{root}/users/yuanzhuo.peng/data/plate/unknow/v220505/4pe/train.rec",
            #     anno_path=f"{root}/users/yuanzhuo.peng/data/plate/unknow/v220505/4pe/train.json",
            #     sample_weight=1,
            # ),
            # dict(
            #     rec_path=f"{root}/users/yuanzhuo.peng/data/plate/0233/v220505/4pe/train.rec",
            #     anno_path=f"{root}/users/yuanzhuo.peng/data/plate/0233/v220505/4pe/train.json",
            #     sample_weight=1,
            # ),
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_4pe/20220715-201117/data.rec",
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_4pe/20220715-201117/data.anno.pb_rec",
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_4pe/20220715-184010/data.rec",
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_4pe/20220715-184010/data.anno.pb_rec",
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_4pe/20220715-162048/data.rec",
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_4pe/20220715-162048/data.anno.pb_rec",
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_4pe/20220715-181100/data.rec",
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_4pe/20220715-181100/data.anno.pb_rec",
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_4pe/20220726-155303/data.rec",
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_4pe/20220726-155303/data.anno.pb_rec",
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_4pe/20220726-143048/data.rec",
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_4pe/20220726-143048/data.anno.pb_rec",
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_as33-5v_all_4pe/20220726-143247/data.rec",
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_as33-5v_all_4pe/20220726-143247/data.anno.pb_rec",
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_cc02-5v_all_4pe/20220726-144014/data.rec",
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_cc02-5v_all_4pe/20220726-144014/data.anno.pb_rec",
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_maxfa-5v_all_4pe/20220726-151851/data.rec",
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_person_face_5v_maxfa-5v_all_4pe/20220726-151851/data.anno.pb_rec",
                sample_weight=1,
            ),
        ],
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
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/2pe_person_pose_classification/Pilot_person_pose_5v_CAMERA_CW_A23GB_A114_L_day/20220623-163511/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/2pe_person_pose_classification/Pilot_person_pose_5v_CAMERA_CW_A23GB_A114_L_day/20220623-163511/data.anno.pb_rec",  # noqa
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
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_senyun_day/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_senyun_day/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_weissen_day/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_weissen_day/data.anno.pb_rec",  # noqa
                sample_weight=1,
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
                rec_path=f"{root}/data/cyclist_detection/majiajia.data/CC02_V5.0/Pilot_cyclist_5v_CO-OX3GB-A100-L_day/20220611-151613/data.rec",  # noqa
                anno_path=f"{root}/data/cyclist_detection/majiajia.data/CC02_V5.0/Pilot_cyclist_5v_CO-OX3GB-A100-L_day/20220611-151613/data.anno.pb_rec",  # noqa
                sample_weight=6,
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
                sample_weight=30,
            ),
            # cyc day
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20220510/cyclist/day/20220511-142409/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20220510/cyclist/day/20220511-142409/data.anno.pb_rec",  # noqa
                sample_weight=30,
            ),
            # as33 all 58k
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/cyclist_detection/Pilot_cyclist_5v_CAMERA_CW_A23GB_A114_L_day_5w/20220804-134412/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/cyclist_detection/Pilot_cyclist_5v_CAMERA_CW_A23GB_A114_L_day_5w/20220804-134412/data.anno.pb_rec",  # noqa
                sample_weight=24,
            ),
            # history nontarget
            dict(
                rec_path=f"{root}/data/cyclist_detection/majiajia.data/NonTarget_exp/exp5/Pilot_cyclist_5v__all/20220712-043858/data.rec",  # noqa
                anno_path=f"{root}/data/cyclist_detection/majiajia.data/NonTarget_exp/exp5/Pilot_cyclist_5v__all/20220712-043858/data.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            # c385 v3
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20220907-200731/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20220907-200731/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # c385 v3
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20220907-200731/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20220907-200731/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # veh head and rear
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/cyclist_detection/fp/Veh_Head/Pilot_cyclist_5v_CW-A23GB-A114-L_day_fp/20220926-150141/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/cyclist_detection/fp/Veh_Head/Pilot_cyclist_5v_CW-A23GB-A114-L_day_fp/20220926-150141/data.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            # c385 v4
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221011-224648/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221011-224648/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # c385 v5 2w
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221124-230709/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221124-230709/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # c385 v5 resize4 trigger 9.2k
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221127-200817/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221127-200817/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # c385 v4 jira
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221014-200602/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221014-200602/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # c385 v5 jira 1.3k
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221127-194742/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221127-194742/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # c385 jira 2k
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20230126-144639/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20230126-144639/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # c385 v6 day pillar fp 4K
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20230126-161858/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20230126-161858/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # c385 v6 day normal fp 1W
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20230126-161135/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20230126-161135/data.anno.pb_rec",  # noqa
                sample_weight=10,
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
            # rare pose day
            dict(
                rec_path=f"{root}/data/pedestrain_detection/majiajia.data/rare_pose_exp/Pilot_person_5v_CW-A23GB-A100-L_day/20220602-180009/data.rec",  # noqa
                anno_path=f"{root}/data/pedestrain_detection/majiajia.data/rare_pose_exp/Pilot_person_5v_CW-A23GB-A100-L_day/20220602-180009/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            dict(
                rec_path=f"{root}/data/pedestrain_detection/majiajia.data/rare_pose_exp/Pilot_person_5v_CW-A23GB-A100-L_day/20220602-180009/data.rec",  # noqa
                anno_path=f"{root}/data/pedestrain_detection/majiajia.data/rare_pose_exp/Pilot_person_5v_CW-A23GB-A100-L_day/20220602-180009/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            dict(
                rec_path=f"{root}/data/pedestrain_detection/majiajia.data/CC02_V5/Pilot_person_5v_CO-OX3GB-A100-L_day/20220611-170110/data.rec",  # noqa
                anno_path=f"{root}/data/pedestrain_detection/majiajia.data/CC02_V5/Pilot_person_5v_CO-OX3GB-A100-L_day/20220611-170110/data.anno.pb_rec",  # noqa
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
            # ass33 big person 4.6k
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A100_L_day_big_person/20220620-201707/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A100_L_day_big_person/20220620-201707/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # toll gate 3.4k
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20220214-031030-0233-toll-gate/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20220214-031030-0233-toll-gate/data.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            # tunnel x3c 2k
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20220329-151248-x3c-tunnel/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20220329-151248-x3c-tunnel/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # tunnel 7k
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/tunnel_ped_by_zhongpeng/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/tunnel_ped_by_zhongpeng/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # as33 all 58k
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A114_L_day_5w/20220804-122458/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A114_L_day_5w/20220804-122458/data.anno.pb_rec",  # noqa
                sample_weight=24,
            ),
            # x3c day&night
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20220121-192622-x3c-all/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20220121-192622-x3c-all/data.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            # person day
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20220510/person/day/20220511-142910/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/20220510/person/day/20220511-142910/data.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            # history nontarget
            dict(
                rec_path=f"{root}/data/pedestrain_detection/majiajia.data/NonTarget_exp/exp5/Pilot_person_5v__all/20220708-063813/data.rec",  # noqa
                anno_path=f"{root}/data/pedestrain_detection/majiajia.data/NonTarget_exp/exp5/Pilot_person_5v__all/20220708-063813/data.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            # c385 v3
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20220907-200310/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20220907-200310/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # veh head and rear
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/fp/Veh_Head/Pilot_person_5v_CW-A23GB-A114-L_day_fp/20220926-152912/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/pedestrain_detection/fp/Veh_Head/Pilot_person_5v_CW-A23GB-A114-L_day_fp/20220926-152912/data.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            # c385 v4
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221011-223223/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221011-223223/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # c385 jira
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221014-201823/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221014-201823/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # child history
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/Pilot_person_5v_all_all_Child/20221015-081347/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/Pilot_person_5v_all_all_Child/20221015-081347/data.anno.pb_rec",  # noqa
                sample_weight=6,
            ),
            # c385 v5 2w
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221124-233317/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221124-233317/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # c385 v5 resize4 fp 9.2k
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221127-201047/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221127-201047/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # c385 jira 1.3k fp
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221127-194559/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221127-194559/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # c385 jira 2k fp
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20230126-144504/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20230126-144504/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # c385 jira 700 fn
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20230126-145220/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20230126-145220/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # c385 v6 day pillar fp 4K
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20230126-161618/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20230126-161618/data.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            # c385 v6 day normal fp 1W
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20230126-160449/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20230126-160449/data.anno.pb_rec",  # noqa
                sample_weight=10,
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
            # X3C 3.3w
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_trace/20211129-163640/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_trace/20211129-163640/data.anno.pb_rec",  # noqa
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
            # x3c 白天 45265
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_DG4W_data_5v_CAMERA_CO_OX3GB_D100_L_TIME_DAY_day/20220401-184015/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_DG4W_data_5v_CAMERA_CO_OX3GB_D100_L_TIME_DAY_day/20220401-184015/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # 逆光 7304
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_LIGHT07_data_5v_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220401-164602/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_LIGHT07_data_5v_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220401-164602/data.anno.pb_rec",  # noqa
                sample_weight=2,
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
            # CW-A23GB-A100-L day 5.6k
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/pilot_vehicle_1v_CW-A23GB-A100-L_day/20220518-142747/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/pilot_vehicle_1v_CW-A23GB-A100-L_day/20220518-142747/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # CW-A23GB-A100-L 大车 35940
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_BIG_CAR_DAY_day/20220621-202410/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_BIG_CAR_DAY_day/20220621-202410/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # CO_OX3GB_A100_L 大车 3219
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CO_OX3GB_A100_L_TIME_BIG_CAR_DAY_day/20220621-192421/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CO_OX3GB_A100_L_TIME_BIG_CAR_DAY_day/20220621-192421/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # CW-A23GB-A100-L day 快递车数据
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_day/20220617-131419/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_day/20220617-131419/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # CW_A23GB_A114_L 常规 白天 10842
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day_normal/20220706-134742/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day_normal/20220706-134742/data.anno.pb_rec",
                sample_weight=2,
            ),
            # CW_A23GB_A114_L 逆光 白天 1900
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day/20220621-161421/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day/20220621-161421/data.anno.pb_rec",
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
            # 逆光 5694
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_pilot_pdt_5v_CW-A23GB-A100-L_train_day_glare_Cam_5V/20220729-121524/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_pilot_pdt_5v_CW-A23GB-A100-L_train_day_glare_Cam_5V/20220729-121524/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # maxfa day special car 10259
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_54665_10259_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220707-160146/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_54665_10259_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220707-160146/data.anno.pb_rec",  # noqa
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
            # c385 6802
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20220902-200336/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20220902-200336/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # 环境误检 AS33 白天 1912
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_SCENE-FP_day/20221115-182150/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_SCENE-FP_day/20221115-182150/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # 白天异型车 8010
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_day_special_vehicle_Cam_5V/20220926-103720/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_day_special_vehicle_Cam_5V/20220926-103720/data.anno.pb_rec",  # noqa
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
            # GALAXY 环境误检 白天 1512
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221115-190526/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221115-190526/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # AS33 jira-badcase 白天/夜晚 369
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_JIRA-badcase_all/20221110-192849/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_JIRA-badcase_all/20221110-192849/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # GALAXY 环境误检 白天 531
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221219-120427/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221219-120427/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # maxfa 白天异型车 17352
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CP_OX3GB_D100_L_special_car_day/20230117-180551/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CP_OX3GB_D100_L_special_car_day/20230117-180551/data.anno.pb_rec",
                sample_weight=2,
            ),
            # c385 白天 7166
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO_OX3GB_D100_L_day_all/20230118-115653/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO_OX3GB_D100_L_day_all/20230118-115653/data.anno.pb_rec",
                sample_weight=1,
            ),
            # c385 白天jirabadcase 714
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO_OX3GB_D100_L_jira_badcase_day_all/20230118-114938/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO_OX3GB_D100_L_jira_badcase_day_all/20230118-114938/data.anno.pb_rec",
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
                sample_weight=9,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_3504_20220111_anno16.2_ws_night.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_3504_20220111_anno16.2_ws_night.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # x3c night
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4987_20220111_anno16.2_x3c_night.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4987_20220111_anno16.2_x3c_night.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4300_20220214_anno16.2_x3c_night.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4300_20220214_anno16.2_x3c_night.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4060_20220428_anno16.2_x3c_night.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4060_20220428_anno16.2_x3c_night.anno.pb_rec",  # noqa
                sample_weight=4,
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
            # c385 x3c 176, ID 86907
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_176_anno16.2_c385_day_86907/20221011-011836/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_176_anno16.2_c385_day_86907/20221011-011836/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # c385 x3c 318, ID 86913
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_318_anno16.2_c385_day_86913/20221011-012023/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_318_anno16.2_c385_day_86913/20221011-012023/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # c385 x3c 604, ID 88317
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_604_anno16.2_c385_day_88317/20221011-010935/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_604_anno16.2_c385_day_88317/20221011-010935/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # c385 x3c 618, ID 88356
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_618_anno16.2_c385_day_88356/20221011-010444/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_618_anno16.2_c385_day_88356/20221011-010444/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # c385 x3c 2770, ID 90719
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2770_anno16.2_c385_day_90719/20221011-022443/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2770_anno16.2_c385_day_90719/20221011-022443/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # c385 x3c 1357, 198347,94273
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1357_anno16.2_c385_day_20221123/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1357_anno16.2_c385_day_20221123/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # 137848，133554，133427，144173
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2665_anno16.2_c385_day/20230126-042950/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2665_anno16.2_c385_day/20230126-042950/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # jira 数据，路中人行道，高速围栏，144701，144682
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_145_anno16.2_c385_day/20230126-005237/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_145_anno16.2_c385_day/20230126-005237/data.anno.pb_rec",  # noqa
                sample_weight=2,
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
                sample_weight=0.2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_num_24851_20220328_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_num_24851_20220328_5_2classes.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_num_5356_20220328_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_num_5356_20220328_5_2classes.anno.pb_rec",  # noqa
                sample_weight=0.2,
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
                sample_weight=0.5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_4481_20220525_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_4481_20220525_5_2classes.anno.pb_rec",  # noqa
                sample_weight=2.3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_0233_CW_A23GB_A114L_num_5460_20220525_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_0233_CW_A23GB_A114L_num_5460_20220525_5_2classes.anno.pb_rec",  # noqa
                sample_weight=2.5,
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
                sample_weight=0.5,
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
                sample_weight=2.0,
            ),
            # day rear 650
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_005334/pilot_lane_parsing_annoset/20221012-012546/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_005334/pilot_lane_parsing_annoset/20221012-012546/data.anno.pb_rec",  # noqa
                sample_weight=2.0,
            ),
            # night side 1427
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_012136/pilot_lane_parsing_annoset/20221012-015617/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_012136/pilot_lane_parsing_annoset/20221012-015617/data.anno.pb_rec",  # noqa
                sample_weight=1.0,
            ),
            # night rear 874
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_014836/pilot_lane_parsing_annoset/20221012-022211/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_014836/pilot_lane_parsing_annoset/20221012-022211/data.anno.pb_rec",  # noqa
                sample_weight=1.0,
            ),
            # day side 4325
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_123750/pilot_lane_parsing_annoset/20221126-132031/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_123750/pilot_lane_parsing_annoset/20221126-132031/data.anno.pb_rec",  # noqa
                sample_weight=2.0,
            ),
            # day rear 2737
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_125221/pilot_lane_parsing_annoset/20221126-132842/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_125221/pilot_lane_parsing_annoset/20221126-132842/data.anno.pb_rec",  # noqa
                sample_weight=2.0,
            ),
            # night side 2013
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_130644/pilot_lane_parsing_annoset/20221126-133734/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_130644/pilot_lane_parsing_annoset/20221126-133734/data.anno.pb_rec",  # noqa
                sample_weight=1.0,
            ),
            # v6.0  day
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20230128_224418_local/pilot_lane_parsing_annoset/20230128-224930/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20230128_224418_local/pilot_lane_parsing_annoset/20230128-224930/data.anno.pb_rec",  # noqa
                sample_weight=2.0,
            ),
            # v6.0  night
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20230128_234420_local/pilot_lane_parsing_annoset/20230128-234745/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20230128_234420_local/pilot_lane_parsing_annoset/20230128-234745/data.anno.pb_rec",  # noqa
                sample_weight=0.5,
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
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_0233/20220119-212808/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_0233/20220119-212808/data.anno.pb_rec",  # noqa
                sample_weight=50,
            ),
            # X3C 白天 34092
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_X3C/20220119-122143/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_X3C/20220119-122143/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # X3C 白天 16632
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_X3C/20220224-131626/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_X3C/20220224-131626/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # X3C 白天 51208
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220315-163131/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220315-163131/data.anno.pb_rec",  # noqa
                sample_weight=6,
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
                sample_weight=50,
            ),
            # X3C 白天 34092
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_X3C/20220119-125418/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_X3C/20220119-125418/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # X3C 白天 16632
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_X3C/20220224-133404/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_X3C/20220224-133404/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # X3C 白天 51208
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220315-163208/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220315-163208/data.anno.pb_rec",  # noqa
                sample_weight=6,
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
            # 维森 白天 330660
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_0233/20220119-012241/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_0233/20220119-012241/data.anno.pb_rec",  # noqa
                sample_weight=50,
            ),
            # X3C 白天 12906
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_X3C/20220224-113552/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_X3C/20220224-113552/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # X3C 白天 15688
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_day/20220315-163248/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_day/20220315-163248/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # X3C 白天 13505
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_day/20220401-140046/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_day/20220401-140046/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # AS33 day 24985
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_AS33_day/20220412-220556/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_AS33_day/20220412-220556/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # PDT 大车 day 3205
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_CW_A23GB_A100_L_big_car_day/20220412-212012/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_5v_CW_A23GB_A100_L_big_car_day/20220412-212012/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # PROJECT_CC02_5V 白天 4910
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_day/20220407-195106/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_day/20220407-195106/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # PROJECT_CC02_5V 白天 8526 annoset(16130)
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_day/20220510-214104/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_day/20220510-214104/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # PROJECT_AS33_5V 白天 2113
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW-A23GB-A114-L_day/20220510-103100/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW-A23GB-A114-L_day/20220510-103100/data.anno.pb_rec",  # noqa
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
                sample_weight=50,
            ),
            # X3C 白天 84312
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_X3C/20220119-120507/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_X3C/20220119-120507/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # X3C 白天 12657
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_X3C/20220224-123226/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_X3C/20220224-123226/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # X3C 白天 15083
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_CAMERA_X3C_day/20220316-102102/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_CAMERA_X3C_day/20220316-102102/data.anno.pb_rec",  # noqa
                sample_weight=2,
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
                sample_weight=50,
            ),
            # X3C 白天 67690
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_X3C/20220119-120509/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_X3C/20220119-120509/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # X3C 白天 15788
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_X3C/20220224-130525/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_X3C/20220224-130525/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # X3C 白天 12323
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_CAMERA_X3C_day/20220315-163229/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_CAMERA_X3C_day/20220315-163229/data.anno.pb_rec",  # noqa
                sample_weight=2,
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

pilot3_night_datapaths = dict(
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
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CP_OX3GB_D100_L_special_car_day/20230117-180551/data.rec",
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CP_OX3GB_D100_L_special_car_day/20230117-180551/data.anno.pb_rec",
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

# galaxy base 2d data
galaxy_datapaths = dict(
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
            # rear day 5961
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedOcclusion/kPedOcclusion_project_galaxy_5v_train_day_Cam_Rear/20220622-171219/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedOcclusion/kPedOcclusion_project_galaxy_5v_train_day_Cam_Rear/20220622-171219/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # side night 3413
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedOcclusion/kPedOcclusion_project_galaxy_5v_train_night_Cam_Side/20220622-171224/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedOcclusion/kPedOcclusion_project_galaxy_5v_train_night_Cam_Side/20220622-171224/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # side day 5689
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedOcclusion/kPedOcclusion_project_galaxy_5v_train_day_Cam_Side/20220622-171019/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedOcclusion/kPedOcclusion_project_galaxy_5v_train_day_Cam_Side/20220622-171019/data.anno.pb_rec",  # noqa
                sample_weight=1,
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
            # rear day 5672
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehOcclusion/kVehOcclusion_project_galaxy_5v_train_day_Cam_Rear/20220622-171638/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehOcclusion/kVehOcclusion_project_galaxy_5v_train_day_Cam_Rear/20220622-171638/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # side night 2618
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehOcclusion/kVehOcclusion_project_galaxy_5v_train_night_Cam_Side/20220622-171803/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehOcclusion/kVehOcclusion_project_galaxy_5v_train_night_Cam_Side/20220622-171803/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # side day 6439
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehOcclusion/kVehOcclusion_project_galaxy_5v_train_day_Cam_Side/20220622-171251/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehOcclusion/kVehOcclusion_project_galaxy_5v_train_day_Cam_Side/20220622-171251/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    cyclist_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            # galaxy 白天  68503
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/Pilot_cyclist_category_5v_CAMERA_CW_A23GB_A114_L_DAY_day/20220926-105914/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/Pilot_cyclist_category_5v_CAMERA_CW_A23GB_A114_L_DAY_day/20220926-105914/data.anno.pb_rec",  # noqa
                sample_weight=7,
            ),
            # galaxy 夜晚  15028
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/Pilot_cyclist_category_5v_CAMERA_CW_A23GB_A114_L_NIGHT_night/20220926-122940/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/Pilot_cyclist_category_5v_CAMERA_CW_A23GB_A114_L_NIGHT_night/20220926-122940/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # as33 白天  120926
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CW_A23GB_A114_L_DAY_day/20221012-135628/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CW_A23GB_A114_L_DAY_day/20221012-135628/data.anno.pb_rec",  # noqa
                sample_weight=13,
            ),
            # as33 夜晚  59137
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CW_A23GB_A114_L_NIGHT_night/20221012-174742/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CW_A23GB_A114_L_NIGHT_night/20221012-174742/data.anno.pb_rec",  # noqa
                sample_weight=6,
            ),
            # c385 白天  15749
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CO-OX3GB-D100-L_DAY_day/20221012-185013/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CO-OX3GB-D100-L_DAY_day/20221012-185013/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # c385 夜晚  4400
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CO-OX3GB-D100-L_NIGHT_night/20221012-195248/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CO-OX3GB-D100-L_NIGHT_night/20221012-195248/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # maxfa 白天  20121
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CP-OX3GB-D100-L_DAY_day/20221012-211105/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CP-OX3GB-D100-L_DAY_day/20221012-211105/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # maxfa 夜晚  5561
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CP-OX3GB-D100-L_NIGHT_night/20221013-012831/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CP-OX3GB-D100-L_NIGHT_night/20221013-012831/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # cc02 白天  20121
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CO-OX3GB-A100-L_DAY_day/20221013-025417/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CO-OX3GB-A100-L_DAY_day/20221013-025417/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # cc02 夜晚  27809
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CO-OX3GB-A100-L_NIGHT_night/20221013-022108/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CO-OX3GB-A100-L_NIGHT_night/20221013-022108/data.anno.pb_rec",  # noqa
                sample_weight=3,
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
                rec_path=f"{root}/multicam_pilot/data/train/person/2pe_cyclist_kps/cyc_kps_zhengwei/20220121-132406/data.pb_rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/2pe_cyclist_kps/cyc_kps_zhengwei/20220121-132406/data.anno.pb_rec",  # noqa
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
                sample_weight=32,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/2pe_person_pose_classification/example_pack_densebox_ped_pose_data_20220121_122405_0233_all/20220121-140616/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/2pe_person_pose_classification/example_pack_densebox_ped_pose_data_20220121_122405_0233_all/20220121-140616/data.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            # rear day 5961
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedPose/kPedPose_project_galaxy_5v_train_day_Cam_Rear/20220622-171352/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedPose/kPedPose_project_galaxy_5v_train_day_Cam_Rear/20220622-171352/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # side night 3413
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedPose/kPedPose_project_galaxy_5v_train_night_Cam_Side/20220622-171245/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedPose/kPedPose_project_galaxy_5v_train_night_Cam_Side/20220622-171245/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # side day 5689
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedPose/kPedPose_project_galaxy_5v_train_day_Cam_Side/20220622-171024/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedPose/kPedPose_project_galaxy_5v_train_day_Cam_Side/20220622-171024/data.anno.pb_rec",  # noqa
                sample_weight=1,
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
    vehicle_category=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_senyun_day/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_senyun_day/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_weissen_day/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_weissen_day/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # rear day 5672
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehCategory/kVehCategory_project_galaxy_5v_train_day_Cam_Rear/20220622-171442/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehCategory/kVehCategory_project_galaxy_5v_train_day_Cam_Rear/20220622-171442/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # side night 2168
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehCategory/kVehCategory_project_galaxy_5v_train_night_Cam_Side/20220622-171219/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehCategory/kVehCategory_project_galaxy_5v_train_night_Cam_Side/20220622-171219/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # side day 6439
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehCategory/kVehCategory_project_galaxy_5v_train_day_Cam_Side/20220622-171417/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehCategory/kVehCategory_project_galaxy_5v_train_day_Cam_Side/20220622-171417/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # side day 4038
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehCategory/kVehCategory_project_galaxy_5v_train_day_Cam_Side/20221209-173155/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehCategory/kVehCategory_project_galaxy_5v_train_day_Cam_Side/20221209-173155/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # side night 1216
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehCategory/kVehCategory_project_galaxy_5v_train_night_Cam_Side/20221209-172748/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehCategory/kVehCategory_project_galaxy_5v_train_night_Cam_Side/20221209-172748/data.anno.pb_rec",  # noqa
                sample_weight=1,
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
            # side night 2797
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedOrientation/kPedOrientation_project_galaxy_5v_train_night_Cam_Side/20220622-171155/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedOrientation/kPedOrientation_project_galaxy_5v_train_night_Cam_Side/20220622-171155/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    cyclist=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            # galaxy day rear annoset(25942) 2660
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220622-145223/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220622-145223/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night rear annoset(25946) 1352
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220622-145424/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220622-145424/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day side annoset(25948) 4404
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20220622-150123/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20220622-150123/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night side annoset(25949) 3413
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Side/20220622-145741/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Side/20220622-145741/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day rear annoset(39526) 12920
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220719-115017/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220719-115017/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy day side annoset(39452) 5169
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20220719-102656/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20220719-102656/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day rear annoset(47652) 12377
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220806-002734/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220806-002734/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy day side annoset(47643) 11171
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20220805-233757/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20220805-233757/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy day side annoset(60042) 11319
            dict(
                rec_path=f"{root}/multicam_pilot/data/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20220928-170533/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20220928-170533/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy night side annoset(60023) 6508
            dict(
                rec_path=f"{root}/multicam_pilot/data/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Side/kCycBBox2D_project_galaxy_5v_train_night_Cam_Side/20220928-155907/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Side/kCycBBox2D_project_galaxy_5v_train_night_Cam_Side/20220928-155907/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day rear annoset(60023) 13343
            dict(
                rec_path=f"{root}/multicam_pilot/data/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220928-172750/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220928-172750/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy night rear annoset(60026) 5861
            dict(
                rec_path=f"{root}/multicam_pilot/data/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Rear/kCycBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220928-160112/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Rear/kCycBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220928-160112/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy day side annoset(66848) 10911
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20221028-222207/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20221028-222207/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy day rear annoset(66875) 10191
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221028-235854/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221028-235854/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy night side annoset(66873) 5549
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Side/20221028-231600/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Side/20221028-231600/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy night rear annoset(66874) 3562
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221028-232324/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221028-232324/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night rear glare
            dict(
                rec_path=f"{root}/data/cyclist_detection/jianhang.he.data/Cutin_exp/debug/Pilot_cyclist_5v_all_night_cutin/20220928-194559/data.rec",  # noqa
                anno_path=f"{root}/data/cyclist_detection/jianhang.he.data/Cutin_exp/debug/Pilot_cyclist_5v_all_night_cutin/20220928-194559/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night rear glare
            dict(
                rec_path=f"{root}/data/cyclist_detection/jianhang.he.data/Cutin_exp/debug/Pilot_cyclist_5v_all_night_cutin/20220928-200950/data.rec",  # noqa
                anno_path=f"{root}/data/cyclist_detection/jianhang.he.data/Cutin_exp/debug/Pilot_cyclist_5v_all_night_cutin/20220928-200950/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy jira annoset(66877) 340
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_JIRA_badcase_Cam_All/20221028-230453/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_JIRA_badcase_Cam_All/20221028-230453/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy underground rear annoset(72627) 657
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_all_Cam_Rear/20221119-132909/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_all_Cam_Rear/20221119-132909/data.anno.pb_rec",  # noqa
                sample_weight=0.25,
            ),
            # galaxy underground side annoset(72632) 431
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_all_Cam_Side/20221119-134819/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_all_Cam_Side/20221119-134819/data.anno.pb_rec",  # noqa
                sample_weight=0.25,
            ),
            # galaxy jira day annoset(93443) 330
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_JIRA_badcase_Cam_All/20230221-104817/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_JIRA_badcase_Cam_All/20230221-104817/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy jira night annoset(72638)
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_JIRA_badcase_Cam_All/20221119-143020/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_JIRA_badcase_Cam_All/20221119-143020/data.anno.pb_rec",  # noqa
                sample_weight=1.5,
            ),
            # galaxy day side annoset(78809) 8833
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_all_Cam_Side/20221212-133412/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_all_Cam_Side/20221212-133412/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day rear annoset(78810) 4634
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_all_Cam_Rear/20221212-134154/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_all_Cam_Rear/20221212-134154/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy glare jira annoset(88777) 3863
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_CO-OX3GB-D100-L_train_night_jira_Cam_5V/20230128-143617/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_CO-OX3GB-D100-L_train_night_jira_Cam_5V/20230128-143617/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # galaxy night rear; glare; fp; hand select; annoset(89129) 2329
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_train_night_Cam_Rear/20230130-211816/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_train_night_Cam_Rear/20230130-211816/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # galaxy night rear; glare; no prediction; hand select; annoset(89149) 7893
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_train_night_Cam_Rear/20230130-221935/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_train_night_Cam_Rear/20230130-221935/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # throw away dataset
            # cyclist lighted badcase night
            dict(
                rec_path=f"{root}/data/cyclist_detection/majiajia.data/MaxFaV10.0/Cyclist_lighted_night/Pilot_cyclist_5v_MaxFaV10_Cyclist_Lighted_Night_night/20220425-182351/data.rec",  # noqa
                anno_path=f"{root}/data/cyclist_detection/majiajia.data/MaxFaV10.0/Cyclist_lighted_night/Pilot_cyclist_5v_MaxFaV10_Cyclist_Lighted_Night_night/20220425-182351/data.anno.pb_rec",  # noqa
                sample_weight=0,
            ),
            # cyclist lighted badcase night
            dict(
                rec_path=f"{root}/data/cyclist_detection/majiajia.data/cyc_light_CW_A23GB_A100_L/pilot_cyclist_packing_data_publish/20220330-211259/data.rec",  # noqa
                anno_path=f"{root}/data/cyclist_detection/majiajia.data/cyc_light_CW_A23GB_A100_L/pilot_cyclist_packing_data_publish/20220330-211259/data.anno.pb_rec",  # noqa
                sample_weight=0,
            ),
            # cyclist glare
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/cyclist_detection/glare/lighted_cyclist/debug/Pilot_cyclist_5v_all_night_glare/20220923-222554/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/cyclist_detection/glare/lighted_cyclist/debug/Pilot_cyclist_5v_all_night_glare/20220923-222554/data.anno.pb_rec",  # noqa
                sample_weight=0,
            ),
            # cyclist glare
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/person/cyclist_detection/glare/Pilot_cyclist_5v_CAMERA_CW_A23GB_A100_L_Night_night/20220929-145754/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/person/cyclist_detection/glare/Pilot_cyclist_5v_CAMERA_CW_A23GB_A100_L_Night_night/20220929-145754/data.anno.pb_rec",  # noqa
                sample_weight=0,
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
            # galaxy day rear annoset(25943) 2660
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220622-145832/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220622-145832/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night rear annoset(25939) 1352
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220622-145002/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220622-145002/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day side annoset(25951) 4404
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20220622-150324/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20220622-150324/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night side annoset(25515) 3413
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Side/20220622-150103/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Side/20220622-150103/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day rear annoset(39527) 12920
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220719-114353/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220719-114353/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy day side annoset(39450) 5169
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20220719-102053/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20220719-102053/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day side annoset(47649) 11171
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20220805-235334/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20220805-235334/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy day rear annoset(47645) 12377
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220805-234732/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220805-234732/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy day side annoset(60039) 11319
            dict(
                rec_path=f"{root}/multicam_pilot/data/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20220928-171508/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20220928-171508/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy night side annoset(60018) 6508
            dict(
                rec_path=f"{root}/multicam_pilot/data/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Side/kPedBBox2D_project_galaxy_5v_train_night_Cam_Side/20220928-155340/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Side/kPedBBox2D_project_galaxy_5v_train_night_Cam_Side/20220928-155340/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy day rear annoset(60043) 13343
            dict(
                rec_path=f"{root}/multicam_pilot/data/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220928-170750/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220928-170750/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy night rear annoset(60021) 5861
            dict(
                rec_path=f"{root}/multicam_pilot/data/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Rear/kPedBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220928-155540/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Rear/kPedBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220928-155540/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy day side annoset(66830) 10911
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20221028-213019/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20221028-213019/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy day rear annoset(66840) 10191
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221028-215546/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221028-215546/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy night side annoset(66867) 5549
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Side/20221028-225802/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Side/20221028-225802/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night rear annoset(66870) 3562
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221028-230726/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221028-230726/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy underground rear annoset(72625) 657
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_all_Cam_Rear/20221119-132747/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_all_Cam_Rear/20221119-132747/data.anno.pb_rec",  # noqa
                sample_weight=0.25,
            ),
            # galaxy underground side annoset(72630) 431
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_all_Cam_Side/20221119-134707/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_all_Cam_Side/20221119-134707/data.anno.pb_rec",  # noqa
                sample_weight=0.25,
            ),
            # galaxy jira day annoset(83957) 180
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_JIRA_badcase_Cam_All/20230101-212215/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_JIRA_badcase_Cam_All/20230101-212215/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy jira night annoset(72637)
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_JIRA_badcase_Cam_All/20221119-142917/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_JIRA_badcase_Cam_All/20221119-142917/data.anno.pb_rec",  # noqa
                sample_weight=1.5,
            ),
            # galaxy day side annoset(78806) 8833
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_all_Cam_Side/20221212-132212/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_all_Cam_Side/20221212-132212/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day rear annoset(78808) 4634
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_all_Cam_Rear/20221212-132630/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_all_Cam_Rear/20221212-132630/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy glare jira annoset(88776) 3863
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_CO-OX3GB-D100-L_train_night_jira_Cam_5V/20230128-143411/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_CO-OX3GB-D100-L_train_night_jira_Cam_5V/20230128-143411/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # galaxy night rear; glare; fp; hand select; annoset(89125) 2329
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_train_night_Cam_Rear/20230130-211620/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_train_night_Cam_Rear/20230130-211620/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # galaxy night rear; glare; no prediction; hand select; annoset(89148) 7893
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_train_night_Cam_Rear/20230130-221528/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_train_night_Cam_Rear/20230130-221528/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    vehicle=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            # galaxy day rear annoset(25957) 5672
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220622-150758/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220622-150758/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night rear annoset(25954) 1473
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220622-145944/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220622-145944/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day side annoset(25956) 6439
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Side/20220622-150945/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Side/20220622-150945/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night side annoset(25952) 2168
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Side/20220622-145829/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Side/20220622-145829/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day rear annoset(39574) 13564
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220719-114309/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220719-114309/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy day side annoset(39575) 10593
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Side/20220719-111957/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Side/20220719-111957/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy night side annoset(39555) 1319
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Side/20220719-115156/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Side/20220719-115156/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night rear annoset(39540) 1345
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220719-115248/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220719-115248/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day side annoset(47642) 3300
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Side/20220805-225859/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Side/20220805-225859/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day rear annoset(47644) 5276
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220805-230512/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220805-230512/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night rear annoset(47646) 3160
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220805-230658/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220805-230658/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night side annoset(47647) 2065
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Side/20220805-230919/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Side/20220805-230919/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # c385 v2.0 63351
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_350703_CAMERA_CO_OX3GB_D100_L_TIME_ALL_all/20220818-202734/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_350703_CAMERA_CO_OX3GB_D100_L_TIME_ALL_all/20220818-202734/data.anno.pb_rec",  # noqa
                sample_weight=6,
            ),
            # galaxy day side annoset(61056) 25547
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Side/20221001-093910/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Side/20221001-093910/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # galaxy night rear annoset(61067) 6701
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221001-093514/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221001-093514/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day rear annoset(61065) 15397
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221001-092648/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221001-092648/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy night side annoset(61061) 10114
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Side/20221001-092213/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Side/20221001-092213/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy JIRA-badcase annoset(66486) 5319
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_JIRA-badcase_all/20221230-101251/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_JIRA-badcase_all/20221230-101251/data.anno.pb_rec",  # noqa
                sample_weight=10,
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
                anno_path=f"{root}/multicam_pilot/data/train/vehicle/vehicle_full_truncation_classification/pilot3.0_0233_data_before_0406_vehicle_truncation_cls/data.anno.pb_rec",  # noqa
                sample_weight=32,
            ),
            # rear day 5672
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehTruncation/kVehTruncation_project_galaxy_5v_train_day_Cam_Rear/20220622-171728/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehTruncation/kVehTruncation_project_galaxy_5v_train_day_Cam_Rear/20220622-171728/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # side day 6439
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehTruncation/kVehTruncation_project_galaxy_5v_train_day_Cam_Side/20220622-171503/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehTruncation/kVehTruncation_project_galaxy_5v_train_day_Cam_Side/20220622-171503/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # side night 2168
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehTruncation/kVehTruncation_project_galaxy_5v_train_night_Cam_Side/20220622-171245/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehTruncation/kVehTruncation_project_galaxy_5v_train_night_Cam_Side/20220622-171245/data.json",  # noqa
                pb_anno_path=f"{root}/multicam_pilot/data/train/kVehTruncation/kVehTruncation_project_galaxy_5v_train_night_Cam_Side/20220622-171245/data.anno.pb_rec",  # noqa
                sample_weight=1,
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
                sample_weight=9,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_3504_20220111_anno16.2_ws_night.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_3504_20220111_anno16.2_ws_night.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # x3c night
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4987_20220111_anno16.2_x3c_night.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4987_20220111_anno16.2_x3c_night.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4300_20220214_anno16.2_x3c_night.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4300_20220214_anno16.2_x3c_night.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4060_20220428_anno16.2_x3c_night.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4060_20220428_anno16.2_x3c_night.anno.pb_rec",  # noqa
                sample_weight=4,
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
            # galaxy day
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2138_anno16.2_galaxy/20220621-165237/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2138_anno16.2_galaxy/20220621-165237/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_3383_anno16.2_galaxy_day/20220718-231244/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_3383_anno16.2_galaxy_day/20220718-231244/data.anno.pb_rec",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1601_anno16.2_galaxy_day/20220809-203210/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1601_anno16.2_galaxy_day/20220809-203210/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1137_anno16.2_galaxy_day/20220909-193537_new/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1137_anno16.2_galaxy_day/20220909-193537_new/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2428_anno16.2_galaxy_day/20220930-170957/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2428_anno16.2_galaxy_day/20220930-170957/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2390_anno16.2_galaxy_day/20221028-174147/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2390_anno16.2_galaxy_day/20221028-174147/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_3693_anno16.2_galaxy_day/20221210-001652/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_3693_anno16.2_galaxy_day/20221210-001652/data.anno.pb_rec",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1364_anno16.2_galaxy_day/20230126-021406/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1364_anno16.2_galaxy_day/20230126-021406/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy night
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_496_anno16.2_galaxy/20220621-151513/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_496_anno16.2_galaxy/20220621-151513/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_558_anno16.2_galaxy_night/20220719-154904/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_558_anno16.2_galaxy_night/20220719-154904/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1316_anno16.2_galaxy_night/20220909-215318/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1316_anno16.2_galaxy_night/20220909-215318/data.anno.pb_rec",  # noqa
                sample_weight=2.5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1015_anno16.2_galaxy_night/20220930-162403/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1015_anno16.2_galaxy_night/20220930-162403/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1207_anno16.2_galaxy_night/20221209-234006/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1207_anno16.2_galaxy_night/20221209-234006/data.anno.pb_rec",  # noqa
                sample_weight=2,
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
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_3500_20210419_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_3500_20210419_5_2classes.anno.pb_rec",  # noqa
                sample_weight=1.5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_8571_20210524_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_8571_20210524_5_2classes.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_3902_20210524_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_3902_20210524_5_2classes.anno.pb_rec",  # noqa
                sample_weight=1.5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_24305_20210711_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_24305_20210711_5_2classes.anno.pb_rec",  # noqa
                sample_weight=12,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_6291_20210711_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_6291_20210711_5_2classes.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_1899_20210711_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_1899_20210711_5_2classes.anno.pb_rec",  # noqa
                sample_weight=0.5,
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
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_1721_20210803_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_1721_20210803_5_2classes.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_5598_20210803_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_5598_20210803_5_2classes.anno.pb_rec",  # noqa
                sample_weight=3,
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
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_20735_20210825_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_20735_20210825_5_2classes.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_3512_20210825_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_3512_20210825_5_2classes.anno.pb_rec",  # noqa
                sample_weight=2,
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
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_13475_20211018_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_13475_20211018_5_2classes.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_6000_20211018_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_6000_20211018_5_2classes.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_36387_20211018_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_36387_20211018_5_2classes.anno.pb_rec",  # noqa
                sample_weight=15,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_4387_20211129_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_4387_20211129_5_2classes.anno.pb_rec",  # noqa
                sample_weight=8,
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
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_wissen_num_653_20211129_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_wissen_num_653_20211129_5_2classes.anno.pb_rec",  # noqa
                sample_weight=0.1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_wissen_num_15689_20211129_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_wissen_num_15689_20211129_5_2classes.anno.pb_rec",  # noqa
                sample_weight=8,
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
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_12116_20211220_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_12116_20211220_5_2classes.anno.pb_rec",  # noqa
                sample_weight=12,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_x3c_num_2802_20211220_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_x3c_num_2802_20211220_5_2classes.anno.pb_rec",  # noqa
                sample_weight=2,
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
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_16876_20220118_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_16876_20220118_5_2classes.anno.pb_rec",  # noqa
                sample_weight=12,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_x3c_num_6284_20220118_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_x3c_num_6284_20220118_5_2classes.anno.pb_rec",  # noqa
                sample_weight=3,
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
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_2909_20220214_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_2909_20220214_5_2classes.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_x3c_num_673_20220214_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_x3c_num_673_20220214_5_2classes.anno.pb_rec",  # noqa
                sample_weight=0.1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_14440_20220214_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_14440_20220214_5_2classes.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_2924_20220214_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_2924_20220214_5_2classes.anno.pb_rec",  # noqa
                sample_weight=0.2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_num_24851_20220328_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_num_24851_20220328_5_2classes.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_num_5356_20220328_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_num_5356_20220328_5_2classes.anno.pb_rec",  # noqa
                sample_weight=0.2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_day_x3c_num_6785_20220328_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_day_x3c_num_6785_20220328_5_2classes.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_night_x3c_num_5356_20220328_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_night_x3c_num_5356_20220328_5_2classes.anno.pb_rec",  # noqa
                sample_weight=0.2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_D100L_num_17734_20220424_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_D100L_num_17734_20220424_5_2classes.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_day_x3c_D100L_num_5286_20220424_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_day_x3c_D100L_num_5286_20220424_5_2classes.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_CO_OX3GB_A100L_num_7589_20220505_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_CO_OX3GB_A100L_num_7589_20220505_5_2classes.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_day_x3c_CO_OX3GB_A100L_num_2629_20220505_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_day_x3c_CO_OX3GB_A100L_num_2629_20220505_5_2classes.anno.pb_rec",  # noqa
                sample_weight=8,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_CO_OX3GB_A100L_num_745_20220505_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_CO_OX3GB_A100L_num_745_20220505_5_2classes.anno.pb_rec",  # noqa
                sample_weight=0.5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_night_x3c_CO_OX3GB_A100L_num_166_20220505_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_night_x3c_CO_OX3GB_A100L_num_166_20220505_5_2classes.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_4481_20220525_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_4481_20220525_5_2classes.anno.pb_rec",  # noqa
                sample_weight=2.3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_0233_CW_A23GB_A114L_num_1777_20220525_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_0233_CW_A23GB_A114L_num_1777_20220525_5_2classes.anno.pb_rec",  # noqa
                sample_weight=1.0,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_0233_CW_A23GB_A114L_num_5460_20220525_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_0233_CW_A23GB_A114L_num_5460_20220525_5_2classes.anno.pb_rec",  # noqa
                sample_weight=2.5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_0233_CW_A23GB_A114L_num_2461_20220525_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_0233_CW_A23GB_A114L_num_2461_20220525_5_2classes.anno.pb_rec",  # noqa
                sample_weight=1.0,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_6271_20220525_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_6271_20220525_5_2classes.anno.pb_rec",  # noqa
                sample_weight=3.1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_0233_CW_A23GB_A114L_num_2193_20220525_5_2classes.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_0233_CW_A23GB_A114L_num_2193_20220525_5_2classes.anno.pb_rec",  # noqa
                sample_weight=1.1,
            ),
            # night rear 1209
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220624_145658/pilot_lane_parsing_annoset/20220624-151527/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220624_145658/pilot_lane_parsing_annoset/20220624-151527/data.anno.pb_rec",  # noqa
                sample_weight=0.6,
            ),
            # day rear 778
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220624_145604/pilot_lane_parsing_annoset/20220624-150814/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220624_145604/pilot_lane_parsing_annoset/20220624-150814/data.anno.pb_rec",  # noqa
                sample_weight=0.5,
            ),
            # night side 4297
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220624_145632/pilot_lane_parsing_annoset/20220624-154111/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220624_145632/pilot_lane_parsing_annoset/20220624-154111/data.anno.pb_rec",  # noqa
                sample_weight=2.2,
            ),
            # day side 1208
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220624_145532/pilot_lane_parsing_annoset/20220624-151249/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220624_145532/pilot_lane_parsing_annoset/20220624-151249/data.anno.pb_rec",  # noqa
                sample_weight=0.5,
            ),
            # v2
            # rear-day 1890
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220718_172921/pilot_lane_parsing_annoset/20220718-180136/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220718_172921/pilot_lane_parsing_annoset/20220718-180136/data.anno.pb_rec",  # noqa
                sample_weight=0.7,
            ),
            # rear-night 1100
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220718_173829/pilot_lane_parsing_annoset/20220718-175945/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220718_173829/pilot_lane_parsing_annoset/20220718-175945/data.anno.pb_rec",  # noqa
                sample_weight=0.6,
            ),
            # side-night 746
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220718_173357/pilot_lane_parsing_annoset/20220718-175942/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220718_173357/pilot_lane_parsing_annoset/20220718-175942/data.anno.pb_rec",  # noqa
                sample_weight=0.5,
            ),
            # side-day 4891
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220718_172304/pilot_lane_parsing_annoset/20220718-182428/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220718_172304/pilot_lane_parsing_annoset/20220718-182428/data.anno.pb_rec",  # noqa
                sample_weight=2.3,
            ),
            # v3
            # rear-day 1286
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220808_145734/pilot_lane_parsing_annoset/20220808-152052/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220808_145734/pilot_lane_parsing_annoset/20220808-152052/data.anno.pb_rec",  # noqa
                sample_weight=0.5,
            ),
            # rear-night 262
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220808_150429/pilot_lane_parsing_annoset/20220808-151403/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220808_150429/pilot_lane_parsing_annoset/20220808-151403/data.anno.pb_rec",  # noqa
                sample_weight=0.2,
            ),
            # v5
            # day rear 303
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220915_115417/pilot_lane_parsing_annoset/20220915-121824/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220915_115417/pilot_lane_parsing_annoset/20220915-121824/data.anno.pb_rec",  # noqa
                sample_weight=0.1,
            ),
            # day side 2500
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220915_125452/pilot_lane_parsing_annoset/20220915-134505/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220915_125452/pilot_lane_parsing_annoset/20220915-134505/data.anno.pb_rec",  # noqa
                sample_weight=1.1,
            ),
            # night rear 1317
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220915_141227/pilot_lane_parsing_annoset/20220915-143020/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220915_141227/pilot_lane_parsing_annoset/20220915-143020/data.anno.pb_rec",  # noqa
                sample_weight=0.7,
            ),
            # night side 1810
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220915_140510/pilot_lane_parsing_annoset/20220915-142425/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220915_140510/pilot_lane_parsing_annoset/20220915-142425/data.anno.pb_rec",  # noqa
                sample_weight=0.9,
            ),
            # v7
            # day rear 871
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221010_003115/pilot_lane_parsing_annoset/20221010-010602/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221010_003115/pilot_lane_parsing_annoset/20221010-010602/data.anno.pb_rec",  # noqa
                sample_weight=0.45,
            ),
            # night rear 1954
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221009_235934/pilot_lane_parsing_annoset/20221010-004456/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221009_235934/pilot_lane_parsing_annoset/20221010-004456/data.anno.pb_rec",  # noqa
                sample_weight=1.0,
            ),
            # day rear 409
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221028_152259/pilot_lane_parsing_annoset/20221028-153922/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221028_152259/pilot_lane_parsing_annoset/20221028-153922/data.anno.pb_rec",  # noqa
                sample_weight=0.2,
            ),
            # night rear 444
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221009_225819/pilot_lane_parsing_annoset/20221010-002433/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221009_225819/pilot_lane_parsing_annoset/20221010-002433/data.anno.pb_rec",  # noqa
                sample_weight=0.2,
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
                sample_weight=50,
            ),
            # X3C 白天 34092
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_day_X3C/20220119-122143/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_day_X3C/20220119-122143/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # X3C 白天 16632
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_day_X3C/20220224-131626/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_day_X3C/20220224-131626/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # X3C 白天 51208
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220315-163131/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220315-163131/data.anno.pb_rec",  # noqa
                sample_weight=6,
            ),
            # CAMERA_X3C 白天 9776 annoset(7184)
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220425-161851/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220425-161851/data.anno.pb_rec",  # noqa
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
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_day_0233/20220119-221935/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_day_0233/20220119-221935/data.anno.pb_rec",  # noqa
                sample_weight=50,
            ),
            # X3C 白天 34092
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_day_X3C/20220119-125418/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_day_X3C/20220119-125418/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # X3C 白天 16632
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_day_X3C/20220224-133404/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_day_X3C/20220224-133404/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # X3C 白天 51208
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220315-163208/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220315-163208/data.anno.pb_rec",  # noqa
                sample_weight=6,
            ),
            # CAMERA_X3C 白天 9776 annoset(12336)
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220425-161825/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220425-161825/data.anno.pb_rec",  # noqa
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
        ],
        val_batch_size_per_ctx=16,
        val_data_paths=None,
    ),
    rear=dict(
        train_batch_size_per_ctx=24,
        train_data_paths=[
            # galaxy day rear annoset(25953) 2618
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220622-150345/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220622-150345/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night rear annoset(26076) 535
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220622-171850/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220622-171850/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day side annoset(25955) 2993
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20220622-150745/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20220622-150745/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night side annoset(26074) 733
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Side/20220622-172038/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Side/20220622-172038/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night side annoset(39573) 1426
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220719-115310/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220719-115310/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day rear annoset(39564) 13286
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220719-114719/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220719-114719/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy day side annoset(39539) 4225
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20220719-111918/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20220719-111918/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night side annoset(47638) 1188
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Side/20220805-224901/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Side/20220805-224901/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night rear annoset(47641) 2115
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220805-230106/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220805-230106/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy day side annoset(47648) 8140
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20220805-233211/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20220805-233211/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day rear annoset(47651) 12466
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220806-000857/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220806-000857/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # PROJECT_C385_5V 泊车 6319 annoset(19300)
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_C385_5V_PARKING_all/20220524-122038/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_C385_5V_PARKING_all/20220524-122038/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # PROJECT_C385_5V 泊车 11926
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_day/20220621-011046/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_day/20220621-011046/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # PROJECT_C385_5V 泊车
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CO-OX3GB-D100-L_day/20220812-004728/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CO-OX3GB-D100-L_day/20220812-004728/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day side annoset(61058) 23870
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20221001-094341/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20221001-094341/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            # galaxy day rear annoset(61064) 17728
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221001-093816/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221001-093816/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # galaxy night rear annoset(61066) 4031
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221001-093022/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221001-093022/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night side annoset(61062) 5948
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Side/20221001-092532/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Side/20221001-092532/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day side contain parking annoset(66501) 5055
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20221027-205819/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20221027-205819/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night side contain parking annoset(66516) 1854
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Side/20221027-205945/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Side/20221027-205945/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy day rear contain parking annoset(66517) 3499
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221027-210957/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221027-210957/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # galaxy night rear contain parking annoset(66520) 1271
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221027-211254/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221027-211254/data.anno.pb_rec",  # noqa
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
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_day_0233/20220119-041133/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_day_0233/20220119-041133/data.anno.pb_rec",  # noqa
                sample_weight=50,
            ),
            # X3C 白天 84312
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_day_X3C/20220119-120507/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_day_X3C/20220119-120507/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # X3C 白天 12657
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_day_X3C/20220224-123226/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_day_X3C/20220224-123226/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # X3C 白天 15083
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_CAMERA_X3C_day/20220316-102102/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_CAMERA_X3C_day/20220316-102102/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # X3C 白天 13505
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_CAMERA_X3C_day/20220401-160904/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_occ_5v_CAMERA_X3C_day/20220401-160904/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # rear day 2618
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearOcclusion/kVehRearOcclusion_project_galaxy_5v_train_day_Cam_Rear/20220622-171123/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearOcclusion/kVehRearOcclusion_project_galaxy_5v_train_day_Cam_Rear/20220622-171123/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # side day 2993
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/kVehRearOcclusion/kVehRearOcclusion_project_galaxy_5v_train_day_Cam_Side/20220622-171126/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/kVehRearOcclusion/kVehRearOcclusion_project_galaxy_5v_train_day_Cam_Side/20220622-171126/data.anno.pb_rec",  # noqa
                sample_weight=1,
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
                sample_weight=50,
            ),
            # X3C 白天 67690
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_day_X3C/20220119-120509/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_day_X3C/20220119-120509/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # X3C 白天 15788
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_day_X3C/20220224-130525/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_day_X3C/20220224-130525/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # X3C 白天 12323
            dict(
                rec_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_CAMERA_X3C_day/20220315-163229/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/tmp/pilot_rear_part_5v_CAMERA_X3C_day/20220315-163229/data.anno.pb_rec",  # noqa
                sample_weight=2,
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

pilot3_day_weight_ratio = dict(
    rear_plate_detection=-1,
    person_face_detection=-1,
    vehicle=1,
    vehicle_occlusion_classification=-1,
    vehicle_category=-1,
    vehicle_truncation_classification=-1,
    vehicle_wheel_kps=-1,
    vehicle_ground_line=-1,
    vehicle_flank=1,
    vehicle_wheel_detection=-1,
    rear=1,
    rear_occlusion_classification=-1,
    rear_part_classification=-1,
    person=1,
    person_occlusion_classification=-1,
    person_pose_classification=-1,
    person_orientation_classification=-1,
    cyclist=1,
    cyclist_classification=-1,
    cyclist_kps=-1,
    semantic_parsing=-1,
    lane_parsing=-1,
)
pilot3_night_weight_ratio = dict(
    rear_plate_detection=-1,
    person_face_detection=-1,
    vehicle=1,
    vehicle_occlusion_classification=-1,
    vehicle_category=-1,
    vehicle_truncation_classification=-1,
    vehicle_wheel_kps=-1,
    vehicle_ground_line=-1,
    vehicle_flank=1,
    vehicle_wheel_detection=-1,
    rear=1,
    rear_occlusion_classification=-1,
    rear_part_classification=-1,
    person=1,
    person_occlusion_classification=-1,
    person_pose_classification=-1,
    person_orientation_classification=-1,
    cyclist=1,
    cyclist_classification=-1,
    cyclist_kps=-1,
    semantic_parsing=-1,
    lane_parsing=-1,
)

pilot3_day_datapaths = EasyDict(pilot3_day_datapaths)
pilot3_night_datapaths = EasyDict(pilot3_night_datapaths)
galaxy_datapaths = EasyDict(galaxy_datapaths)
append_data_not_exists_in_galaxy(
    input_datapaths=pilot3_day_datapaths,
    output_datapaths=galaxy_datapaths,
    weight_append_ratio=pilot3_day_weight_ratio,
)
append_data_not_exists_in_galaxy(
    input_datapaths=pilot3_night_datapaths,
    output_datapaths=galaxy_datapaths,
    weight_append_ratio=pilot3_night_weight_ratio,
)

datapaths = galaxy_datapaths
