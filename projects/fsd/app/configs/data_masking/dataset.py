import os

from easydict import EasyDict

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
old_root = os.path.join(bucket_root, "matrix")
root = os.path.join(bucket_root, "matrix2")
sd_root = os.path.join(bucket_root, "SD_Algorithm")


if not os.path.isdir(root):
    raise FileNotFoundError("matrix2 bucket")

if not os.path.isdir(old_root):
    raise FileNotFoundError("matrix bucket")

if not os.path.isdir(sd_root):
    raise FileNotFoundError("SD_Algorithm bucket")


datapaths = dict(
    plate=dict(
        train_batch_size_per_ctx=24,
        train_data_paths=[
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
            # 8.9w
            dict(
                rec_path=f"{root}/users/jinchuan01.xiao/data/plate/plate_v211108/train/train.rec",  # noqa
                anno_path=f"{root}/users/jinchuan01.xiao/data/plate/plate_v211108/train/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # # 鱼眼
            # dict(
            #     rec_path=f"{sd_root}/03_perception_single_camera/02_user/kai.liu/20230119-162655/data.rec",  # noqa
            #     anno_path=f"{sd_root}/03_perception_single_camera/02_user/kai.liu/20230119-162655/data.anno.pb_rec",  # noqa
            #     sample_weight=1,
            # ),
            # 2022-07-16_2022-08-16 9704
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_X8B_all_2pe_2022-07-16_2022-08-16/20220818-205449/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_X8B_all_2pe_2022-07-16_2022-08-16/20220818-205449/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # (66192) 5902
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/mono_rear_plate_5v_X8B_all_4pe/20220719-152400/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/mono_rear_plate_5v_X8B_all_4pe/20220719-152400/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # (55710 55712 55713 63134 63135) 8233
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/mono_rear_plate_5v_X8B_all_4pe/20220719-170628/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/mono_rear_plate_5v_X8B_all_4pe/20220719-170628/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
        ],
    ),
    face=dict(
        train_batch_size_per_ctx=24,
        train_data_paths=[
            dict(  # 6392
                rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_as33-5v_all_4pe_2022-11-06_2022-12-15/20221216-221754/data.rec",  # noqa
                anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_as33-5v_all_4pe_2022-11-06_2022-12-15/20221216-221754/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(  # 6644
                rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_c385-5v_all_4pe_2022-11-06_2022-11-30/20221217-002324/data.rec",  # noqa
                anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_c385-5v_all_4pe_2022-11-06_2022-11-30/20221217-002324/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(  # 771
                rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_c385-5v_all_4pe_2022-12-01_2022-12-15/20221217-011848/data.rec",  # noqa
                anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_c385-5v_all_4pe_2022-12-01_2022-12-15/20221217-011848/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(  # 743
                rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_cc02-5v_all_4pe_2022-11-06_2022-12-15/20221217-013950/data.rec",  # noqa
                anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_cc02-5v_all_4pe_2022-11-06_2022-12-15/20221217-013950/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(  # 2786
                rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_pilot-pdt-5v_all_4pe_2022-11-06_2022-12-15/20221217-015438/data.rec",  # noqa
                anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_pilot-pdt-5v_all_4pe_2022-11-06_2022-12-15/20221217-015438/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(  # 1.1w
                rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_galaxy-5v_all_4pe_2022-11-06_2022-12-15/20221217-020614/data.rec",  # noqa
                anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_galaxy-5v_all_4pe_2022-11-06_2022-12-15/20221217-020614/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(  # 1.1w
                rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_pilot_all_4pe_2022-12-21_2022-12-23/20230315-234346/data.rec",  # noqa
                anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_pilot_all_4pe_2022-12-21_2022-12-23/20230315-234346/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # mono数据
            # dict(
            #     rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_0820_all_4pe_2022-02-02_2023-03-01/20230307-162917/data.rec",  # noqa
            #     anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_0820_all_4pe_2022-02-02_2023-03-01/20230307-162917/data.anno.pb_rec",  # noqa
            #     sample_weight=5,
            # ),
            # dict(
            #     rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_0820_all_4pe_2022-12-16_2023-01-05/20230228-164156/data.rec",  # noqa
            #     anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_0820_all_4pe_2022-12-16_2023-01-05/20230228-164156/data.anno.pb_rec",  # noqa
            #     sample_weight=5,
            # ),
            # dict(
            #     rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_X8B_all_4pe_2022-11-06_2022-12-15/20221216-200426/data.rec",  # noqa
            #     anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_X8B_all_4pe_2022-11-06_2022-12-15/20221216-200426/data.anno.pb_rec",  # noqa
            #     sample_weight=5,
            # ),
        ],
    ),
    plate_front=dict(
        train_batch_size_per_ctx=24,
        train_data_paths=[
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
            # 8.9w
            dict(
                rec_path=f"{root}/users/jinchuan01.xiao/data/plate/plate_v211108/train/train.rec",  # noqa
                anno_path=f"{root}/users/jinchuan01.xiao/data/plate/plate_v211108/train/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # # 鱼眼
            # dict(
            #     rec_path=f"{sd_root}/03_perception_single_camera/02_user/kai.liu/20230119-162655/data.rec",  # noqa
            #     anno_path=f"{sd_root}/03_perception_single_camera/02_user/kai.liu/20230119-162655/data.anno.pb_rec",  # noqa
            #     sample_weight=1,
            # ),
            # 2022-07-16_2022-08-16 9704
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_X8B_all_2pe_2022-07-16_2022-08-16/20220818-205449/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_X8B_all_2pe_2022-07-16_2022-08-16/20220818-205449/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # (66192) 5902
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/mono_rear_plate_5v_X8B_all_4pe/20220719-152400/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/mono_rear_plate_5v_X8B_all_4pe/20220719-152400/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # (55710 55712 55713 63134 63135) 8233
            dict(
                rec_path=f"{root}/users/yuanzhuo.peng/tmp/mono_rear_plate_5v_X8B_all_4pe/20220719-170628/data.rec",  # noqa
                anno_path=f"{root}/users/yuanzhuo.peng/tmp/mono_rear_plate_5v_X8B_all_4pe/20220719-170628/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
        ],
    ),
    face_front=dict(
        train_batch_size_per_ctx=24,
        train_data_paths=[
            dict(  # 6392
                rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_as33-5v_all_4pe_2022-11-06_2022-12-15/20221216-221754/data.rec",  # noqa
                anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_as33-5v_all_4pe_2022-11-06_2022-12-15/20221216-221754/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(  # 6644
                rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_c385-5v_all_4pe_2022-11-06_2022-11-30/20221217-002324/data.rec",  # noqa
                anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_c385-5v_all_4pe_2022-11-06_2022-11-30/20221217-002324/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(  # 771
                rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_c385-5v_all_4pe_2022-12-01_2022-12-15/20221217-011848/data.rec",  # noqa
                anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_c385-5v_all_4pe_2022-12-01_2022-12-15/20221217-011848/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(  # 743
                rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_cc02-5v_all_4pe_2022-11-06_2022-12-15/20221217-013950/data.rec",  # noqa
                anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_cc02-5v_all_4pe_2022-11-06_2022-12-15/20221217-013950/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(  # 2786
                rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_pilot-pdt-5v_all_4pe_2022-11-06_2022-12-15/20221217-015438/data.rec",  # noqa
                anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_pilot-pdt-5v_all_4pe_2022-11-06_2022-12-15/20221217-015438/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(  # 1.1w
                rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_galaxy-5v_all_4pe_2022-11-06_2022-12-15/20221217-020614/data.rec",  # noqa
                anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_galaxy-5v_all_4pe_2022-11-06_2022-12-15/20221217-020614/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(  # 1.1w
                rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_pilot_all_4pe_2022-12-21_2022-12-23/20230315-234346/data.rec",  # noqa
                anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_pilot_all_4pe_2022-12-21_2022-12-23/20230315-234346/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # mono数据
            dict(
                rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_0820_all_4pe_2022-02-02_2023-03-01/20230307-162917/data.rec",  # noqa
                anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_0820_all_4pe_2022-02-02_2023-03-01/20230307-162917/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_0820_all_4pe_2022-12-16_2023-01-05/20230228-164156/data.rec",  # noqa
                anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_0820_all_4pe_2022-12-16_2023-01-05/20230228-164156/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_X8B_all_4pe_2022-11-06_2022-12-15/20221216-200426/data.rec",  # noqa
                anno_path=f"{old_root}/users/jinchuan01.xiao/data/face/pilot_person_face_5v_X8B_all_4pe_2022-11-06_2022-12-15/20221216-200426/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
        ],
    ),
)

datapaths = EasyDict(datapaths)
buckets = ["matrix", "matrix2", "SD_Algorithm"]
