from easydict import EasyDict
from hatbc.filestream.bucket.client import get_bucket_mount_root

bucket2mount_root = get_bucket_mount_root()
root = bucket2mount_root.get("mono", None)
if root is None:
    raise FileNotFoundError("mono bucket")

mono_data_paths = {
    "person": {
        "train_batch_size_per_ctx": 48,
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/person/20220412_20220808_release_v20220816/20220412_20220808_release_v20220816_train.rec",
                "anno_path": f"{root}/data/person/20220412_20220808_release_v20220816/20220412_20220808_release_v20220816_train.anno.pb_rec",
                "length": 385093,
                "img_shape": [[2160, 3840], [940, 1824]],
            },
            {
                "rec_path": f"{root}/data/person/20210906_v20210930/20210906_v20210930_train.rec",
                "anno_path": f"{root}/data/person/20210906_v20210930/20210906_v20210930_train.anno.pb_rec",
                "sample_weight": 3,
                "length": 130070,
                "img_shape": [[2160, 3840], [940, 1824]],
            },
            {
                "rec_path": f"{root}/data/person/ped_20181210/train_12cam.rec",
                "anno_path": f"{root}/data/person/ped_20181210/train_12cam.anno.pb_rec",
                "sample_weight": 5,
                "length": 242367,
                "img_shape": [[720, 1280]],
            },
            {
                "rec_path": f"{root}/data/person/ped_20190605/train_part_2.rec",
                "anno_path": f"{root}/data/person/ped_20190605/train_part_2.anno.pb_rec",
                "sample_weight": 5,
                "length": 317580,
                "img_shape": [[720, 1280]],
            },
            # grey images
            # {
            #     "rec_path": f"{root}/data/person/ped_20180814/train_gray.rec",
            #     "anno_path": f"{root}/data/person/ped_20180814/train_gray.anno.pb_rec",
            #     "sample_weight": 4,
            #     "length": 295096,
            #     "img_shape": [[720, 1280]],
            # },
            {
                "rec_path": f"{root}/data/person/0323_0220_820_10652_AEB_copy_paste_v20210804_adjRainNight/0323_0220_820_10652_AEB_copy_paste_v20210804_adjRainNight.rec",
                "anno_path": f"{root}/data/person/0323_0220_820_10652_AEB_copy_paste_v20210804_adjRainNight/0323_0220_820_10652_AEB_copy_paste_v20210804_adjRainNight.anno.pb_rec",
                "sample_weight": 8,
                "length": 555901,
                "img_shape": [
                    [896, 2048],
                    [1080, 2048],
                    [2160, 3840],
                    [940, 1824],
                ],
            },
            {
                "rec_path": f"{root}/data/person/0323_10635_quad_20200722/0323_10635_quad_20200722.rec",
                "anno_path": f"{root}/data/person/0323_10635_quad_20200722/0323_10635_quad_20200722.anno.pb_rec",
                "sample_weight": 7,
                "length": 364704,
                "img_shape": [[1080, 1920], [720, 1280], [1080, 2280]],
            },
            {
                "rec_path": f"{root}/data/person/20210802_20210906_all_night_v20211119_cfgv6.3/20210802_20210906_all_night_v20211119_cfgv6.3_train.rec",
                "anno_path": f"{root}/data/person/20210802_20210906_all_night_v20211119_cfgv6.3/20210802_20210906_all_night_v20211119_cfgv6.3_train.anno.pb_rec",
                "sample_weight": 3,
                "length": 154449,
                "img_shape": [
                    [896, 2048],
                    [2160, 3840],
                    [1080, 2048],
                    [940, 1824],
                ],
            },
            {
                "rec_path": f"{root}/data/person/20220304_multitask_v20220510/20220304_multitask_v20220510_train.rec",
                "anno_path": f"{root}/data/person/20220304_multitask_v20220510/20220304_multitask_v20220510_train.anno.pb_rec",
                "length": 682385,
                "img_shape": [[2160, 3840], [1080, 2048], [940, 1824]],
                "roi_list": f"{root}/data/person/20220304_multitask_v20220510/20220304_multitask_v20220510_train.roi.json",
            },
            # {
            #     "rec_path": f"{root}/data/person/20220412_release_20220711/20220412_release_20220711_train.rec",
            #     "anno_path": f"{root}/data/person/20220412_release_20220711/20220412_release_20220711_train.anno.pb_rec",
            #     "length": 297093,
            #     "img_shape": [[2160, 3840], [940, 1824]],
            #     "roi_list": f"{root}/data/person/20220412_release_20220711/20220412_release_20220711_train.roi.json",
            # },
            {
                "rec_path": f"{root}/data/person/20220112_multitask_v20220225/0323_0220_820_10652_AEB_v20210804_adjRainNight_v20220119_train.rec",
                "anno_path": f"{root}/data/person/20220112_multitask_v20220225/0323_0220_820_10652_AEB_v20210804_adjRainNight_v20220119_train.anno.pb_rec",
                "length": 452863,
                "img_shape": [[2160, 3840], [1080, 2048], [940, 1824]],
                "roi_list": f"{root}/data/person/20220112_multitask_v20220225/0323_0220_820_10652_AEB_v20210804_adjRainNight_v20220119_train.roi.json",
            },
            {
                "rec_path": f"{root}/data/person/20211209-night-cfgv7.0/20211209-night_train.rec",
                "anno_path": f"{root}/data/person/20211209-night-cfgv7.0/20211209-night_train.anno.pb_rec",
                "length": 165277,
                "img_shape": [
                    [1080, 2280],
                    [940, 1824],
                    [234, 464],
                    [1080, 2048],
                    [2160, 3840],
                    [720, 1280],
                    [1080, 1920],
                ],
                "roi_list": f"{root}/data/person/20211209-night-cfgv7.0/20211209-night_train.roi.json",
            },
            # {
            #     "rec_path": f"{root}/data/person/20220412_release_20220624/20220412_release_20220624_train.rec",
            #     "anno_path": f"{root}/data/person/20220412_release_20220624/20220412_release_20220624_train.anno.pb_rec",
            #     "length": 272871,
            #     "img_shape": [[2160, 3840], [940, 1824]],
            #     "roi_list": f"{root}/data/person/20220412_release_20220624/20220412_release_20220624_train.roi.json",
            # },
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_vehicle/newdata_rec_10635/densebox.val.basenewv1.rec",
                "anno_path": f"{root}/data/4pe_vehicle/newdata_rec_10635/densebox.val.basenewv1.anno.pb_rec",
                "sample_weight": 1,
                "length": 2119,
                "img_shape": [[2160, 3840], [940, 1824]],
            },
            {
                "rec_path": f"{root}/data/person/eval_20200609/eval_20200609.rec",
                "anno_path": f"{root}/data/person/eval_20200609/eval_20200609.anno.pb_rec",
                "length": 19880,
                "img_shape": [[720, 1280]],
            },
        ],
    },
    "cyclist": {
        "train_batch_size_per_ctx": 48,
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/person/0323_10635_quad_cyc_20200723/0323_10635_quad_cyc_20200723.rec",
                "anno_path": f"{root}/data/person/0323_10635_quad_cyc_20200723/0323_10635_quad_cyc_20200723.anno.pb_rec",
                "sample_weight": 4,
                "length": 122902,
                "img_shape": [[1080, 1920], [720, 1280], [1080, 2280]],
            },
            {
                "rec_path": f"{root}/data/person/20190522-remove-truck/20190522-remove-truck.rec",
                "anno_path": f"{root}/data/person/20190522-remove-truck/20190522-remove-truck.anno.pb_rec",
                "sample_weight": 2.5,
                "length": 246575,
                "img_shape": [[720, 1280]],
            },
            # # there are a large proportion of images with no targets in them
            # this set is a subset of 20220412_20220808_release_v20220816_train.rec
            # {
            #     "rec_path": f"{root}/data/person/20220412_release_20220711/20220412_release_20220711_train.rec",
            #     "anno_path": f"{root}/data/person/20220412_release_20220711/20220412_release_20220711_train.anno.pb_rec",
            #     "length": 297093,
            #     "img_shape": [[2160, 3840], [940, 1824]],
            #     "roi_list": f"{root}/data/person/20220412_release_20220711/20220412_release_20220711_train.roi.json",
            # },
            # there are a large proportion of images with no targets in them
            {
                "rec_path": f"{root}/data/person/0323_0220_820_10652_AEB_copy_paste_v20210804_adjRainNight/0323_0220_820_10652_AEB_copy_paste_v20210804_adjRainNight.rec",
                "anno_path": f"{root}/data/person/0323_0220_820_10652_AEB_copy_paste_v20210804_adjRainNight/0323_0220_820_10652_AEB_copy_paste_v20210804_adjRainNight.anno.pb_rec",
                "sample_weight": 4,
                "length": 555901,
                "img_shape": [
                    [896, 2048],
                    [1080, 2048],
                    [2160, 3840],
                    [940, 1824],
                ],
            },
            {
                "rec_path": f"{root}/data/person/20210906_v20210930/20210906_v20210930_train.rec",
                "anno_path": f"{root}/data/person/20210906_v20210930/20210906_v20210930_train.anno.pb_rec",
                "sample_weight": 1.5,
                "length": 130070,
                "img_shape": [[2160, 3840], [940, 1824]],
            },
            {
                "rec_path": f"{root}/data/person/20220304_multitask_v20220510/20220304_multitask_v20220510_train.rec",
                "anno_path": f"{root}/data/person/20220304_multitask_v20220510/20220304_multitask_v20220510_train.anno.pb_rec",
                "length": 682385,
                "img_shape": [[2160, 3840], [1080, 2048], [940, 1824]],
                "roi_list": f"{root}/data/person/20220304_multitask_v20220510/20220304_multitask_v20220510_train.roi.json",
            },
            # there are a large proportion of images with no targets in them
            {
                "rec_path": f"{root}/data/person/20220112_multitask_v20220225/0323_0220_820_10652_AEB_v20210804_adjRainNight_v20220119_train.rec",
                "anno_path": f"{root}/data/person/20220112_multitask_v20220225/0323_0220_820_10652_AEB_v20210804_adjRainNight_v20220119_train.anno.pb_rec",
                "length": 452863,
                "img_shape": [[2160, 3840], [1080, 2048], [940, 1824]],
                "roi_list": f"{root}/data/person/20220112_multitask_v20220225/0323_0220_820_10652_AEB_v20210804_adjRainNight_v20220119_train.roi.json",
            },
            {
                "rec_path": f"{root}/data/person/20220412_20220808_release_v20220816/20220412_20220808_release_v20220816_train.rec",
                "anno_path": f"{root}/data/person/20220412_20220808_release_v20220816/20220412_20220808_release_v20220816_train.anno.pb_rec",
                "length": 385093,
                "img_shape": [[2160, 3840], [940, 1824]],
            },
            {
                "rec_path": f"{root}/data/person_det_multitask/pedestrian_cyclist_det_4pe_online_packing_20220909/20220909-141246/data.rec",
                "anno_path": f"{root}/data/person_det_multitask/pedestrian_cyclist_det_4pe_online_packing_20220909/20220909-141246/data.anno.pb_rec",
                "train_roi_list": "/data/person_det_multitask/pedestrian_cyclist_det_4pe_online_packing_20220909/20220909-141246/data_roilist.json",
                "length": 74839,
            },
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [],
    },
    # # newly packed vehicle data with cycles included
    # "vehicle": {
    #     "train_batch_size_per_ctx": 48,
    #     "train_data_paths": [
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_1/20230104-064152/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_1/20230104-064152/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 6308,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_2/20230104-064954/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_2/20230104-064954/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 6759,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_3/20230104-065720/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_3/20230104-065720/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 6053,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_4/20230104-213637/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_4/20230104-213637/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 8795,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_5/20230104-071925/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_5/20230104-071925/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 12360,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_6/20230104-072325/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_6/20230104-072325/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 8246,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_7/20230104-073154/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_7/20230104-073154/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 7060,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_8/20230104-074243/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_8/20230104-074243/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 9399,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_9/20230104-074656/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_9/20230104-074656/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 7157,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_night_202208_202212_attribute_44cls_1/20230104-075109/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_night_202208_202212_attribute_44cls_1/20230104-075109/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 3884,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_night_202208_202212_attribute_44cls_2/20230104-075952/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_night_202208_202212_attribute_44cls_2/20230104-075952/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 5284,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_night_202208_202212_attribute_44cls_3/20230104-080634/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_night_202208_202212_attribute_44cls_3/20230104-080634/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 2141,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_night_202208_202212_attribute_44cls_4/20230104-081523/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_night_202208_202212_attribute_44cls_4/20230104-081523/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 3212,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_2/20230226-165525/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_2/20230226-165525/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 138525,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_3/20230226-163651/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_3/20230226-163651/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 119524,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_4/20230226-172712/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_4/20230226-172712/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 126862,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_7/20230226-201517/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_7/20230226-201517/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 144832,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_8/20230226-211033/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_8/20230226-211033/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 135098,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_9/20230227-000910/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_9/20230227-000910/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 131555,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_10/20230226-213125/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_10/20230226-213125/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 134736,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_11/20230226-220322/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_11/20230226-220322/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 52602,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_12/20230226-234943/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_12/20230226-234943/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 24540,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_13/20230227-081832/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_13/20230227-081832/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 24765,
    #             "img_shape": [[2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_14/20230226-185326/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_14/20230226-185326/data.anno.pb_rec",
    #             "sample_weight": 7,
    #             "length": 24167,
    #             "img_shape": [[2160, 3840]],
    #         },
    #     ],
    #     "val_batch_size_per_ctx": 4,
    #     "val_data_paths": [],
    # },
    "vehicle": {
        "train_batch_size_per_ctx": 48,
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0820/det_vehicle_full_v1_0-20210501_v1/train.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0820/det_vehicle_full_v1_0-20210501_v1/train.anno.pb_rec",
                "sample_weight": 7,
                "length": 104242,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0323/det_vehicle_full_v1_0-20210311_day_v1/train.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0323/det_vehicle_full_v1_0-20210311_day_v1/train.rec.anno.pb_rec",
                "length": 246863,
                "img_shape": [[1080, 2048], [1080, 2280], [940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/newdata_rec_10635/densebox.train.basenewv1.rec",
                "anno_path": f"{root}/data/4pe_vehicle/newdata_rec_10635/densebox.train.basenewv1.anno.pb_rec",
                "sample_weight": 8,
                "length": 346190,
                "img_shape": [[720, 1280], [736, 1280]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/newdata_rec_10635/densebox.train.12camnewv1.rec",
                "anno_path": f"{root}/data/4pe_vehicle/newdata_rec_10635/densebox.train.12camnewv1.anno.pb_rec",
                "sample_weight": 15,
                "length": 346218,
                "img_shape": [[720, 1280]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/newdata2_rec_10635/densebox.train.newdatav2.rec",
                "anno_path": f"{root}/data/4pe_vehicle/newdata2_rec_10635/densebox.train.newdatav2.anno.pb_rec",
                "sample_weight": 2,
                "length": 60047,
                "img_shape": [[720, 1280]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0323/det_vehicle_full_v1_0-20201118_v1/train.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0323/det_vehicle_full_v1_0-20201118_v1/train.anno.pb_rec",
                "sample_weight": 14,
                "length": 289414,
                "img_shape": [[1080, 2048], [1080, 2280]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_10652/det_vehicle_full_v1_0-20210521_v1/train.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_10652/det_vehicle_full_v1_0-20210521_v1/train.anno.pb_rec",
                "sample_weight": 12,
                "length": 314200,
                "img_shape": [[1080, 1920], [940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0820/det_vehicle_full_v1_0-20210430_daypart1_v1/train.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0820/det_vehicle_full_v1_0-20210430_daypart1_v1/train.anno.pb_rec",
                "sample_weight": 7,
                "length": 106851,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_10652/det_vehicle_full_v1_0-20210917_v1/train.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_10652/det_vehicle_full_v1_0-20210917_v1/train.anno.pb_rec",
                "sample_weight": 13,
                "length": 170616,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_10652/det_vehicle_full_v1_0-20211202_v1/train.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_10652/det_vehicle_full_v1_0-20211202_v1/train.anno.pb_rec",
                "sample_weight": 5,
                "length": 61127,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_10652/det_vehicle_full_v1_0-20211229_v1/train.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_10652/det_vehicle_full_v1_0-20211229_v1/train.anno.pb_rec",
                "sample_weight": 5,
                "length": 54884,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0820/det_vehicle_full_v1_0-20210525_v1/train.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0820/det_vehicle_full_v1_0-20210525_v1/train.anno.pb_rec",
                "length": 164582,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0820/det_vehicle_full_v1_0-20210608_v1/train.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0820/det_vehicle_full_v1_0-20210608_v1/train.anno.pb_rec",
                "length": 134869,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0820/det_vehicle_full_v1_0-20210729_rainy_v1/train.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0820/det_vehicle_full_v1_0-20210729_rainy_v1/train.anno.pb_rec",
                "length": 107298,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_x8b/det_vehicle_full_v5-20220524-day_under_park_v1/train.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_x8b/det_vehicle_full_v5-20220524-day_under_park_v1/train.anno.pb_rec",
                "length": 73893,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_x8b/det_vehicle_full_v7-20220711-badcase_jira_v1/train.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_x8b/det_vehicle_full_v7-20220711-badcase_jira_v1/train.rec.anno.pb_rec",
                "length": 56513,
                "img_shape": [[2160, 3840], [940, 1824]],
            },
            # {
            #     "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_10652/det_vehicle_full_v1_0-20220407_v1/train.rec",
            #     "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_10652/det_vehicle_full_v1_0-20220407_v1/train.anno.pb_rec",
            #     "length": 86868,
            #     "img_shape": [[940, 1824]],
            # },
            # only tricycles are labelled
            # {
            #     "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_x8b/det_vehicle_full_v6-20220616-tricycle_v2/train.rec",
            #     "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_x8b/det_vehicle_full_v6-20220616-tricycle_v2/train.rec.anno.pb_rec",
            #     "length": 97162,
            #     "img_shape": [[2160, 3840], [940, 1824]],
            # },
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_1080/det_vehicle_full_v1_0-20200820_v1/train.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_1080/det_vehicle_full_v1_0-20200820_v1/train.rec.anno.pb_rec",
                "length": 140519,
                "img_shape": [[1080, 1920], [940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0820/det_vehicle_full_v1_0-20210608_night_v1/train.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0820/det_vehicle_full_v1_0-20210608_night_v1/train.rec.anno.pb_rec",
                "length": 248674,
                "img_shape": [
                    [1080, 2048],
                    [1080, 2280],
                    [2160, 3840],
                    [940, 1824],
                ],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0820/det_vehicle_full_v1_0-20210430_daypart2_v1/train.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0820/det_vehicle_full_v1_0-20210430_daypart2_v1/train.rec.anno.pb_rec",
                "length": 87717,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_x8b/det_vehicle_full_v5-20220415-night_v1/train.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_x8b/det_vehicle_full_v5-20220415-night_v1/train.rec.anno.pb_rec",
                "length": 50145,
                "img_shape": [[2160, 3840]],
            },
            # {
            #     "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_10652/det_vehicle_full_v1_0-20220624_v1/train.rec",
            #     "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_10652/det_vehicle_full_v1_0-20220624_v1/train.anno.pb_rec",
            #     "length": 92462,
            #     "img_shape": [[940, 1824]],
            # },
            # {
            #     "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_10652/det_vehicle_full_v1_0-20220114_v1/train.rec",
            #     "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_10652/det_vehicle_full_v1_0-20220114_v1/train.anno.pb_rec",
            #     "length": 82334,
            #     "img_shape": [[940, 1824]],
            # },
            # {
            #     "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_10652/det_vehicle_full_v1_0-20210521_day_v1/train.rec",
            #     "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_10652/det_vehicle_full_v1_0-20210521_day_v1/train.rec.anno.pb_rec",
            #     "length": 256704,
            #     "img_shape": [[1080, 1920], [940, 1824]],
            # },
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_vehicle/newdata_rec_10635/densebox.val.basenewv1.rec",
                "anno_path": f"{root}/data/4pe_vehicle/newdata_rec_10635/densebox.val.basenewv1.anno.pb_rec",
                "sample_weight": 1,
                "length": 2119,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/newdata_rec_10635/densebox.val.12camnewv1.rec",
                "anno_path": f"{root}/data/4pe_vehicle/newdata_rec_10635/densebox.val.12camnewv1.anno.pb_rec",
                "length": 2000,
                "img_shape": [[720, 1280]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/newdata2_rec_10635/densebox.val.newdatav2.rec",
                "anno_path": f"{root}/data/4pe_vehicle/newdata2_rec_10635/densebox.val.newdatav2.anno.pb_rec",
                "length": 5154,
                "img_shape": [[720, 1280]],
            },
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0323/det_vehicle_full_v1_0-20200630_v1/val.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_0323/det_vehicle_full_v1_0-20200630_v1/val.anno.pb_rec",
                "length": 7321,
                "img_shape": [[1080, 2280]],
            },
        ],
    },
    "rear": {
        "train_batch_size_per_ctx": 48,
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20220129-0820-train-rainy/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20220129-0820-train-rainy/data.anno.pb_rec",
                "length": 98527,
                "img_shape": [[2160, 3840]],
            },
            # grey images
            # {
            #     "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v8-0323_10635_day-20210729/data.rec",
            #     "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v8-0323_10635_day-20210729/data.anno.pb_rec",
            #     "sample_weight": 324742,
            #     "length": 324740,
            #     "img_shape": [
            #         [1080, 1920],
            #         [720, 1280],
            #         [1080, 2280],
            #         [1080, 2048],
            #     ],
            # },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v7-20190726-train_base_and_night/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v7-20190726-train_base_and_night/data.anno.pb_rec",
                "sample_weight": 370503,
                "length": 370503,
                "img_shape": [[720, 1280]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220425-0323_10635_night/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220425-0323_10635_night/data.anno.pb_rec",
                "length": 229923,
                "img_shape": [
                    [1080, 2280],
                    [1080, 2048],
                    [720, 1280],
                    [910, 1920],
                    [1080, 1920],
                ],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20211105-0220_10652_day/data.rec",
                "anno_path": f"{root}//data/4pe_rear/det_vehicle_rear_v10-20211105-0220_10652_day/data.anno.pb_rec",
                "sample_weight": 153764,
                "length": 151198,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20211105-0220_10652_0820_night/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20211105-0220_10652_0820_night/data.anno.pb_rec",
                "sample_weight": 163741,
                "length": 163741,
                "img_shape": [[2160, 3840], [940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v9-20211202-0820-train-day/train.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v9-20211202-0820-train-day/train.anno.pb_rec",
                "sample_weight": 147600,
                "length": 147664,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v8-0820-split_big_car-20210902/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v8-0820-split_big_car-20210902/data.anno.pb_rec",
                "sample_weight": 189850,
                "length": 189850,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v8-0220_0323_10635_bigcar-20210917/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v8-0220_0323_10635_bigcar-20210917/data.anno.pb_rec",
                "sample_weight": 200000,
                "length": 397407,
                "img_shape": [
                    [1080, 2280],
                    [940, 1824],
                    [1080, 2048],
                    [720, 1280],
                    [1080, 1920],
                ],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20211229-10652_0820_side_fp_data/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20211229-10652_0820_side_fp_data/data.anno.pb_rec",
                "sample_weight": 120000,
                "length": 112099,
                "img_shape": [[2160, 3840], [940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20211229V2-jira_badcase_close_vehicle_data/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20211229V2-jira_badcase_close_vehicle_data/data.anno.pb_rec",
                "sample_weight": 45000,
                "length": 44298,
                "img_shape": [[1080, 1920], [940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20211229-10652_badcase_and_tricycle/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20211229-10652_badcase_and_tricycle/data.anno.pb_rec",
                "sample_weight": 87518,
                "length": 87518,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20211229-10652-split_special_and_BigTrucks/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20211229-10652-split_special_and_BigTrucks/data.anno.pb_rec",
                "sample_weight": 81000,
                "length": 81792,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220425-0323_10635_day/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220425-0323_10635_day/data.anno.pb_rec",
                "length": 384259,
                "img_shape": [
                    [1080, 2280],
                    [1080, 2048],
                    [720, 1280],
                    [910, 1920],
                    [1080, 1920],
                ],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220425-0220_10652_day/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220425-0220_10652_day/data.anno.pb_rec",
                "length": 145882,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220425-0820_0220_10652_night/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220425-0820_0220_10652_night/data.anno.pb_rec",
                "length": 162309,
                "img_shape": [[2160, 3840], [940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20220129-0820-train-day/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20220129-0820-train-day/data.anno.pb_rec",
                "length": 144906,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20220129-0820-split_special_and_BigTrucks/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20220129-0820-split_special_and_BigTrucks/data.anno.pb_rec",
                "length": 108999,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220426-0220_0323_special_and_BigTrucks/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220426-0220_0323_special_and_BigTrucks/data.anno.pb_rec",
                "length": 240433,
                "img_shape": [
                    [1080, 1920],
                    [1080, 2280],
                    [1080, 2048],
                    [940, 1824],
                ],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20220114-10652_0820_side_fp_data/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20220114-10652_0820_side_fp_data/data.anno.pb_rec",
                "length": 112825,
                "img_shape": [[2160, 3840], [940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20220114-10652_badcase_from_collect_car/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20220114-10652_badcase_from_collect_car/data.anno.pb_rec",
                "length": 87009,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20220129-jira_and_close_vehicle_data/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v10-20220129-jira_and_close_vehicle_data/data.anno.pb_rec",
                "length": 64006,
                "img_shape": [
                    [1080, 2280],
                    [940, 1824],
                    [1080, 2048],
                    [2160, 3840],
                    [1080, 1920],
                ],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220129-10652_tricycle/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220129-10652_tricycle/data.anno.pb_rec",
                "length": 42774,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220711-x8b-badcase/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220711-x8b-badcase/data.anno.pb_rec",
                "length": 66932,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220524-x8b-day_and_night/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220524-x8b-day_and_night/data.anno.pb_rec",
                "length": 96134,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220325-canno-train-all/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220325-canno-train-all/data.anno.pb_rec",
                "length": 73772,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220325-10652_0820_side_fp_data/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220325-10652_0820_side_fp_data/data.anno.pb_rec",
                "length": 154698,
                "img_shape": [[2160, 3840], [940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220407-10652_badcase_from_collect_car/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220407-10652_badcase_from_collect_car/data.anno.pb_rec",
                "length": 87719,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220624-jira_and_close_vehicle_data/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220624-jira_and_close_vehicle_data/data.anno.pb_rec",  # ***
                "length": 143007,
                "img_shape": [
                    [1080, 2280],
                    [940, 1824],
                    [1080, 2048],
                    [2160, 3840],
                    [1080, 1920],
                ],
            },
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v8-10652_0820_rainy_tunnel-20210906/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v8-10652_0820_rainy_tunnel-20210906/data.anno.pb_rec",
                "sample_weight": 98527,
                "length": 98527,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v1-20190726-val_base/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v1-20190726-val_base/data.anno.pb_rec",
                "length": 17729,
                "img_shape": [[720, 1280]],
            },
        ],
    },
    "traffic_light": {
        "train_batch_size_per_ctx": 48,
        # "train_data_paths": [
        #     # {
        #     #     "rec_path": f"{root}/data/4pe_traffic_light/20220614_220_det_batch_1-28_filter_balck/train.rec",
        #     #     "anno_path": f"{root}/data/4pe_traffic_light/20220614_220_det_batch_1-28_filter_balck/train.anno.pb_rec",
        #     #     "length": 93443,
        #     #     "img_shape": [[1080, 2048], [940, 1824]],
        #     #     "roi_list": f"{root}/data/4pe_traffic_light/20220614_220_det_batch_1-28_filter_balck/train.roi.json",
        #     # },
        #     {
        #         "rec_path": f"{root}/data/4pe_traffic_light/20220614_323_det_all_filter_balck/train.rec",
        #         "anno_path": f"{root}/data/4pe_traffic_light/20220614_323_det_all_filter_balck/train.anno.pb_rec",
        #         "length": 134226,
        #         "img_shape": [[1080, 2048], [1080, 2280]],
        #         "roi_list": f"{root}/data/4pe_traffic_light/20220614_323_det_all_filter_balck/train.roi.json",
        #     },
        #     {
        #         "rec_path": f"{root}/data/4pe_traffic_light/20211115_820_det_batch_1-12_filter_balck/train.rec",
        #         "anno_path": f"{root}/data/4pe_traffic_light/20211115_820_det_batch_1-12_filter_balck/train.anno.pb_rec",
        #         "sample_weight": 25,
        #         "length": 128322,
        #         "img_shape": [[2160, 3840]],
        #     },
        #     {
        #         "rec_path": f"{root}/data/4pe_traffic_light/20211115_10652_det_batch_1-14_filter_balck/train.rec",
        #         "anno_path": f"{root}/data/4pe_traffic_light/20211115_10652_det_batch_1-14_filter_balck/train.anno.pb_rec",
        #         "sample_weight": 7,
        #         "length": 74711,
        #         "img_shape": [[940, 1824]],
        #     },
        #     {
        #         "rec_path": f"{root}/data/4pe_traffic_light/densebox_traffic_light_color.train.zx.tl.cn.0810.cn.1080p.qud.filter.all.black.rec",
        #         "anno_path": f"{root}/data/4pe_traffic_light/densebox_traffic_light_color.train.zx.tl.cn.0810.cn.1080p.qud.filter.all.black.anno.pb_rec",
        #         "sample_weight": 10,
        #         "length": 81962,
        #         "img_shape": [[940, 1824]],
        #     },
        #     {
        #         "rec_path": f"{root}/data/4pe_traffic_light/20220614_820_det_batch_1-13_filter_balck/train.rec",
        #         "anno_path": f"{root}/data/4pe_traffic_light/20220614_820_det_batch_1-13_filter_balck/train.anno.pb_rec",
        #         "length": 138255,
        #         "img_shape": [[2160, 3840]],
        #         "roi_list": f"{root}/data/4pe_traffic_light/20220614_820_det_batch_1-13_filter_balck/train.roi.json",
        #     },
        #     {
        #         "rec_path": f"{root}/data/4pe_traffic_light/20220614_10652_det_batch_1-15_filter_balck/train.rec",
        #         "anno_path": f"{root}/data/4pe_traffic_light/20220614_10652_det_batch_1-15_filter_balck/train.anno.pb_rec",  # ***
        #         "length": 83515,
        #         "img_shape": [[940, 1824]],
        #         "roi_list": f"{root}/data/4pe_traffic_light/20220614_10652_det_batch_1-15_filter_balck/train.roi.json",
        #     },
        #     {
        #         "rec_path": f"{root}/data/4pe_traffic_light/20220614_x8b_det_batch_1-23_filter_balck/train.rec",
        #         "anno_path": f"{root}/data/4pe_traffic_light/20220614_x8b_det_batch_1-23_filter_balck/train.anno.pb_rec",
        #         "length": 52979,
        #         "img_shape": [[2160, 3840]],
        #         "roi_list": f"{root}/data/4pe_traffic_light/20220614_x8b_det_batch_1-23_filter_balck/train.roi.json",
        #     },
        #     {
        #         "rec_path": f"{root}/data/4pe_traffic_light/20220614_x8b_det_badcase_batch_1_filter_balck/train.rec",
        #         "anno_path": f"{root}/data/4pe_traffic_light/20220614_x8b_det_badcase_batch_1_filter_balck/train.anno.pb_rec",
        #         "length": 985,
        #         "img_shape": [[2160, 3840]],
        #         "roi_list": f"{root}/data/4pe_traffic_light/20220614_x8b_det_badcase_batch_1_filter_balck/train.roi.json",
        #     },
        #     {
        #         "rec_path": f"{root}/data/4pe_traffic_light_night/20220608_mix_cn_det_night_820_c4_x8b_c11_filter_balck/train.rec",
        #         "anno_path": f"{root}/data/4pe_traffic_light_night/20220608_mix_cn_det_night_820_c4_x8b_c11_filter_balck/train.anno.pb_rec",
        #         "length": 72371,
        #         "img_shape": [[2160, 3840]],
        #         "roi_list": f"{root}/data/4pe_traffic_light_night/20220608_mix_cn_det_night_820_c4_x8b_c11_filter_balck/train.roi.json",
        #     },
        # ],
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/traffic_light_det_roilist/20220809_220_det_batch_all/20220919-180838/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/data/traffic_light_det_roilist/20220809_220_det_batch_all/20220919-180838/data.anno.pb_rec",
                "length": 1,
                "img_shape": [[1080, 2048], [1080, 2280]],
                "roi_list": f"{root}/data/traffic_light_det_roilist/20220809_220_det_batch_all/20220919-180838/data_roilist.json",
            },
            {
                "rec_path": f"{root}/data/traffic_light_det_roilist/20220811_323_det_batch_all/20220816-172434/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/data/traffic_light_det_roilist/20220811_323_det_batch_all/20220816-172434/data.anno.pb_rec",
                "length": 2,
                "img_shape": [[1080, 2048], [1080, 2280]],
                "roi_list": f"{root}/data/traffic_light_det_roilist/20220811_323_det_batch_all/20220816-172434/data_roilist.json",
            },
            {
                "rec_path": f"{root}/data/traffic_light_det_roilist/20220809_820_det_batch_lx/20220818-053825/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/data/traffic_light_det_roilist/20220809_820_det_batch_lx/20220818-053825/data.anno.pb_rec",
                "length": 3,
                "img_shape": [[1080, 2048], [1080, 2280]],
                "roi_list": f"{root}/data/traffic_light_det_roilist/20220809_820_det_batch_lx/20220818-053825/data_roilist.json",
            },
            {
                "rec_path": f"{root}/data/traffic_light_det_roilist/20220809_10652_det_batch_all/20220811-230602/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/data/traffic_light_det_roilist/20220809_10652_det_batch_all/20220811-230602/data.anno.pb_rec",
                "length": 4,
                "img_shape": [[1080, 2048], [1080, 2280]],
                "roi_list": f"{root}/data/traffic_light_det_roilist/20220809_10652_det_batch_all/20220811-230602/data_roilist.json",
            },
            {
                "rec_path": f"{root}/data/traffic_light_det_roilist/20220809_x8b_det_batch_all/20220817-102819/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/data/traffic_light_det_roilist/20220809_x8b_det_batch_all/20220817-102819/data.anno.pb_rec",
                "length": 5,
                "img_shape": [[1080, 2048], [1080, 2280]],
                "roi_list": f"{root}/data/traffic_light_det_roilist/20220809_x8b_det_batch_all/20220817-102819/data_roilist.json",
            },
            {
                "rec_path": f"{root}/data/traffic_light_det_roilist/20221011_badcase_det_batch_46_66/20221012-123756/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/data/traffic_light_det_roilist/20221011_badcase_det_batch_46_66/20221012-123756/data.anno.pb_rec",
                "length": 6,
                "img_shape": [[1080, 2048], [1080, 2280]],
                "roi_list": f"{root}/data/traffic_light_det_roilist/20221011_badcase_det_batch_46_66/20221012-123756/data_roilist.json",
            },
            {
                "rec_path": f"{root}/data/traffic_light_det_roilist/20220809_220_det_batch_all/20220811-073101/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/data/traffic_light_det_roilist/20220809_220_det_batch_all/20220811-073101/data.anno.pb_rec",
                "length": 7,
                "img_shape": [[1080, 2048], [1080, 2280]],
                "roi_list": f"{root}/data/traffic_light_det_roilist/20220809_220_det_batch_all/20220811-073101/data_roilist.json",
            },
            {
                "rec_path": f"{root}/data/traffic_light_det_roilist/20220811_323_det_batch_all/20220816-172434/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/data/traffic_light_det_roilist/20220811_323_det_batch_all/20220816-172434/data.anno.pb_rec",
                "length": 8,
                "img_shape": [[1080, 2048], [1080, 2280]],
                "roi_list": f"{root}/data/traffic_light_det_roilist/20220811_323_det_batch_all/20220816-172434/data_roilist.json",
            },
            {
                "rec_path": f"{root}/data/traffic_light_det_roilist/20220809_820_det_batch_lx/20220818-053825/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/data/traffic_light_det_roilist/20220809_820_det_batch_lx/20220818-053825/data.anno.pb_rec",
                "length": 9,
                "img_shape": [[1080, 2048], [1080, 2280]],
                "roi_list": f"{root}/data/traffic_light_det_roilist/20220809_820_det_batch_lx/20220818-053825/data_roilist.json",
            },
            {
                "rec_path": f"{root}/data/traffic_light_det_roilist/20220809_10652_det_batch_all/20220811-230602/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/data/traffic_light_det_roilist/20220809_10652_det_batch_all/20220811-230602/data.anno.pb_rec",
                "length": 10,
                "img_shape": [[1080, 2048], [1080, 2280]],
                "roi_list": f"{root}/data/traffic_light_det_roilist/20220809_10652_det_batch_all/20220811-230602/data_roilist.json",
            },
            {
                "rec_path": f"{root}/data/traffic_light_det_roilist/20220809_x8b_det_batch_all/20220817-102819/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/data/traffic_light_det_roilist/20220809_x8b_det_batch_all/20220817-102819/data.anno.pb_rec",
                "length": 11,
                "img_shape": [[1080, 2048], [1080, 2280]],
                "roi_list": f"{root}/data/traffic_light_det_roilist/20220809_x8b_det_batch_all/20220817-102819/data_roilist.json",
            },
            {
                "rec_path": f"{root}/data/traffic_light_det_roilist/20221011_badcase_det_batch_46_66/20221012-123756/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/data/traffic_light_det_roilist/20221011_badcase_det_batch_46_66/20221012-123756/data.anno.pb_rec",
                "length": 12,
                "img_shape": [[1080, 2048], [1080, 2280]],
                "roi_list": f"{root}/../adas/big_model/train_dataset/data/traffic_light_det_roilist/20221011_badcase_det_batch_46_66/20221012-123756/data_roilist.json",
            },
            {
                "rec_path": f"{root}/data/traffic_light_night_det_roilist/20221011_cn_night_det_batch_14_21/20221012-182422/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/data/traffic_light_night_det_roilist/20221011_cn_night_det_batch_14_21/20221012-182422/data.anno.pb_rec",
                "length": 13,
                "img_shape": [[1080, 2048], [1080, 2280]],
                "roi_list": f"{root}/data/traffic_light_night_det_roilist/20221011_cn_night_det_batch_14_21/20221012-182422/data_roilist.json",
            },
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [],
    },
    "traffic_sign": {
        "train_batch_size_per_ctx": 48,
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_x8b_90205_20220609/train.rec",
                "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_x8b_90205_20220609/train.anno.pb_rec",
                "length": 90205,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_10652_part1_43_140991_20220425/train.rec",
                "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_10652_part1_43_140991_20220425/train.anno.pb_rec",
                "length": 140991,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_0820_part1_34_288977_20220425/train.rec",
                "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_0820_part1_34_288977_20220425/train.anno.pb_rec",
                "length": 288976,
                "img_shape": [[2160, 3840]],
            },
            # partly labelled
            # {
            #     "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_10635_part1_4_406765_20220425/train.rec",
            #     "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_10635_part1_4_406765_20220425/train.anno.pb_rec",
            #     "length": 406765,
            #     "img_shape": [[720, 1280]],
            # },
            {
                "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_0323_part1_26_164468_20220425/train.rec",
                "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_0323_part1_26_164468_20220425/train.anno.pb_rec",
                "length": 164468,
                "img_shape": [[1080, 2280], [1080, 2048], [910, 1920]],
            },
            {
                "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_0220_part_1_23_162724_20220425/train.rec",
                "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_0220_part_1_23_162724_20220425/train.anno.pb_rec",
                "length": 162724,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_traffic_sign/mono_longtail_data_26637_20210129/train.rec",
                "anno_path": f"{root}/data/4pe_traffic_sign/mono_longtail_data_26637_20210129/train.anno.pb_rec",
                "sample_weight": 1,
                "length": 26637,
                "img_shape": [
                    [1080, 1920],
                    [720, 1280],
                    [1080, 2048],
                    [940, 1824],
                ],
            },
            # only small signs are labelled
            # {
            #     "rec_path": f"{root}/data/4pe_traffic_sign/mono_10652_realcar_badcase_31071_20210906/train.rec",
            #     "anno_path": f"{root}/data/4pe_traffic_sign/mono_10652_realcar_badcase_31071_20210906/train.anno.pb_rec",
            #     "sample_weight": 1,
            #     "length": 31071,
            #     "img_shape": [[940, 1824]],
            # },
            {
                "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_10652_realcar_badcase_58461_20220425/train.rec",
                "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_10652_realcar_badcase_58461_20220425/train.anno.pb_rec",
                "length": 49339,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_0820_0323_0220_night_65126_20220425/train.rec",
                "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_0820_0323_0220_night_65126_20220425/train.anno.pb_rec",
                "length": 65126,
                "img_shape": [[1080, 2048], [2160, 3840], [940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_assist_140154_20220425/train.rec",
                "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_assist_140154_20220425/train.anno.pb_rec",
                "length": 140154,
                "img_shape": [
                    [940, 1824],
                    [1080, 2048],
                    [720, 1280],
                    [2160, 3840],
                    [1080, 1920],
                ],
            },
            {
                "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_longtail_30696_20220425/train.rec",
                "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_traindata_mergebbox_longtail_30696_20220425/train.anno.pb_rec",
                "length": 30696,
                "img_shape": [
                    [940, 1824],
                    [1080, 2048],
                    [720, 1280],
                    [2160, 3840],
                    [1080, 1920],
                ],
            },
            # only circle signs are labelled
            # {
            #     "rec_path": f"{root}/data/4pe_traffic_sign/mono_assist_v4_55067_20210129/train.rec",
            #     "anno_path": f"{root}/data/4pe_traffic_sign/mono_assist_v4_55067_20210129/train.anno.pb_rec",
            #     "length": 55067,
            #     "img_shape": [
            #         [1080, 2280],
            #         [940, 1824],
            #         [1080, 2048],
            #         [720, 1280],
            #         [1080, 1920],
            #     ],
            # },
            # # below val set is used as training set
            # {
            #     "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_0220_part6_710301_20220425/train.rec",
            #     "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_0220_part6_710301_20220425/train.anno.pb_rec",
            #     "length": 10301,
            #     "img_shape": [[940, 1824]],
            # },
            # {
            #     "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_10652_25047_20220425/train.rec",
            #     "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_10652_25047_20220425/train.anno.pb_rec",
            #     "length": 25047,
            #     "img_shape": [[940, 1824]],
            # },
            # {
            #     "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_10652_badcase_22302_20220425/train.rec",
            #     "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_10652_badcase_22302_20220425/train.anno.pb_rec",
            #     "length": 22302,
            #     "img_shape": [[940, 1824]],
            # },
            # {
            #     "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_0820_13537_20220425/train.rec",
            #     "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_0820_13537_20220425/train.anno.pb_rec",
            #     "length": 13537,
            #     "img_shape": [[2160, 3840]],
            # },
            # {
            #     "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_x8b_31860_20220425/train.rec",
            #     "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_x8b_31860_20220425/train.anno.pb_rec",
            #     "length": 31860,
            #     "img_shape": [[2160, 3840]],
            # },
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [
            # {
            #     "rec_path": f"{root}/data/4pe_traffic_sign/mono_assist_v4_55067_20210129/train.rec",
            #     "anno_path": f"{root}/data/4pe_traffic_sign/mono_assist_v4_55067_20210129/train.anno.pb_rec",
            #     "sample_weight": 1,
            #     "length": 55067,
            #     "img_shape": [[940, 1824]],
            # },
            # {
            #     "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_0323_part4_7_11319_20220425/train.rec",
            #     "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_0323_part4_7_11319_20220425/train.anno.pb_rec",
            #     "length": 11319,
            #     "img_shape": [[910, 1920]],
            # },
            # {
            #     "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_0220_part6_710301_20220425/train.rec",
            #     "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_0220_part6_710301_20220425/train.anno.pb_rec",
            #     "length": 10301,
            #     "img_shape": [[940, 1824]],
            # },
            # {
            #     "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_10652_25047_20220425/train.rec",
            #     "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_10652_25047_20220425/train.anno.pb_rec",
            #     "length": 25047,
            #     "img_shape": [[940, 1824]],
            # },
            # {
            #     "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_10652_badcase_22302_20220425/train.rec",
            #     "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_10652_badcase_22302_20220425/train.anno.pb_rec",
            #     "length": 22302,
            #     "img_shape": [[940, 1824]],
            # },
            # {
            #     "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_0820_13537_20220425/train.rec",
            #     "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_0820_13537_20220425/train.anno.pb_rec",
            #     "length": 13537,
            #     "img_shape": [[2160, 3840]],
            # },
            # {
            #     "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_x8b_31860_20220425/train.rec",
            #     "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_valdata_mergebbox_x8b_31860_20220425/train.anno.pb_rec",
            #     "length": 31860,
            #     "img_shape": [[2160, 3840]],
            # },
        ],
    },
    "lane": {
        "train_batch_size_per_ctx": 16,
        "train_data_paths": [
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_x8b_train/20230203-221044/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_x8b_train/20230203-221044/data.anno.pb_rec",
                "length": 28273,
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_0323_train/20230220-193655/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_0323_train/20230220-193655/data.anno.pb_rec",
                "length": 187370,
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_mono_hardcase_train/20230221-001608/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_mono_hardcase_train/20230221-001608/data.anno.pb_rec",
                "length": 68353,
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_0820_train/20230221-061039/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_0820_train/20230221-061039/data.anno.pb_rec",
                "length": 59487,
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_mono_train_20220803_20221201/20230208-133727/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_mono_train_20220803_20221201/20230208-133727/data.anno.pb_rec",
                "length": 14213,
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_mono_train_20220803_part1/20230217-210324/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_mono_train_20220803_part1/20230217-210324/data.anno.pb_rec",
                "length": 50008,
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_mono_train_20220803_part2/20230217-225735/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_mono_train_20220803_part2/20230217-225735/data.anno.pb_rec",
                "length": 45165,
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_mono_train_20220803_part3/20230218-004548/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_mono_train_20220803_part3/20230218-004548/data.anno.pb_rec",
                "length": 51193,
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_mono_train_20220803_part4/20230218-030625/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_mono_train_20220803_part4/20230218-030625/data.anno.pb_rec",
                "length": 43350,
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_mono_train_20220803_part5/20230218-061252/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/lane_instanceseg/lane_instanceseg_mono_train_20220803_part5/20230218-061252/data.anno.pb_rec",
                "length": 40085,
            },
        ],
    },
    "lane_parsing_6cls": {
        "train_batch_size_per_ctx": 16,
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/lane_6cls_wide_dec/10652_6cls_102503_day.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/lane_6cls_wide_dec/10652_6cls_102503_day.anno.pb_rec",
                "sample_weight": 1,
                "length": 102503,
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/x8b/x8b_20085_6cls.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/x8b/x8b_20085_6cls.json",
                "sample_weight": 1,
                "length": 20085,
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/lane_6cls_wide_dec/0220_6cls_132355.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/lane_6cls_wide_dec/0220_6cls_132355.anno.pb_rec",
                "sample_weight": 1,
                "length": 132355,
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/lane_6cls_wide_dec/0323_6cls_184093.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/lane_6cls_wide_dec/0323_6cls_184093.anno.pb_rec",
                "sample_weight": 1,
                "length": 184093,
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/lane_6cls_wide_dec/0820_6cls_59488.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/lane_6cls_wide_dec/0820_6cls_59488.anno.pb_rec",
                "sample_weight": 1,
                "length": 59488,
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/lane_6cls_wide_dec/hadcase_6cls_65566.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/lane_6cls_wide_dec/hadcase_6cls_65566.anno.pb_rec",
                "sample_weight": 1,
                "length": 65566,
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/x8b/x8b_29003_6cls.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/x8b/x8b_29003_6cls.anno.pb_rec",
                "length": 29003,
                "img_shape": [[2160, 3840]],
            },
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/x8b/X8B_val/Rhode_X8B_Test_Dataset_LaneParsing_Urban_Day_6cls_1500_val.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/x8b/X8B_val/Rhode_X8B_Test_Dataset_LaneParsing_Urban_Day_6cls_1500_val.json",
                "sample_weight": 1,
                "length": 1500,
            },
        ],
    },
    "lane_parsing_4cls": {
        "train_batch_size_per_ctx": 16,
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/shun.wang/lane_parsing_America_new_v3/train.rec",
                "anno_path": f"{root}/data/shun.wang/lane_parsing_America_new_v3/train.anno.pb_rec",
                "sample_weight": 1,
                "length": 43500,
            },
            {
                "rec_path": f"{root}/data/shun.wang/lane_parsing_America_new_v2/train.rec",
                "anno_path": f"{root}/data/shun.wang/lane_parsing_America_new_v2/train.anno.pb_rec",
                "sample_weight": 1,
                "length": 8410,
            },
            {
                "rec_path": f"{root}/data/shun.wang/lane_parsing_America_new/train.rec",
                "anno_path": f"{root}/data/shun.wang/lane_parsing_America_new/train.anno.pb_rec",
                "sample_weight": 1,
                "length": 10707,
            },
            {
                "rec_path": f"{root}/data/shun.wang/lane_parsing_America_old/matrix_2.x_us_1080p_train_pinhole_num_44488_shuffle_20200814_4classes/train.rec",
                "anno_path": f"{root}/data/shun.wang/lane_parsing_America_old/matrix_2.x_us_1080p_train_pinhole_num_44488_shuffle_20200814_4classes/train.anno.pb_rec",
                "sample_weight": 1,
                "length": 44488,
            },
            {
                "rec_path": f"{root}/data/shun.wang/lane_parsing_America_old/matrix_2.x_us_1080p_train_pinhole_front_day_num_3046_shuffle_20201010_4classes/train.rec",
                "anno_path": f"{root}/data/shun.wang/lane_parsing_America_old/matrix_2.x_us_1080p_train_pinhole_front_day_num_3046_shuffle_20201010_4classes/train.anno.pb_rec",
                "sample_weight": 1,
                "length": 3046,
            },
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [
            {
                "rec_path": f"{root}/../adas/big_model/eval_segmentation/lane_parsing_us_4class.rec",
                "anno_path": f"{root}/../adas/big_model/eval_segmentation/lane_parsing_us_4class.anno.pb_rec",
                "sample_weight": 1,
            },
        ],
    },
    "lane_parsing_5cls": {
        "train_batch_size_per_ctx": 16,
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/10652/10652_day_102503.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/10652/10652_day_102503.anno.pb_rec",
                "sample_weight": 1.875,
                "length": 102503,
                "img_shape": [[2160, 3840], [940, 1824]],  # [H, W]
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/323/anno_mono_5cls_wide_720p_201223/train.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/323/anno_mono_5cls_wide_720p_201223/train.anno.pb_rec",
                "sample_weight": 1,
                "length": 132307,
                "img_shape": [[720, 1280]],  # [H, W]
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/323/anno_mono_5cls_wide_0323_20210218/train.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/323/anno_mono_5cls_wide_0323_20210218/train.anno.pb_rec",
                "sample_weight": 1.25,
                "length": 151144,
                "img_shape": [[1080, 2048], [1080, 2280], [910, 1920]]
                # [H, W]
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/323/anno_mono_5cls_0323_rampdata_20210416/train.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/323/anno_mono_5cls_0323_rampdata_20210416/train.anno.pb_rec",
                "sample_weight": 0.625,
                "length": 32957,
                "img_shape": [[1080, 2048], [1080, 2280], [910, 1920]]
                # [H, W]
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/0220/220_5cls_132355_1.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/0220/220_5cls_132355_1.anno.pb_rec",
                "sample_weight": 1.25,
                "length": 132355,
                "img_shape": [[940, 1824]],  # [H, W]
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/lane_question/hardcase_85175_5cls.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/lane_question/hardcase_85175_5cls.anno.pb_rec",
                "sample_weight": 1.25,
                "length": 85175,
                "img_shape": [
                    [940, 1824],
                    [1080, 2048],
                    [720, 1280],
                    [2160, 3840],
                    [1080, 1920],
                ],  # [H, W]
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/lane_question/deceleration_wide_58801.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/lane_question/deceleration_wide_58801.anno.pb_rec",
                "sample_weight": 0.625,
                "length": 58801,
                "img_shape": [
                    [1080, 2280],
                    [940, 1824],
                    [720, 1280],
                    [910, 1920],
                    [1080, 2048],
                ],  # [H, W]
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/0820/train_820_46113_5cls.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/0820/train_820_46113_5cls.anno.pb_rec",
                "sample_weight": 1.25,
                "length": 46113,
                "img_shape": [[2160, 3840]],  # [H, W]
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/lane_question/hardcase_26310_5cls.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/lane_question/hardcase_26310_5cls.anno.pb_rec",
                "sample_weight": 0.625,
                "length": 26310,
                "img_shape": [
                    [1080, 1920],
                    [720, 1280],
                    [1080, 2048],
                    [2160, 3840],
                ],  # [H, W]
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/x8b/x8b_18717_5cls.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/x8b/x8b_18717_5cls.anno.pb_rec",
                "length": 18717,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/323/anno_mono_5cls_wide_0323_only_night_20210312/train.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/323/anno_mono_5cls_wide_0323_only_night_20210312/train.anno.pb_rec",
                "length": 65272,
                "img_shape": [[1080, 2280], [1080, 2048], [910, 1920]],
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/323/anno_mono_5cls_onlywide_0323_20210317_heishizi/train.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/323/anno_mono_5cls_onlywide_0323_20210317_heishizi/train.anno.pb_rec",
                "length": 27919,
                "img_shape": [[1080, 2280], [1080, 2048], [910, 1920]],
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/0220/0220_data_night_5cls.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/0220/0220_data_night_5cls.anno.pb_rec",
                "length": 26319,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/323/train_anno_mono_5cls_10652_night_20210625_n13953.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/323/train_anno_mono_5cls_10652_night_20210625_n13953.anno.pb_rec",
                "length": 13674,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/0820/0820_night_5cls_14282.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/0820/0820_night_5cls_14282.anno.pb_rec",
                "length": 14282,
                "img_shape": [[2160, 3840]],
            },
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_lane_parsing/323/train_anno_mono_5cls_10652_night_20210625_n13953.rec",
                "anno_path": f"{root}/data/4pe_lane_parsing/323/train_anno_mono_5cls_10652_night_20210625_n13953.anno.pb_rec",
                "sample_weight": 1,
                "length": 13674,
                "img_shape": [
                    [1080, 1920],
                    [720, 1280],
                    [1080, 2048],
                    [2160, 3840],
                ],  # [H, W]
            }
        ],
    },
    "semantic_parsing_40cls": {
        "train_batch_size_per_ctx": 16,
        "train_data_paths": [
            # {  # overlap with parsing-1080p, used for fast debug
            #     "rec_path": f"{root}/../adas/big_model/train_dataset/default_parsing_40Cls/mono_0820_40cls_num_11818/20230224-233846/data.rec",
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/default_parsing_40Cls/mono_0820_40cls_num_11818/20230224-233846/data.anno.pb_rec",
            #     "sample_weight": 1,
            #     "length": 11884,
            # },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/default_parsing_40Cls/parsing-720p10635/20230227-201911/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/default_parsing_40Cls/parsing-720p10635/20230227-201911/data.anno.pb_rec",
                "sample_weight": 1,
                "length": 54117,
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/default_parsing_40Cls/parsing-1080p/20230228-034731/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/default_parsing_40Cls/parsing-1080p/20230228-034731/data.anno.pb_rec",
                "sample_weight": 1,
                "length": 44602,
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/default_parsing_40Cls/parsing-1080p-underground/20230228-062706/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/default_parsing_40Cls/parsing-1080p-underground/20230228-062706/data.anno.pb_rec",
                "sample_weight": 1,
                "length": 29273,
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/default_parsing_40Cls/parsing-1080p-badcase/20230228-081222/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/default_parsing_40Cls/parsing-1080p-badcase/20230228-081222/data.anno.pb_rec",
                "sample_weight": 1,
                "length": 11594,
            },
        ],
    },
    "semantic_parsing_33cls": {
        "train_batch_size_per_ctx": 16,
        "train_data_paths": [
            {
                "rec_path": f"{root}/../adas/big_model/semantic_parsing_33cls/0820_20211201_33cls_12143.rec",
                "anno_path": f"{root}/../adas/big_model/semantic_parsing_33cls/0820_20211201_33cls_12143.anno.pb_rec",
                "sample_weight": 1,
                "length": 12143,
            },
            {
                "rec_path": f"{root}/../adas/big_model/semantic_parsing_33cls/720p_33cls_59066.rec",
                "anno_path": f"{root}/../adas/big_model/semantic_parsing_33cls/720p_33cls_59066.anno.pb_rec",
                "sample_weight": 1,
                "length": 59066,
            },
            {
                "rec_path": f"{root}/../adas/big_model/semantic_parsing_33cls/10652_20211203_33cls_30756.rec",
                "anno_path": f"{root}/../adas/big_model/semantic_parsing_33cls/10652_20211203_33cls_30756.anno.pb_rec",
                "sample_weight": 1,
                "length": 30756,
            },
        ],
    },
    "semantic_parsing_16cls": {
        "train_batch_size_per_ctx": 16,
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_default_parsing/0820/anno_16.2_merge_car/data_0820_x8b_2022-06-15/train-1080p-id20220615_66631.rec",
                "anno_path": f"{root}/data/4pe_default_parsing/0820/anno_16.2_merge_car/data_0820_x8b_2022-06-15/train-1080p-id20220615_66631.anno.pb_rec",
                "sample_weight": 8,
                "length": 66631,
            },
            {
                "rec_path": f"{root}/data/4pe_default_parsing/10635/anno_16.2_merge_car/parsing-720p10635-train-anno16.2_59070.rec",
                "anno_path": f"{root}/data/4pe_default_parsing/10635/anno_16.2_merge_car/parsing-720p10635-train-anno16.2_59070.anno.pb_rec",
                "sample_weight": 8,
                "length": 59070,
            },
            {
                "rec_path": f"{root}/data/4pe_default_parsing/0820/anno_16.2_merge_car/data_0820_x8b_2022-06-15/train-1080p-id20220615-underground_30398.rec",
                "anno_path": f"{root}/data/4pe_default_parsing/0820/anno_16.2_merge_car/data_0820_x8b_2022-06-15/train-1080p-id20220615-underground_30398.anno.pb_rec",
                "sample_weight": 8,
                "length": 30398,
            },
            # {
            #     "rec_path": f"{root}/data/4pe_default_parsing/0820/anno_16.2_merge_car/data_0820_x8b_2022-07-11/train-1080p-id20220711_68701.rec",
            #     "anno_path": f"{root}/data/4pe_default_parsing/0820/anno_16.2_merge_car/data_0820_x8b_2022-07-11/train-1080p-id20220711_68701.anno.pb_rec",
            #     "sample_weight": 8,
            #     "length": 68701,
            #     "img_shape": [
            #         [1080, 2280],
            #         [940, 1824],
            #         [1080, 2048],
            #         [2160, 3840],
            #         [1080, 1920],
            #     ],
            # },
            # {
            #     "rec_path": f"{root}/data/4pe_default_parsing/0820/anno_16.2_merge_car/data_0820_x8b_2022-07-11/train-1080p-id20220711-underground_31275.rec",
            #     "anno_path": f"{root}/data/4pe_default_parsing/0820/anno_16.2_merge_car/data_0820_x8b_2022-07-11/train-1080p-id20220711-underground_31275.anno.pb_rec",
            #     "sample_weight": 8,
            #     "length": 31275,
            #     "img_shape": [[720, 1280], [2160, 3840]],
            # },
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [
            {
                "rec_path": f"{root}/../adas/big_model/eval_segmentation/X8b_0820_Test_Dataset_SemanticSeg_Cls16_2523.rec",
                "anno_path": f"{root}/../adas/big_model/eval_segmentation/X8b_0820_Test_Dataset_SemanticSeg_Cls16_2523.anno.pb_rec",
                "sample_weight": 1,
                "length": 2523,
            },
        ],
    },
    "semantic_parsing_12cls": {
        "train_batch_size_per_ctx": 16,
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_default_parsing/10652/anno_12.5_merge_car/data_10652_2021-11-05/train-1080p-id20211104_81036.rec",
                "anno_path": f"{root}/data/4pe_default_parsing/10652/anno_12.5_merge_car/data_10652_2021-11-05/train-1080p-id20211104_81036.anno.pb_rec",
                "sample_weight": 8,
                "length": 81036,
                "img_shape": [
                    [1080, 2280],
                    [940, 1824],
                    [1080, 2048],
                    [2160, 3840],
                    [1080, 1920],
                ],  # [H, W]
            },
            {
                "rec_path": f"{root}/data/4pe_default_parsing/10652/anno_12.5_merge_car/data_10652_2021-09-06/train-720p-id20200315_59486.rec",
                "anno_path": f"{root}/data/4pe_default_parsing/10652/anno_12.5_merge_car/data_10652_2021-09-06/train-720p-id20200315_59486.anno.pb_rec",
                "sample_weight": 8,
                "length": 59486,
                "img_shape": [[720, 1280]],  # [H, W]
            },
            {
                "rec_path": f"{root}/data/4pe_default_parsing/10652/anno_12.5_merge_car/data_10652_2022-06-24/train-1080p-id20220623_81762.rec",
                "anno_path": f"{root}/data/4pe_default_parsing/10652/anno_12.5_merge_car/data_10652_2022-06-24/train-1080p-id20220623_81762.anno.pb_rec",
                "length": 81762,
                "img_shape": [
                    [1080, 2280],
                    [940, 1824],
                    [1080, 2048],
                    [2160, 3840],
                    [1080, 1920],
                ],
            },
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_default_parsing/10652/anno_12.5_merge_car/data_10652_2021-09-06/val-1080p-id20200320_2201.rec",
                "anno_path": f"{root}/data/4pe_default_parsing/10652/anno_12.5_merge_car/data_10652_2021-09-06/val-1080p-id20200320_2201.anno.pb_rec",
                "sample_weight": 1,
                "length": 2201,
            },
            {
                "rec_path": f"{root}/data/4pe_default_parsing/10652/anno_12.5_merge_car/data_10652_2021-11-05/train-1080p-id20211104_81036.rec",
                "anno_path": f"{root}/data/4pe_default_parsing/10652/anno_12.5_merge_car/data_10652_2021-11-05/train-1080p-id20211104_81036.anno.pb_rec",
                "sample_weight": 1,
                "length": 81036,
                "img_shape": [
                    [1080, 2280],
                    [940, 1824],
                    [1080, 2048],
                    [2160, 3840],
                    [1080, 1920],
                ],  # [H, W]
            },
            {
                "rec_path": f"{root}/data/4pe_default_parsing/10652/anno_12.5_merge_car/data_10652_2021-09-06/val-720p-id20200320_4183.rec",
                "anno_path": f"{root}/data/4pe_default_parsing/10652/anno_12.5_merge_car/data_10652_2021-09-06/val-720p-id20200320_4183.anno.pb_rec",
                "length": 4183,
                "img_shape": [[720, 1280]],
            },
        ],
    },
    "ihbc": {
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_ihbc/ihbcv2_det_10652_47103_22.03.24_densebox/train.rec",
                "anno_path": f"{root}/data/4pe_ihbc/ihbcv2_det_10652_47103_22.03.24_densebox/train_attrs.anno.pb_rec",
                "length": 47103,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_ihbc/ihbcv2_0220_14943_21.05.20_densebox/train.rec",
                "anno_path": f"{root}/data/4pe_ihbc/ihbcv2_0220_14943_21.05.20_densebox/train_attrs.anno.pb_rec",
                "length": 14943,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_ihbc/ihbcv2_0323_21514_21.03.17_densebox/train.rec",
                "anno_path": f"{root}/data/4pe_ihbc/ihbcv2_0323_21514_21.03.17_densebox/train_attrs.anno.pb_rec",
                "length": 21514,
                "img_shape": [[1080, 2048], [1080, 2280]],
            },
            {
                "rec_path": f"{root}/data/4pe_ihbc/ihbcv2_0820_28479_21.06.16_densebox/train.rec",
                "anno_path": f"{root}/data/4pe_ihbc/ihbcv2_0820_28479_21.06.16_densebox/train_attrs.anno.pb_rec",
                "length": 28479,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_ihbc/ihbcv2_10652_4136_21.05.08_densebox/train.rec",
                "anno_path": f"{root}/data/4pe_ihbc/ihbcv2_10652_4136_21.05.08_densebox/train_attrs.anno.pb_rec",
                "length": 4136,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_ihbc/ihbcv2_det_10652_4180_22.04.02_badcase_densebox/train.rec",
                "anno_path": f"{root}/data/4pe_ihbc/ihbcv2_det_10652_4180_22.04.02_badcase_densebox/train_attrs2.anno.pb_rec",
                "length": 4180,
                "img_shape": [[940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_ihbc/ihbcv2_det_10652_9676_22.06.24_badcase_densebox/train.rec",
                "anno_path": f"{root}/data/4pe_ihbc/ihbcv2_det_10652_9676_22.06.24_badcase_densebox/train_attrs.anno.pb_rec",
                "length": 9675,
                "img_shape": [[940, 1824]],
            },
        ],
        "val_data_paths": [],
    },
    "road_arrow": {
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_road_arrow/4pe_road_arrow-train_dataset-CN0323train/data.rec",
                "anno_path": f"{root}/data/4pe_road_arrow/4pe_road_arrow-train_dataset-CN0323train/data.anno.pb_rec",
                "length": 97055,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/xiaojin_data/road_arrow_all/4pe/4pe_road_arrow_day_20210415/train.rec",
                "anno_path": f"{root}/xiaojin_data/road_arrow_all/4pe/4pe_road_arrow_day_20210415/train.anno.pb_rec",
                "length": 71661,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/xiaojin_data/road_arrow_all/4pe/4pe_road_arrow_night_20210428-20210601/train.rec",
                "anno_path": f"{root}/xiaojin_data/road_arrow_all/4pe/4pe_road_arrow_night_20210428-20210601/train.anno.pb_rec",
                "length": 55169,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/xiaojin_data/road_arrow_all/4pe_addx8b_roilist/4pe_road_arrow_CNX8B_day_20220328/train.rec",
                "anno_path": f"{root}/xiaojin_data/road_arrow_all/4pe_addx8b_roilist/4pe_road_arrow_CNX8B_day_20220328/train.anno.pb_rec",
                "length": 14387,
                "img_shape": [[2160, 3840]],
                "roi_list": f"{root}/xiaojin_data/road_arrow_all/4pe_addx8b_roilist/4pe_road_arrow_CNX8B_day_20220328/train.roi.json",
            },
            {
                "rec_path": f"{root}/xiaojin_data/road_arrow_all/4pe_addx8b_roilist/4pe_road_arrow_CNX8B_under_park_20220223/train.rec",
                "anno_path": f"{root}/xiaojin_data/road_arrow_all/4pe_addx8b_roilist/4pe_road_arrow_CNX8B_under_park_20220223/train.anno.pb_rec",
                "length": 375,
                "img_shape": [[2160, 3840]],
                "roi_list": f"{root}/xiaojin_data/road_arrow_all/4pe_addx8b_roilist/4pe_road_arrow_CNX8B_under_park_20220223/train.roi.json",
            },
            # {
            #     "rec_path": f"{root}/xiaojin_data/road_arrow_all/4pe_addx8b_roilist/4pe_road_arrow_CNX8B_night_20220328/train.rec",
            #     "anno_path": f"{root}/xiaojin_data/road_arrow_all/4pe_addx8b_roilist/4pe_road_arrow_CNX8B_night_20220328/train.anno.pb_rec",
            #     "length": 1123,
            #     "img_shape": [[2160, 3840]],
            #     "roi_list": f"{root}/xiaojin_data/road_arrow_all/4pe_addx8b_roilist/4pe_road_arrow_CNX8B_night_20220328/train.roi.json",
            # },
            # {
            #     "rec_path": f"{root}/xiaojin_data/road_arrow_all/4pe_addx8b_roilist/4pe_road_arrow_CNX8B_day_20220223/train.rec",
            #     "anno_path": f"{root}/xiaojin_data/road_arrow_all/4pe_addx8b_roilist/4pe_road_arrow_CNX8B_day_20220223/train.anno.pb_rec",
            #     "length": 3051,
            #     "img_shape": [[2160, 3840]],
            #     "roi_list": f"{root}/xiaojin_data/road_arrow_all/4pe_addx8b_roilist/4pe_road_arrow_CNX8B_day_20220223/train.roi.json",
            # },
            {
                "rec_path": f"{root}/xiaojin_data/road_arrow_all/4pe_addx8b_roilist/4pe_road_arrow_CNX8B_day_20211208/train.rec",
                "anno_path": f"{root}/xiaojin_data/road_arrow_all/4pe_addx8b_roilist/4pe_road_arrow_CNX8B_day_20211208/train.anno.pb_rec",
                "length": 5064,
                "img_shape": [[2160, 3840]],
                "roi_list": f"{root}/xiaojin_data/road_arrow_all/4pe_addx8b_roilist/4pe_road_arrow_CNX8B_day_20211208/train.roi.json",
            },
            {
                "rec_path": f"{root}/xiaojin_data/road_arrow_all/4pe/4pe_road_arrow_day_20210512-20210901/train.rec",
                "anno_path": f"{root}/xiaojin_data/road_arrow_all/4pe/4pe_road_arrow_day_20210512-20210901/train.anno.pb_rec",
                "length": 36913,
                "img_shape": [[2160, 3840], [940, 1824]],
            },
            {
                "rec_path": f"{root}/xiaojin_data/road_arrow_all/4pe/4pe_road_arrow_night_20210611-20210901/train.rec",
                "anno_path": f"{root}/xiaojin_data/road_arrow_all/4pe/4pe_road_arrow_night_20210611-20210901/train.anno.pb_rec",
                "length": 53180,
                "img_shape": [[2160, 3840]],
            },
        ],
        "val_data_paths": [],
    },
    "traffic_cone": {
        "train_data_paths": [
            {
                "rec_path": f"{root}/jianyu.hong/dataset/train_val/date20220606/train/CN_underground/4pe_traffic_cone/train.rec",
                "anno_path": f"{root}/jianyu.hong/dataset/train_val/date20220606/train/CN_underground/4pe_traffic_cone/train.anno.pb_rec",
                "length": 74139,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/jianyu.hong/dataset/train_val/date20220606/train/CN_day/4pe_traffic_cone/train.rec",
                "anno_path": f"{root}/jianyu.hong/dataset/train_val/date20220606/train/CN_day/4pe_traffic_cone/train.anno.pb_rec",
                "length": 250171,
                "img_shape": [[2160, 3840], [1080, 2048], [940, 1824]],
            },
            {
                "rec_path": f"{root}/data/traffic_cone_4pe/CN_night_4pe_traffic_cone_train/20220812-224346/data.rec",
                "anno_path": f"{root}/data/traffic_cone_4pe/CN_night_4pe_traffic_cone_train/20220812-224346/data.anno.pb_rec",
                "length": 141820,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/jianyu.hong/dataset/train_val/date20220606/train/US_day/4pe_traffic_cone/train.rec",
                "anno_path": f"{root}/jianyu.hong/dataset/train_val/date20220606/train/US_day/4pe_traffic_cone/train.anno.pb_rec",
                "length": 40366,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/jianyu.hong/dataset/train_val/date20220606/train/US_night/4pe_traffic_cone/train.rec",
                "anno_path": f"{root}/jianyu.hong/dataset/train_val/date20220606/train/US_night/4pe_traffic_cone/train.anno.pb_rec",
                "length": 5636,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/traffic_cone_4pe/badcase_4pe_traffic_cone_train/20220907-162008/data.rec",
                "anno_path": f"{root}/data/traffic_cone_4pe/badcase_4pe_traffic_cone_train/20220907-162008/data.anno.pb_rec",
                "length": 19806,
                "img_shape": [[2160, 3840]],
            },
            # # SuperParking data begins
            # # parking_column
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/column/pack_densebox_column_data_2022-03-28_2022-09-30/train/train.rec",  # noqa
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/SuperParking/LastVersion/Dataset/SOD/V1.9.3/column/pack_densebox_column_data_2022-03-28_2022-09-30/train/train.anno.pb_rec",  # noqa
            #     "length": 44099,
            #     "img_shape": [[1280, 1920]],
            # },
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.0/column/pack_densebox_column_data_2022-10-15_2022-10-31/train/train.rec",  # noqa
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/SuperParking/LastVersion/Dataset/SOD/V1.9.0/column/pack_densebox_column_data_2022-10-15_2022-10-31/train/train.anno.pb_rec",  # noqa
            #     "length": 3983,
            #     "img_shape": [[1280, 1920]],
            # },
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/column/pack_densebox_column_data_2022-10-01_2022-10-09/train/train.rec",  # noqa
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/SuperParking/LastVersion/Dataset/SOD/V1.9.3/column/pack_densebox_column_data_2022-10-01_2022-10-09/train/train.anno.pb_rec",  # noqa
            #     "length": 5781,
            #     "img_shape": [[1280, 1920]],
            # },
            # # parking_lock_open
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/parking_lock_open/pack_densebox_parking_lock_data_2022-03-22_2022-09-30/train/train.rec",  # noqa
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/SuperParking/LastVersion/Dataset/SOD/V1.9.3/parking_lock_open/pack_densebox_parking_lock_data_2022-03-22_2022-09-30/train/train.anno.pb_rec",  # noqa
            #     "length": 10269,
            #     "img_shape": [[1280, 1920]],
            # },
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.0/parking_lock_open/pack_densebox_parking_lock_data_2022-10-15_2022-10-31/train/train.rec",  # noqa
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/SuperParking/LastVersion/Dataset/SOD/V1.9.0/parking_lock_open/pack_densebox_parking_lock_data_2022-10-15_2022-10-31/train/train.anno.pb_rec",  # noqa
            #     "length": 1770,
            #     "img_shape": [[1280, 1920]],
            # },
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/parking_lock_open/pack_densebox_parking_lock_data_2022-10-01_2022-10-09/train/train.rec",  # noqa
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/SuperParking/LastVersion/Dataset/SOD/V1.9.3/parking_lock_open/pack_densebox_parking_lock_data_2022-10-01_2022-10-09/train/train.anno.pb_rec",  # noqa
            #     "length": 977,
            #     "img_shape": [[1280, 1920]],
            # },
            # # parking_lock_close
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/parking_lock_close/pack_densebox_parking_lock_data_2022-03-22_2022-09-30/train/train.rec",  # noqa
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/SuperParking/LastVersion/Dataset/SOD/V1.9.3/parking_lock_close/pack_densebox_parking_lock_data_2022-03-22_2022-09-30/train/train.anno.pb_rec",  # noqa
            #     "length": 22706,
            #     "img_shape": [[1280, 1920]],
            # },
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.0/parking_lock_close/pack_densebox_parking_lock_data_2022-10-15_2022-10-31/train/train.rec",  # noqa
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/SuperParking/LastVersion/Dataset/SOD/V1.9.0/parking_lock_close/pack_densebox_parking_lock_data_2022-10-15_2022-10-31/train/train.anno.pb_rec",  # noqa
            #     "length": 2594,
            #     "img_shape": [[1280, 1920]],
            # },
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/parking_lock_close/pack_densebox_parking_lock_data_2022-10-01_2022-10-09/train/train.rec",  # noqa
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/SuperParking/LastVersion/Dataset/SOD/V1.9.3/parking_lock_close/pack_densebox_parking_lock_data_2022-10-01_2022-10-09/train/train.anno.pb_rec",  # noqa
            #     "length": 2056,
            #     "img_shape": [[1280, 1920]],
            # },
            # # traffic_bollard
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.0/traffic_bollard/pack_densebox_traffic_bollard_data_2022-10-15_2022-10-31/train/train.rec",  # noqa
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/SuperParking/LastVersion/Dataset/SOD/V1.9.0/traffic_bollard/pack_densebox_traffic_bollard_data_2022-10-15_2022-10-31/train/train.anno.pb_rec",  # noqa
            #     "length": 3342,
            #     "img_shape": [[1280, 1920]],
            # },
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/traffic_bollard/pack_densebox_traffic_bollard_data_2022-03-22_2022-09-30/train/train.rec",  # noqa
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/SuperParking/LastVersion/Dataset/SOD/V1.9.3/traffic_bollard/pack_densebox_traffic_bollard_data_2022-03-22_2022-09-30/train/train.anno.pb_rec",  # noqa
            #     "length": 57690,
            #     "img_shape": [[1280, 1920]],
            # },
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/traffic_bollard/pack_densebox_traffic_bollard_data_2022-10-01_2022-10-08/train/train.rec",  # noqa
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/SuperParking/LastVersion/Dataset/SOD/V1.9.3/traffic_bollard/pack_densebox_traffic_bollard_data_2022-10-01_2022-10-08/train/train.anno.pb_rec",  # noqa
            #     "length": 4469,
            #     "img_shape": [[1280, 1920]],
            # },
            # # traffic_cone
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.0/traffic_cone/pack_densebox_traffic_cone_data_2022-10-15_2022-10-31/train/train.rec",  # noqa
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/SuperParking/LastVersion/Dataset/SOD/V1.9.0/traffic_cone/pack_densebox_traffic_cone_data_2022-10-15_2022-10-31/train/train.anno.pb_rec",  # noqa
            #     "length": 11812,
            #     "img_shape": [[1280, 1920]],
            # },
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/traffic_cone/pack_densebox_traffic_cone_data_2022-03-22_2022-09-30/train/train.rec",  # noqa
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/SuperParking/LastVersion/Dataset/SOD/V1.9.3/traffic_cone/pack_densebox_traffic_cone_data_2022-03-22_2022-09-30/train/train.anno.pb_rec",  # noqa
            #     "length": 80868,
            #     "img_shape": [[1280, 1920]],
            # },
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/traffic_cone/pack_densebox_traffic_cone_data_2022-10-01_2022-10-08/train/train.rec",  # noqa
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/SuperParking/LastVersion/Dataset/SOD/V1.9.3/traffic_cone/pack_densebox_traffic_cone_data_2022-10-01_2022-10-08/train/train.anno.pb_rec",  # noqa
            #     "length": 12854,
            #     "img_shape": [[1280, 1920]],
            # },
            # # aframe_sign
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.0/AFrame/pack_densebox_aframe_data_2022-10-15_2022-10-31/train/train.rec",  # noqa
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/SuperParking/LastVersion/Dataset/SOD/V1.9.0/AFrame/pack_densebox_aframe_data_2022-10-15_2022-10-31/train/train.anno.pb_rec",  # noqa
            #     "length": 710,
            #     "img_shape": [[1280, 1920]],
            # },
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/AFrame/pack_densebox_aframe_data_2022-03-22_2022-09-30/train/train.rec",  # noqa
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/SuperParking/LastVersion/Dataset/SOD/V1.9.3/AFrame/pack_densebox_aframe_data_2022-03-22_2022-09-30/train/train.anno.pb_rec",  # noqa
            #     "length": 4240,
            #     "img_shape": [[1280, 1920]],
            # },
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/AFrame/pack_densebox_aframe_data_2022-10-01_2022-10-08/train/train.rec",  # noqa
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/SuperParking/LastVersion/Dataset/SOD/V1.9.3/AFrame/pack_densebox_aframe_data_2022-10-01_2022-10-08/train/train.anno.pb_rec",  # noqa
            #     "length": 596,
            #     "img_shape": [[1280, 1920]],
            # },
            # # SuperParking data ends
        ],
        "val_data_paths": [
            {
                "rec_path": f"{root}/jianyu.hong/dataset/train_val/date20220606/val/CN0820/4pe_traffic_cone/val.rec",
                "anno_path": f"{root}/jianyu.hong/dataset/train_val/date20220606/val/CN0820/4pe_traffic_cone/val.anno.pb_rec",
                "length": 19806,
                "img_shape": [[2160, 3840]],
            },
        ],
    },
    "vehicle_wheel": {
        "train_data_paths": [
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/wheel_attribute_cls/wheel_all_data_before_202208_attribute_1/20230108-225515/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/wheel_attribute_cls/wheel_all_data_before_202208_attribute_1/20230108-225515/data.anno.pb_rec",
                "length": 48802,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/wheel_attribute_cls/wheel_all_data_before_202208_attribute_2/20230109-055714/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/wheel_attribute_cls/wheel_all_data_before_202208_attribute_2/20230109-055714/data.anno.pb_rec",
                "length": 64461,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/wheel_attribute_cls/wheel_all_data_before_202208_attribute_3/20230106-195058/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/wheel_attribute_cls/wheel_all_data_before_202208_attribute_3/20230106-195058/data.anno.pb_rec",
                "length": 50904,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/wheel_attribute_cls/wheel_all_data_before_202208_attribute_4/20230107-015945/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/wheel_attribute_cls/wheel_all_data_before_202208_attribute_4/20230107-015945/data.anno.pb_rec",
                "length": 49988,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/wheel_attribute_cls/wheel_all_data_before_202208_attribute_5/20230111-172921/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/wheel_attribute_cls/wheel_all_data_before_202208_attribute_5/20230111-172921/data.anno.pb_rec",
                "length": 43176,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/wheel_attribute_cls/wheel_all_data_before_202208_attribute_6/20230106-190506/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/wheel_attribute_cls/wheel_all_data_before_202208_attribute_6/20230106-190506/data.anno.pb_rec",
                "length": 59757,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/wheel_attribute_cls/wheel_all_data_before_202208_attribute_7/20230106-232606/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/wheel_attribute_cls/wheel_all_data_before_202208_attribute_7/20230106-232606/data.anno.pb_rec",
                "length": 104911,
                "img_shape": [[1080, 2048]],
            },
        ],
        "val_data_paths": [],
    },
    "person_head": {
        "train_data_paths": [
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/person_head_attribute_cls/person_head_all_data_before_202208_attribute_1/20230109-103115/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/person_head_attribute_cls/person_head_all_data_before_202208_attribute_1/20230109-103115/data.anno.pb_rec",
                "length": 74064,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/person_head_attribute_cls/person_head_all_data_before_202208_attribute_2/20230106-195926/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/person_head_attribute_cls/person_head_all_data_before_202208_attribute_2/20230106-195926/data.anno.pb_rec",
                "length": 74284,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/person_head_attribute_cls/person_head_all_data_before_202208_attribute_3/20230106-211927/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/person_head_attribute_cls/person_head_all_data_before_202208_attribute_3/20230106-211927/data.anno.pb_rec",
                "length": 80882,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/person_head_attribute_cls/person_head_all_data_before_202208_attribute_4/20230108-230956/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/person_head_attribute_cls/person_head_all_data_before_202208_attribute_4/20230108-230956/data.anno.pb_rec",
                "length": 73009,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/person_head_attribute_cls/person_head_all_data_before_202208_attribute_5/20230109-023249/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/person_head_attribute_cls/person_head_all_data_before_202208_attribute_5/20230109-023249/data.anno.pb_rec",
                "length": 94859,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/person_head_attribute_cls/person_head_all_data_before_202208_attribute_6/20230109-052630/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/person_head_attribute_cls/person_head_all_data_before_202208_attribute_6/20230109-052630/data.anno.pb_rec",
                "length": 82532,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/person_head_attribute_cls/person_head_all_data_before_202208_attribute_9/20230109-105719/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/person_head_attribute_cls/person_head_all_data_before_202208_attribute_9/20230109-105719/data.anno.pb_rec",
                "length": 46083,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/person_head_4pe/person_head_20220803_20221201_1/20230214-161419/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/person_head_4pe/person_head_20220803_20221201_1/20230214-161419/data.anno.pb_rec",
                "length": 29482,
                "img_shape": [[1080, 2048]],
            },
        ],
        "val_data_paths": [],
    },
    "cyclist_wheel": {
        "train_data_paths": [
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/cyclist_wheel_4pe/cyclist_wheel_1/20230209-224819/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/cyclist_wheel_4pe/cyclist_wheel_1/20230209-224819/data.anno.pb_rec",
                "length": 47669,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/cyclist_wheel_4pe/cyclist_wheel_2/20230209-204003/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/cyclist_wheel_4pe/cyclist_wheel_2/20230209-204003/data.anno.pb_rec",
                "length": 2800,
                "img_shape": [[1080, 2048]],
            },
        ],
        "val_data_paths": [],
    },
    "person_face": {
        "train_data_paths": [
            {
                "rec_path": f"{root}/../matrix/users/jinchuan01.xiao/data/face/pilot_person_face_5v_X8B_all_4pe_2022-11-06_2022-12-15/20221216-200426/data.rec",
                "anno_path": f"{root}/../matrix/users/jinchuan01.xiao/data/face/pilot_person_face_5v_X8B_all_4pe_2022-11-06_2022-12-15/20221216-200426/data.anno.pb_rec",
                "length": 13984,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../matrix/users/jinchuan01.xiao/data/face/pilot_person_face_5v_as33-5v_all_4pe_2022-11-06_2022-12-15/20221216-221754/data.rec",
                "anno_path": f"{root}/../matrix/users/jinchuan01.xiao/data/face/pilot_person_face_5v_as33-5v_all_4pe_2022-11-06_2022-12-15/20221216-221754/data.anno.pb_rec",
                "length": 6392,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../matrix/users/jinchuan01.xiao/data/face/pilot_person_face_5v_c385-5v_all_4pe_2022-11-06_2022-11-30/20221217-002324/data.rec",
                "anno_path": f"{root}/../matrix/users/jinchuan01.xiao/data/face/pilot_person_face_5v_c385-5v_all_4pe_2022-11-06_2022-11-30/20221217-002324/data.anno.pb_rec",
                "length": 6644,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../matrix/users/jinchuan01.xiao/data/face/pilot_person_face_5v_c385-5v_all_4pe_2022-12-01_2022-12-15/20221217-011848/data.rec",
                "anno_path": f"{root}/../matrix/users/jinchuan01.xiao/data/face/pilot_person_face_5v_c385-5v_all_4pe_2022-12-01_2022-12-15/20221217-011848/data.anno.pb_rec",
                "length": 771,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../matrix/users/jinchuan01.xiao/data/face/pilot_person_face_5v_cc02-5v_all_4pe_2022-11-06_2022-12-15/20221217-013950/data.rec",
                "anno_path": f"{root}/../matrix/users/jinchuan01.xiao/data/face/pilot_person_face_5v_cc02-5v_all_4pe_2022-11-06_2022-12-15/20221217-013950/data.anno.pb_rec",
                "length": 743,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../matrix/users/jinchuan01.xiao/data/face/pilot_person_face_5v_pilot-pdt-5v_all_4pe_2022-11-06_2022-12-15/20221217-015438/data.rec",
                "anno_path": f"{root}/../matrix/users/jinchuan01.xiao/data/face/pilot_person_face_5v_pilot-pdt-5v_all_4pe_2022-11-06_2022-12-15/20221217-015438/data.anno.pb_rec",
                "length": 2786,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../matrix/users/jinchuan01.xiao/data/face/pilot_person_face_5v_galaxy-5v_all_4pe_2022-11-06_2022-12-15/20221217-020614/data.rec",
                "anno_path": f"{root}/../matrix/users/jinchuan01.xiao/data/face/pilot_person_face_5v_galaxy-5v_all_4pe_2022-11-06_2022-12-15/20221217-020614/data.anno.pb_rec",
                "length": 1001,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../matrix/users/jinchuan01.xiao/data/face/pilot_person_face_5v_0820_all_4pe_2022-12-16_2023-01-05/20230228-164156/data.rec",
                "anno_path": f"{root}/../matrix/users/jinchuan01.xiao/data/face/pilot_person_face_5v_0820_all_4pe_2022-12-16_2023-01-05/20230228-164156/data.anno.pb_rec",
                "length": 9009,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../matrix/users/jinchuan01.xiao/data/face/pilot_person_face_5v_0820_all_4pe_2022-02-02_2023-03-01/20230307-162917/data.rec",
                "anno_path": f"{root}/../matrix/users/jinchuan01.xiao/data/face/pilot_person_face_5v_0820_all_4pe_2022-02-02_2023-03-01/20230307-162917/data.anno.pb_rec",
                "length": 11627,
                "img_shape": [[1080, 2048]],
            },
        ],
        "val_data_paths": [],
    },
    "traffic_light_len": {
        "train_data_paths": [
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_1_1/20230213-130331/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_1_1/20230213-130331/data.anno.pb_rec",
                "length": 105606,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_1_2/20230213-165700/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_1_2/20230213-165700/data.anno.pb_rec",
                "length": 111110,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_1_3/20230213-235408/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_1_3/20230213-235408/data.anno.pb_rec",
                "length": 141161,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_1_4/20230213-171511/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_1_4/20230213-171511/data.anno.pb_rec",
                "length": 98150,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_1_5/20230213-161120/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_1_5/20230213-161120/data.anno.pb_rec",
                "length": 89931,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_2/20230213-172806/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_2/20230213-172806/data.anno.pb_rec",
                "length": 1497,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_3/20230213-173558/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_3/20230213-173558/data.anno.pb_rec",
                "length": 6609,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_4/20230213-173058/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_4/20230213-173058/data.anno.pb_rec",
                "length": 7136,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_5/20230214-122419/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_5/20230214-122419/data.anno.pb_rec",
                "length": 11410,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_6/20230213-193007/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_6/20230213-193007/data.anno.pb_rec",
                "length": 60047,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_20220803_20221201/20230210-193505/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_light_len_4pe/traffic_light_len_20220803_20221201/20230210-193505/data.anno.pb_rec",
                "length": 12799,
                "img_shape": [[1080, 2048]],
            },
        ],
        "val_data_paths": [],
    },
    "cycle": {
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/person_det_multitask/cycle_det_4pe_20230209/20230209-183136/data.rec",
                "anno_path": f"{root}/data/person_det_multitask/cycle_det_4pe_20230209/20230209-183136/data.anno.pb_rec",
                "length": 16216,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_two_wheel_car/20221222_cycle_in_vehicle_data/train.rec",
                "anno_path": f"{root}/data/4pe_two_wheel_car/20221222_cycle_in_vehicle_data/train.anno.pb_rec",
                "length": 194585,
                "img_shape": [[2160, 3840]],
            },
        ],
        "val_data_paths": [],
    },
    # "face": {
    #     "train_data_paths": [
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/face_attribute_cls/face_attribute_1/20221214-115257/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/face_attribute_cls/face_attribute_1/20221214-115257/data.anno.pb_rec",
    #             "length": 41180,
    #             "img_shape": [[1080, 2048]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/face_attribute_cls/face_attribute_3/20221214-200355/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/face_attribute_cls/face_attribute_3/20221214-200355/data.anno.pb_rec",
    #             "length": 38605,
    #             "img_shape": [[1080, 2048]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/face_attribute_cls/face_attribute_5/20221214-033811/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/face_attribute_cls/face_attribute_5/20221214-033811/data.anno.pb_rec",
    #             "length": 39145,
    #             "img_shape": [[1080, 2048]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/face_attribute_cls/face_attribute_6/20221214-131324/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/face_attribute_cls/face_attribute_6/20221214-131324/data.anno.pb_rec",
    #             "length": 32646,
    #             "img_shape": [[1080, 2048]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/face_attribute_cls/face_attribute_7/20221214-163837/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/face_attribute_cls/face_attribute_7/20221214-163837/data.anno.pb_rec",
    #             "length": 21632,
    #             "img_shape": [[1080, 2048]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/face_attribute_cls/face_attribute_8/20221214-060654/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/face_attribute_cls/face_attribute_8/20221214-060654/data.anno.pb_rec",
    #             "length": 34495,
    #             "img_shape": [[1080, 2048]],
    #         },
    #         {
    #             "rec_path": f"{root}/../adas/big_model/train_dataset/face_attribute_cls/face_attribute_9/20221214-132353/data.rec",
    #             "anno_path": f"{root}/../adas/big_model/train_dataset/face_attribute_cls/face_attribute_9/20221214-132353/data.anno.pb_rec",
    #             "length": 31712,
    #             "img_shape": [[1080, 2048]],
    #         },
    #     ],
    #     "val_data_paths": [],
    # },
    "vehicle_light": {
        "train_data_paths": [
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_light_4pe/light_0/20230201-171208/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_light_4pe/light_0/20230201-171208/data.anno.pb_rec",
                "length": 6111,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_light_4pe/light_1/20230201-172625/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_light_4pe/light_1/20230201-172625/data.anno.pb_rec",
                "length": 27646,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_light_4pe/light_2/20230201-174128/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_light_4pe/light_2/20230201-174128/data.anno.pb_rec",
                "length": 33278,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_light_4pe/light_3/20230202-112923/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_light_4pe/light_3/20230202-112923/data.anno.pb_rec",
                "length": 13120,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_light_4pe/vehicle_light_0/20230201-210533/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_light_4pe/vehicle_light_0/20230201-210533/data.anno.pb_rec",
                "length": 4673,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_light_4pe/vehicle_light_1/20230201-203425/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_light_4pe/vehicle_light_1/20230201-203425/data.anno.pb_rec",
                "length": 46572,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_light_4pe/vehicle_light_2/20230201-204406/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_light_4pe/vehicle_light_2/20230201-204406/data.anno.pb_rec",
                "length": 76777,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_light_4pe/vehicle_light_3/20230202-165100/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_light_4pe/vehicle_light_3/20230202-165100/data.anno.pb_rec",
                "length": 78731,
                "img_shape": [[1080, 2048]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/vehicle_light_4pe/vehicle_light_4/20230202-161915/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/vehicle_light_4pe/vehicle_light_4/20230202-161915/data.anno.pb_rec",
                "length": 21437,
                "img_shape": [[1080, 2048]],
            },
        ],
        "val_data_paths": [],
    },
    "vehicle_plate": {
        "train_data_paths": [
            {
                "rec_path": f"{root}/../matrix/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_X8B_all_2pe_2022-07-16_2022-08-16/20220818-205449/data.rec",
                "anno_path": f"{root}/../matrix/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_X8B_all_2pe_2022-07-16_2022-08-16/20220818-205449/data.anno.pb_rec",
                "length": 9704,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../matrix/users/yuanzhuo.peng/tmp/mono_rear_plate_5v_X8B_all_4pe/20220719-152400/data.rec",
                "anno_path": f"{root}/../matrix/users/yuanzhuo.peng/tmp/mono_rear_plate_5v_X8B_all_4pe/20220719-152400/data.anno.pb_rec",
                "length": 5902,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../matrix/users/yuanzhuo.peng/tmp/mono_rear_plate_5v_X8B_all_4pe/20220719-170628/data.rec",
                "anno_path": f"{root}/../matrix/users/yuanzhuo.peng/tmp/mono_rear_plate_5v_X8B_all_4pe/20220719-170628/data.anno.pb_rec",
                "length": 8233,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../matrix/users/jinchuan01.xiao/data/plate/0820/v220305/4pe/train.rec",
                "anno_path": f"{root}/../matrix/users/jinchuan01.xiao/data/plate/0820/v220305/4pe/train.anno.pb_rec",
                "length": 14741,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../matrix/users/jinchuan01.xiao/data/plate/unknow/v220305/4pe/train.rec",
                "anno_path": f"{root}/../matrix/users/jinchuan01.xiao/data/plate/unknow/v220305/4pe/train.anno.pb_rec",
                "length": 38185,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../matrix/users/jinchuan01.xiao/data/plate/wissen/v220305/4pe/train.rec",
                "anno_path": f"{root}/../matrix/users/jinchuan01.xiao/data/plate/wissen/v220305/4pe/train.anno.pb_rec",
                "length": 28297,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../matrix/users/jinchuan01.xiao/data/plate/x3c/v220305/4pe/train.rec",
                "anno_path": f"{root}/../matrix/users/jinchuan01.xiao/data/plate/x3c/v220305/4pe/train.anno.pb_rec",
                "length": 37759,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../matrix/users/jinchuan01.xiao/data/plate/plate_v211108/train/train.rec",
                "anno_path": f"{root}/../matrix/users/jinchuan01.xiao/data/plate/plate_v211108/train/train.anno.pb_rec",
                "length": 89366,
                "img_shape": [[2160, 3840]],
            },
        ],
        "val_data_paths": [],
    },
    # there is only cone class in these datasets
    "cone": {
        "train_data_paths": [
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_cone_4pe_sd_cn_day/20230328-182615/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_cone_4pe_sd_cn_day/20230328-182615/data.anno.pb_rec",
                "length": 142372,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_cone_4pe_sd_cn_night/20230328-164556/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_cone_4pe_sd_cn_night/20230328-164556/data.anno.pb_rec",
                "length": 73227,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_cone_4pe_sd_cn_underground/20230328-160256/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_cone_4pe_sd_cn_underground/20230328-160256/data.anno.pb_rec",
                "length": 23915,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_cone_4pe_sd_us_day/20230328-163728/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_cone_4pe_sd_us_day/20230328-163728/data.anno.pb_rec",
                "length": 14972,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_cone_4pe_sd_us_night/20230328-164501/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_cone_4pe_sd_us_night/20230328-164501/data.anno.pb_rec",
                "length": 1654,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_cone_4pe_sd_badcase/20230328-170021/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_cone_4pe_sd_badcase/20230328-170021/data.anno.pb_rec",
                "length": 8290,
                "img_shape": [[2160, 3840]],
            },
            # SuperParking data
            {
                "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.0/traffic_cone/pack_densebox_traffic_cone_data_2022-10-15_2022-10-31/train/train.rec",  # noqa
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/traffic_cone_superparking_data_multiple_classes/horizon-bucket/SuperParking/LastVersion/Dataset/SOD/V1.9.0/traffic_cone/pack_densebox_traffic_cone_data_2022-10-15_2022-10-31/train/train.anno.pb_rec",  # noqa
                "length": 11812,
                "img_shape": [[1280, 1920]],
            },
            {
                "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/traffic_cone/pack_densebox_traffic_cone_data_2022-03-22_2022-09-30/train/train.rec",  # noqa
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/traffic_cone_superparking_data_multiple_classes/horizon-bucket/SuperParking/LastVersion/Dataset/SOD/V1.9.3/traffic_cone/pack_densebox_traffic_cone_data_2022-03-22_2022-09-30/train/train.anno.pb_rec",  # noqa
                "length": 80868,
                "img_shape": [[1280, 1920]],
            },
            {
                "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/traffic_cone/pack_densebox_traffic_cone_data_2022-10-01_2022-10-08/train/train.rec",  # noqa
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/traffic_cone_superparking_data_multiple_classes/horizon-bucket/SuperParking/LastVersion/Dataset/SOD/V1.9.3/traffic_cone/pack_densebox_traffic_cone_data_2022-10-01_2022-10-08/train/train.anno.pb_rec",  # noqa
                "length": 12854,
                "img_shape": [[1280, 1920]],
            },
        ],
        "val_data_paths": [],
    },
    "traffic_bollard": {
        "train_data_paths": [
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_bollard_4pe_sd_cn_day/20230328-214050/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_bollard_4pe_sd_cn_day/20230328-214050/data.anno.pb_rec",
                "length": 16092,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_bollard_4pe_sd_cn_night/20230328-202858/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_bollard_4pe_sd_cn_night/20230328-202858/data.anno.pb_rec",
                "length": 6504,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_bollard_4pe_sd_cn_underground/20230328-202356/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_bollard_4pe_sd_cn_underground/20230328-202356/data.anno.pb_rec",
                "length": 6400,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_bollard_4pe_sd_us_day/20230328-205718/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_bollard_4pe_sd_us_day/20230328-205718/data.anno.pb_rec",
                "length": 7637,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_bollard_4pe_sd_us_night/20230328-203535/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_bollard_4pe_sd_us_night/20230328-203535/data.anno.pb_rec",
                "length": 792,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_bollard_4pe_sd_badcase/20230328-204753/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/traffic_bollard_4pe_sd_badcase/20230328-204753/data.anno.pb_rec",
                "length": 1389,
                "img_shape": [[2160, 3840]],
            },
            # SuperParking data
            {
                "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.0/traffic_bollard/pack_densebox_traffic_bollard_data_2022-10-15_2022-10-31/train/train.rec",  # noqa
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/traffic_cone_superparking_data_multiple_classes/horizon-bucket/SuperParking/LastVersion/Dataset/SOD/V1.9.0/traffic_bollard/pack_densebox_traffic_bollard_data_2022-10-15_2022-10-31/train/train.anno.pb_rec",  # noqa
                "length": 3342,
                "img_shape": [[1280, 1920]],
            },
            {
                "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/traffic_bollard/pack_densebox_traffic_bollard_data_2022-03-22_2022-09-30/train/train.rec",  # noqa
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/traffic_cone_superparking_data_multiple_classes/horizon-bucket/SuperParking/LastVersion/Dataset/SOD/V1.9.3/traffic_bollard/pack_densebox_traffic_bollard_data_2022-03-22_2022-09-30/train/train.anno.pb_rec",  # noqa
                "length": 57690,
                "img_shape": [[1280, 1920]],
            },
            {
                "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/traffic_bollard/pack_densebox_traffic_bollard_data_2022-10-01_2022-10-08/train/train.rec",  # noqa
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/traffic_cone_superparking_data_multiple_classes/horizon-bucket/SuperParking/LastVersion/Dataset/SOD/V1.9.3/traffic_bollard/pack_densebox_traffic_bollard_data_2022-10-01_2022-10-08/train/train.anno.pb_rec",  # noqa
                "length": 4469,
                "img_shape": [[1280, 1920]],
            },
        ],
        "val_data_paths": [],
    },
    "isolation_bollard": {
        "train_data_paths": [
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/isolation_bollard_4pe_sd_cn_day/20230330-141954/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/isolation_bollard_4pe_sd_cn_day/20230330-141954/data.anno.pb_rec",
                "length": 137542,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/isolation_bollard_4pe_sd_cn_night/20230328-225133/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/isolation_bollard_4pe_sd_cn_night/20230328-225133/data.anno.pb_rec",
                "length": 72994,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/isolation_bollard_4pe_sd_us_day/20230328-220851/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/isolation_bollard_4pe_sd_us_day/20230328-220851/data.anno.pb_rec",
                "length": 4110,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/isolation_bollard_4pe_sd_us_night/20230328-221507/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/isolation_bollard_4pe_sd_us_night/20230328-221507/data.anno.pb_rec",
                "length": 701,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/isolation_bollard_4pe_sd_cn_underground_negative_examples_small/20230406-113028/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/isolation_bollard_4pe_sd_cn_underground_negative_examples_small/20230406-113028/data.anno.pb_rec",
                "length": 12837,
                "img_shape": [[2160, 3840]],
            },
            # {
            #     "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/isolation_bollard_4pe_sd_cn_underground_negative_examples/20230404-144723/data.rec",
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/isolation_bollard_4pe_sd_cn_underground_negative_examples/20230404-144723/data.anno.pb_rec",
            #     "length": 63686,
            #     "img_shape": [[2160, 3840]],
            # },
            # there are unwanted classes in this dataset
            # {
            #     "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/isolation_bollard_4pe_sd_badcase/20230329-153053/data.rec",
            #     "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/isolation_bollard_4pe_sd_badcase/20230329-153053/data.anno.pb_rec",
            #     "length": 3497,
            #     "img_shape": [[2160, 3840]],
            # },
        ],
        "val_data_paths": [],
    },
    "crash_barrel": {
        "train_data_paths": [
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/crash_barrel_4pe_sd_cn_day/20230329-011957/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/crash_barrel_4pe_sd_cn_day/20230329-011957/data.anno.pb_rec",
                "length": 41210,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/crash_barrel_4pe_sd_cn_night/20230329-004142/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/crash_barrel_4pe_sd_cn_night/20230329-004142/data.anno.pb_rec",
                "length": 16362,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/crash_barrel_4pe_sd_cn_underground/20230329-012819/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/crash_barrel_4pe_sd_cn_underground/20230329-012819/data.anno.pb_rec",
                "length": 585,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/crash_barrel_4pe_sd_badcase/20230329-014404/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/crash_barrel_4pe_sd_badcase/20230329-014404/data.anno.pb_rec",
                "length": 1756,
                "img_shape": [[2160, 3840]],
            },
            # aframe_sign negative examples for removing aframe_sign FPS
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/crash_barrel_aframe_sign_negative_examples/20230412-205144/data.rec",
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/crash_barrel_aframe_sign_negative_examples/20230412-205144/data.anno.pb_rec",
                "length": 7162,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/aframe_sign_4pe_sd_mono_20230331/20230331-163905/data.rec",  # noqa
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/aframe_sign_4pe_sd_mono_20230331/20230331-163905/data.anno.pb_rec",  # noqa
                "length": 9204,
                "img_shape": [[1280, 1920]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/aframe_sign_4pe_sd_mono_20230407/20230412-165523/data.rec",  # noqa
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/aframe_sign_4pe_sd_mono_20230407/20230412-165523/data.anno.pb_rec",  # noqa
                "length": 22687,
                "img_shape": [[1280, 1920]],
            },
            # below is superparking data
            {
                "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.0/AFrame/pack_densebox_aframe_data_2022-10-15_2022-10-31/train/train.rec",  # noqa
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/traffic_cone_superparking_data_multiple_classes/horizon-bucket/SuperParking/LastVersion/Dataset/SOD/V1.9.0/AFrame/pack_densebox_aframe_data_2022-10-15_2022-10-31/train/train.anno.pb_rec",  # noqa
                "length": 710,
                "img_shape": [[1280, 1920]],
            },
            {
                "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/AFrame/pack_densebox_aframe_data_2022-03-22_2022-09-30/train/train.rec",  # noqa
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/traffic_cone_superparking_data_multiple_classes/horizon-bucket/SuperParking/LastVersion/Dataset/SOD/V1.9.3/AFrame/pack_densebox_aframe_data_2022-03-22_2022-09-30/train/train.anno.pb_rec",  # noqa
                "length": 4240,
                "img_shape": [[1280, 1920]],
            },
            {
                "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/AFrame/pack_densebox_aframe_data_2022-10-01_2022-10-08/train/train.rec",  # noqa
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/traffic_cone_superparking_data_multiple_classes/horizon-bucket/SuperParking/LastVersion/Dataset/SOD/V1.9.3/AFrame/pack_densebox_aframe_data_2022-10-01_2022-10-08/train/train.anno.pb_rec",  # noqa
                "length": 596,
                "img_shape": [[1280, 1920]],
            },
        ],
        "val_data_paths": [],
    },
    "aframe_sign": {
        "train_data_paths": [
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/aframe_sign_4pe_sd_mono_20230331/20230331-163905/data.rec",  # noqa
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/aframe_sign_4pe_sd_mono_20230331/20230331-163905/data.anno.pb_rec",  # noqa
                "length": 9204,
                "img_shape": [[1280, 1920]],
            },
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/aframe_sign_4pe_sd_mono_20230407/20230412-165523/data.rec",  # noqa
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/aframe_sign_4pe_sd_mono_20230407/20230412-165523/data.anno.pb_rec",  # noqa
                "length": 22687,
                "img_shape": [[1280, 1920]],
            },
            # below is superparking data
            {
                "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.0/AFrame/pack_densebox_aframe_data_2022-10-15_2022-10-31/train/train.rec",  # noqa
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/traffic_cone_superparking_data_multiple_classes/horizon-bucket/SuperParking/LastVersion/Dataset/SOD/V1.9.0/AFrame/pack_densebox_aframe_data_2022-10-15_2022-10-31/train/train.anno.pb_rec",  # noqa
                "length": 710,
                "img_shape": [[1280, 1920]],
            },
            {
                "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/AFrame/pack_densebox_aframe_data_2022-03-22_2022-09-30/train/train.rec",  # noqa
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/traffic_cone_superparking_data_multiple_classes/horizon-bucket/SuperParking/LastVersion/Dataset/SOD/V1.9.3/AFrame/pack_densebox_aframe_data_2022-03-22_2022-09-30/train/train.anno.pb_rec",  # noqa
                "length": 4240,
                "img_shape": [[1280, 1920]],
            },
            {
                "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/AFrame/pack_densebox_aframe_data_2022-10-01_2022-10-08/train/train.rec",  # noqa
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_superparking/traffic_cone_superparking_data_multiple_classes/horizon-bucket/SuperParking/LastVersion/Dataset/SOD/V1.9.3/AFrame/pack_densebox_aframe_data_2022-10-01_2022-10-08/train/train.anno.pb_rec",  # noqa
                "length": 596,
                "img_shape": [[1280, 1920]],
            },
            # crash_barrel negative examples for removing crash_barrel FPS
            {
                "rec_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/aframe_sign_crash_barrel_negative_examples/20230413-164941/data.rec",  # noqa
                "anno_path": f"{root}/../adas/big_model/train_dataset/traffic_cone_4pe_sd/aframe_sign_crash_barrel_negative_examples/20230413-164941/data.anno.pb_rec",  # noqa
                "length": 59250,
                "img_shape": [[1280, 1920]],
            },
        ],
        "val_data_paths": [],
    },
    "vehicle_side": {
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/vehicle_side_4pe_REMOVE_add_veh_type/vehicle_side_4pe_0820_jn/20230506-033229/data.rec",
                "anno_path": f"{root}/data/vehicle_side_4pe_REMOVE_add_veh_type/vehicle_side_4pe_0820_jn/20230506-033229/data.anno.pb_rec",
                "length": 33297,
                "img_shape": [[0, 0]],
            },
            {
                "rec_path": f"{root}/data/vehicle_side_4pe_REMOVE_add_veh_type/vehicle_side_4pe_x8b_jn/20230506-023641/data.rec",
                "anno_path": f"{root}/data/vehicle_side_4pe_REMOVE_add_veh_type/vehicle_side_4pe_x8b_jn/20230506-023641/data.anno.pb_rec",
                "length": 80097,
                "img_shape": [[0, 0]],
            },
            {
                "rec_path": f"{root}/data/vehicle_side_4pe_REMOVE_add_veh_type_v2/vehicle_side_4pe_x8b_jn/20230601-171803/data.rec",
                "anno_path": f"{root}/data/vehicle_side_4pe_REMOVE_add_veh_type_v2/vehicle_side_4pe_x8b_jn/20230601-171803/data.anno.pb_rec",
                "length": 107169,
                "img_shape": [[0, 0]],
            },
            {
                "rec_path": f"{root}/data/vehicle_side_4pe_REMOVE_add_veh_type/vehicle_side_4pe_x8b_jn/20230506-025006/data.rec",
                "anno_path": f"{root}/data/vehicle_side_4pe_REMOVE_add_veh_type/vehicle_side_4pe_x8b_jn/20230506-025006/data.anno.pb_rec",
                "length": 2340,
                "img_shape": [[0, 0]],
            },
            {
                "rec_path": f"{root}/data/vehicle_side_4pe_REMOVE_add_veh_type_v2/vehicle_side_4pe_x8b_jn/20230602-003934/data.rec",
                "anno_path": f"{root}/data/vehicle_side_4pe_REMOVE_add_veh_type_v2/vehicle_side_4pe_x8b_jn/20230602-003934/data.anno.pb_rec",
                "length": 126696,
                "img_shape": [[0, 0]],
            },
            {
                "rec_path": f"{root}/data/vehicle_side_4pe_REMOVE_add_veh_type_v2_fake_veh/vehicle_side_4pe_x8b_jn/20230601-212045/data.rec",
                "anno_path": f"{root}/data/vehicle_side_4pe_REMOVE_add_veh_type_v2_fake_veh/vehicle_side_4pe_x8b_jn/20230601-212045/data.anno.pb_rec",
                "length": 4127,
                "img_shape": [[0, 0]],
            },
            {
                "rec_path": f"{root}/data/vehicle_side_4pe_REMOVE_add_veh_type_v2/vehicle_side_4pe_x8b_jn/20230622-234104/data.rec",
                "anno_path": f"{root}/data/vehicle_side_4pe_REMOVE_add_veh_type_v2/vehicle_side_4pe_x8b_jn/20230622-234104/data.anno.pb_rec",
                "length": 53184,
                "img_shape": [[0, 0]],
            },
        ],
        "val_data_paths": [],
    },
}


mono_data_paths = EasyDict(mono_data_paths)
buckets = ["mono"]
