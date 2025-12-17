from easydict import EasyDict
from hatbc.filestream.bucket.client import get_bucket_mount_root

bucket2mount_root = get_bucket_mount_root()
root = bucket2mount_root.get("mono", None)
if root is None:
    raise FileNotFoundError("mono bucket")

canoo_data_paths = {
    "cyclist": {
        "train_batch_size_per_ctx": 8,
        "train_data_paths": [
            # {
            #     "rec_path": f"{root}/data/person_fz/20220412_multitask/20220328_ped_v20220412_train.rec",
            #     "anno_path": f"{root}/data/person_fz/20220412_multitask/20220328_ped_v20220412_train.anno.pb_rec",
            #     "roi_list_path": f"{root}/data/person_fz/20220412_multitask/20220328_ped_v20220412_train.roi.json",
            #     "sample_weight": 8.0,
            #     "length": 166492,
            #     "img_shape": [[2160, 3840]],
            # }
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [
            {
                "rec_path": f"{root}/data/person/eval_cyc_20200609/eval_cyc_20200609.rec",
                "anno_path": f"{root}/data/person/eval_cyc_20200609/eval_cyc_20200609.anno.pb_rec",
                "sample_weight": 1,
                "length": 7043,
                "img_shape": [[720, 1280]],
            }
        ],
    },
    "traffic_light": {
        "train_batch_size_per_ctx": 8,
        "train_data_paths": [
            # {
            #     "rec_path": f"{root}/data/4pe_traffic_light/20220112_220_det_batch_1-28_filter_balck/train.rec",
            #     "anno_path": f"{root}/data/4pe_traffic_light/20220112_220_det_batch_1-28_filter_balck/train.anno.pb_rec",
            #     "roi_list_path": None,
            #     "sample_weight": 1.7560975609756098,
            #     "length": 93443,
            #     "img_shape": [[1080, 2048], [940, 1824]],
            # },
            # {
            #     "rec_path": f"{root}/data/4pe_traffic_light/20220204_820_det_batch_1-13_filter_balck/train.rec",
            #     "anno_path": f"{root}/data/4pe_traffic_light/20220204_820_det_batch_1-13_filter_balck/train.anno.pb_rec",
            #     "roi_list_path": None,
            #     "sample_weight": 2.731707317073171,
            #     "length": 138255,
            #     "img_shape": [[2160, 3840]],
            # },
            # {
            #     "rec_path": f"{root}/data/4pe_traffic_light/20220107_10652_det_batch_1-15_filter_balck/train.rec",
            #     "anno_path": f"{root}/data/4pe_traffic_light/20220107_10652_det_batch_1-15_filter_balck/train.anno.pb_rec",
            #     "roi_list_path": None,
            #     "sample_weight": 1.5609756097560976,
            #     "length": 83515,
            #     "img_shape": [[940, 1824]],
            # },
            {
                "rec_path": f"{root}/data/4pe_traffic_light/20220402_x8b_det_batch_1-10_filter_balck/train.rec",
                "anno_path": f"{root}/data/4pe_traffic_light/20220402_x8b_det_batch_1-10_filter_balck/train.anno.pb_rec",
                "roi_list_path": None,
                "sample_weight": 0.5853658536585366,
                "length": 27698,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_traffic_light/20220402_820_us_det_batch_1-15_filter_balck/train.rec",
                "anno_path": f"{root}/data/4pe_traffic_light/20220402_820_us_det_batch_1-15_filter_balck/train.anno.pb_rec",
                "roi_list_path": None,
                "sample_weight": 1.3658536585365855,
                "length": 64989,
                "img_shape": [[2160, 3840]],
            },
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_traffic_light/densebox_traffic_light_color.val.aptiv.7.2.rec",
                "anno_path": f"{root}/data/4pe_traffic_light/densebox_traffic_light_color.val.aptiv.7.2.rec.anno.pb_rec",
                "sample_weight": 1,
                "length": 6806,
                "img_shape": [[720, 1280]],
            }
        ],
    },
    "person": {
        "train_batch_size_per_ctx": 24,
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/person_fz/20220412_multitask/20220328_ped_v20220412_train.rec",
                "anno_path": f"{root}/data/person_fz/20220412_multitask/20220328_ped_v20220412_train.anno.pb_rec",
                "roi_list_path": f"{root}/data/person_fz/20220412_multitask/20220328_ped_v20220412_train.roi.json",
                "sample_weight": 24.0,
                "length": 166492,
                "img_shape": [[2160, 3840]],
            }
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [],
    },
    "vehicle": {
        "train_batch_size_per_ctx": 17,
        "train_data_paths": [
            # {
            #     "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_x8b/det_vehicle_full_v4-20220415-day_under_park_v1/train.rec",
            #     "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_x8b/det_vehicle_full_v4-20220415-day_under_park_v1/train.rec.anno.pb_rec",
            #     "roi_list_path": None,
            #     "sample_weight": 7.285714285714286,
            #     "length": 62204,
            #     "img_shape": [[2160, 3840]],
            # },
            {
                "rec_path": f"{root}/data/4pe_vehicle/vehicle_full_data_canno/det_vehicle_full_v2-20220421-day_night_v1/train.rec",
                "anno_path": f"{root}/data/4pe_vehicle/vehicle_full_data_canno/det_vehicle_full_v2-20220421-day_night_v1/train.rec.anno.pb_rec",
                "roi_list_path": None,
                "sample_weight": 9.714285714285714,
                "length": 87987,
                "img_shape": [[2160, 3840]],
            },
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [],
    },
    "traffic_sign": {
        "train_batch_size_per_ctx": 7,
        "train_data_paths": [
            #     {
            #         "rec_path": f"{root}/data/4pe_traffic_sign_us/390item26sum34214day20220419_train_type/train.rec",
            #         "anno_path": f"{root}/data/4pe_traffic_sign_us/390item26sum34214day20220419_train_type/train.anno.pb_rec",
            #         "roi_list_path": None,
            #         "sample_weight": 1.75,
            #         "length": 34214,
            #         "img_shape": [[1080, 1920]],
            #     },
            #     {
            #         "rec_path": f"{root}/data/4pe_traffic_sign_us/D3RCMitem17sum27946day20220419_train_type/train.rec",
            #         "anno_path": f"{root}/data/4pe_traffic_sign_us/D3RCMitem17sum27946day20220419_train_type/train.anno.pb_rec",
            #         "roi_list_path": None,
            #         "sample_weight": 1.75,
            #         "length": 27946,
            #         "img_shape": [[1080, 1920]],
            #     },
            {
                "rec_path": f"{root}/data/4pe_traffic_sign_us/820item17sum64393day20220424_train_type/train.rec",
                "anno_path": f"{root}/data/4pe_traffic_sign_us/820item17sum64393day20220424_train_type/train.anno.pb_rec",
                "roi_list_path": None,
                "sample_weight": 3.5,
                "length": 64393,
                "img_shape": [[2160, 3840]],
            },
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_data_820_valdata_13537_20210604/train.rec",
                "anno_path": f"{root}/data/4pe_traffic_sign_j3mono/complete_data_820_valdata_13537_20210604/train.anno.pb_rec",
                "sample_weight": 1,
                "length": 13537,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_traffic_sign_us/val.rec",
                "anno_path": f"{root}/data/4pe_traffic_sign_us/val.anno.pb_rec",
                "sample_weight": 1,
                "length": 3918,
                "img_shape": [[1080, 1920], [2160, 3840]],
            },
        ],
    },
    "road_arrow": {
        "train_batch_size_per_ctx": 4,
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_road_arrow/20211105/rec_20211008/train.rec",
                "anno_path": f"{root}/data/4pe_road_arrow/20211105/rec_20211008/train.anno.pb_rec",
                "roi_list_path": None,
                "sample_weight": 0.8,
                "length": 170184,
                "img_shape": [[1080, 2048], [2160, 3840], [940, 1824]],
            },
            {
                "rec_path": f"{root}/data/4pe_road_arrow/20220420/v0.1.2/train.rec",
                "anno_path": f"{root}/data/4pe_road_arrow/20220420/v0.1.2/train.anno.pb_rec",
                "roi_list_path": None,
                "sample_weight": 3.2,
                "length": 62346,
                "img_shape": [[2160, 3840]],
            },
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_road_arrow/20210724/4pe_road_arrow-val_dataset-CN0820val/data.rec",
                "anno_path": f"{root}/data/4pe_road_arrow/20210724/4pe_road_arrow-val_dataset-CN0820val/data.anno.pb_rec",
                "sample_weight": 1,
                "length": 2971,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_road_arrow/20220420/v0.1.2/val.rec",
                "anno_path": f"{root}/data/4pe_road_arrow/20220420/v0.1.2/val.anno.pb_rec",
                "sample_weight": 1,
                "length": 6756,
                "img_shape": [[2160, 3840]],
            },
        ],
    },
    "rear": {
        "train_batch_size_per_ctx": 24,
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220415-x8b-cicd_and_under_park/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220415-x8b-cicd_and_under_park/data.anno.pb_rec",
                "roi_list_path": None,
                "sample_weight": 1.2496029238467714,
                "length": 9146,
                "img_shape": [[2160, 3840]],
            },
            # {
            #     "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220415-x8b-day_and_night/data.rec",
            #     "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220415-x8b-day_and_night/data.anno.pb_rec",
            #     "roi_list_path": None,
            #     "sample_weight": 9.801854729902823,
            #     "length": 71741,
            #     "img_shape": [[2160, 3840]],
            # },
            {
                "rec_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220421-canno-train-all/data.rec",
                "anno_path": f"{root}/data/4pe_rear/det_vehicle_rear_v13-20220421-canno-train-all/data.anno.pb_rec",
                "roi_list_path": None,
                "sample_weight": 12.948542346250406,
                "length": 94772,
                "img_shape": [[2160, 3840]],
            },
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [],
    },
    "semantic_parsing": {
        "train_batch_size_per_ctx": 10,
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/4pe_default_parsing/0820/anno_16.2_merge_car/data_0820_canoo_2022-04-14/train-1080p-id20220414_9258.rec",
                "anno_path": f"{root}/data/4pe_default_parsing/0820/anno_16.2_merge_car/data_0820_canoo_2022-04-14/train-1080p-id20220414_9258.anno.pb_rec",
                "roi_list_path": None,
                "sample_weight": 5.0,
                "length": 9258,
                "img_shape": [[1080, 1920], [2160, 3840]],
            },
            {
                "rec_path": f"{root}/data/4pe_default_parsing/0820/anno_16.2_merge_car/data_0820_canoo_2022-04-14/train-1080p-id20220414_56459.rec",
                "anno_path": f"{root}/data/4pe_default_parsing/0820/anno_16.2_merge_car/data_0820_canoo_2022-04-14/train-1080p-id20220414_56459.anno.pb_rec",
                "roi_list_path": None,
                "sample_weight": 5.0,
                "length": 56459,
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
        "val_data_paths": [],
    },
    "lane_parsing": {
        "train_batch_size_per_ctx": 10,
        "train_data_paths": [
            {
                "rec_path": f"{root}/data/shun.wang/lane_parsing_America_new_v4/train.rec",
                "anno_path": f"{root}/data/shun.wang/lane_parsing_America_new_v4/train.anno.pb_rec",
                "roi_list_path": None,
                "sample_weight": 10.0,
                "length": 14104,
                "img_shape": [[2160, 3840]],
            }
        ],
        "val_batch_size_per_ctx": 4,
        "val_data_paths": [
            {
                "rec_path": f"{root}/data/shun.wang/lane_parsing_America_new/val.rec",
                "anno_path": f"{root}/data/shun.wang/lane_parsing_America_new/val.anno.pb_rec",
                "sample_weight": 1,
                "length": 89,
                "img_shape": [[2160, 3840]],
            }
        ],
    },
    # "traffic_cone": {
    #     "train_batch_size_per_ctx": 4,
    #     "train_data_paths": [
    #         {
    #             "rec_path": f"{root}/data/4pe_traffic_cone/train_dataset/20220224/10652/4pe_traffic_cone/train.rec",
    #             "anno_path": f"{root}/data/4pe_traffic_cone/train_dataset/20220224/10652/4pe_traffic_cone/train.anno.pb_rec",
    #             "roi_list_path": None,
    #             "sample_weight": 2.2892655568732,
    #             "length": 192561,
    #             "img_shape": [[2160, 3840], [940, 1824]],
    #         },
    #         {
    #             "rec_path": f"{root}/data/4pe_traffic_cone/train_dataset/20220224/323_820_day/4pe_traffic_cone/train.rec",
    #             "anno_path": f"{root}/data/4pe_traffic_cone/train_dataset/20220224/323_820_day/4pe_traffic_cone/train.anno.pb_rec",
    #             "roi_list_path": None,
    #             "sample_weight": 1.0788595341482914,
    #             "length": 90748,
    #             "img_shape": [[1080, 2048], [2160, 3840]],
    #         },
    #         {
    #             "rec_path": f"{root}/data/4pe_traffic_cone/train_dataset/20220224/x8b_us820/4pe_traffic_cone/train.rec",
    #             "anno_path": f"{root}/data/4pe_traffic_cone/train_dataset/20220224/x8b_us820/4pe_traffic_cone/train.anno.pb_rec",
    #             "roi_list_path": None,
    #             "sample_weight": 0.6318749089785085,
    #             "length": 53150,
    #             "img_shape": [[2160, 3840]],
    #         },
    #     ],
    #     "val_batch_size_per_ctx": 4,
    #     "val_data_paths": [
    #         {
    #             "rec_path": f"{root}/data/4pe_traffic_cone/eval_dataset/4pe_traffic_cone/val.rec",
    #             "anno_path": f"{root}/data/4pe_traffic_cone/eval_dataset/4pe_traffic_cone/val.anno.pb_rec",
    #             "sample_weight": 1,
    #             "length": 13052,
    #             "img_shape": [[2160, 3840]],
    #         }
    #     ],
    # },
}

canoo_data_paths = EasyDict(canoo_data_paths)
buckets = ["mono"]
