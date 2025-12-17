from easydict import EasyDict

adb_light_train_data_paths = {
    "adb_light": {
        "train_data_paths": [
            {
                "rec_path": f"/horizon-bucket/mono/data/4pe_ihbc/SD_20230228/20230228_SD_normal_43330_task_28817_add_washing_data_reflect/train.rec",  # noqa
                "anno_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/data_4pe_ihbc_SD_20230228_20230228_SD_normal_43330_task_28817_add_washing_data_reflect_train.anno.pb_rec",  # noqa
                "length": 1,
                "img_shape": [[1280, 1920]],
            },
            {
                "rec_path": f"/horizon-bucket/mono/data/4pe_ihbc/SD_20230208/20230207_SD_normal_66400_task_83277_90769_91381_92101_92109_add_washing_data_reflect/train.rec",  # noqa
                "anno_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/data_4pe_ihbc_SD_20230208_20230207_SD_normal_66400_task_83277_90769_91381_92101_92109_add_washing_data_reflect_train.anno.pb_rec",  # noqa
                "length": 2,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"/horizon-bucket/mono/data/4pe_ihbc/SD_20230115/20230115_SD_normal_41162_task_95489_head_add_washing_data_reflect_all/train.rec",  # noqa
                "anno_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/data_4pe_ihbc_SD_20230115_20230115_SD_normal_41162_task_95489_head_add_washing_data_reflect_all_train.anno.pb_rec",  # noqa
                "length": 3,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"/horizon-bucket/mono/data/4pe_ihbc/ADB_20221128/20221128_ADB_normal_7024_task_97312_light_oppo_direction/train.rec",  # noqa
                "anno_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/data_4pe_ihbc_ADB_20221128_20221128_ADB_normal_7024_task_97312_light_oppo_direction_train.anno.pb_rec",  # noqa
                "length": 4,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"/horizon-bucket/mono/data/4pe_ihbc/SD_20230228/20230228_SD_badcase_5495_task_81892_add_washing_data_reflect/train.rec",  # noqa
                "anno_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/data_4pe_ihbc_SD_20230228_20230228_SD_badcase_5495_task_81892_add_washing_data_reflect_train.anno.pb_rec",  # noqa
                "length": 5,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"/horizon-bucket/mono/data/4pe_ihbc/ADB_20221027/20221028_ADB_badcase_3856_task_CICD_Wuling_87625/train.rec",  # noqa
                "anno_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/data_4pe_ihbc_ADB_20221027_20221028_ADB_badcase_3856_task_CICD_Wuling_87625_train.anno.pb_rec",  # noqa
                "length": 6,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"/horizon-bucket/mono/data/4pe_ihbc/ADB_20221231/20221231_ADB_badcase_8640_wuling_ADB_task_135547_134574_135548_140452/train.rec",  # noqa
                "anno_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/data_4pe_ihbc_ADB_20221231_20221231_ADB_badcase_8640_wuling_ADB_task_135547_134574_135548_140452_train.anno.pb_rec",  # noqa
                "length": 7,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"/horizon-bucket/mono/data/4pe_ihbc/train/145322/train.rec",  # noqa
                "anno_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/data_4pe_ihbc_train_145322_train.anno.pb_rec",  # noqa
                "length": 8,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"/horizon-bucket/mono/data/4pe_ihbc/train/145323/train.rec",  # noqa
                "anno_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/data_4pe_ihbc_train_145323_train.anno.pb_rec",  # noqa
                "length": 9,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"/horizon-bucket/mono/data/4pe_ihbc/train/152943/train.rec",  # noqa
                "anno_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/data_4pe_ihbc_train_152943_train.anno.pb_rec",  # noqa
                "length": 10,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"/horizon-bucket/mono/data/4pe_ihbc/train/152944/train.rec",  # noqa
                "anno_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/data_4pe_ihbc_train_152944_train.anno.pb_rec",  # noqa
                "length": 11,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"/horizon-bucket/mono/data/4pe_ihbc/train/152945/train.rec",  # noqa
                "anno_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/data_4pe_ihbc_train_152945_train.anno.pb_rec",  # noqa
                "length": 12,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"/horizon-bucket/mono/data/4pe_ihbc/train/145317/train.rec",  # noqa
                "anno_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/data_4pe_ihbc_train_145317_train.anno.pb_rec",  # noqa
                "length": 13,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"/horizon-bucket/mono/data/4pe_ihbc/mono_data/rec/train/wash_20220518/train.rec",  # noqa
                "anno_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/data_4pe_ihbc_mono_data_rec_train_wash_20220518_train.anno.pb_rec",  # noqa
                "length": 14,
                "img_shape": [[2160, 3840]],
            },
            {
                "rec_path": f"/horizon-bucket/mono/data/4pe_ihbc/train/jira/train.rec",  # noqa
                "anno_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/data_4pe_ihbc_train_jira_train.anno.pb_rec",  # noqa
                "length": 15,
                "img_shape": [[2160, 3840]],
            },
        ],
        "val_data_paths": [],
    },
}
adb_light_val_data_paths = {
    "adb_light": {
        "val_data_paths": [
            {
                "rec_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/head/train.rec",  # noqa
                "anno_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/head_train.anno.pb_rec",  # noqa
                "coco_anno_json": f"/horizon-bucket/adas/big_model/val_big_model_coco/val_mono_head_light.json",  # noqa
                "length": 16553,
                "eval_id": 1,
            },
            {
                "rec_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/tail/train.rec",  # noqa
                "anno_path": f"/horizon-bucket/adas/big_model/train_dataset/adb_light/tail_train.anno.pb_rec",  # noqa
                "coco_anno_json": f"/horizon-bucket/adas/big_model/val_big_model_coco/val_mono_tail_light.json",  # noqa
                "length": 16553,
                "eval_id": 2,
            },
        ]
    },
}

adb_light_train_data_paths = EasyDict(adb_light_train_data_paths)
adb_light_val_data_paths = EasyDict(adb_light_val_data_paths)
