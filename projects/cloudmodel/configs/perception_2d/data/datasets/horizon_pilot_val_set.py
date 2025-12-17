from easydict import EasyDict
from hatbc.filestream.bucket.client import get_bucket_mount_root

bucket2mount_root = get_bucket_mount_root()
root = bucket2mount_root.get("adas", None)
if root is None:
    raise FileNotFoundError("adas bucket")

pilot_val_data_paths = {
    "vehicle": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/vehicle_4pe/eval_rec/20230529-183205/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/vehicle_4pe/eval_rec/20230529-183205/data.anno.pb_rec",
                # "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_pilot_vehicle_6037154.json",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_pilot_vehicle_6037154_ignore_abandon.json",
                "length": 18211,
                "eval_id": 6037154,
            },
        ]
    },
    "rear": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/rear_4pe/eval_rec/20230526-183724/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/rear_4pe/eval_rec/20230526-183724/data.anno.pb_rec",
                # "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_pilot_vehicle_rear_6037566.json",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_pilot_vehicle_rear_6037566_ignore_abandon.json",
                "length": 15330,
                "eval_id": 6037566,
            },
        ]
    },
    "person": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/person_4pe/eval_rec/20230529-180012/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/person_4pe/eval_rec/20230529-180012/data.anno.pb_rec",
                # "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_pilot_person_6037573.json",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_pilot_person_6037573_ignore_abandon.json",
                "length": 7802,
                "eval_id": 6037573,
            },
        ]
    },
    "cyclist": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/cyclist_4pe/eval_rec/20230529-180525/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/cyclist_4pe/eval_rec/20230529-180525/data.anno.pb_rec",
                # "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_pilot_cyclist_6037135.json",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_pilot_cyclist_6037135_ignore_abandon.json",
                "length": 7802,
                "eval_id": 6037135,
            },
        ]
    },
    "semantic_parsing_40cls": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/default_parsing_40Cls/wissen_day_eval_40cls_num_5527/20230226-104004/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/default_parsing_40Cls/wissen_day_eval_40cls_num_5527/20230226-104004/data.anno.pb_rec",
                "sample_weight": 1,
                "length": 5527,
            },
            {
                "rec_path": f"{root}/big_model/eval_dataset/default_parsing_40Cls/wissen_night_eval_40cls_num_3328/20230226-110744/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/default_parsing_40Cls/wissen_night_eval_40cls_num_3328/20230226-110744/data.anno.pb_rec",
                "sample_weight": 1,
                "length": 3328,
            },
        ],
    },
    "lane": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/lane_instanceseg/lane_instanceseg_pilot_eval/20230208-183850/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/lane_instanceseg/lane_instanceseg_pilot_eval/20230208-183850/data.anno.pb_rec",
                "length": 5011,
            },
        ]
    },
}
pilot_val_data_paths = EasyDict(pilot_val_data_paths)
