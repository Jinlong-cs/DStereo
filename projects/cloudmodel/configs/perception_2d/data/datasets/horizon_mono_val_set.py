from easydict import EasyDict
from hatbc.filestream.bucket.client import get_bucket_mount_root

bucket2mount_root = get_bucket_mount_root()
root = bucket2mount_root.get("adas", None)
if root is None:
    raise FileNotFoundError("adas bucket")

mono_val_data_paths = {
    "vehicle": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/vehicle_4pe/eval_rec/20220811-214610/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/vehicle_4pe/eval_rec/20220811-214610/data.anno.pb_rec",
                # "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_vehicle_6028372.json",
                # "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_vehicle_6028372_ignore_bbox.json",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_vehicle_6028372_ignore_abandon.json",
                "length": 17128,
                "eval_id": 6028372,
            },
        ]
    },
    "rear": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/rear_4pe/eval_rec/20220812-132253/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/rear_4pe/eval_rec/20220812-132253/data.anno.pb_rec",
                # "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_rear_6028397.json",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_rear_6028397_ignore_abandon.json",
                "length": 20869,
                "eval_id": 6028397,
            },
        ]
    },
    "person": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/person_4pe/eval_rec/20220812-154919/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/person_4pe/eval_rec/20220812-154919/data.anno.pb_rec",
                # "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_person_6029315.json",
                # "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_person_6029315_remove_crowded.json",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_person_6029315_ignore_abandon.json",
                "length": 25113,
                "eval_id": 6029315,
            },
        ]
    },
    "cyclist": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/cyclist_4pe/eval_rec/20220812-155502/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/cyclist_4pe/eval_rec/20220812-155502/data.anno.pb_rec",
                # "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_cyclist_6029316.json",
                # "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_cyclist_remove_crowded.json",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_cyclist_6029316_ignore_abandon.json",
                "length": 24977,
                "eval_id": 6029316,
            },
        ]
    },
    "traffic_sign": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/traffic_sign_4pe/eval_rec/20220812-171440/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/traffic_sign_4pe/eval_rec/20220812-171440/data.anno.pb_rec",
                # "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_traffic_sign_6028564.json",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_traffic_sign_6028564_ignore_abandon.json",
                "length": 31860,
                "eval_id": 6028564,
            },
        ]
    },
    "road_arrow": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/road_arrow_4pe/eval_rec/20220812-173139/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/road_arrow_4pe/eval_rec/20220812-173139/data.anno.pb_rec",
                # "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_road_arrow_6028431.json",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_road_arrow_6028431_ignore_abandon.json",
                "length": 9470,
                "eval_id": 6028431,
            },
        ]
    },
    "traffic_cone": {
        "val_data_paths": [
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/parking_lock_open/pack_densebox_parking_lock_data_2022-03-22_2022-09-30/val/val.rec",  # noqa
            #     "anno_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/parking_lock_open/pack_densebox_parking_lock_data_2022-03-22_2022-09-30/val/val.anno.pb_rec",  # noqa
            #     "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_traffic_cone_6029527.json",
            #     "length": 25141,
            #     "eval_id": 6029527,
            # },
            # {
            #     "rec_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3/parking_lock_close/pack_densebox_parking_lock_data_2022-03-22_2022-09-30/val/val.rec",  # noqa
            #     "anno_path": f"{root}/../SuperParking/LastVersion/Dataset/SOD/V1.9.3//parking_lock_close/pack_densebox_parking_lock_data_2022-03-22_2022-09-30/val/val.anno.pb_rec",  # noqa
            #     "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_traffic_cone_6029527.json",
            #     "length": 25141,
            #     "eval_id": 6029527,
            # },
            {
                "rec_path": f"{root}/big_model/eval_dataset/traffic_cone_4pe/eval_rec/20220812-201845/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/traffic_cone_4pe/eval_rec/20220812-201845/data.anno.pb_rec",
                # "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_traffic_cone_6029527.json",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_traffic_cone_6029527_ignore_abandon.json",
                "length": 25141,
                "eval_id": 6029527,
            },
        ]
    },
    "traffic_light": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/traffic_light_4pe/eval_rec/20220817-152730/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/traffic_light_4pe/eval_rec/20220817-152730/data.anno.pb_rec",
                # "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_traffic_light_6028502.json",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_traffic_light_6028502_ignore_abandon.json",
                "length": 16165,
                "eval_id": 6028502,
            },
        ]
    },
    "vehicle_light": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/vehicle_light_4pe/20230202-111531/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/vehicle_light_4pe/20230202-111531/data.anno.pb_rec",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_vehicle_light.json",
                "length": 4018,
                "eval_id": 0000000,
            },
        ]
    },
    "vehicle_wheel": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/vehicle_wheel_4pe/vehicle_wheel_eval/20230203-143332/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/vehicle_wheel_4pe/vehicle_wheel_eval/20230203-143332/data.anno.pb_rec",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_vehicle_wheel.json",
                "length": 1094,
                "eval_id": 0000000,
            },
        ]
    },
    "person_head": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/person_head_4pe/person_head_eval/20230203-170648/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/person_head_4pe/person_head_eval/20230203-170648/data.anno.pb_rec",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_person_head.json",
                "length": 1729,
                "eval_id": 0000000,
            },
        ]
    },
    "cyclist_wheel": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/cyclist_wheel_4pe/cyclist_wheel_eval/20230209-191445/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/cyclist_wheel_4pe/cyclist_wheel_eval/20230209-191445/data.anno.pb_rec",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_cyclist_wheel.json",
                "length": 406,
                "eval_id": 0000000,
            },
        ]
    },
    "traffic_light_len": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/traffic_light_len_4pe/traffic_light_len_eval/20230210-193523/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/traffic_light_len_4pe/traffic_light_len_eval/20230210-193523/data.anno.pb_rec",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_traffic_light_len.json",
                "length": 1501,
                "eval_id": 0000000,
            },
        ]
    },
    "person_face": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/person_face_4pe/person_face_eval/20230213-200138/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/person_face_4pe/person_face_eval/20230213-200138/data.anno.pb_rec",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_person_face.json",
                "length": 634,
                "eval_id": 6039183,
            },
        ]
    },
    "cycle": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/cycle_4pe/cycle_eval/20230214-174623/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/cycle_4pe/cycle_eval/20230214-174623/data.anno.pb_rec",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_cycle.json",
                "length": 3228,
                "eval_id": 6039584,
            },
        ]
    },
    "vehicle_plate": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/vehicle_plate_4pe/eval_rec/mono_rear_plate_X8B_day_4pe/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/vehicle_plate_4pe/eval_rec/mono_rear_plate_X8B_day_4pe/data.anno.pb_rec",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_vehicle_plate_6040434.json",
                "length": 16646,
                "eval_id": 6040434,
            },
        ]
    },
    "cone": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/cone_4pe/eval_rec/20230328-171850/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/cone_4pe/eval_rec/20230328-171850/data.anno.pb_rec",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_cone.json",
                "length": 7893,
                "eval_id": 1,
            },
        ]
    },
    "traffic_bollard": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/traffic_bollard_4pe/eval_rec/20230328-210835/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/traffic_bollard_4pe/eval_rec/20230328-210835/data.anno.pb_rec",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_traffic_bollard.json",
                "length": 1049,
                "eval_id": 1,
            },
        ]
    },
    "isolation_bollard": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/isolation_bollard_4pe/eval_rec/20230328-224404/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/isolation_bollard_4pe/eval_rec/20230328-224404/data.anno.pb_rec",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_isolation_bollard.json",
                "length": 15028,
                "eval_id": 1,
            },
        ]
    },
    "crash_barrel": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/crash_barrel_4pe/eval_rec/20230329-020458/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/crash_barrel_4pe/eval_rec/20230329-020458/data.anno.pb_rec",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_crash_barrel.json",
                "length": 2705,
                "eval_id": 2,
            },
        ]
    },
    "aframe_sign": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/aframe_sign_4pe/eval_rec/val.rec",
                "anno_path": f"{root}/big_model/eval_dataset/aframe_sign_4pe/eval_rec/val.anno.pb_rec",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_aframe_sign.json",
                "length": 230,
                "eval_id": 1,
            },
            {
                "rec_path": f"{root}/big_model/eval_dataset/aframe_sign_4pe/aframe_sign_4pe_sd_mono_eval/20230414-182839/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/aframe_sign_4pe/aframe_sign_4pe_sd_mono_eval/20230414-182839/data.anno.pb_rec",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_aframe_sign_eval.json",
                "length": 5600,
                "eval_id": 2,
            },
        ]
    },
    "adb_light": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/train_dataset/adb_light/head/train.rec",
                "anno_path": f"{root}/big_model/train_dataset/adb_light/head_train.anno.pb_rec",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_head_light.json",
                "length": 16553,
                "eval_id": 1,
            },
            {
                "rec_path": f"{root}/big_model/train_dataset/adb_light/tail/train.rec",
                "anno_path": f"{root}/big_model/train_dataset/adb_light/tail_train.anno.pb_rec",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_tail_light.json",
                "length": 16553,
                "eval_id": 2,
            },
        ]
    },
    "semantic_parsing_40cls": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/default_parsing_40Cls/Rhode_CA82GB_Test_Dataset_SemanticSeg_Cls40_2251/20230226-100039/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/default_parsing_40Cls/Rhode_CA82GB_Test_Dataset_SemanticSeg_Cls40_2251/20230226-100039/data.anno.pb_rec",
                "length": 2251,
            },
            {
                "rec_path": f"{root}/big_model/eval_dataset/default_parsing_40Cls/Canoo_0820_Test_Dataset_SemanticSeg_Day_Cls40_1713/20230226-092427/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/default_parsing_40Cls/Canoo_0820_Test_Dataset_SemanticSeg_Day_Cls40_1713/20230226-092427/data.anno.pb_rec",
                "length": 1816,
            },
        ]
    },
    "semantic_parsing_33cls": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_segmentation/Rhode_CA82GB_Test_Dataset_SemanticSeg_Cls33_2251.rec",
                "anno_path": f"{root}/big_model/eval_segmentation/Rhode_CA82GB_Test_Dataset_SemanticSeg_Cls33_2251.anno.pb_rec",
                "length": 2251,
                "eval_id": 6027718,
            },
            {
                "rec_path": f"{root}/big_model/eval_segmentation/Canoo_0820_Test_Dataset_SemanticSeg_Day_Cls33_1095.rec",
                "anno_path": f"{root}/big_model/eval_segmentation/Canoo_0820_Test_Dataset_SemanticSeg_Day_Cls33_1095.anno.pb_rec",
                "length": 1095,
                "eval_id": 6027717,
            },
        ],
    },
    "lane": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/lane_instanceseg/lane_instanceseg_x8b_eval/20230203-224715/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/lane_instanceseg/lane_instanceseg_x8b_eval/20230203-224715/data.anno.pb_rec",
                "length": 4220,
            },
        ]
    },
    "vehicle_side": {
        "val_data_paths": [
            {
                "rec_path": f"{root}/big_model/eval_dataset/vehicle_side_4pe_REMOVE_add_veh_type/vehicle_side_6040553_eval_normal/20230713-172321/data.rec",
                "anno_path": f"{root}/big_model/eval_dataset/vehicle_side_4pe_REMOVE_add_veh_type/vehicle_side_6040553_eval_normal/20230713-172321/data.anno.pb_rec",
                "coco_anno_json": f"{root}/big_model/val_big_model_coco/val_mono_vehicle_side_6040553.json",
                "length": 12952,
                "eval_id": 6040553,
            },
        ]
    },
}
mono_val_data_paths = EasyDict(mono_val_data_paths)
