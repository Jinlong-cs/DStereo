dataset_ids = dict(
    detection={
        "vehicle_detection": [
            # nio 常规
            "6041419",
            # nio 异性车
            "6042169",  # 943
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            # 白天
            "6030530",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            # 白天
            "6030545",
        ],
        ("vehicle_truncation_classification", "vehicle_detection"): [
            # 白天
            "6030564",
        ],
        ("vehicle_wheel_detection", "vehicle_detection"): [
            "6029565",  # x3c
        ],
        ("vehicle_flank", "vehicle_detection"): [
            "6037360",  # galaxy day
        ],
        "rear_detection": [
            # nio 常规
            "6041430",
        ],
        ("rear_part_classification", "rear_detection"): [
            # 白天
            "6027896",  # X3C
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            # 白天
            "6030576",
        ],
        ("rear_plate_detection", "rear_detection"): [
            # 白天
            "6037604",
        ],
        "person_detection": [
            # nio 常规
            "6041486",
            # nio 柱状物
            "6043396",
        ],
        ("person_pose_classification", "person_detection"): [
            # 白天
            "6030583",
        ],
        ("person_occlusion_classification", "person_detection"): [
            # 白天
            "6030589",
        ],
        ("person_orientation_classification", "person_detection"): [
            # 白天
            "6037563",
        ],
        ("person_face_detection", "person_detection"): [
            # 白天
            "6037600",
        ],
        "cyclist_detection": [
            # nio 常规
            "6041487",
        ],
        ("cyclist_classification", "cyclist_detection"): [
            # 白天
            "6041860",
        ],
        ("person_roi_3d", "person_detection"): [
            # 白天
            "6041180",
            # c385 general eval day
            # "6042292",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            # 白天
            "6041182",
            # c385 general eval day
            # "6042293",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            "6041037",
            # truncation
            "6042094",
            # special
            "6042085",
            # occlusion
            "6042083",
            # rotation
            "6042081",
            # c385 general eval day
            # "6042278",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            # 白天
            "6042313",
        ],
        "lane_segmentation": [
            "6041476",
        ],
    },
)
