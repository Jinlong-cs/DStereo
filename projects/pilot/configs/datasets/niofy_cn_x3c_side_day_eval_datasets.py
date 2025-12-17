dataset_ids = dict(
    detection={
        "vehicle_detection": [
            # nio 常规
            "6041418",
            # nio 异性车
            "6042170",  # 3159
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            # 白天
            "6030216",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            # 白天
            "6030220",
        ],
        ("vehicle_truncation_classification", "vehicle_detection"): [
            # 白天
            "6030229",
        ],
        # ("vehicle_wheel_kps", "vehicle_detection"): [
        #     "6027903",  # x3c day
        # ],
        ("vehicle_wheel_detection", "vehicle_detection"): [
            "6029565",  # x3c day
        ],
        ("vehicle_ground_line", "vehicle_detection"): [
            "6027898",  # x3c day
        ],
        ("vehicle_flank", "vehicle_detection"): [
            "6037360",  # galaxy day
        ],
        "rear_detection": [
            # nio 常规
            "6041426",
        ],
        ("rear_part_classification", "rear_detection"): [
            # 白天
            "6027896",  # X3C
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            # 白天
            "6030238",
        ],
        ("rear_plate_detection", "rear_detection"): [
            # 白天
            "6037603",
        ],
        "person_detection": [
            # nio 常规
            "6041484",
            # nio 柱状物
            "6043379",
        ],
        ("person_pose_classification", "person_detection"): [
            # 白天
            "6030240",
        ],
        ("person_occlusion_classification", "person_detection"): [
            # 白天
            "6030244",
        ],
        ("person_orientation_classification", "person_detection"): [
            # 白天
            "6037558",
        ],
        ("person_face_detection", "person_detection"): [
            # 白天
            "6037601",
        ],
        "cyclist_detection": [
            # nio 常规
            "6041485",
        ],
        ("cyclist_classification", "cyclist_detection"): [
            # 白天
            "6041848",
        ],
        ("person_roi_3d", "person_detection"): [
            # 白天
            "6041184",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            # 白天
            "6041183",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            # 白天
            "6041128",
            # truncation
            "6042106",
            # special
            "6042103",
            # occlusion
            "6042096",
            # rotation
            "6042093",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            "6042311",
        ],
        "lane_segmentation": [
            # 白天
            "6041474",
        ],
    },
)
