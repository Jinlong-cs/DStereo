dataset_ids = dict(
    detection={
        "vehicle_detection": [
            "6040744",  # 常规 3338
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            "6037415",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            "6037412",
        ],
        ("vehicle_truncation_classification", "vehicle_detection"): [
            "6037416",
        ],
        ("vehicle_wheel_detection", "vehicle_detection"): [],
        ("vehicle_ground_line", "vehicle_detection"): [],
        ("vehicle_roi_3d", "vehicle_detection"): [],
        "rear_detection": [
            "6040746",  # 常规 3168
        ],
        ("rear_part_classification", "rear_detection"): [],
        ("rear_occlusion_classification", "rear_detection"): ["6037612"],
        ("rear_plate_detection", "rear_detection"): [],
        "person_detection": [
            "6037451",
        ],
        ("person_pose_classification", "person_detection"): [
            "6037575",
        ],
        ("person_occlusion_classification", "person_detection"): [
            "6037576",
        ],
        ("person_orientation_classification", "person_detection"): [
            "6037577",
        ],
        ("person_face_detection", "person_detection"): [],
        ("person_roi_3d", "person_detection"): [],
        "cyclist_detection": [
            "6037450",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [],
    },
    semantic_segmentation={
        "default_segmentation": [],
        "lane_segmentation": [],
    },
)
