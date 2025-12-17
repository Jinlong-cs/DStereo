dataset_ids = dict(
    detection={
        "vehicle_detection": [
            "6040743",  # 常规3136
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            "6037413",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            "6037411",
        ],
        ("vehicle_truncation_classification", "vehicle_detection"): [
            "6037414",
        ],
        ("vehicle_wheel_detection", "vehicle_detection"): [],
        ("vehicle_ground_line", "vehicle_detection"): [],
        ("vehicle_roi_3d", "vehicle_detection"): [],
        "rear_detection": [
            "6040751",  # 常规 2929
        ],
        ("rear_part_classification", "rear_detection"): [],
        ("rear_occlusion_classification", "rear_detection"): [],
        ("rear_plate_detection", "rear_detection"): [],
        "person_detection": [],
        ("person_pose_classification", "person_detection"): [],
        ("person_occlusion_classification", "person_detection"): [],
        ("person_orientation_classification", "person_detection"): [],
        ("person_face_detection", "person_detection"): [],
        ("person_roi_3d", "person_detection"): [],
        "cyclist_detection": [],
        ("cyclist_roi_3d", "cyclist_detection"): [],
    },
    semantic_segmentation={
        "default_segmentation": [],
        "lane_segmentation": [],
    },
)
