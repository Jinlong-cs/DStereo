dataset_ids = dict(
    detection={
        "vehicle_detection": [
            "6042330",
        ],
        ("vehicle_category_classification", "vehicle_detection"): [],
        ("vehicle_occlusion_classification", "vehicle_detection"): [],
        "rear_detection": [
            "6042333",
        ],
        ("rear_occlusion_classification", "rear_detection"): [],
        "person_detection": ["6042317", "6042750"],
        ("person_pose_classification", "person_detection"): ["6042208"],
        ("person_occlusion_classification", "person_detection"): ["6042207"],
        "cyclist_detection": [
            "6042318",
            "6042357",
        ],
        ("cyclist_classification", "cyclist_detection"): [
            # 夜晚
            "6042206",
        ],
    },
)
