dataset_ids = dict(
    detection={
        "vehicle_detection": [
            # weisen day
            "6027008",
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            # weisen day
            "6027009",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [],
        "rear_detection": [
            # AS33 常规
            "6029011",
            # 森云 常规
            "6027908",
            # 维森 常规
            "6027007",
        ],
        ("rear_occlusion_classification", "rear_detection"): [],
        "person_detection": [
            # AS33 day
            "6029016",
            # weisen day
            "6026998",
        ],
        ("person_pose_classification", "person_detection"): [],
        ("person_occlusion_classification", "person_detection"): [],
        "cyclist_detection": [
            # AS33 day
            "6029017",
            # weisen day
            "6026999",
        ],
    },
)
