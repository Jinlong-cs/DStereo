dataset_ids = dict(
    detection={
        "vehicle_detection": [
            # 维森夜晚
            "6028031",
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            # 维森夜晚
            "6028032",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            # 维森夜晚
            "6028033",
        ],
        "rear_detection": [
            # as33 常规
            "6029012",
            # 维森 常规
            "6027909",
        ],
        ("rear_occlusion_classification", "rear_detection"): [],
        "person_detection": [
            # AS33 night
            "6029015",
            # weisen night
            "6028029",
        ],
        ("person_pose_classification", "person_detection"): [],
        ("person_occlusion_classification", "person_detection"): [],
        "cyclist_detection": [
            # AS33 night
            "6029014",
            # weisen night
            "6028030",
        ],
    },
)
