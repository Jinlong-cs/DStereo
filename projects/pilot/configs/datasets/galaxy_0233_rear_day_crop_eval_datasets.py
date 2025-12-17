dataset_ids = dict(
    detection={
        "vehicle_detection": [
            # X03
            "6040327",
            # 白天
            "6036263",
            "6030290",
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            # 白天
            "6030429",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            # 白天
            "6030436",
        ],
        "rear_detection": [
            # X03
            "6040332",
            # 白天
            "6036258",
            "6030298",
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            # 白天
            "6030460",
        ],
        "person_detection": [
            # 白天
            "6036188",
            "6030302",
        ],
        ("person_pose_classification", "person_detection"): [
            # 白天
            "6030483",
        ],
        ("person_occlusion_classification", "person_detection"): [
            # 白天
            "6030494",
        ],
        "cyclist_detection": [
            # 白天
            "6036204",
            "6030304",
        ],
    },
)
