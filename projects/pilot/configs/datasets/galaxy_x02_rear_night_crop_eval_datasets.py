dataset_ids = dict(
    detection={
        "vehicle_detection": [
            "6041310",
            # 夜晚
            "6036251",
            "6030292",
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            # 夜晚
            "6030432",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            # 夜晚
            "6030437",
        ],
        "rear_detection": [
            "6041456",
            # 夜晚
            "6036243",
            "6030299",
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            # 夜晚
            "6030482",
        ],
        "person_detection": [
            # 夜晚
            "6036195",
            "6030306",
            # glare
            "6040308",
        ],
        ("person_pose_classification", "person_detection"): [
            # 夜晚
            "6030492",
        ],
        ("person_occlusion_classification", "person_detection"): [
            # 夜晚
            "6030510",
        ],
        "cyclist_detection": [
            # 夜晚
            "6036235",
            "6030318",
            # glare
            "6040307",
        ],
    },
)
