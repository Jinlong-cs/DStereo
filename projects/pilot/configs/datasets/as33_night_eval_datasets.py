dataset_ids = dict(
    detection={
        "vehicle_detection": [
            "6031015",
            "6026688",
            "6028007",
            "6029004",
            "6029416",
            "6031498",
            "6036555",
            "6031648",
            "6032411",
            "6036818",  # 异型车 1794
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            "6028298",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            "6026795",
        ],
        # ("vehicle_wheel_kps", "vehicle_detection"): [
        #     "6026749",
        # ],
        ("vehicle_wheel_detection", "vehicle_detection"): [
            "6029463",
            "6026789",
        ],
        ("vehicle_ground_line", "vehicle_detection"): [
            "6027703",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            "6036055",
            "6036271",
            "6036272",
            "6029010",
            "6029261",
            "6029047",
        ],
        "rear_detection": [
            "6028918",
            "6027117",
            "6035918",
        ],
        ("rear_part_classification", "rear_detection"): [
            "6036372",
            "6036368",
            "6027119",
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            "6027118",
        ],
        ("rear_plate_detection", "rear_detection"): [
            "6028207",
        ],
        "person_detection": [
            "6028983",
            "6026693",
            # "6028609",
            # "6029386",
            "6029355",
        ],
        ("person_pose_classification", "person_detection"): [
            "6026756",
            "6031320",
        ],
        ("person_occlusion_classification", "person_detection"): [
            "6026758",
            "6031321",
        ],
        ("person_orientation_classification", "person_detection"): [
            "6026761",
            "6031371",
        ],
        # ("person_face_detection", "person_detection"): [
        #     "6034106",
        # ],
        ("person_roi_3d", "person_detection"): [
            "6029020",
            "6036260",
        ],
        "cyclist_detection": [
            "6028985",
            "6026695",
            "6027176",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            "6029019",
            "6036230",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            # as33
            "6030958",
            "6030959",
            "6031074",
            "6030960",
            "6030961",
            # 为森
            "6026687",
            "6026689",
            "6026690",
            "6026691",
            "6026692",
        ],
        "lane_segmentation": [
            "6029449",
            "6029450",
            "6029452",
            "6029456",
            "6029459",
            # bad case
            "6035988",
            "6035989",
            "6035991",
            "6035992",
        ],
    },
)
