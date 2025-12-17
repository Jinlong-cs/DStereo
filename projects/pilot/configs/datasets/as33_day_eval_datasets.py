dataset_ids = dict(
    detection={
        "vehicle_detection": [
            "6029359",
            "6026506",
            "6026847",
            "6027456",
            "6028855",
            "6028996",
            "6029415",
            "6037190",
            "6032394",
            "6032411",
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            "6027716",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            "6026508",
        ],
        ("vehicle_truncation_classification", "vehicle_detection"): [
            "6026601",
        ],
        # ("vehicle_wheel_kps", "vehicle_detection"): [
        #     "6026493",
        #     "6026849",
        # ],
        ("vehicle_wheel_detection", "vehicle_detection"): [
            "6029463",
            "6026789",
        ],
        ("vehicle_ground_line", "vehicle_detection"): [
            "6027703",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            "6036059",
            "6036276",
            "6036275",
            "6029013",
            "6029258",
            "6029045",
            "6029385",
        ],
        "rear_detection": [
            "6028916",
            "6027199",
            "6027121",
            "6027111",
            "6027003",
        ],
        ("rear_part_classification", "rear_detection"): [
            "6036373",
            "6036374",
            "6027115",
            "6027123",
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            "6027112",
            "6027201",
        ],
        ("rear_plate_detection", "rear_detection"): [
            "6028207",
        ],
        "person_detection": [
            "6029088",
            # "6028666",
            "6029101",
            "6028981",
            "6026368",
            "6026947",
            "6027363",
            "6029356",
        ],
        ("person_pose_classification", "person_detection"): [
            "6026486",
            "6026949",
        ],
        ("person_occlusion_classification", "person_detection"): [
            "6026488",
            "6026952",
        ],
        ("person_orientation_classification", "person_detection"): [
            "6026499",
            "6026981",
        ],
        # ("person_face_detection", "person_detection"): [
        #     "6034106",
        # ],
        ("person_roi_3d", "person_detection"): [
            "6029018",
            "6036274",
        ],
        "cyclist_detection": [
            "6028984",
            "6026388",
            "6026951",
            "6026851",
            "6027190",
            "6037186",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            "6029021",
            "6036234",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            # as33
            "6029405",
            "6029404",
            "6029403",
            "6029005",
            "6029006",
            # 为森
            "6026882",
            "6026887",
            "6026888",
            "6026889",
            "6026890",
            # 雨天
            "6029423",
            # 炫光
            "6029277",
            # 雪天
            "6029325",
        ],
        "lane_segmentation": [
            "6029002",
            "6028999",
            "6028998",
            "6028997",
            "6028995",
            "6029442",
            "6029443",
            "6029444",
            "6029445",
            "6029447",
            # bad case
            "6035932",
            "6035933",
            "6035934",
            "6035935",
            "6035936",
            "6035937",
            "6035938",
        ],
    },
)
