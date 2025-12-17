dataset_ids = dict(
    # detection_3d={
    #     "vehicle_heatmap_3d_detection":[
    #         "6036026",
    #         "6036021",
    #         "6036769",
    #         "6036783",
    #         "6036961",
    #         "6036767",
    #         "6038525",
    #     ],
    #     "ped_cyc_heatmap_3d_detection": [
    #         # person
    #         "6036034",
    #         "6036030",
    #         # cyclist
    #         "6035999",
    #         "6035997",
    #         "6038541",
    #     ],
    # },
    detection={
        "vehicle_detection": [
            "6036083",  # normal 4370
            "6036735",  # normal 8915
            "6028646",  # 晴天逆光 1231
            "6029412",  # 大车 3044
            "6029418",  # 异型车 1909
            "6039083",  # 环境误检 1670
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            "6036741",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            "6036742",
        ],
        ("vehicle_truncation_classification", "vehicle_detection"): [
            "6036740",
        ],
        # ("vehicle_wheel_kps", "vehicle_detection"): [
        #     "6027903",
        # ],
        ("vehicle_wheel_detection", "vehicle_detection"): [
            "6029565",
        ],
        ("vehicle_ground_line", "vehicle_detection"): [
            "6027898",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            "6036026",
            "6036021",
            "6036769",
            "6036783",
            "6036961",
            "6036767",
            "6038525",
        ],
        "rear_detection": [
            "6036121",
        ],
        ("rear_part_classification", "rear_detection"): [
            "6027896",
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            "6027854",
        ],
        ("rear_plate_detection", "rear_detection"): [
            "6037582",
            "6036689",
        ],
        "person_detection": [
            "6036832",  # normal 6.7K
            "6038147",  # x3c fp
            "6036087",  # normal 2.6K
            "6036092",
            "6039363",  # x3c fp resize4
            "6039683",  # jira both for train and val
        ],
        ("person_pose_classification", "person_detection"): [
            "6027941",
        ],
        ("person_occlusion_classification", "person_detection"): [
            "6027945",
        ],
        ("person_orientation_classification", "person_detection"): [
            "6027944",
        ],
        ("person_face_detection", "person_detection"): ["6034106"],
        ("person_roi_3d", "person_detection"): [
            "6036034",
            "6036030",
        ],
        "cyclist_detection": [
            "6036836",  # normal 6.7K
            "6036088",  # normal 2.6K
            "6036093",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            "6035999",
            "6035997",
            "6038541",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            "6036596",
            "6036597",
            "6039768",
        ],
        "lane_segmentation": [
            "6036133",
            "6036131",
            "6036130",
            "6036128",
            "6036127",
        ],
    },
)
