dataset_ids = dict(
    # detection_3d={
    #     "vehicle_heatmap_3d_detection":[
    #         "6029920",
    #         "6029370",
    #     ],
    #     "ped_cyc_heatmap_3d_detection": [
    #         "6030224",  # person
    #         "6029998",  # cyclist
    #     ],
    # },
    detection={
        "vehicle_detection": [
            # "6029378",  # 常规评测集 4k+
            "6036628",  # 常规评测集 5k+
            "6036644",  # 墙壁误检评测集 1k
            "6037878",  # 常规评测集 10k
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            "6030143",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            "6030146",
        ],
        ("vehicle_truncation_classification", "vehicle_detection"): [
            "6030141",
        ],
        # ("vehicle_wheel_kps", "vehicle_detection"): [
        #     "6027903",
        # ],
        ("vehicle_wheel_detection", "vehicle_detection"): [
            "6029565",
            "6026789",
        ],
        ("vehicle_ground_line", "vehicle_detection"): [
            "6027898",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            "6029920",
            "6029370",
        ],
        "rear_detection": [
            "6029372",
            "6036844",
        ],
        ("rear_part_classification", "rear_detection"): [
            "6027896",
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            "6027854",
        ],
        ("rear_plate_detection", "rear_detection"): [
            "6037582",
        ],
        "person_detection": [
            "6029351",
            "6039731",  # jira for both train and val
            # "6029088",
            # "6029101",
        ],
        ("person_pose_classification", "person_detection"): [
            "6029369",
        ],
        ("person_occlusion_classification", "person_detection"): [
            "6029352",
        ],
        ("person_orientation_classification", "person_detection"): [
            "6038773",
        ],
        ("person_face_detection", "person_detection"): [
            "6034106",
        ],
        ("person_roi_3d", "person_detection"): [
            "6030224",
        ],
        "cyclist_detection": [
            "6029350",
            # cc02
            "6028935",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            "6029998",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            "6029346",
            "6029347",
            "6029348",
            "6029345",
            "6029353",
            "6039766",
        ],
    },
)
