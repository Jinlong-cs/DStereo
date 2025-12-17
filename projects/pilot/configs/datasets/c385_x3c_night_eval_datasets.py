dataset_ids = dict(
    # detection_3d={
    #     "vehicle_heatmap_3d_detection":[
    #         "6036019",
    #         "6036028",
    #         "6036771",
    #         "6036960",
    #         "6036949",
    #         "6038527",
    #     ],
    #     "ped_cyc_heatmap_3d_detection": [
    #         # person
    #         "6036046",
    #         "6036025",
    #         # cyclist
    #         "6036008",
    #         "6036029",
    #         "6038542",
    #     ],
    # },
    detection={
        "vehicle_detection": [
            "6036082",  # rainy 776
            "6036086",  # normal 1652
            "6036727",  # normal 4057
            "6028652",  # 收费站 126
            "6028754",  # 低照2115
            "6036587",  # 异型车1702
            "6030755",  # 大车 931
            "6037953",  # 眩光 891
            "6039083",  # 环境误检 1670
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            "6036733",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            "6036731",
        ],
        ("vehicle_truncation_classification", "vehicle_detection"): [
            "6036732",
        ],
        # ("vehicle_wheel_kps", "vehicle_detection"): [
        #     "6027905",
        # ],
        ("vehicle_wheel_detection", "vehicle_detection"): [
            "6029565",
        ],
        ("vehicle_ground_line", "vehicle_detection"): [
            "6027899",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            "6036019",
            "6036028",
            "6036771",
            "6036960",
            "6036949",
            "6038527",
        ],
        "rear_detection": [
            "6036105",
            "6036117",
        ],
        ("rear_part_classification", "rear_detection"): [
            "6027897",
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            "6027860",
        ],
        ("rear_plate_detection", "rear_detection"): [
            "6037582",
            "6036689",
        ],
        "person_detection": [
            "6036835",  # normal 1.4k
            "6036094",  # normal 386
            "6036096",
        ],
        ("person_pose_classification", "person_detection"): [
            "6027938",
        ],
        ("person_occlusion_classification", "person_detection"): [
            "6027939",
        ],
        ("person_orientation_classification", "person_detection"): [
            "6027948",
        ],
        ("person_face_detection", "person_detection"): ["6034106"],
        ("person_roi_3d", "person_detection"): [
            "6036046",
            "6036025",
        ],
        "cyclist_detection": [
            "6036837",  # normal 1.4k
            "6036095",  # normal 386
            "6036097",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            "6036008",
            "6036029",
            "6038542",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            "6036598",
            "6036599",
            "6039767",
        ],
        "lane_segmentation": [
            "6036116",
            "6036122",
            "6036123",
            "6036124",
            "6036125",
        ],
    },
)
