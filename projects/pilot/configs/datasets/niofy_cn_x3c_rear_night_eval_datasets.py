dataset_ids = dict(
    # 3d 任务 withbn评测有关，默认注释
    # detection_3d = {
    #     "vehicle_heatmap_3d_detection":[
    #     ],
    #     "ped_cyc_heatmap_3d_detection":[
    #     ],
    # },
    detection={
        "vehicle_detection": [
            # nio 常规
            "6041422",
            # nio 异性车
            "6042167",  # 249
            # nio 眩光
            "6042166",  # 135
            # nio 环境误检
            "6042218",  # 832
            # nio 眩光
            "6042996",  # 701
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            # galaxy_x02 夜晚
            "6030536",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            # galaxy_x02 夜晚
            "6030553",
        ],
        ("vehicle_truncation_classification", "vehicle_detection"): [
            # galaxy_x02 夜晚
            "6030568",
        ],
        ("vehicle_wheel_detection", "vehicle_detection"): [
            "6029565",  # x3c
        ],
        ("vehicle_flank", "vehicle_detection"): [
            "6037362",  # galaxy night
        ],
        "rear_detection": [
            # nio 常规
            "6041429",
        ],
        ("rear_part_classification", "rear_detection"): [
            # galaxy_x02 夜晚
            "6027897",  # X3C
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            # galaxy_x02 夜晚
            "6030578",
        ],
        ("rear_plate_detection", "rear_detection"): [],
        "person_detection": [
            # nio 常规
            "6042312",
            # nio 柱状物
            "6043397",
            # nio 车身
            "6043400",
        ],
        ("person_pose_classification", "person_detection"): [
            # 夜晚
            "6030588",
        ],
        ("person_occlusion_classification", "person_detection"): [
            # 夜晚
            "6030594",
        ],
        ("person_orientation_classification", "person_detection"): [
            # 夜晚
            "6037557",
        ],
        ("person_face_detection", "person_detection"): [],
        "cyclist_detection": [
            # nio 常规
            "6042314",
            # nio 车身
            "6043408",
        ],
        ("cyclist_classification", "cyclist_detection"): [
            # 夜晚
            "6042202",
        ],
        ("person_roi_3d", "person_detection"): [
            "6041177",
            # c385 general eval night
            # "6042653",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            "6041176",
            # c385 general eval night
            # "6042655",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            "6041066",
            # truncation
            "6042091",
            # special
            "6042084",
            # # occlusion len not enough
            # "6042087",
            # rotation
            "6042082",
            # night glare
            "6042835",
            # c385 general eval night
            # "6042659",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            "6041467",
        ],
        "lane_segmentation": [
            "6041478",
        ],
    },
)
