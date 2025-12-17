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
            "6041424",
            # nio 异性车
            "6042168",  # 942
            # nio 眩光
            "6042171",  # 414
            # nio 环境误检
            "6042218",  # 832
            # nio 眩光
            "6042997",  # 1031
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            # galaxy_x02 夜晚
            "6030218",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            # galaxy_x02 夜晚
            "6030221",
        ],
        ("vehicle_truncation_classification", "vehicle_detection"): [
            # galaxy_x02 夜晚
            "6030231",
        ],
        ("vehicle_wheel_detection", "vehicle_detection"): [
            "6029565",  # x3c
        ],
        ("vehicle_flank", "vehicle_detection"): [
            "6037362",  # galaxy night
        ],
        "rear_detection": [
            # nio 常规
            "6041428",
        ],
        ("rear_part_classification", "rear_detection"): [
            # galaxy_x02 夜晚
            "6027897",  # X3C
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            # galaxy_x02 夜晚
            "6030239",
        ],
        ("rear_plate_detection", "rear_detection"): [],
        "person_detection": [
            # nio 常规
            "6042316",
            # nio 柱状物
            "6043398",
            # nio 车身
            "6043399",
        ],
        ("person_pose_classification", "person_detection"): [
            # 夜晚
            "6030241",
        ],
        ("person_occlusion_classification", "person_detection"): [
            # 夜晚
            "6030242",
        ],
        ("person_orientation_classification", "person_detection"): [
            # 夜晚
            "6037556",
        ],
        ("person_face_detection", "person_detection"): [],
        "cyclist_detection": [
            # nio 常规
            "6042315",
            # nio 车身
            "6043410",
        ],
        ("cyclist_classification", "cyclist_detection"): [
            # 夜晚
            "6042203",
        ],
        ("person_roi_3d", "person_detection"): ["6041181"],
        ("cyclist_roi_3d", "cyclist_detection"): ["6041179"],
        ("vehicle_roi_3d", "vehicle_detection"): [
            "6041127",
            # truncation
            "6042105",
            # special
            "6042099",
            # occlusion
            "6042095",
            # rotation
            "6042092",
            # night glare
            "6042834",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            "6041471",
        ],
        "lane_segmentation": [
            "6041475",
        ],
    },
)
