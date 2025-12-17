dataset_ids = dict(
    # 3d 任务 withbn评测有关，默认注释
    # detection_3d={
    #     "vehicle_heatmap_3d_detection": [
    #         # x02
    #         # 白天
    #         "6036169",
    #         # rotation badcase
    #         "6040329",
    #         # special car
    #         "6036773",
    #         # truncate badcase
    #         "6039660",
    #     ],
    #     "ped_cyc_heatmap_3d_detection":[
    #         # 行人
    #         # 白天
    #         "6036445",
    #         # 骑行人
    #         # 白天
    #         "6036165",
    #     ],
    # },
    detection={
        "vehicle_detection": [
            # 交付
            "6037154",
            # 白天
            "6036259",
            # 白天隧道 0233
            # "6027456",
            # 白天逆光
            "6028646",
            # 白天收费站 0233
            # "6028653",
            # 白天大车
            "6029412",
            # 白天异型车
            "6029418",
            "6040955",
            # 环境误检
            "6040640",
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            # 白天
            "6030216",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            # 白天
            "6030220",
        ],
        ("vehicle_truncation_classification", "vehicle_detection"): [
            # 白天
            "6030229",
        ],
        # ("vehicle_wheel_kps", "vehicle_detection"): [
        #     "6027903",  # x3c day
        # ],
        ("vehicle_wheel_detection", "vehicle_detection"): [
            "6029565",  # x3c day
        ],
        ("vehicle_ground_line", "vehicle_detection"): [
            "6027898",  # x3c day
        ],
        ("vehicle_flank", "vehicle_detection"): [
            "6037360",  # galaxy day
        ],
        "rear_detection": [
            # 交付
            "6037566",
            # 白天
            "6036266",
            # 白天异型车
            "6028752",
            # 白天大车
            "6028753",
        ],
        ("rear_part_classification", "rear_detection"): [
            # 白天
            "6027896",  # X3C
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            # 白天
            "6030238",
        ],
        ("rear_plate_detection", "rear_detection"): [
            # 白天
            "6037603",
        ],
        "person_detection": [
            # 交付
            "6037573",
            # 白天
            "6036190",
        ],
        ("person_pose_classification", "person_detection"): [
            # 白天
            "6030240",
        ],
        ("person_occlusion_classification", "person_detection"): [
            # 白天
            "6030244",
        ],
        ("person_orientation_classification", "person_detection"): [
            # 白天
            "6037558",
        ],
        ("person_face_detection", "person_detection"): [
            # 白天
            "6037601",
        ],
        "cyclist_detection": [
            # 交付
            "6037135",
            # 白天
            "6036261",
        ],
        ("person_roi_3d", "person_detection"): [
            # 白天 x02
            # normal
            "6042391",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            # 白天 x02
            # normal
            "6042392",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            # 白天 x02
            # normal
            "6036169",
            # x02&x03 badcase
            # rotation
            "6042803",
            # speical_vehicle
            "6042792",
            # truncate
            "6042800",
            # curve
            "6042848",
            # curve_jira
            "6042032",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            # 白天 量产模组
            "6036214",
            "6036242",
            "6039765",
            "6037611",
            # 白天
            "6030247",
            "6030249",
            "6030246",
            "6030248",
        ],
        "lane_segmentation": [
            # 白天
            "6036750",
            "6036172",
            "6036174",
            "6036177",
            "6030430",
            "6030381",
            "6030435",
            "6030434",
        ],
    },
)
