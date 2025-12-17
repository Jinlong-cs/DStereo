dataset_ids = dict(
    # 3d 任务 withbn评测有关，默认注释
    # detection_3d={
    #     "vehicle_heatmap_3d_detection": [
    #         # x03
    #         "6039769",
    #         # x03角度
    #         "6040164",
    #         # truncate
    #         "6040169",
    #         # sr
    #         "6040184",
    #     ],
    #     "ped_cyc_heatmap_3d_detection":[
    #         # person
    #         # x03
    #         "6039752",
    #         # cyclist
    #         # x03
    #         "6039757",
    #     ],
    # },
    detection={
        "vehicle_detection": [
            # X03
            "6039600",
            # 卡阈值
            "6036259",
        ],
        ("vehicle_category_classification", "vehicle_detection"): [],
        ("vehicle_occlusion_classification", "vehicle_detection"): [],
        ("vehicle_truncation_classification", "vehicle_detection"): [],
        ("vehicle_wheel_detection", "vehicle_detection"): [],
        ("vehicle_ground_line", "vehicle_detection"): [],
        ("vehicle_flank", "vehicle_detection"): [],
        "rear_detection": [
            # X03
            "6039630",
            # 卡阈值
            "6036266",
        ],
        ("rear_part_classification", "rear_detection"): [],
        ("rear_occlusion_classification", "rear_detection"): [],
        ("rear_plate_detection", "rear_detection"): [
            # 卡阈值
            "6037603",
        ],
        "person_detection": [
            # x03 白天
            "6039590",
            # 卡阈值
            "6036190",
            # x03 白天 v3
            "6042158",
            # 车身
            "6042639",
            # 柱状物
            "6042871",
        ],
        ("person_pose_classification", "person_detection"): [],
        ("person_occlusion_classification", "person_detection"): [],
        ("person_orientation_classification", "person_detection"): [],
        ("person_face_detection", "person_detection"): [
            # 卡阈值
            "6037601",
        ],
        "cyclist_detection": [
            # x03 白天
            "6039588",
            # 卡阈值
            "6036261",
            # x03 白天 v3
            "6042160",
        ],
        ("person_roi_3d", "person_detection"): [
            # x03 白天
            "6042373",
            # 卡阈值 x02
            "6042391",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            # x03 白天
            "6042382",
            # 卡阈值 x02
            "6042392",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            # x03 白天
            "6042395",
            # 卡阈值 x02
            "6036169",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            # 白天 量产模组
            "6036214",
            "6036242",
            "6039765",
            "6037611",
            "6030247",
            "6030249",
            "6030246",
            "6030248",
        ],
        "lane_segmentation": [
            # X03 白天
            "6039567",
            "6039568",
        ],
    },
)
