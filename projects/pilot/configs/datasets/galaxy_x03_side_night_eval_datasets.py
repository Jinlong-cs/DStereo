dataset_ids = dict(
    # 3d 任务 withbn评测有关，默认注释
    # detection_3d={
    #     "vehicle_heatmap_3d_detection": [
    #         # x03
    #         "6039770",
    #         # x03角度
    #         "6040165",
    #         # truncate
    #         "6040170",
    #         # sr
    #         "6040349",
    #         # night glare
    #         "6040352",
    #     ],
    #     "ped_cyc_heatmap_3d_detection":[
    #         # person
    #         # x03
    #         "6039753",
    #         # cyclist
    #         # x03
    #         "6039755",
    #     ],
    # },
    detection={
        "vehicle_detection": [
            # X03
            "6039601",
            # 卡阈值
            "6036255",
        ],
        ("vehicle_category_classification", "vehicle_detection"): [],
        ("vehicle_occlusion_classification", "vehicle_detection"): [],
        ("vehicle_truncation_classification", "vehicle_detection"): [],
        ("vehicle_wheel_detection", "vehicle_detection"): [],
        ("vehicle_ground_line", "vehicle_detection"): [],
        ("vehicle_flank", "vehicle_detection"): [],
        "rear_detection": [
            # X03
            "6039598",
            # 卡阈值
            "6036245",
        ],
        ("rear_part_classification", "rear_detection"): [],
        ("rear_occlusion_classification", "rear_detection"): [],
        ("rear_plate_detection", "rear_detection"): [],
        "person_detection": [
            # x03 夜晚
            "6039591",
            # 卡阈值
            "6036197",
            # x03 夜晚 v3
            "6042163",
            # 柱状物
            "6042638",
        ],
        ("person_pose_classification", "person_detection"): [],
        ("person_occlusion_classification", "person_detection"): [],
        ("person_orientation_classification", "person_detection"): [],
        ("person_face_detection", "person_detection"): [],
        "cyclist_detection": [
            # x03 夜晚
            "6039589",
            # 卡阈值
            "6036240",
            # x03 夜晚 v3
            "6042164",
        ],
        ("person_roi_3d", "person_detection"): [
            # x03 夜晚
            "6042925",
            # 卡阈值 x02
            "6042390",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            # x03 夜晚
            "6042927",
            # 卡阈值 x02
            "6042388",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            # x03 夜晚
            "6039770",
            # 卡阈值 x02
            "6042394",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            # 夜晚 量产模组
            "6036256",
            "6039764",
            "6037610",
            "6037609",
            "6030253",
            "6030251",
            "6030250",
            "6030252",
        ],
        "lane_segmentation": [
            # X03 夜晚
            "6039561",
            "6039563",
        ],
    },
)
