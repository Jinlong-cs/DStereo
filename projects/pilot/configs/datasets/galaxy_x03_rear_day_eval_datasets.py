dataset_ids = dict(
    # 3d 任务 withbn评测有关，默认注释
    # detection_3d = {
    #     "vehicle_heatmap_3d_detection":[
    #         # x03
    #         "6039054",
    #         "6039654",
    #     ],
    #     "ped_cyc_heatmap_3d_detection":[
    #         # 行人 x03
    #         "6039072",
    #         # 骑行人 x03
    #         "6039069",
    #     ],
    # },
    detection={
        "vehicle_detection": [
            # X03
            "6038662",
            # X03 V2
            "6039606",
            # 卡阈值
            "6036267",
        ],
        ("vehicle_category_classification", "vehicle_detection"): [],
        ("vehicle_occlusion_classification", "vehicle_detection"): [],
        ("vehicle_truncation_classification", "vehicle_detection"): [],
        ("vehicle_wheel_kps", "vehicle_detection"): [],
        ("vehicle_wheel_detection", "vehicle_detection"): [],
        ("vehicle_ground_line", "vehicle_detection"): [],
        ("vehicle_flank", "vehicle_detection"): [],
        "rear_detection": [
            # X03
            "6038664",
            # X03 V2
            "6039604",
            # 卡阈值
            "6036262",
        ],
        ("rear_part_classification", "rear_detection"): [],
        ("rear_occlusion_classification", "rear_detection"): [],
        ("rear_plate_detection", "rear_detection"): [
            # 卡阈值
            "6037604",
        ],
        "person_detection": [
            # 白天 x03 v3
            "6042138",
            # 白天 x03 v2
            "6039530",
            # 白天 x03
            "6038688",
            # 卡阈值
            "6036189",
        ],
        ("person_pose_classification", "person_detection"): [],
        ("person_occlusion_classification", "person_detection"): [],
        ("person_orientation_classification", "person_detection"): [],
        ("person_face_detection", "person_detection"): [
            # 卡阈值
            "6037600"
        ],
        "cyclist_detection": [
            # 白天 x03 v3
            "6042139",
            # 白天 x03 v2
            "6039532",
            # 白天 x03
            "6038690",
            # 卡阈值
            "6036257",
        ],
        ("person_roi_3d", "person_detection"): [
            # x03
            "6042365",
            # 卡阈值
            "6042359",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            # x03
            "6042366",
            # 卡阈值
            "6042352",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            # x03
            "6042398",
            # 卡阈值
            "6040836",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            # x03 白天
            "6039628",
        ],
        "lane_segmentation": [
            # X03 白天
            "6039566",
        ],
    },
)
