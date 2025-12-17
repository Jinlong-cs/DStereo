dataset_ids = dict(
    # 3d 任务 withbn评测有关，默认注释
    # detection_3d = {
    #     "vehicle_heatmap_3d_detection":[
    #         "6039058",
    #         "6039655",
    #     ],
    #     "ped_cyc_heatmap_3d_detection":[
    #         # 行人 x03
    #         "6039073",
    #         # 骑行人 x03
    #         "6039062",
    #     ],
    # },
    detection={
        "vehicle_detection": [
            "6041265",
            # X03
            "6038663",
            # X03 V2
            "6039607",
            # 卡阈值
            "6040926",
        ],
        ("vehicle_category_classification", "vehicle_detection"): [],
        ("vehicle_occlusion_classification", "vehicle_detection"): [],
        ("vehicle_truncation_classification", "vehicle_detection"): [],
        ("vehicle_wheel_kps", "vehicle_detection"): [],
        ("vehicle_wheel_detection", "vehicle_detection"): [],
        ("vehicle_ground_line", "vehicle_detection"): [],
        ("vehicle_flank", "vehicle_detection"): [],
        "rear_detection": [
            "6041271",
            # X03
            "6038665",
            # X03 V2
            "6039605",
            # 卡阈值
            "6036244",
        ],
        ("rear_part_classification", "rear_detection"): [],
        ("rear_occlusion_classification", "rear_detection"): [],
        ("rear_plate_detection", "rear_detection"): [],
        "person_detection": [
            # 夜晚 x03 v3
            "6042156",
            # 夜晚 x03 v2
            "6039531",
            # 夜晚 x03
            "6038689",
            # 卡阈值
            "6036196",
        ],
        ("person_pose_classification", "person_detection"): [],
        ("person_occlusion_classification", "person_detection"): [],
        ("person_orientation_classification", "person_detection"): [],
        ("person_face_detection", "person_detection"): [],
        "cyclist_detection": [
            # 夜晚 x03 v3
            "6042157",
            # 夜晚 x03 v2
            "6039533",
            # 夜晚 x03
            "6038691",
            # 卡阈值
            "6036239",
        ],
        ("person_roi_3d", "person_detection"): [
            # x03
            "6042926",
            "6043391",
            # 卡阈值
            "6042358",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            # x03
            "6042928",
            # 卡阈值
            "6042362",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            # x03
            "6039058",
            # 卡阈值
            "6042396",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            # x03 夜晚
            "6039771",
        ],
        "lane_segmentation": [
            # X03夜晚
            "6039560",
        ],
    },
)
