dataset_ids = dict(
    # 3d 任务 withbn评测有关，默认注释
    # detection_3d = {
    #     "vehicle_heatmap_3d_detection":[
    #         # 白天
    #         # x02
    #         "6040836",
    #     ],
    #     "ped_cyc_heatmap_3d_detection":[
    #         # 行人 x02
    #         "6040846",
    #         # 骑行人 x02
    #         "6040838",
    #     ],
    # },
    detection={
        "vehicle_detection": [
            # 交付
            "6037439",
            # 白天
            "6036267",
            # 白天隧道
            "6027456",
            # 白天眩光
            "6028855",
            # 白天异型车
            "6029415",
            "6040954",
            # 白天大车
            "6032394",
            # 白天小车挡大车
            "6032411",
            # 白天标志牌
            "6029359",
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            # 白天
            "6030530",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            # 白天
            "6030545",
        ],
        ("vehicle_truncation_classification", "vehicle_detection"): [
            # 白天
            "6030564",
        ],
        ("vehicle_wheel_kps", "vehicle_detection"): [],
        ("vehicle_wheel_detection", "vehicle_detection"): [],
        ("vehicle_ground_line", "vehicle_detection"): [],
        ("vehicle_flank", "vehicle_detection"): [
            "6037356",
        ],
        "rear_detection": [
            # 交付
            "6037444",
            # 白天
            "6036262",
        ],
        ("rear_part_classification", "rear_detection"): [
            # 白天
            "6027896",  # X3C
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            # 白天
            "6030576",
        ],
        ("rear_plate_detection", "rear_detection"): [
            # 白天
            "6037604",
        ],
        "person_detection": [
            # 交付
            "6038426",
            # 白天
            "6036189",
        ],
        ("person_pose_classification", "person_detection"): [
            # 白天
            "6030583",
        ],
        ("person_occlusion_classification", "person_detection"): [
            # 白天
            "6030589",
        ],
        ("person_orientation_classification", "person_detection"): [
            # 白天
            "6037563",
            "6026499",
            "6026981",
        ],
        ("person_face_detection", "person_detection"): [
            # 白天
            "6037600",
        ],
        "cyclist_detection": [
            # 交付
            "6038386",
            # 白天
            "6036257",
        ],
        ("person_roi_3d", "person_detection"): [
            # 白天
            # x02
            "6042359",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            # 白天
            # x02
            "6042352",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            # 白天
            # x02
            "6040836",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            # 白天
            "6036265",
            "6036058",
        ],
        "lane_segmentation": [
            # X02 白天
            "6036168",
            "6035995",
        ],
    },
)
