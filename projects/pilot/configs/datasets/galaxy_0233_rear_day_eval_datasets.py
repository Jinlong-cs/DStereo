dataset_ids = dict(
    detection={
        "vehicle_detection": [
            # 交付
            "6037439",
            # X01
            "6039954",
            # X03
            "6038662",
            # X03 V2
            "6039606",
            # 白天
            "6036267",
            # 白天隧道
            "6027456",
            # 白天眩光
            "6028855",
            # 白天异型车
            "6029415",
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
            # X01
            "6039956",
            # X03
            "6038664",
            # X03 V2
            "6039604",
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
            # 白天 x03 v2
            "6039530",
            # 白天 x03
            "6038688",
            # 白天 x03 v3
            "6042138",
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
            # 白天 x03 v2
            "6039532",
            # 白天 x03
            "6038690",
            # 白天 x03 v3
            "6042139",
        ],
        ("person_roi_3d", "person_detection"): [
            # x03
            "6042365",
            # x02 卡阈值
            "6042359",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            # x03
            "6042366",
            # x02 卡阈值
            "6042352",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            # x03
            "6042398",
            # x02 卡阈值
            "6040836",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            # 白天
            "6036265",
            "6036058",
            # x03 白天
            "6039628",
        ],
        "lane_segmentation": [
            # X02 白天
            "6036168",
            "6035995",
            # X03 白天
            "6039566",
        ],
    },
)
