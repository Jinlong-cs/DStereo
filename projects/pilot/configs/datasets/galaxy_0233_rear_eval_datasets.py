dataset_ids = dict(
    # 3d 任务 withbn评测有关，默认注释
    # detection_3d = {
    #     "vehicle_heatmap_3d_detection":[
    #         # 白天
    #         # "6040836",
    #         # 夜晚
    #         # "6040837",
    #     ],
    #     "ped_cyc_heatmap_3d_detection":[
    #         # 行人
    #         # 白天
    #         # "6040846",
    #         # 夜晚
    #         # "6040843",
    #         # 骑行人
    #         # 白天
    #         # "6040838",
    #         # 夜晚
    #         # "6040840",
    #     ],
    # },
    detection={
        "vehicle_detection": [
            # 交付
            "6037439",
            "6038032",
            # X01
            "6039954",
            "6039955",
            # X03
            "6038662",
            "6038663",
            # X03 V2
            "6039606",
            "6039607",
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
            # 夜晚
            "6040926",
            "6036253",
            "6030467",
            # 夜晚八达岭
            "6028007",
            # 夜晚异型车
            "6036818",
            # 夜晚车灯眩光
            "6031498",
            # 夜晚大车
            "6031648",
            # 环境误检
            "6037934",
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            # 白天
            "6030530",
            # 夜晚
            "6030536",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            # 白天
            "6030545",
            # 夜晚
            "6030553",
        ],
        ("vehicle_truncation_classification", "vehicle_detection"): [
            # 白天
            "6030564",
            # 夜晚
            "6030568",
        ],
        ("vehicle_wheel_kps", "vehicle_detection"): [],
        ("vehicle_wheel_detection", "vehicle_detection"): [],
        ("vehicle_ground_line", "vehicle_detection"): [],
        ("vehicle_flank", "vehicle_detection"): [
            # 白天
            "6037356",
            # 夜晚
            "6037361",
        ],
        "rear_detection": [
            # 交付
            "6037444",
            "6038061",
            # X01
            "6039956",
            "6039957",
            # X03
            "6038664",
            "6038665",
            # X03 V2
            "6039604",
            "6039605",
            # 白天
            "6036262",
            # 夜晚
            "6036244",
            # 夜晚眩光
            "6035918",
        ],
        ("rear_part_classification", "rear_detection"): [
            # 白天
            "6027896",  # X3C
            # 夜晚
            "6027897",  # X3C
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            # 白天
            "6030576",
            # 夜晚
            "6030578",
        ],
        ("rear_plate_detection", "rear_detection"): [
            # 白天
            "6037604",
        ],
        "person_detection": [
            # 交付
            "6038426",
            "6038306",
            # 白天
            "6036189",
            # 白天 x03 v3
            "6042138",
            # 白天 x03 v2
            "6039530",
            # 白天 x03
            "6038688",
            # 夜晚
            "6036196",
            # 夜晚 x03 v3
            "6042156",
            # 夜晚 x03 v2
            "6039531",
            # 夜晚 x03
            "6038689",
            # 夜晚 jira both for train and eval
            "6039741",
            # 夜晚 glare fp both for train and eval
            "6039745",
            # 夜晚 glare eval
            "6040123",
        ],
        ("person_pose_classification", "person_detection"): [
            # 白天
            "6030583",
            # 夜晚
            "6030588",
        ],
        ("person_occlusion_classification", "person_detection"): [
            # 白天
            "6030589",
            # 夜晚
            "6030594",
        ],
        ("person_orientation_classification", "person_detection"): [
            # 白天
            "6037563",
            "6026499",
            "6026981",
            # 夜晚
            "6037557",
            "6026761",
        ],
        ("person_face_detection", "person_detection"): [
            # 白天
            "6037600",
        ],
        "cyclist_detection": [
            # 交付
            "6038386",
            "6038288",
            # 白天
            "6036257",
            # 白天 x03 v3
            "6042139",
            # 白天 x03 v2
            "6039532",
            # 白天 x03
            "6038690",
            # 夜晚
            "6036239",
            # 夜晚 x03 v3
            "6042157",
            # 夜晚 x03 v2
            "6039533",
            # 夜晚 x03
            "6038691",
            # 夜晚 jira both for train and eval
            "6039742",
            # 夜晚 glare fp both for train and eval
            "6039744",
            # 夜晚 glare eval
            "6040124",
        ],
        ("person_roi_3d", "person_detection"): [
            # 白天：
            # x03
            "6042365",
            # x02 卡阈值
            "6042359",
            # 夜晚：
            # x03
            "6042926",
            # x02 卡阈值
            "6042358",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            # 白天
            # x03
            "6042366",
            # x02 卡阈值
            "6042352",
            # 夜晚
            # x03
            "6042928",
            # x02 卡阈值
            "6042362",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            # 白天
            # x03
            "6042398",
            # x02 卡阈值
            "6040836",
            # 夜晚
            # x03
            "6039058",
            # x02 卡阈值
            "6042396",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            # 白天
            "6036265",
            "6036058",
            # 夜晚
            "6036252",
            "6036057",
            # x03 白天
            "6039628",
        ],
        "lane_segmentation": [
            # 白天
            "6036168",
            "6035995",
            # 夜晚
            "6035974",
            "6036586",
            # X03 白天
            "6039566",
            # X03夜晚
            "6039560",
        ],
    },
)
