dataset_ids = dict(
    detection={
        "vehicle_detection": [
            # 交付
            "6038032",
            # X01
            "6039955",
            # X03
            "6038663",
            # X03 V2
            "6039607",
            # 夜晚
            "6040926",
            "6036253",
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
            # 夜晚
            "6030536",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            # 夜晚
            "6030553",
        ],
        ("vehicle_truncation_classification", "vehicle_detection"): [
            # 夜晚
            "6030568",
        ],
        ("vehicle_wheel_kps", "vehicle_detection"): [],
        ("vehicle_wheel_detection", "vehicle_detection"): [],
        ("vehicle_ground_line", "vehicle_detection"): [],
        ("vehicle_flank", "vehicle_detection"): [
            "6037361",
        ],
        "rear_detection": [
            # 交付
            "6038061",
            # X01
            "6039957",
            # X03
            "6038665",
            # X03 V2
            "6039605",
            # 夜晚
            "6036244",
            # 夜晚眩光
            "6035918",
        ],
        ("rear_part_classification", "rear_detection"): [
            # 夜晚
            "6027897",  # X3C
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            # 夜晚
            "6030578",
        ],
        ("rear_plate_detection", "rear_detection"): [],
        "person_detection": [
            # 交付
            "6038306",
            # 夜晚
            "6036196",
            # 夜晚 x03 v2
            "6039531",
            # 夜晚 x03
            "6038689",
            # 夜晚 x03 v3
            "6042156",
            # 夜晚 jira both for train and eval
            "6039741",
            # 夜晚 glare fp both for train and eval
            "6039745",
            # 夜晚 glare eval
            "6040123",
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
            "6026761",
        ],
        ("person_face_detection", "person_detection"): [],
        "cyclist_detection": [
            # 交付
            "6038288",
            # 夜晚
            "6036239",
            # 夜晚 x03 v2
            "6039533",
            # 夜晚 x03
            "6038691",
            # 夜晚 x03 v3
            "6042157",
            # 夜晚 jira both for train and eval
            "6039742",
            # 夜晚 glare fp both for train and eval
            "6039744",
            # 夜晚 glare eval
            "6040124",
        ],
        ("person_roi_3d", "person_detection"): [
            # x03
            "6042926",
            # x02 卡阈值
            "6042358",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            # x03
            "6042928",
            # x02 卡阈值
            "6042362",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            # x03
            "6039058",
            # x02 卡阈值
            "6042396",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            # 夜晚
            "6036252",
            "6036057",
            # x03 夜晚
            "6039771",
        ],
        "lane_segmentation": [
            # X02夜晚
            "6035974",
            "6036586",
            # X03夜晚
            "6039560",
        ],
    },
)
