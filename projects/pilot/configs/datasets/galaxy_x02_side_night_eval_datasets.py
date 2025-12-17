dataset_ids = dict(
    # 3d 任务 withbn评测有关，默认注释
    detection_3d={
        # "vehicle_heatmap_3d_detection": [
        #     # 夜晚
        #     "6036184",
        #     # rotation badcase
        #     "6036784",
        #     # truncate badcase
        #     "6039661",
        #     # x02夜晚炫光
        #     "6040139",
        # ],
        # "ped_cyc_heatmap_3d_detection":[
        #     # person
        #     "6036446",
        #     # cyclist
        #     "6036164",
        # ],
    },
    detection={
        "vehicle_detection": [
            # 交付
            "6037142",
            # 夜晚
            "6036255",
            # 夜晚八达岭 0233
            # "6028007",
            # 夜晚收费站 0233
            # "6028652",
            # 夜晚 dark night
            "6028754",
            # 夜晚异型车
            "6036587",
            "6040957",
            # 夜晚大车
            "6030755",
            # 夜晚眩光
            "6037953",
            "6040696",
            # 地库
            "6029378",
            # 环境误检
            "6039083",
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            # 夜晚
            "6030218",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            # 夜晚
            "6030221",
        ],
        ("vehicle_truncation_classification", "vehicle_detection"): [
            # 夜晚
            "6030231",
        ],
        # ("vehicle_wheel_kps", "vehicle_detection"): [
        #     "6027905",  # x3c night
        # ],
        ("vehicle_wheel_detection", "vehicle_detection"): [
            "6029565",  # x3c day
        ],
        ("vehicle_ground_line", "vehicle_detection"): [
            "6027899",  # x3c night
        ],
        ("vehicle_flank", "vehicle_detection"): [
            "6037362",  # galaxy night
        ],
        "rear_detection": [
            # 交付
            "6037570",
            # 夜晚
            "6036245",
            # 夜晚 dark night
            "6028774",
            # 地库
            "6029372",
        ],
        ("rear_part_classification", "rear_detection"): [
            # 夜晚
            "6027897",  # X3C
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            # 夜晚
            "6030239",
        ],
        ("rear_plate_detection", "rear_detection"): [],
        "person_detection": [
            # 交付
            "6037574",
            # 夜晚
            "6036197",
            # 地库
            "6029351",
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
            # 交付
            "6036813",
            # 夜晚
            "6036240",
            # 地库
            "6029350",
        ],
        ("person_roi_3d", "person_detection"): [
            # 夜晚 x02
            # normal
            "6042390",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            # 夜晚 x02
            # normal
            "6042388",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            # 夜晚 x02
            # normal
            "6042394",
            # x02&x03 badcase
            # rotation
            "6042827",
            # speical_vehicle
            "6042797",
            # truncate
            "6042799",
            # night_glare
            "6042785",
            # night_glare_jira
            "6040720",
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
            # 夜晚
            "6036748",
            "6036915",
            "6036183",
            "6036179",
            "6030349",
            "6030354",
            "6030357",
            "6030788",
        ],
    },
)
