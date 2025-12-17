dataset_ids = dict(
    # 3d 任务 withbn评测有关，默认注释
    # detection_3d = {
    #     "vehicle_heatmap_3d_detection":[
    #         "6036169",
    #         "6036184",
    #     ],
    #     "ped_cyc_heatmap_3d_detection":[
    #         # 行人
    #         # 白天
    #         "6036445",
    #         # 夜晚
    #         "6036446",
    #         # 骑行人
    #         # 白天
    #         "6036165",
    #         # 夜晚
    #         "6036164",
    #     ],
    # },
    detection={
        "vehicle_detection": [
            # X03
            "6039600",
            "6039601",
            # 交付
            "6037154",
            "6037142",
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
            # 夜晚大车
            "6030755",
            # 夜晚眩光
            "6037953",
            # 地库
            "6029378",
            # 环境误检
            "6039083",
            # 环境误检
            "6040640",
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            # 白天
            "6030216",
            # 夜晚
            "6030218",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            # 白天
            "6030220",
            # 夜晚
            "6030221",
        ],
        ("vehicle_truncation_classification", "vehicle_detection"): [
            # 白天
            "6030229",
            # 夜晚
            "6030231",
        ],
        # ("vehicle_wheel_kps", "vehicle_detection"): [
        #     "6027903",  # x3c day
        #     "6027905",  # x3c night
        # ],
        ("vehicle_wheel_detection", "vehicle_detection"): [
            "6029565",  # x3c day
        ],
        ("vehicle_ground_line", "vehicle_detection"): [
            "6027898",  # x3c day
            "6027899",  # x3c night
        ],
        ("vehicle_flank", "vehicle_detection"): [
            "6037360",  # galaxy day
            "6037362",  # galaxy night
        ],
        "rear_detection": [
            # X03
            "6039630",
            "6039598",
            # 交付
            "6037566",
            "6037570",
            # 白天
            "6036266",
            # 白天异型车
            "6028752",
            # 白天大车
            "6028753",
            # 夜晚
            "6036245",
            # 夜晚 dark night
            "6028774",
            # 地库
            "6029372",
        ],
        ("rear_part_classification", "rear_detection"): [
            # 白天
            "6027896",  # X3C
            # 夜晚
            "6027897",  # X3C
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            # 白天
            "6030238",
            # 夜晚
            "6030239",
        ],
        ("rear_plate_detection", "rear_detection"): [
            # 白天
            "6037603",
        ],
        "person_detection": [
            # 交付
            "6037573",
            "6037574",
            # 白天
            "6036190",
            # 夜晚
            "6036197",
            # 地库
            "6029351",
            # x03 白天
            "6039590",
            # x03 夜晚
            "6039591",
            # x03 白天 v3
            "6042158",
            # x03 夜晚 v3
            "6042163",
        ],
        ("person_pose_classification", "person_detection"): [
            # 白天
            "6030240",
            # 夜晚
            "6030241",
        ],
        ("person_occlusion_classification", "person_detection"): [
            # 白天
            "6030244",
            # 夜晚
            "6030242",
        ],
        ("person_orientation_classification", "person_detection"): [
            # 白天
            "6037558",
            # 夜晚
            "6037556",
        ],
        ("person_face_detection", "person_detection"): [
            # 白天
            "6037601",
        ],
        "cyclist_detection": [
            # 交付
            "6037135",
            "6036813",
            # 白天
            "6036261",
            # 夜晚
            "6036240",
            # 地库
            "6029350",
            # x03 白天
            "6039588",
            # x03 夜晚
            "6039589",
            # x03 白天 v3
            "6042160",
            # x03 夜晚 v3
            "6042164",
        ],
        ("person_roi_3d", "person_detection"): [
            # 白天
            # x03 白天
            "6042373",
            # 卡阈值 x02
            "6042391",
            # 夜晚
            # x03 夜晚
            "6042925",
            # 卡阈值 x02
            "6042390",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            # 白天
            # x03 白天
            "6042382",
            # 卡阈值 x02
            "6042392",
            # 夜晚
            # x03 夜晚
            "6042927",
            # 卡阈值 x02
            "6042388",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            # 白天
            # x03 白天
            "6042395",
            # 卡阈值 x02
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
            # 夜晚
            # x03 夜晚
            "6039770",
            # 卡阈值 x02
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
            # 夜晚 量产模组
            "6036256",
            "6039764",
            "6037610",
            "6037609",
            # 夜晚
            "6030253",
            "6030251",
            "6030250",
            "6030252",
        ],
        "lane_segmentation": [
            # 白天
            "6036748",
            "6036172",
            "6036174",
            "6036177",
            "6030430",
            "6030381",
            "6030435",
            "6030434",
            # 夜晚
            "6036750",
            "6036915",
            "6036183",
            "6036179",
            "6030349",
            "6030354",
            "6030357",
            "6030788",
            # X03 白天
            "6039567",
            "6039568",
            # X03 夜晚
            "6039561",
            "6039563",
        ],
    },
)
