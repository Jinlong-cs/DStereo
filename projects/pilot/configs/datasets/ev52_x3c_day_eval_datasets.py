dataset_ids = dict(
    # detection_3d={
    #     "vehicle_heatmap_3d_detection":[
    #         "6042183",  # normal mix ev073&035
    #         "6041366",  # badcase special
    #         "6041364",  # badcase rotation
    #         "6042431",  # badcase truncation
    #     ],
    #     "ped_cyc_heatmap_3d_detection": [
    #         "6042192",  # person
    #         "6042195",  # cyclist
    #     ],
    # },
    detection={
        "vehicle_detection": [
            "6040738",  # 常规 6985
            "6028646",  # 晴天逆光 1231
            "6028653",  # 收费站白天 396
            "6029412",  # 大车 3044
            "6029418",  # 异型车 1909
            "6039083",  # 环境误检 1670
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            "6027983",  # X3C
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            "6027984",  # X3C
        ],
        ("vehicle_truncation_classification", "vehicle_detection"): [
            "6036740",
        ],
        ("vehicle_wheel_kps", "vehicle_detection"): [],
        ("vehicle_wheel_detection", "vehicle_detection"): [
            "6029565",
        ],
        ("vehicle_ground_line", "vehicle_detection"): [
            "6027898",
        ],
        "rear_detection": [
            "6040745",  # 常规 6479
        ],
        ("rear_part_classification", "rear_detection"): [
            "6027896",  # X3C
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            "6027854",  # X3C
        ],
        ("rear_plate_detection", "rear_detection"): [
            "6037582",
            "6036689",
        ],
        "person_detection": [
            # ev52
            "6041581",  # 常规
            "6041585",  # rare pose and worker
            # other
            "6038147",  # x3c fp
            "6039363",  # x3c fp resize4
            # maxfa
            "6028707",  # day_urban_sunny
            "6028706",  # day_urban_snowy
            "6028609",  # 隧道
        ],
        # not found
        # ("person_pose_classification", "person_detection"): [
        #     "6027941",
        # ],
        # ("person_occlusion_classification", "person_detection"): [
        #     "6027945",
        # ],
        # ("person_orientation_classification", "person_detection"): [
        #     "6027944",
        # ],
        ("person_face_detection", "person_detection"): [
            "6034106",
        ],
        "cyclist_detection": [
            # ev52
            "6041583",  # 常规
            # maxfa
            "6028708",  # day_urban_sunny
            "6028705",  # day_urban_snowy
        ],
        ("person_roi_3d", "person_detection"): [
            # "6027779",
            # ev52
            "6042192",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            # "6027781",
            # ev52
            "6042195",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            # "6027799",
            # "6028262",
            # "6028786",
            # "6028787",
            # "6028790",
            # "6028791",
            # "6028792",
            # "6028793",
            # "6028807",
            # "6028800",
            # "6028799",
            # "6028798",
            # "6028795",
            # ev52
            "6042183",  # normal mix ev073&035
            "6041366",  # badcase special
            "6041364",  # badcase rotation
            "6042431",  # badcase truncation
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            # ev52
            "6040688",
            "6040685",
            # maxfa 常规
            "6027725",
            # "6027724",  # eval error
            "6027920",
            "6027921",
            "6027922",
            # maxfa 场景
            "6028744",
            "6028743",
            "6028742",
            "6028721",
            "6028720",
            "6028718",
            "6028712",
            "6028709",
            "6028704",
            "6028703",
        ],
        "lane_segmentation": [
            "6040810",
            "6040811",
            "6027736",
            "6027738",
            "6027740",
            "6027742",
            "6027744",
            "6028692",
            "6028694",
            "6028695",
            "6028722",
            "6028732",
            "6028738",
            "6028740",
            "6028745",
            "6028749",
        ],
    },
)
