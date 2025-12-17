dataset_ids = dict(
    # detection_3d={
    #     "vehicle_heatmap_3d_detection":[
    #         "6042182",  # normal mix ev073&035
    #         "6041365",  # badcase rotation
    #         "6041363",  # badcase special
    #         "6042428",  # badcase glare
    #         "6042429",  # badcase truncation
    #     ],
    #     "ped_cyc_heatmap_3d_detection": [
    #         "6042190",  # person
    #         "6042194",  # cyclist
    #     ],
    # },
    detection={
        "vehicle_detection": [
            "6040741",  # 常规 5913
            "6028652",  # 收费站 126
            "6028754",  # 低照2115
            "6036587",  # 异型车1702
            "6030755",  # 大车 931
            "6040686",  # 眩光 1872
            "6039083",  # 环境误检 1670
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            "6027983",  # X3C
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            "6036731",
        ],
        ("vehicle_truncation_classification", "vehicle_detection"): [
            "6036732",
        ],
        # ("vehicle_wheel_kps", "vehicle_detection"): [
        #     "6027905",
        # ],
        ("vehicle_wheel_detection", "vehicle_detection"): [
            "6029565",
        ],
        ("vehicle_ground_line", "vehicle_detection"): [
            "6027899",
        ],
        "rear_detection": [
            "6040753",  # 常规 5288
        ],
        ("rear_part_classification", "rear_detection"): [
            "6027897",
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            "6027860",  # X3C
        ],
        ("rear_plate_detection", "rear_detection"): [
            "6037582",
            "6036689",
        ],
        "person_detection": [
            # ev52
            "6041582",  # 常规
            # maxfa
            "6028711",  # night_urban_snowy
            "6027750",  # 常规
            "6028609",  # 隧道
        ],
        ("person_pose_classification", "person_detection"): [
            "6027938",
        ],
        ("person_occlusion_classification", "person_detection"): [
            "6027939",
        ],
        ("person_orientation_classification", "person_detection"): [
            "6027948",
        ],
        ("person_face_detection", "person_detection"): ["6034106"],
        "cyclist_detection": [
            # ev52
            "6041584",  # 常规
            # other
            "6039321",  # fp
            # maxfa
            "6028710",  # night_urban_snowy
            "6028655",  # cyclist_lighted
            "6027748",  # 常规
        ],
        ("person_roi_3d", "person_detection"): [
            # "6027773",
            # ev52
            "6042190",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            # "6027772",
            # "6028782",
            # "6028788",
            # "6028783",
            # "6028784",
            # "6028794",
            # "6028796",
            # "6028801",
            # ev52
            "6042182",  # normal mix ev073&035
            "6041365",  # badcase rotation
            "6041363",  # badcase special
            "6042428",  # badcase glare
            "6042429",  # badcase truncation
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            # "6027776",
            # ev52
            "6042194",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            # ev52
            "6040694",
            "6040693",
            # maxfa 常规
            "6027729",
            "6027728",
            "6028385",
            "6028386",
            "6028387",
            # maxfa 场景
            "6028748",
            "6028747",
            "6028746",
            "6028715",
            "6028713",
            "6028714",
        ],
        "lane_segmentation": [
            "6040812",
            "6040813",
            "6027735",
            "6027737",
            "6027739",
            "6027741",
            "6027743",
            "6028750",
            "6028737",
            "6028733",
            "6028724",
            "6028723",
            "6028693",
        ],
    },
)
