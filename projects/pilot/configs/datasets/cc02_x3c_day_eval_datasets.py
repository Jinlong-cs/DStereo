dataset_ids = dict(
    # detection_3d={
    #     "vehicle_heatmap_3d_detection":[
    #         "6029266",
    #     ],
    #     "ped_cyc_heatmap_3d_detection": [
    #         # person
    #         "6028977",
    #         # cyclist
    #         "6028978",
    #     ],
    # },
    detection={
        "vehicle_detection": [
            # "6026688",
            # "6028007",
            "6028901",
            # "6027695",
            # "6028652",
            # "6028725",
            # "6028726",
            # "6028729",
            # "6028731",
            # "6028754",
            # "6029417",
        ],
        ("vehicle_category_classification", "vehicle_detection"): [
            "6028298",
        ],
        ("vehicle_occlusion_classification", "vehicle_detection"): [
            "6026795",
        ],
        # ("vehicle_truncation_classification", "vehicle_detection"): [
        #     "",
        # ],
        ("vehicle_wheel_kps", "vehicle_detection"): [
            "6027905",
        ],
        ("vehicle_wheel_detection", "vehicle_detection"): [
            "6029565",
            # "6026789",
        ],
        ("vehicle_ground_line", "vehicle_detection"): [
            "6027899",
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            "6029266",
            # "6029262",
            # "6029324",
        ],
        "rear_detection": [
            "6028880",
            #     # "6027853",
            #     # "6028757",
            #     # "6028758",
            #     # "6028759",
            #     # "6028760",
            #     # "6028767",
            #     # "6028774",
            #     # "6027746",
        ],
        ("rear_part_classification", "rear_detection"): [
            "6027897",
        ],
        ("rear_occlusion_classification", "rear_detection"): [
            "6027860",
        ],
        # # ("rear_plate_detection", "rear_detection"): [
        # #     "",
        # # ],
        "person_detection": [
            # "6029101",
            #     # "6029088",
            #     # "6028925",
            "6028609",
        ],
        ("person_pose_classification", "person_detection"): [
            "6026756",
        ],
        ("person_occlusion_classification", "person_detection"): [
            "6026758",
        ],
        ("person_orientation_classification", "person_detection"): [
            "6026761",
        ],
        # ("person_face_detection", "person_detection"): [
        #     "",
        # ],
        ("person_roi_3d", "person_detection"): [
            "6028977",
        ],
        "cyclist_detection": [
            "6028655",
            # "6028923",
        ],
        ("cyclist_roi_3d", "cyclist_detection"): [
            "6028978",
        ],
    },
    semantic_segmentation={
        "default_segmentation": [
            "6028896",
            #     "6028897",
            #     "6029308",
            #     "6029309",
            #     "6029307",
            #     "6028748",
            #     "6028747",
            #     "6028746",
            #     "6028715",
            #     "6028713",
            #     "6028714",
        ],
        "lane_segmentation": [
            "6028893",
            #     "6029295",
        ],
    },
)
