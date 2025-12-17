# Only important task datasets
dataset_ids = dict(
    detection={
        "vehicle_detection": [
            "6037439",  # 1000
        ],
        ("vehicle_roi_3d", "vehicle_detection"): [
            # "6040836",  # 2500
            "6036773",  # 1157
        ],
        "person_detection": [
            "6038688",  # 625
        ],
        ("person_roi_3d", "person_detection"): [
            "6040353",  # 939
        ],
    },
    semantic_segmentation={
        "lane_segmentation": ["6036750"],
    },
)
