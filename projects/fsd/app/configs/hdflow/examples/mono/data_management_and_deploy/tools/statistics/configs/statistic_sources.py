from mono.data_management_and_deploy.tools.statistics.utils import (
    StatisticItem,
    StatisticSource,
)

# -----------------------------------------
# source configs
# -----------------------------------------
statistic_items = [
    StatisticItem(
        source_type=StatisticSource.leaderboard_id,
        values=[6042257],
        item_name="test_with_6042257",
        collect_keys=[
            # attrs
            "attrs.tags.time",
            "attrs.tags.weather",
            "attrs.tags.sensor",
            "attrs.tags.fov",
            "attrs.tags.isp",
            "attrs.tags.scene",
            # class types
            "vehicle.attrs.Orientation",
            "vehicle.attrs.confidence",
            "vehicle.attrs.ignore",
            "vehicle.attrs.occlusion",
            "vehicle.attrs.type",
        ],
    ),
    StatisticItem(
        source_type=StatisticSource.leaderboard_id,
        values=[6040946],
        task="segmentation",
        item_name="test_with_6042257",
        collect_keys=[
            "plate",
            "sensor",
            "camera_name",
            "time",
            "weather",
            "scene",
            "illumination",
        ],
    ),
    StatisticItem(
        source_type=StatisticSource.densebox_json,
        values=[
            "/horizon-bucket/mono/data/4pe_rear/det_vehicle_rear_v13-20220325-canno-train-all/data.json"  # noqa
        ],
        item_name="test_with_det_vehicle_rear_v8-10652-split_big_car-20210902",
        collect_keys=[
            "plate",
            "sensor",
            "camera_name",
            "time",
            "weather",
            "scene",
            "illumination",
        ],
    ),
]
