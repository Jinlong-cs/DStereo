from projects.pilot.configs.project_utils.enum import (
    BEVModelSetting,
    SensorName,
)

BEV_SENSOR_MODULES = {
    BEVModelSetting.pilot51_master: {
        "per_view_shape": {
            SensorName.camera_front: (2160, 3840),
            SensorName.camera_front_left: (1280, 1920),
            SensorName.camera_front_right: (1280, 1920),
            SensorName.camera_rear_left: (1280, 1920),
            SensorName.camera_rear_right: (1280, 1920),
            SensorName.camera_rear: (1280, 1920),
            SensorName.fisheye_front: (1536, 1920),
            SensorName.fisheye_rear: (1536, 1920),
            SensorName.fisheye_left: (1536, 1920),
            SensorName.fisheye_right: (1536, 1920),
            SensorName.camera_front_30fov: (2160, 3840),
        },
        "img_load_size": [(1920, 1080)]
        + [(960, 640)] * 5
        + [(960, 768)] * 4
        + [(1920, 1080)],
        "transforms": {
            "ANCResize3DV": dict(
                type="ANCResize3DV",
                size=[(540, 960)]
                + [(640, 960)] * 5
                + [(768, 960)] * 4
                + [(540, 960)],
            ),
            "ANCCrop3DV": dict(
                type="ANCCrop3DV",
                height=[512] + [640] * 5 + [768] * 4 + [512],
                width=[960] + [960] * 5 + [960] * 4 + [960],
                top=[0] + [0] * 5 + [0] * 4 + [0],
                left=[0] + [0] * 5 + [0] * 4 + [0],
            ),
        },
        "flag_for_group": 3,
    },
    BEVModelSetting.ek_bev: {
        "per_view_shape": {
            SensorName.camera_front: (2160, 3840),
            SensorName.camera_front_left: (1280, 1920),
            SensorName.camera_front_right: (1280, 1920),
            SensorName.camera_rear_left: (1280, 1920),
            SensorName.camera_rear_right: (1280, 1920),
            SensorName.camera_rear: (1280, 1920),
            SensorName.fisheye_front: (1280, 1920),
            SensorName.fisheye_rear: (1280, 1920),
            SensorName.fisheye_left: (1280, 1920),
            SensorName.fisheye_right: (1280, 1920),
            SensorName.camera_front_30fov: (2160, 3840),
        },
        "img_load_size": [(1920, 1080)]
        + [(960, 640)] * 5
        + [(960, 640)] * 4
        + [(1920, 1080)],
        "transforms": {
            "ANCResize3DV": dict(
                type="ANCResize3DV",
                size=[(540, 960)]
                + [(640, 960)] * 5
                + [(640, 960)] * 4
                + [(540, 960)],
            ),
            "ANCCrop3DV": dict(
                type="ANCCrop3DV",
                height=[512] + [640] * 5 + [640] * 4 + [512],
                width=[960] + [960] * 5 + [960] * 4 + [960],
                top=[0] + [0] * 5 + [0] * 4 + [0],
                left=[0] + [0] * 5 + [0] * 4 + [0],
            ),
        },
        "flag_for_group": 3,
    },
}
