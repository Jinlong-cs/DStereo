import re
from typing import Set

version_matcher = re.compile(r"v\d+(\.\d+){2}$", re.IGNORECASE)


class SimpleEnum:
    @classmethod
    def values(cls) -> Set[str]:
        _values = {
            getattr(cls, attr)
            for attr in dir(cls)
            if not callable(getattr(cls, attr)) and not attr.startswith("__")
        }

        return _values


# ======================ATTENTION========================#
# 以下内容如需修改, 必须软件同学协商确认清楚!!!
# =======================================================#
class BEVModelName(SimpleEnum):
    # bev large range
    bev_multitask_stage1_fov120 = "bev_multitask_stage1_fov120"
    bev_multitask_stage1_side = "bev_multitask_stage1_fov100"
    bev_multitask_stage1_narrow = "bev_multitask_stage1_narrow"
    bev_multitask_stage2 = "bev_multitask_stage2"
    bev_multitask_stage2_temporal = "bev_multitask_stage2_temporal"
    bev_multitask_stage2_temporal_e2e = "bev_multitask_stage2_temporal_e2e"
    bev_multitask_stage2_trackernet = "beve2e_stage2_trackernet"
    # bev samll range
    bev_multitask_small_stage1_fov120 = "bev_multitask_small_stage1_fov120"
    bev_multitask_small_stage1_fisheye = "bev_multitask_small_stage1_fisheye"
    bev_multitask_small_stage2 = "bev_multitask_small_stage2"
    bev_multitask_small_stage2_temporal = "bev_multitask_small_stage2_temporal"
    # iqa
    iqa_parsing_fov120 = "iqa_parsing_fov120"
    iqa_parsing_fov100 = "iqa_parsing_fov100"
    iqa_parsing_fisheye = "iqa_parsing_fisheye"
    # lane_parsing
    lane_parsing_fov100 = "lane_parsing_fov100"
    lane_parsing_fisheye = "lane_parsing_fisheye"
    # datamasking
    datamasking_fov120 = "datamasking_fov120"
    datamasking_fov100 = "datamasking_fov100"
    datamasking_fisheye = "datamasking_fisheye"


class BEVModelPackName(SimpleEnum):
    bev_small_range = "fsd_bev_small_range.hbm"
    bev_large_range = "fsd_bev_large_range.hbm"


class BEVModelReleaseName(SimpleEnum):
    master_bev = "MCP5.1-AlgoModelBEV-Master"
    master_bev_temporal = "MCP5.1-AlgoModelBEVTemporal-Master"
    master_bev_e2e = "MCP5.1-AlgoModelBEVE2E-Master"
    ek_bev = "MCP5.1-AlgoModelBEV-EK"
    ek_bev_temporal = "MCP5.1-AlgoModelBEVTemporal-EK"
    ek_bev_e2e = "MCP5.1-AlgoModelBEVE2E-EK"


# ======================ATTENTION========================#
# 以下内容算法内部enum，非必要不修改
# =======================================================#
class BEVModelType(SimpleEnum):
    bev_7v = "bev_7v"
    bev_5v = "bev_5v"
    bev_7v_temporal = "bev_7v_temporal"
    bev_5v_temporal = "bev_5v_temporal"


class BEVModelSetting(SimpleEnum):
    pilot51_master = "pilot5.1_master"
    ek_bev = "ek_bev"


class BEVSubPorject(SimpleEnum):
    master_bev = "master_bev"
    master_bev_temporal = "master_bev_temporal"
    ek_bev = "ek_bev"
    ek_bev_temporal = "ek_bev_temporal"


# ======================Sensor Module========================#
class SensorName(SimpleEnum):
    camera_front = "camera_front"
    camera_front_left = "camera_front_left"
    camera_front_right = "camera_front_right"
    camera_rear_left = "camera_rear_left"
    camera_rear_right = "camera_rear_right"
    camera_rear = "camera_rear"
    fisheye_front = "fisheye_front"
    fisheye_rear = "fisheye_rear"
    fisheye_left = "fisheye_left"
    fisheye_right = "fisheye_right"
    camera_front_30fov = "camera_front_30fov"
