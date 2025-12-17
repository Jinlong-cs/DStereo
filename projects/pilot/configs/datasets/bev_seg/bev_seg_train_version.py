# flake8: noqa
import os

from hat.utils import Config
from projects.pilot.configs.datasets.bev_seg.utils import (
    del_item_in_dict,
    update_item_in_dict,
)

only_use_update_version = False
# Example:
v1_0_0 = {
    "BJ": {
        "BYD": {
            "BYD19_1": {"sample_interval": 10},
            "BYD19_2": None,
        },
        "LS912": {
            "LS912_1": None,
            "LS912_2": None,
        },
    }
}

v1_0_1_del = {
    "BJ": {
        "BYD": {
            "BYD19_2": None,
        },
        "LS912": {
            "LS912_1": None,
        },
    }
}

v1_0_1_update = {
    "BJ": {
        "BYD": {
            "BYD19_2": None,
        },
        "LS912": {
            "LS912_1": None,
        },
    }
}

v1_0_1 = del_item_in_dict(v1_0_0, v1_0_1_del)
v1_0_1 = update_item_in_dict(v1_0_0, v1_0_1_update, only_use_update_version)
# version:
pipeline_test = {
    "11V_FULL": {
        "BYD": {"BYD19_20221211_by_arrow_road_1": {"sample_interval": 10}}
    }
}
FULL_11v_V1_0_0 = {
    "11V_FULL": {
        "BYD": {
            "BYD19_20221211_by_arrow_road_1": None,
            "BYD19_20221130_by_arrow_road_0": None,
            "BYD19_20221212_by_arrow_road_1": None,
            "BYD72_20221128_by_arrow_road_0": None,
            "BYD19_20221205_by_arrow_road_0": None,
            "BYD18_20221202_by_arrow_road_1": None,
            "BYD72_20221119_by_arrow_road_0": None,
            "BYD72_20221129_by_arrow_road_3": None,
            "BYD72_20221130_by_arrow_road_1": None,
            "BYD19_20221129_by_arrow_road_0": None,
            "BYD19_20221128_by_arrow_road_0": None,
            "BYD18_20221201_by_arrow_road_1": None,
            "BYD72_20221109_by_arrow_road_1": None,
            "BYD19_20221211_by_arrow_road_0": None,
            "BYD19_20221210_by_arrow_road_1": None,
            "BYD72_20221129_by_arrow_road_1": None,
            "BYD18_20221202_by_arrow_road_0": None,
            "BYD72_20221130_by_arrow_road_0": None,
            "BYD19_20221212_by_arrow_road_0": None,
            "BYD19_20221206_by_arrow_road_1": None,
            "BYD19_20221201_by_arrow_road_0": None,
            "BYD19_20221210_by_arrow_road_0": None,
            "BYD72_20221129_by_arrow_road_0": None,
            "BYD19_20221206_by_arrow_road_3": None,
            "BYD72_20221109_by_arrow_road_0": None,
            "BYD18_20221201_by_arrow_road_0": None,
        },
        "LS912": {
            "LS912_20221109_by_arrow_road_0": None,
            "LS912_20221114_by_arrow_road_0": None,
            "LS912_20221117_by_arrow_road_0": None,
            "LS912_20221116_by_arrow_road_0": None,
            "LS912_20221107_by_arrow_road_1": None,
            "LS912_20221111_by_arrow_road_0": None,
            "LS912_20221109_by_arrow_road_1": None,
            "LS912_20221107_by_arrow_road_0": None,
            "LS912_20221112_by_arrow_road_1": None,
        },
        "LX165": {
            "LX165_20230227_by_arrow_road_2": None,
            "LX165_20230228_by_arrow_road_2": None,
            "LX165_20230307_by_arrow_road_0": None,
            "LX165_20230304_by_arrow_road_1": None,
            "LX165_20230308_by_arrow_road_1": None,
            "LX165_20230226_by_arrow_road_0": None,
            "LX165_20230227_by_arrow_road_0": None,
            "LX165_20230306_by_arrow_road_2": None,
            "LX165_20230307_by_arrow_road_2": None,
            "LX165_20230226_by_arrow_road_2": None,
            "LX165_20230308_by_arrow_road_0": None,
            "LX165_20230226_by_arrow_road_1": None,
            "LX165_20230307_by_arrow_road_1": None,
            "LX165_20230306_by_arrow_road_1": None,
            "LX165_20230304_by_arrow_road_0": None,
            "LX165_20230308_by_arrow_road_2": None,
            "LX165_20230301_by_arrow_road_1": None,
            "LX165_20230228_by_arrow_road_1": None,
            "LX165_20230227_by_arrow_road_1": None,
            "LX165_20230304_by_arrow_road_2": None,
        },
        "UT126": {
            "UT126_20221026_by_arrow_road_1": None,
            "UT126_20221023_by_arrow_road_0": None,
            "UT126_20221026_by_arrow_road_0": None,
            "UT126_20220927_by_arrow_road_2": None,
        },
        "UT1R3": {
            "UT1R3_20221123_by_arrow_road_0": None,
            "UT1R3_20230206_by_arrow_road_0": None,
            "UT1R3_20221126_by_arrow_road_0": None,
            "UT1R3_20230207_by_arrow_road_0": None,
            "UT1R3_20230208_by_arrow_road_0": None,
            "UT1R3_20230204_by_arrow_road_0": None,
            "UT1R3_20230205_by_arrow_road_0": None,
            "UT1R3_20230209_by_arrow_road_0": None,
        },
        "UT263": {
            "UT263_20221015_by_arrow_road_0": None,
            "UT263_20220926_by_arrow_road_0": None,
            "UT263_20221010_by_arrow_road_1": None,
            "UT263_20220928_by_arrow_road_2": None,
            "UT263_20220925_by_arrow_road_0": None,
            "UT263_20220924_by_arrow_road_3": None,
            "UT263_20220923_by_arrow_road_0": None,
            "UT263_20221103_by_arrow_road_1": None,
            "UT263_20220926_by_arrow_road_2": None,
            "UT263_20220925_by_arrow_road_2": None,
            "UT263_20221012_by_arrow_road_0": None,
            "UT263_20220928_by_arrow_road_0": None,
            "UT263_20220923_by_arrow_road_2": None,
            "UT263_20221011_by_arrow_road_1": None,
            "UT263_20220924_by_arrow_road_2": None,
            "UT263_20221103_by_arrow_road_0": None,
            "UT263_20221015_by_arrow_road_1": None,
            "UT263_20220926_by_arrow_road_1": None,
            "UT263_20221010_by_arrow_road_0": None,
            "UT263_20220924_by_arrow_road_0": None,
            "UT263_20221011_by_arrow_road_0": None,
            "UT263_20220926_by_arrow_road_3": None,
            "UT263_20220928_by_arrow_road_1": None,
            "UT263_20220925_by_arrow_road_3": None,
            "UT263_20221012_by_arrow_road_1": None,
        },
        "UTHS6": {"UTHS6_20221114_by_arrow_road_0": None},
    }
}
