# flake8: noqa
from projects.pilot.configs.datasets.bev_discrete_obj.utils import (
    del_item_in_dict,
    reduce_sample_interval,
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
pipeline_test = {"BJ": {"UT126": {"0005_UT126_20220825_ml_41map": None}}}
HDE_lidar_v1_2_2_small = {
    "BJ": {
        "UT126": {
            "0005_UT126_20220825_ml_41map": None,
            "0006_UT126_20220830_ml_17map": None,
            "0008_UT126_20220906_ml_13map": None,
            "0010_UT126_20220909_ml_14map": None,
            "0011_UT126_20220916_ml_31map": None,
            "0012_UT126_20220920_ml_12map": None,
            "0016_UT126_20220924_ml_18map": None,
            "0017_UT126_20220930_ml_34map": None,
            "0019_UT126_20221010_ml_5map": None,
            "0021_UT126_20221013_ml_20map": None,
            "0025_UT126_20221021_ml_33map": None,
            "0028_UT126_20221027_ml_5map": None,
        },
        "NC109": {
            "0035_NC109_20221104_ml_11map": None,
            "0040_NC109_20221111_ml_13map": None,
            "0045_NC109_20221118_ml_26map": None,
            "0058_NC109_20221202_ml_4map": None,
        },
        "NC110": {
            "0046_NC110_20221118_ml_19map": None,
            "0059_NC110_20221202_ml_4map": None,
        },
        "UT0Q9": {
            "0052_UT0Q9_20221128real_ml_12map": None,
            "0074_UT0Q9_20221216_ml_5map": None,
        },
        "UT1R3": {
            "0047_UT1R3_20221118_ml_14map": None,
            "0051_UT1R3_20221128real_ml_19map": None,
            "0061_UT1R3_20221202_ml_12map": None,
            "0062_UT1R3_20221209_ml_53map": None,
        },
        "UT3J5": {
            "0048_UT3J5_20221118_ml_20map": None,
            "0057_UT3J5_20221202_ml_11map": None,
            "0064_UT3J5_20221209_ml_59map": None,
        },
        "UTHS6": {
            "0036_UTHS6_20221104_ml_15map": None,
            "0039_UTHS6_20221111_ml_25map": None,
            "0044_UTHS6_20221118_ml_23map": None,
            "0054_UTHS6_20221128real_ml_2map": None,
            "0060_UTHS6_20221202_ml_13map": None,
            "0063_UTHS6_20221209_ml_30map": None,
        },
    },
    "11v_driving_data_small": {
        "LX165": {"merge_LX165_det_v5a_20230226_20230301_LX165_11v": None},
        "UT263": {
            "merge_UT263_det_v5a_20221116_det_ut126_ut263_v2_11v": None,
            "20221018_711931_1_FSD_Site_UT263_20220925": None,
            "merge_UT263_det_v5a_20221025_20221025_UT263_11v": None,
            "merge_UT263_det_v5a_20221031_20221031_UT263_11v": None,
            "merge_UT263_det_v5a_20221026_20221109_UT263_11v": None,
            "20221116_719153_1_FSD_Site_UT263_20221103": None,
            "merge_UT263_det_v5a_20221215_20221215_UT263_11v": None,
            "merge_UT263_det_v5a_20221223_20221224_UT263_11v": None,
        },
        "UT126": {
            "merge_UT126_det_v5a_20221116_det_ut126_ut263_v2_11v": None,
            "20221020_712561_1_FSD_Site_UT126_20220927": None,
            "20221017_711698_2_FSD_Site_UT126_20220807": None,
            "merge_UT126_det_v5a_20221025_1031_UT126_11v": None,
            "20221111_717960_1_FSD_Site_UT126_20221023": None,
            "20221115_718929_1_FSD_Site_UT126_20221026": None,
            "20221115_718929_1_FSD_Site_UT126_20221025": None,
            "merge_UT126_det_v5a_20221019_20221103_UT126_11v": None,
            "20221125_721390_1_FSD_Site_UT126_20221103": None,
        },
        "UT1R3": {
            "merge_UT1R3_det_v5a_20221108_1109_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20221028_20221111_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20221113_20221130_UT1R3_11v": None,
            "20221206_723762_1_FSD_Site_UT1R3_20221123": None,
            "merge_UT1R3_det_v5a_20221213_20221215_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20221217_20221220_UT1R3_11v": None,
        },
        "UTHS6": {
            "20221111_718174_1_FSD_Site_UTHS6_20221101": None,
            "merge_UTHS6_det_v5a_20221028_1102_UTHS6_11v": None,
            "merge_UTHS6_det_v5a_20221104_20221105_UTHS6_11v": None,
            "merge_UTHS6_det_v5a_20221024_20221027_UTHS6_11v": None,
            "merge_UTHS6_det_v5a_20221019_20221108_UTHS6_11v": None,
            "merge_UTHS6_det_v5a_20221114_20221129_UTHS6_11v": None,
            "20221206_723762_1_FSD_Site_UTHS6_20221114": None,
            "merge_UTHS6_det_v5a_20221213_20221215_UTHS6_11v": None,
        },
        "UT0Q9": {
            "merge_UT0Q9_det_v5a_20221016_20221020_UT0Q9_11v": None,
            "merge_UT0Q9_det_v5a_20221216_20221218_UT0Q9_11v": None,
            "merge_UT0Q9_det_v5a_20221026_20221027_UT0Q9_11v": None,
            "merge_UT0Q9_det_v5a_20221024_20221024_UT0Q9_11v": None,
            "merge_UT0Q9_det_v5a_20221026_20221027_UT0Q9_val_11v": None,
            "merge_UT0Q9_det_v5a_20221216_20221218_UT0Q9_val_11v": None,
        },
        "LS912": {
            "merge_LS912_det_v5a_20221119_20221130_LS912_11v": None,
            "20221215_725926_1_FSD_Site_LS912_20221126": None,
            "merge_LS912_det_v5a_20221203_LS912_11v": None,
            "merge_LS912_det_v5a_20221107_20221118_LS912_11v": None,
            "20221202_723057_1_FSD_Site_LS912_20221115": None,
            "merge_LS912_det_v5a_20221211_20221215_LS912_11v": None,
        },
        "NC110": {
            "merge_NC110_det_v5a_20221213_20221218_NC110_11v": None,
            "merge_NC110_det_v5a_20221213_20221218_NC110_val_11v": None,
        },
        "UT3J5": {
            "merge_UT3J5_det_v5a_20221210_20221212_UT3J5_11v": None,
            "merge_UT3J5_det_v5a_20221210_20221212_UT3J5_val_11v": None,
        },
    },
    "byd_7v_driving_data_small": {
        "BYD72": {"merge_BYD72_det_v5a_20221225_20230107_BYD72_7v": None},
        "BYD18": {
            "merge_BYD18_det_v5a_20221203_20221203_BYD18_7v": None,
            "merge_BYD18_det_v5a_20230102_20230105_BYD18_7v": None,
        },
        "BYD19": {
            "merge_BYD19_det_v5a_20221218_20230104_BYD19_7v": None,
            "merge_BYD19_det_v5a_20221127_20221127_BYD19_7v": None,
        },
    },
    "11v_driving_auto_show_small": {
        "LX165": {
            "merge_LX165_det_v5a_20230226_20230301_LX165_11v": None,
            "20230308_742574_1_background_0_FSD_Site_LX165_20230227": None,
            "merge_LX165_det_v5a_20230304_20230308_LX165_11v": None,
        },
        "UT1R3": {
            "20230220_738837_1_FSD_Site_UT1R3_20230209": None,
            "20230220_738832_1_FSD_Site_UT1R3_20230204": None,
            "20230220_738832_1_FSD_Site_UT1R3_20230205": None,
            "20230220_738832_1_FSD_Site_UT1R3_20230206": None,
            "20230220_738836_1_FSD_Site_UT1R3_20230205": None,
            "20230220_738836_1_FSD_Site_UT1R3_20230207": None,
            "20230220_738836_1_FSD_Site_UT1R3_20230208": None,
            "20230308_742574_1_background_0_FSD_Site_UT1R3_20230213": None,
        },
    },
    "11v_driving_background_autoshow": {
        "LX165": {
            "merge_LX165_det_v5a_20230228_20230301_LX165_11v": None,
            "20230308_742574_1_background_1_FSD_Site_LX165_20230227": None,
        },
        "UT1R3": {
            "20230220_738832_1_FSD_Site_UT1R3_20230205": None,
            "20230220_738832_1_FSD_Site_UT1R3_20230206": None,
            "20230220_738832_1_FSD_Site_UT1R3_20230204": None,
            "20230220_738836_1_FSD_Site_UT1R3_20230208": None,
            "20230220_738836_1_FSD_Site_UT1R3_20230205": None,
            "20230220_738836_1_FSD_Site_UT1R3_20230207": None,
            "20230220_738837_1_FSD_Site_UT1R3_20230209": None,
            "merge_UT1R3_det_v5a_20230209_20230209_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20230205_20230209_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20230205_20230205_UT1R3_11v": None,
            "20230308_742574_1_background_1_FSD_Site_UT1R3_20230213": None,
        },
    },
    "11v_driving_background": {
        "UT263": {
            "20230112_732484_1_FSD_Site_UT263_20221223": None,
            "merge_UT263_det_v5a_20221217_20221221_UT263_11v": None,
        },
        "UT1R3": {
            "merge_UT1R3_det_v5a_20221217_20221220_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20221213_20221224_UT1R3_11v": None,
        },
        "UT0Q9": {"merge_UT0Q9_det_v5a_20221217_20221218_UT0Q9_11v": None},
        "UT126": {"merge_UT126_det_v5a_20221213_20221220_UT126_11v": None},
        "LS912": {"merge_LS912_det_v5a_20221211_20221215_LS912_11v": None},
        "LX165": {"merge_LX165_det_v5a_20230226_20230301_LX165_11v": None},
    },
    "byd7v_background": {
        "BYD72": {
            "merge_BYD72_det_v5a_20221123_20221209_BYD72_7v": None,
            "merge_BYD72_det_v5a_20221217_20221227_BYD72_7v": None,
            "merge_BYD72_det_v5a_20221225_20230107_BYD72_7v": None,
            "merge_BYD72_det_v5a_20221123_20221212_BYD72_7v": None,
        },
        "BYD18": {
            "merge_BYD18_det_v5a_20221208_20221208_BYD18_7v": None,
            "merge_BYD18_det_v5a_20221228_20221230_BYD18_7v": None,
            "merge_BYD18_det_v5a_20221203_20221203_BYD18_7v": None,
            "merge_BYD18_det_v5a_20230102_20230105_BYD18_7v": None,
            "merge_BYD18_det_v5a_20221122_20221208_BYD18_7v": None,
        },
        "BYD19": {
            "merge_BYD19_det_v5a_20221219_20221230_BYD19_7v": None,
            "merge_BYD19_det_v5a_20221127_20221127_BYD19_7v": None,
            "merge_BYD19_det_v5a_20221218_20230104_BYD19_7v": None,
            "merge_BYD19_det_v5a_20221116_20221213_BYD19_7v": None,
        },
    },
}
HDE_lidar_v1_2_2_wide = {
    "BJ": {
        "UT126": {
            "0005_UT126_20220825_ml_41map": None,
            "0006_UT126_20220830_ml_17map": None,
            "0008_UT126_20220906_ml_13map": None,
            "0010_UT126_20220909_ml_14map": None,
            "0011_UT126_20220916_ml_31map": None,
            "0012_UT126_20220920_ml_12map": None,
            "0017_UT126_20220930_ml_34map": None,
            "0019_UT126_20221010_ml_5map": None,
            "0021_UT126_20221013_ml_20map": None,
            "0025_UT126_20221021_ml_33map": None,
            "0028_UT126_20221027_ml_5map": None,
        },
        "UT3J5": {
            "0048_UT3J5_20221118_ml_20map": None,
            "0057_UT3J5_20221202_ml_11map": None,
            "0064_UT3J5_20221209_ml_59map": None,
        },
        "UTHS6": {
            "0036_UTHS6_20221104_ml_15map": None,
            "0039_UTHS6_20221111_ml_25map": None,
            "0044_UTHS6_20221118_ml_23map": None,
            "0054_UTHS6_20221128real_ml_2map": None,
            "0060_UTHS6_20221202_ml_13map": None,
            "0063_UTHS6_20221209_ml_30map": None,
        },
    },
    "11v_driving_data_wide": {
        "LX165": {"merge_LX165_det_v5a_20230226_20230301_LX165_11v": None},
        "UT263": {
            "merge_UT263_det_v5a_20221116_det_ut126_ut263_v2_11v": None,
            "20221018_711931_1_FSD_Site_UT263_20220925": None,
            "merge_UT263_det_v5a_20221025_20221025_UT263_11v": None,
            "merge_UT263_det_v5a_20221031_20221031_UT263_11v": None,
            "merge_UT263_det_v5a_20221026_20221109_UT263_11v": None,
            "20221116_719153_1_FSD_Site_UT263_20221103": None,
            "merge_UT263_det_v5a_20221215_20221215_UT263_11v": None,
            "merge_UT263_det_v5a_20221223_20221224_UT263_11v": None,
            "merge_UT263_det_v5a_20221217_20221221_UT263_11v": None,
        },
        "UT126": {
            "merge_UT126_det_v5a_20221116_det_ut126_ut263_v2_11v": None,
            "20221020_712561_1_FSD_Site_UT126_20220927": None,
            "20221017_711698_2_FSD_Site_UT126_20220807": None,
            "merge_UT126_det_v5a_20221025_1031_UT126_11v": None,
            "20221111_717960_1_FSD_Site_UT126_20221023": None,
            "20221115_718929_1_FSD_Site_UT126_20221026": None,
            "20221115_718929_1_FSD_Site_UT126_20221025": None,
            "merge_UT126_det_v5a_20221019_20221103_UT126_11v": None,
            "20221125_721390_1_FSD_Site_UT126_20221103": None,
            "merge_UT126_det_v5a_20221213_20221220_UT126_11v": None,
        },
        "UT1R3": {
            "merge_UT1R3_det_v5a_20221108_1109_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20221028_20221111_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20221113_20221130_UT1R3_11v": None,
            "20221206_723762_1_FSD_Site_UT1R3_20221123": None,
            # "merge_UT1R3_det_v5a_20221125_20221127_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20221213_20221215_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20221217_20221220_UT1R3_11v": None,
        },
        "UTHS6": {
            "20221111_718174_1_FSD_Site_UTHS6_20221101": None,
            "merge_UTHS6_det_v5a_20221028_1102_UTHS6_11v": None,
            "merge_UTHS6_det_v5a_20221104_20221105_UTHS6_11v": None,
            "merge_UTHS6_det_v5a_20221024_20221027_UTHS6_11v": None,
            "merge_UTHS6_det_v5a_20221019_20221108_UTHS6_11v": None,
            "merge_UTHS6_det_v5a_20221114_20221129_UTHS6_11v": None,
            # "20221212_724921_1_FSD_Site_UTHS6_20221126": None,
            "20221206_723762_1_FSD_Site_UTHS6_20221114": None,
            # "merge_UTHS6_det_v5a_20221124_20221126_UTHS6_11v": None,
            "merge_UTHS6_det_v5a_20221213_20221215_UTHS6_11v": None,
        },
        "UT0Q9": {
            "merge_UT0Q9_det_v5a_20221016_20221020_UT0Q9_11v": None,
            "merge_UT0Q9_det_v5a_20221216_20221218_UT0Q9_11v": None,
            "merge_UT0Q9_det_v5a_20221026_20221027_UT0Q9_11v": None,
            "merge_UT0Q9_det_v5a_20221024_20221024_UT0Q9_11v": None,
            "merge_UT0Q9_det_v5a_20221026_20221027_UT0Q9_val_11v": None,
            "merge_UT0Q9_det_v5a_20221216_20221218_UT0Q9_val_11v": None,
            "merge_UT0Q9_det_v5a_20221217_20221218_UT0Q9_11v": None,
        },
        "LS912": {
            "merge_LS912_det_v5a_20221119_20221130_LS912_11v": None,
            "20221215_725926_1_FSD_Site_LS912_20221126": None,
            "merge_LS912_det_v5a_20221203_LS912_11v": None,
            "merge_LS912_det_v5a_20221107_20221118_LS912_11v": None,
            "20221202_723057_1_FSD_Site_LS912_20221115": None,
            "merge_LS912_det_v5a_20221211_20221215_LS912_11v": None,
        },
        "NC110": {
            "merge_NC110_det_v5a_20221213_20221218_NC110_11v": None,
            "merge_NC110_det_v5a_20221213_20221218_NC110_val_11v": None,
            "merge_NC110_det_v5a_20221213_20221214_NC110_11v": None,
        },
        "UT3J5": {
            "merge_UT3J5_det_v5a_20221210_20221212_UT3J5_11v": None,
            "merge_UT3J5_det_v5a_20221210_20221212_UT3J5_val_11v": None,
        },
    },
    "byd_7v_driving_data_wide": {
        "BYD72": {
            "merge_BYD72_det_v5a_20221109_20221130_BYD72_7v": None,
            "merge_BYD72_det_v5a_20221225_20230107_BYD72_7v": None,
        },
        "BYD18": {
            "merge_BYD18_det_v5a_20221114_20221202_BYD18_7v": None,
            "merge_BYD18_det_v5a_20221114_20221202_BYD18_val_7v": None,
            "merge_BYD18_det_v5a_20221203_20221203_BYD18_7v": None,
            "merge_BYD18_det_v5a_20230102_20230105_BYD18_7v": None,
        },
        "BYD19": {
            "merge_BYD19_det_v5a_20221210_20221210_BYD19_7v": None,
            "merge_BYD19_det_v5a_20221106_20221212_BYD19_7v": None,
            "merge_BYD19_det_v5a_20221106_20221212_BYD19_val_7v": None,
            "merge_BYD19_det_v5a_20221218_20230104_BYD19_7v": None,
            "merge_BYD19_det_v5a_20221127_20221127_BYD19_7v": None,
        },
    },
    "11v_driving_auto_show_wide": {
        "LX165": {
            "20230308_742574_1_background_0_FSD_Site_LX165_20230227": None,
            "merge_LX165_det_v5a_20230304_20230308_LX165_11v": None,
        },
        "UT1R3": {
            "20230308_742574_1_background_0_FSD_Site_UT1R3_20230213": None
        },
    },
    "11v_driving_auto_show_wide_resample": {
        "LX165": {"merge_LX165_det_v5a_20230226_20230301_LX165_11v": None},
        "UT1R3": {
            "20230220_738837_1_FSD_Site_UT1R3_20230209": None,
            "20230220_738832_1_FSD_Site_UT1R3_20230204": None,
            "20230220_738832_1_FSD_Site_UT1R3_20230205": None,
            "20230220_738832_1_FSD_Site_UT1R3_20230206": None,
            "20230220_738836_1_FSD_Site_UT1R3_20230205": None,
            "20230220_738836_1_FSD_Site_UT1R3_20230207": None,
            "20230220_738836_1_FSD_Site_UT1R3_20230208": None,
        },
    },
    "11v_driving_background_autoshow": {
        "LX165": {
            "merge_LX165_det_v5a_20230228_20230301_LX165_11v": None,
            "20230308_742574_1_background_1_FSD_Site_LX165_20230227": None,
        },
        "UT1R3": {
            "20230220_738832_1_FSD_Site_UT1R3_20230205": None,
            "20230220_738832_1_FSD_Site_UT1R3_20230206": None,
            "20230220_738832_1_FSD_Site_UT1R3_20230204": None,
            "20230220_738836_1_FSD_Site_UT1R3_20230208": None,
            "20230220_738836_1_FSD_Site_UT1R3_20230205": None,
            "20230220_738836_1_FSD_Site_UT1R3_20230207": None,
            "20230220_738837_1_FSD_Site_UT1R3_20230209": None,
            "merge_UT1R3_det_v5a_20230209_20230209_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20230205_20230209_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20230205_20230205_UT1R3_11v": None,
            "20230308_742574_1_background_1_FSD_Site_UT1R3_20230213": None,
        },
    },
    "11v_driving_background": {
        "UT263": {
            "20230112_732484_1_FSD_Site_UT263_20221223": None,
            "merge_UT263_det_v5a_20221217_20221221_UT263_11v": None,
        },
        "UT1R3": {
            "merge_UT1R3_det_v5a_20221217_20221220_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20221213_20221224_UT1R3_11v": None,
        },
        "UT0Q9": {"merge_UT0Q9_det_v5a_20221217_20221218_UT0Q9_11v": None},
        "UT126": {"merge_UT126_det_v5a_20221213_20221220_UT126_11v": None},
        "LS912": {"merge_LS912_det_v5a_20221211_20221215_LS912_11v": None},
        "LX165": {"merge_LX165_det_v5a_20230226_20230301_LX165_11v": None},
    },
    "byd7v_background": {
        "BYD72": {
            "merge_BYD72_det_v5a_20221123_20221209_BYD72_7v": None,
            "merge_BYD72_det_v5a_20221217_20221227_BYD72_7v": None,
            "merge_BYD72_det_v5a_20221225_20230107_BYD72_7v": None,
            "merge_BYD72_det_v5a_20221123_20221212_BYD72_7v": None,
        },
        "BYD18": {
            "merge_BYD18_det_v5a_20221208_20221208_BYD18_7v": None,
            "merge_BYD18_det_v5a_20221228_20221230_BYD18_7v": None,
            "merge_BYD18_det_v5a_20221203_20221203_BYD18_7v": None,
            "merge_BYD18_det_v5a_20230102_20230105_BYD18_7v": None,
            "merge_BYD18_det_v5a_20221122_20221208_BYD18_7v": None,
        },
        "BYD19": {
            "merge_BYD19_det_v5a_20221219_20221230_BYD19_7v": None,
            "merge_BYD19_det_v5a_20221127_20221127_BYD19_7v": None,
            "merge_BYD19_det_v5a_20221218_20230104_BYD19_7v": None,
            "merge_BYD19_det_v5a_20221116_20221213_BYD19_7v": None,
        },
    },
}

# small vcs data of new version
v1_2_3_small_update = {
    "11v_driving_background": {
        "UTHS6": {
            "merge_UTHS6_det_v5a_20221223_20221228_UTHS6_11v": None,
        },
        "UT1R3": {
            "merge_UT1R3_det_v5a_20221216_20221228_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20221231_20221231_UT1R3_11v": None,
            "merge_UT1R3_badcase_v5a_20221201_20221230_UT1R3_11v": None,
        },
        "LS912": {
            "merge_LS912_det_v5a_20221217_20221217_LS912_11v": None,
        },
        "UT263": {
            "merge_UT263_det_v5a_20221222_20221227_UT263_11v": None,
        },
        "UT126": {
            "merge_UT126_det_v5a_20221223_20221226_UT126_11v": None,
            "merge_UT126_det_v5a_20221227_20221229_UT126_11v": None,
        },
        "NC110": {
            "merge_NC110_det_v5a_20221231_20221231_NC110_11v": None,
        },
    },
    "byd7v_background": {
        "BYD72": {
            "merge_BYD72_det_v5a_20221205_20221212_BYD72_7v": None,
        },
        "BYD18": {
            "merge_BYD18_det_v5a_20221122_20221208_BYD18_7v_v2": None,
        },
        "BYD19": {
            "merge_BYD19_det_v5a_20221122_20221213_BYD19_7v": None,
        },
    },
}
# update data
HDE_lidar_v1_2_3_small = update_item_in_dict(
    HDE_lidar_v1_2_2_small, v1_2_3_small_update, only_use_update_version
)

# wide vcs data of new version
v1_2_3_wide_update = {
    "BJ": {
        "NC109": {
            "0035_NC109_20221104_ml_11map": None,
            "0040_NC109_20221111_ml_13map": None,
            "0045_NC109_20221118_ml_26map": None,
            "0058_NC109_20221202_ml_4map": None,
        },
        "NC110": {
            "0046_NC110_20221118_ml_19map": None,
            "0059_NC110_20221202_ml_4map": None,
        },
        "UT0Q9": {
            "0052_UT0Q9_20221128real_ml_12map": None,
            "0074_UT0Q9_20221216_ml_5map": None,
        },
        "UT1R3": {
            "0047_UT1R3_20221118_ml_14map": None,
            "0051_UT1R3_20221128real_ml_19map": None,
            "0061_UT1R3_20221202_ml_12map": None,
            "0062_UT1R3_20221209_ml_53map": None,
        },
    },
    "11v_driving_background": {
        "UTHS6": {
            "merge_UTHS6_det_v5a_20221223_20221228_UTHS6_11v": None,
        },
        "UT1R3": {
            "merge_UT1R3_det_v5a_20221216_20221228_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20221231_20221231_UT1R3_11v": None,
            "merge_UT1R3_badcase_v5a_20221201_20221230_UT1R3_11v": None,
        },
        "LS912": {
            "merge_LS912_det_v5a_20221217_20221217_LS912_11v": None,
        },
        "UT263": {
            "merge_UT263_det_v5a_20221222_20221227_UT263_11v": None,
        },
        "UT126": {
            "merge_UT126_det_v5a_20221223_20221226_UT126_11v": None,
            "merge_UT126_det_v5a_20221227_20221229_UT126_11v": None,
        },
        "NC110": {
            "merge_NC110_det_v5a_20221231_20221231_NC110_11v": None,
        },
    },
    "byd7v_background": {
        "BYD72": {
            "merge_BYD72_det_v5a_20221205_20221212_BYD72_7v": None,
        },
        "BYD18": {
            "merge_BYD18_det_v5a_20221122_20221208_BYD18_7v_v2": None,
        },
        "BYD19": {
            "merge_BYD19_det_v5a_20221122_20221213_BYD19_7v": None,
        },
    },
}
# update data
HDE_lidar_v1_2_3_wide = update_item_in_dict(
    HDE_lidar_v1_2_2_wide, v1_2_3_wide_update, only_use_update_version
)


# small vcs data of new version
v1_2_4_small_update = {
    "11v_driving_data_small": {
        "LX165": {
            "merge_LX165_det_v5a_20230302_20230317_LX165_11v": None,
            "merge_LX165_det_v5a_20230318_20230323_LX165_11v": None,
            "merge_LX165_det_v5a_20230324_20230328_LX165_11v": None,
        },
        "NC109": {
            "merge_NC109_det_v5a_20221201_20221205_NC109_11v": None,
            "merge_NC109_det_v5a_20221201_20221205_NC109_val_11v": None,
            "merge_NC109_det_v5a_20221223_20221231_NC109_11v": None,
            "merge_NC109_det_v5a_20230108_20230113_NC109_11v": None,
            "merge_NC109_det_v5a_20230108_20230113_NC109_val_11v": None,
        },
        "NC110": {
            "merge_NC110_det_v5a_20230106_20230106_NC110_11v": None,
            "merge_NC110_det_v5a_20230118_20230118_NC110_11v": None,
        },
        "UT126": {
            "merge_UT126_det_v5a_20230108_20230213_UT126_11v": None,
            "merge_UT126_det_v5a_20230108_20230213_UT126_val_11v": None,
            "merge_UT126_det_v5a_20230201_20230207_UT126_11v": None,
            "merge_UT126_det_v5a_20230201_20230207_UT126_val_11v": None,
        },
        "UT263": {
            "merge_UT263_det_v5a_20230112_20230114_UT263_11v": None,
            "merge_UT263_det_v5a_20230112_20230114_UT263_val_11v": None,
        },
    },
    "byd_7v_driving_data_small": {
        "BYD18": {
            "merge_BYD18_det_v5a_20230112_20230221_BYD18_7v": None,
        },
        "BYD19": {
            "merge_BYD19_det_v5a_20230205_20230221_BYD19_7v": None,
        },
        "BYD72": {
            "merge_BYD72_det_v5a_20230114_20230114_BYD72_7v": None,
        },
    },
}
# update data
HDE_lidar_v1_2_4_small = update_item_in_dict(
    HDE_lidar_v1_2_3_small, v1_2_4_small_update, only_use_update_version
)

# wide vcs data of new version
v1_2_4_wide_update = {
    "11v_driving_data_wide": {
        "LX165": {
            "merge_LX165_det_v5a_20230302_20230317_LX165_11v": None,
            "merge_LX165_det_v5a_20230318_20230323_LX165_11v": None,
            "merge_LX165_det_v5a_20230324_20230328_LX165_11v": None,
        },
        "NC109": {
            "merge_NC109_det_v5a_20221201_20221205_NC109_11v": None,
            "merge_NC109_det_v5a_20221201_20221205_NC109_val_11v": None,
            "merge_NC109_det_v5a_20221223_20221231_NC109_11v": None,
            "merge_NC109_det_v5a_20230108_20230113_NC109_11v": None,
            "merge_NC109_det_v5a_20230108_20230113_NC109_val_11v": None,
        },
        "NC110": {
            "merge_NC110_det_v5a_20230106_20230106_NC110_11v": None,
            "merge_NC110_det_v5a_20230118_20230118_NC110_11v": None,
        },
        "UT126": {
            "merge_UT126_det_v5a_20230108_20230213_UT126_11v": None,
            "merge_UT126_det_v5a_20230108_20230213_UT126_val_11v": None,
            "merge_UT126_det_v5a_20230201_20230207_UT126_11v": None,
            "merge_UT126_det_v5a_20230201_20230207_UT126_val_11v": None,
        },
        "UT263": {
            "merge_UT263_det_v5a_20230112_20230114_UT263_11v": None,
            "merge_UT263_det_v5a_20230112_20230114_UT263_val_11v": None,
        },
    },
    "byd_7v_driving_data_wide": {
        "BYD18": {
            "merge_BYD18_det_v5a_20230112_20230221_BYD18_7v": None,
        },
        "BYD19": {
            "merge_BYD19_det_v5a_20230205_20230221_BYD19_7v": None,
        },
        "BYD72": {
            "merge_BYD72_det_v5a_20230114_20230114_BYD72_7v": None,
        },
    },
}
# update data
HDE_lidar_v1_2_4_wide = update_item_in_dict(
    HDE_lidar_v1_2_3_wide, v1_2_4_wide_update, only_use_update_version
)


# small vcs data of new version
v1_2_5_small_update = {
    "11v_driving_data_small": {
        "NC109": {
            "20230217_738491_2_background_0_FSD_Site_NC109_20221207": None,
            "20230217_738491_2_background_0_FSD_Site_NC109_20221213": None,
            "20230217_738491_2_background_0_FSD_Site_NC109_20221215": None,
            "20230217_738491_2_background_0_FSD_Site_NC109_20221216": None,
            "20230217_738491_2_background_0_FSD_Site_NC109_20221217": None,
            "20230217_738491_2_background_0_FSD_Site_NC109_20221227": None,
            "20230217_738550_2_background_0_FSD_Site_NC109_20221206": None,
            "20230217_738550_2_background_0_FSD_Site_NC109_20221208": None,
            "20230217_738550_2_background_0_FSD_Site_NC109_20221209": None,
            "20230217_738550_2_background_0_FSD_Site_NC109_20221210": None,
            "20230217_738550_2_background_0_FSD_Site_NC109_20221211": None,
            "20230217_738550_2_background_0_FSD_Site_NC109_20221212": None,
            "20230313_743562_1_background_0_FSD_Site_NC109_20230107": None,
            "20230313_743562_1_background_0_FSD_Site_NC109_20230114": None,
            "20230313_743562_1_background_0_FSD_Site_NC109_20230115": None,
            "20230313_743562_1_background_0_FSD_Site_NC109_20230116": None,
        },
        "UT1R3": {
            "20230306_741981_1_background_0_FSD_Site_UT1R3_20230210": None,
            "20230306_741981_1_background_0_FSD_Site_UT1R3_20230211": None,
            "20230423_753178_2_background_0_FSD_Site_UT1R3_20230221": None,
            "20230423_753189_2_background_0_FSD_Site_UT1R3_20230221": None,
        },
        "UT263": {
            "20230422_752973_2_background_0_FSD_Site_UT263_20230228": None,
            "20230422_752974_2_background_0_FSD_Site_UT263_20230228": None,
        },
    },
    "byd_7v_driving_data_small": {
        "BYD18": {
            "20230306_742087_1_background_0_FSD_Site_BYD18_20221226": None,
            "20230314_743997_1_background_0_FSD_Site_BYD18_20230218": None,
            "20230314_743997_1_background_0_FSD_Site_BYD18_20230219": None,
            "20230314_743997_1_background_0_FSD_Site_BYD18_20230220": None,
            "20230314_744019_1_background_0_FSD_Site_BYD18_20230224": None,
            "20230314_744019_1_background_0_FSD_Site_BYD18_20230225": None,
            "20230314_744019_1_background_0_FSD_Site_BYD18_20230227": None,
            "20230411_750223_3_background_0_FSD_Site_BYD18_20230325": None,
            "20230411_750223_3_background_0_FSD_Site_BYD18_20230327": None,
        },
        "BYD19": {
            "20230314_743997_1_background_0_FSD_Site_BYD19_20230205": None,
            "20230314_744019_1_background_0_FSD_Site_BYD19_20230224": None,
            "20230411_750223_3_background_0_FSD_Site_BYD19_20230324": None,
        },
        "BYD72": {
            "20230306_742087_1_background_0_FSD_Site_BYD72_20221211": None,
            "20230306_742087_1_background_0_FSD_Site_BYD72_20221229": None,
        },
    },
    "11v_driving_background": {
        "UT3J5": {
            "merged__22_160__20221110_20221115": None,
            "merged__22_160__20230212_20230222": None,
        },
        "UT0F3": {"merged__22_160__20221212_20221229": None},
        "UTHS6": {
            "merged__22_160__20221106_20221116": None,
            "merged__22_160__20230207": None,
            "merged__22_160__20230301_20230306": None,
            "merge_UTHS6_det_v5a_20221218_20221220_UTHS6_11v": None,
            "merge_UTHS6_det_v5a_20230103_20230109_UTHS6_11v": None,
        },
        "UT0Q9": {
            "merged__22_160__20221202_20221205": None,
            "merged__22_160__20221207_20221215": None,
            "merged__22_160__20230202_20230304": None,
            "merge_UT0Q9_det_v5a_20230102_20230103_UT0Q9_11v": None,
        },
        "LX165": {
            "merge_LX165_det_v5a_20230226_20230301_LX165_11v_v2": None,
        },
        "UT126": {
            "merged__22_160__20221022_20221115": None,
            "merged__22_160__20221214": None,
        },
        "UT263": {
            "merged__22_160__20220929_20221008": None,
            "merged__22_160__20221027": None,
            "merged__22_160__20221129_20221205": None,
            "merged__22_160__20221207_20221215": None,
            "merged__22_160__20230109": None,
            "merge_UT263_det_v5a_20221228_20230102_UT263_11v": None,
        },
        "UT1R3": {
            "merged__22_160__20221107_20221203": None,
            "merge_UT1R3_det_v5a_20221219_20221220_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20221230_20230103_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20230204_20230206_UT1R3_11v": None,
        },
    },
    "byd7v_background": {
        "BYD72": {
            "merge_BYD72_det_v5a_20221229_20221229_BYD72_7v": None,
        },
        "BYD18": {
            "merge_BYD18_det_v5a_20221226_20221226_BYD18_7v": None,
            "merge_BYD18_det_v5a_20230218_20230227_BYD18_7v": None,
            "merge_BYD18_det_v5a_20230325_20230327_BYD18_7v": None,
        },
        "BYD19": {
            "merge_BYD19_det_v5a_20230205_20230224_BYD19_7v": None,
            "merge_BYD19_det_v5a_20230324_20230324_BYD19_7v": None,
        },
    },
}
# update data
HDE_lidar_v1_2_5_small = update_item_in_dict(
    HDE_lidar_v1_2_4_small, v1_2_5_small_update, only_use_update_version
)

# wide vcs data of new version
v1_2_5_wide_update = {
    "11v_driving_data_wide": {
        "NC109": {
            "20230217_738491_2_background_0_FSD_Site_NC109_20221207": None,
            "20230217_738491_2_background_0_FSD_Site_NC109_20221213": None,
            "20230217_738491_2_background_0_FSD_Site_NC109_20221215": None,
            "20230217_738491_2_background_0_FSD_Site_NC109_20221216": None,
            "20230217_738491_2_background_0_FSD_Site_NC109_20221217": None,
            "20230217_738491_2_background_0_FSD_Site_NC109_20221227": None,
            "20230217_738550_2_background_0_FSD_Site_NC109_20221206": None,
            "20230217_738550_2_background_0_FSD_Site_NC109_20221208": None,
            "20230217_738550_2_background_0_FSD_Site_NC109_20221209": None,
            "20230217_738550_2_background_0_FSD_Site_NC109_20221210": None,
            "20230217_738550_2_background_0_FSD_Site_NC109_20221211": None,
            "20230217_738550_2_background_0_FSD_Site_NC109_20221212": None,
            "20230313_743562_1_background_0_FSD_Site_NC109_20230107": None,
            "20230313_743562_1_background_0_FSD_Site_NC109_20230114": None,
            "20230313_743562_1_background_0_FSD_Site_NC109_20230115": None,
            "20230313_743562_1_background_0_FSD_Site_NC109_20230116": None,
        },
        "UT1R3": {
            "20230306_741981_1_background_0_FSD_Site_UT1R3_20230210": None,
            "20230306_741981_1_background_0_FSD_Site_UT1R3_20230211": None,
            "20230423_753178_2_background_0_FSD_Site_UT1R3_20230221": None,
            "20230423_753189_2_background_0_FSD_Site_UT1R3_20230221": None,
        },
        "UT263": {
            "20230422_752973_2_background_0_FSD_Site_UT263_20230228": None,
            "20230422_752974_2_background_0_FSD_Site_UT263_20230228": None,
        },
    },
    "byd_7v_driving_data_wide": {
        "BYD18": {
            "20230306_742087_1_background_0_FSD_Site_BYD18_20221226": None,
            "20230314_743997_1_background_0_FSD_Site_BYD18_20230219": None,
            "20230314_743997_1_background_0_FSD_Site_BYD18_20230220": None,
            "20230314_744019_1_background_0_FSD_Site_BYD18_20230224": None,
            "20230314_744019_1_background_0_FSD_Site_BYD18_20230225": None,
            "20230314_744019_1_background_0_FSD_Site_BYD18_20230227": None,
            "20230411_750223_3_background_0_FSD_Site_BYD18_20230325": None,
            "20230411_750223_3_background_0_FSD_Site_BYD18_20230327": None,
        },
        "BYD19": {
            "20230314_743997_1_background_0_FSD_Site_BYD19_20230205": None,
            "20230314_744019_1_background_0_FSD_Site_BYD19_20230224": None,
            "20230411_750223_3_background_0_FSD_Site_BYD19_20230324": None,
        },
        "BYD72": {
            "20230306_742087_1_background_0_FSD_Site_BYD72_20221211": None,
            "20230306_742087_1_background_0_FSD_Site_BYD72_20221229": None,
        },
    },
    "11v_driving_background": {
        "UT3J5": {
            "merged__22_160__20221110_20221115": None,
            "merged__22_160__20230212_20230222": None,
        },
        "UT0F3": {"merged__22_160__20221212_20221229": None},
        "UTHS6": {
            "merged__22_160__20221106_20221116": None,
            "merged__22_160__20230207": None,
            "merged__22_160__20230301_20230306": None,
            "merge_UTHS6_det_v5a_20221218_20221220_UTHS6_11v": None,
            "merge_UTHS6_det_v5a_20230103_20230109_UTHS6_11v": None,
        },
        "UT0Q9": {
            "merged__22_160__20221202_20221205": None,
            "merged__22_160__20221207_20221215": None,
            "merged__22_160__20230202_20230304": None,
            "merge_UT0Q9_det_v5a_20230102_20230103_UT0Q9_11v": None,
        },
        "LX165": {
            "merge_LX165_det_v5a_20230226_20230301_LX165_11v_v2": None,
        },
        "UT126": {
            "merged__22_160__20221022_20221115": None,
            "merged__22_160__20221214": None,
        },
        "UT263": {
            "merged__22_160__20220929_20221008": None,
            "merged__22_160__20221027": None,
            "merged__22_160__20221129_20221205": None,
            "merged__22_160__20221207_20221215": None,
            "merged__22_160__20230109": None,
            "merge_UT263_det_v5a_20221228_20230102_UT263_11v": None,
        },
        "UT1R3": {
            "merged__22_160__20221107_20221203": None,
            "merge_UT1R3_det_v5a_20221219_20221220_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20221230_20230103_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20230204_20230206_UT1R3_11v": None,
        },
    },
    "byd7v_background": {
        "BYD72": {
            "merge_BYD72_det_v5a_20221229_20221229_BYD72_7v": None,
        },
        "BYD18": {
            "merge_BYD18_det_v5a_20221226_20221226_BYD18_7v": None,
            "merge_BYD18_det_v5a_20230218_20230227_BYD18_7v": None,
            "merge_BYD18_det_v5a_20230325_20230327_BYD18_7v": None,
        },
        "BYD19": {
            "merge_BYD19_det_v5a_20230205_20230224_BYD19_7v": None,
            "merge_BYD19_det_v5a_20230324_20230324_BYD19_7v": None,
        },
    },
}
# update data
HDE_lidar_v1_2_5_wide = update_item_in_dict(
    HDE_lidar_v1_2_4_wide, v1_2_5_wide_update, only_use_update_version
)

# small vcs data of new version
# delete driving data in wide vcs range model
v1_2_6_small_del = {
    "11v_driving_auto_show_small": {
        "LX165": {
            "merge_LX165_det_v5a_20230226_20230301_LX165_11v": None,
            "20230308_742574_1_background_0_FSD_Site_LX165_20230227": None,
            "merge_LX165_det_v5a_20230304_20230308_LX165_11v": None,
        },
        "UT1R3": {
            "20230220_738837_1_FSD_Site_UT1R3_20230209": None,
            "20230220_738832_1_FSD_Site_UT1R3_20230204": None,
            "20230220_738832_1_FSD_Site_UT1R3_20230205": None,
            "20230220_738832_1_FSD_Site_UT1R3_20230206": None,
            "20230220_738836_1_FSD_Site_UT1R3_20230205": None,
            "20230220_738836_1_FSD_Site_UT1R3_20230207": None,
            "20230220_738836_1_FSD_Site_UT1R3_20230208": None,
            "20230308_742574_1_background_0_FSD_Site_UT1R3_20230213": None,
        },
    },
    "byd_7v_driving_data_small": {
        "BYD18": {
            "20230306_742087_1_background_0_FSD_Site_BYD18_20221226": None,
            "20230314_743997_1_background_0_FSD_Site_BYD18_20230218": None,
            "20230314_743997_1_background_0_FSD_Site_BYD18_20230219": None,
            "20230314_743997_1_background_0_FSD_Site_BYD18_20230220": None,
            "20230314_744019_1_background_0_FSD_Site_BYD18_20230224": None,
            "20230314_744019_1_background_0_FSD_Site_BYD18_20230225": None,
            "20230314_744019_1_background_0_FSD_Site_BYD18_20230227": None,
            "20230411_750223_3_background_0_FSD_Site_BYD18_20230325": None,
            "20230411_750223_3_background_0_FSD_Site_BYD18_20230327": None,
            "merge_BYD18_det_v5a_20230112_20230221_BYD18_7v": None,
            "merge_BYD18_det_v5a_20221203_20221203_BYD18_7v": None,
            "merge_BYD18_det_v5a_20230102_20230105_BYD18_7v": None,
        },
        "BYD19": {
            "20230314_743997_1_background_0_FSD_Site_BYD19_20230205": None,
            "20230314_744019_1_background_0_FSD_Site_BYD19_20230224": None,
            "20230411_750223_3_background_0_FSD_Site_BYD19_20230324": None,
            "merge_BYD19_det_v5a_20230205_20230221_BYD19_7v": None,
            "merge_BYD19_det_v5a_20221218_20230104_BYD19_7v": None,
            "merge_BYD19_det_v5a_20221127_20221127_BYD19_7v": None,
        },
        "BYD72": {
            "20230306_742087_1_background_0_FSD_Site_BYD72_20221211": None,
            "20230306_742087_1_background_0_FSD_Site_BYD72_20221229": None,
            "merge_BYD72_det_v5a_20230114_20230114_BYD72_7v": None,
            "merge_BYD72_det_v5a_20221225_20230107_BYD72_7v": None,
        },
    },
    "byd7v_background": {
        "BYD72": {
            "merge_BYD72_det_v5a_20221123_20221209_BYD72_7v": None,
            "merge_BYD72_det_v5a_20221217_20221227_BYD72_7v": None,
            "merge_BYD72_det_v5a_20221225_20230107_BYD72_7v": None,
            "merge_BYD72_det_v5a_20221123_20221212_BYD72_7v": None,
            "merge_BYD72_det_v5a_20221205_20221212_BYD72_7v": None,
            "merge_BYD72_det_v5a_20221229_20221229_BYD72_7v": None,
        },
        "BYD18": {
            "merge_BYD18_det_v5a_20221208_20221208_BYD18_7v": None,
            "merge_BYD18_det_v5a_20221228_20221230_BYD18_7v": None,
            "merge_BYD18_det_v5a_20221203_20221203_BYD18_7v": None,
            "merge_BYD18_det_v5a_20230102_20230105_BYD18_7v": None,
            "merge_BYD18_det_v5a_20221122_20221208_BYD18_7v": None,
            "merge_BYD18_det_v5a_20221122_20221208_BYD18_7v_v2": None,
            "merge_BYD18_det_v5a_20221226_20221226_BYD18_7v": None,
            "merge_BYD18_det_v5a_20230218_20230227_BYD18_7v": None,
            "merge_BYD18_det_v5a_20230325_20230327_BYD18_7v": None,
        },
        "BYD19": {
            "merge_BYD19_det_v5a_20221219_20221230_BYD19_7v": None,
            "merge_BYD19_det_v5a_20221127_20221127_BYD19_7v": None,
            "merge_BYD19_det_v5a_20221218_20230104_BYD19_7v": None,
            "merge_BYD19_det_v5a_20221116_20221213_BYD19_7v": None,
            "merge_BYD19_det_v5a_20221122_20221213_BYD19_7v": None,
            "merge_BYD19_det_v5a_20230205_20230224_BYD19_7v": None,
            "merge_BYD19_det_v5a_20230324_20230324_BYD19_7v": None,
        },
    },
}
HDE_lidar_v1_2_6_small = del_item_in_dict(
    HDE_lidar_v1_2_5_small, v1_2_6_small_del
)

v1_2_6_small_update = {
    "BJ": {
        "LS830": {
            "merged__0_180__20230302_20230309": {
                "sample_interval": 1,
            },
            "merged__0_180__20230316_20230320": {
                "sample_interval": 1,
            },
            "merged__0_180__20230322_20230330": {
                "sample_interval": 1,
            },
            "merged__181_204__20230327_20230408": {
                "sample_interval": 1,
            },
        },
        "NC109": {
            "merged__0_180__20221021_20221111": {
                "sample_interval": 1,
            },
        },
        "NC110": {
            "merged__0_180__20221104_20221116": {
                "sample_interval": 1,
            },
            "merged__0_180__20221213_20221221": {
                "sample_interval": 1,
            },
            "merged__0_180__20221228_20230112": {
                "sample_interval": 1,
            },
        },
        "UT0Q9": {
            "merged__0_180__20221024_20221026": {
                "sample_interval": 1,
            },
            "merged__0_180__20221129_20221206": {
                "sample_interval": 1,
            },
            "merged__0_180__20221207_20230103": {
                "sample_interval": 1,
            },
            "merged__0_180__20230105_20230228": {
                "sample_interval": 1,
            },
        },
        "UT126": {
            "merged__0_180__20220803_20220829": {
                "sample_interval": 1,
            },
            "merged__0_180__20220830_20220914": {
                "sample_interval": 1,
            },
            "merged__0_180__20220920_20220922": {
                "sample_interval": 1,
            },
            "merged__0_180__20220923_20221206": {
                "sample_interval": 1,
            },
            "merged__0_180__20221207_20230128": {
                "sample_interval": 1,
            },
            "merged__181_204__20230328_20230331": {
                "sample_interval": 1,
            },
        },
        "UT1R3": {
            "merged__0_180__20221104_20221116": {
                "sample_interval": 1,
            },
            "merged__0_180__20221211_20230104": {
                "sample_interval": 1,
            },
            "merged__0_180__20230130_20230225": {
                "sample_interval": 1,
            },
        },
        "UT263": {
            "merged__0_180__20220921_20221008": {
                "sample_interval": 1,
            },
            "merged__0_180__20221028_20221110": {
                "sample_interval": 1,
            },
            "merged__0_180__20221111_20221121": {
                "sample_interval": 1,
            },
            "merged__0_180__20221122": {
                "sample_interval": 1,
            },
            "merged__0_180__20221210_20230102": {
                "sample_interval": 1,
            },
            "merged__0_180__20230204_20230214": {
                "sample_interval": 1,
            },
            "merged__0_180__20230224_20230321": {
                "sample_interval": 1,
            },
        },
        "UTHS6": {
            "merged__0_180__20221027_20221030": {
                "sample_interval": 1,
            },
            "merged__0_180__20221101_20221206": {
                "sample_interval": 1,
            },
            "merged__0_180__20230210_20230228": {
                "sample_interval": 1,
            },
        },
    },
    "11v_driving_background": {
        "UTHS6": {
            "merge_UTHS6_det_v5a_20221223_20221228_UTHS6_11v": {
                "sample_interval": 10
            },
        },
        "UT1R3": {
            "merge_UT1R3_det_v5a_20221216_20221228_UT1R3_11v": {
                "sample_interval": 10
            },
            "merge_UT1R3_det_v5a_20221231_20221231_UT1R3_11v": {
                "sample_interval": 10
            },
            "merge_UT1R3_badcase_v5a_20221201_20221230_UT1R3_11v": {
                "sample_interval": 10
            },
        },
        "LS912": {
            "merge_LS912_det_v5a_20221217_20221217_LS912_11v": {
                "sample_interval": 10
            },
        },
        "UT263": {
            "merge_UT263_det_v5a_20221222_20221227_UT263_11v": {
                "sample_interval": 10
            },
        },
        "UT126": {
            "merge_UT126_det_v5a_20221223_20221226_UT126_11v": {
                "sample_interval": 10
            },
            "merge_UT126_det_v5a_20221227_20221229_UT126_11v": {
                "sample_interval": 10
            },
        },
        "NC110": {
            "merge_NC110_det_v5a_20221231_20221231_NC110_11v": {
                "sample_interval": 10
            },
        },
    },
    "11v_driving_background_autoshow": {
        "LX165": {
            "merge_LX165_det_v5a_20230228_20230301_LX165_11v": {
                "sample_interval": 10
            },
            "20230308_742574_1_background_1_FSD_Site_LX165_20230227": {
                "sample_interval": 10
            },
        },
        "UT1R3": {
            "20230220_738832_1_FSD_Site_UT1R3_20230205": {
                "sample_interval": 20
            },
            "20230220_738832_1_FSD_Site_UT1R3_20230206": {
                "sample_interval": 20
            },
            "20230220_738832_1_FSD_Site_UT1R3_20230204": {
                "sample_interval": 20
            },
            "20230220_738836_1_FSD_Site_UT1R3_20230208": {
                "sample_interval": 20
            },
            "20230220_738836_1_FSD_Site_UT1R3_20230205": {
                "sample_interval": 20
            },
            "20230220_738836_1_FSD_Site_UT1R3_20230207": {
                "sample_interval": 20
            },
            "20230220_738837_1_FSD_Site_UT1R3_20230209": {
                "sample_interval": 20
            },
            "merge_UT1R3_det_v5a_20230209_20230209_UT1R3_11v": {
                "sample_interval": 20
            },
            "merge_UT1R3_det_v5a_20230205_20230209_UT1R3_11v": {
                "sample_interval": 20
            },
            "merge_UT1R3_det_v5a_20230205_20230205_UT1R3_11v": {
                "sample_interval": 20
            },
            "20230308_742574_1_background_1_FSD_Site_UT1R3_20230213": {
                "sample_interval": 20
            },
        },
    },
    "11v_driving_data_small": {
        "NC109": {
            "merge_NC109_det_v5a_20221201_20221205_NC109_11v": {
                "sample_interval": 10
            },
            "merge_NC109_det_v5a_20221201_20221205_NC109_val_11v": {
                "sample_interval": 10
            },
            "merge_NC109_det_v5a_20221223_20221231_NC109_11v": {
                "sample_interval": 10
            },
            "merge_NC109_det_v5a_20230108_20230113_NC109_11v": {
                "sample_interval": 10
            },
            "merge_NC109_det_v5a_20230108_20230113_NC109_val_11v": {
                "sample_interval": 10
            },
            "20230217_738491_2_background_0_FSD_Site_NC109_20221207": {
                "sample_interval": 2
            },
            "20230217_738491_2_background_0_FSD_Site_NC109_20221213": {
                "sample_interval": 2
            },
            "20230217_738491_2_background_0_FSD_Site_NC109_20221215": {
                "sample_interval": 2
            },
            "20230217_738491_2_background_0_FSD_Site_NC109_20221216": {
                "sample_interval": 2
            },
            "20230217_738491_2_background_0_FSD_Site_NC109_20221217": {
                "sample_interval": 2
            },
            "20230217_738491_2_background_0_FSD_Site_NC109_20221227": {
                "sample_interval": 2
            },
            "20230217_738550_2_background_0_FSD_Site_NC109_20221206": {
                "sample_interval": 2
            },
            "20230217_738550_2_background_0_FSD_Site_NC109_20221208": {
                "sample_interval": 2
            },
            "20230217_738550_2_background_0_FSD_Site_NC109_20221209": {
                "sample_interval": 2
            },
            "20230217_738550_2_background_0_FSD_Site_NC109_20221210": {
                "sample_interval": 2
            },
            "20230217_738550_2_background_0_FSD_Site_NC109_20221211": {
                "sample_interval": 2
            },
            "20230217_738550_2_background_0_FSD_Site_NC109_20221212": {
                "sample_interval": 2
            },
            "20230313_743562_1_background_0_FSD_Site_NC109_20230107": {
                "sample_interval": 2
            },
            "20230313_743562_1_background_0_FSD_Site_NC109_20230114": {
                "sample_interval": 2
            },
            "20230313_743562_1_background_0_FSD_Site_NC109_20230115": {
                "sample_interval": 2
            },
            "20230313_743562_1_background_0_FSD_Site_NC109_20230116": {
                "sample_interval": 2
            },
        },
        "UT0F3": {
            "20230316_744558_1_background_0_FSD_Site_UT0F3_20230128": None,
            "20230316_744558_1_background_0_FSD_Site_UT0F3_20230129": None,
            "20230316_744558_1_background_0_FSD_Site_UT0F3_20230130": None,
            "20230316_744558_1_background_0_FSD_Site_UT0F3_20230131": None,
            "merge_UT0F3_det_v5a_20230220_20230222_UT0F3_11v": None,
        },
        "LS830": {
            "merge_LS830_det_v5a_20230221_20230221_LS830_11v": None,
        },
        "LX165": {
            "merge_LX165_det_v5a_20230226_20230301_LX165_11v": {
                "sample_interval": 10
            },
            "merge_LX165_det_v5a_20230302_20230317_LX165_11v": {
                "sample_interval": 10
            },
            "merge_LX165_det_v5a_20230318_20230323_LX165_11v": {
                "sample_interval": 10
            },
            "merge_LX165_det_v5a_20230324_20230328_LX165_11v": {
                "sample_interval": 10
            },
            "merge_LX165_det_v5a_20230404_20230408_LX165_11v": None,
            "merge_LX165_det_v5a_20230426_20230427_LX165_11v": None,
        },
        "UT263": {
            "merge_UT263_det_v5a_20221116_det_ut126_ut263_v2_11v": {
                "sample_interval": 15
            },
            "20221018_711931_1_FSD_Site_UT263_20220925": {
                "sample_interval": 15
            },
            "merge_UT263_det_v5a_20221025_20221025_UT263_11v": {
                "sample_interval": 15
            },
            "merge_UT263_det_v5a_20221031_20221031_UT263_11v": {
                "sample_interval": 15
            },
            "merge_UT263_det_v5a_20221026_20221109_UT263_11v": {
                "sample_interval": 15
            },
            "20221116_719153_1_FSD_Site_UT263_20221103": {
                "sample_interval": 15
            },
            "merge_UT263_det_v5a_20221215_20221215_UT263_11v": {
                "sample_interval": 15
            },
            "merge_UT263_det_v5a_20221223_20221224_UT263_11v": {
                "sample_interval": 15
            },
            "merge_UT263_det_v5a_20230112_20230114_UT263_11v": {
                "sample_interval": 15
            },
            "merge_UT263_det_v5a_20230112_20230114_UT263_val_11v": {
                "sample_interval": 15
            },
            "20230316_744495_1_background_0_FSD_Site_UT263_20230206": None,
            "20230316_744495_1_background_0_FSD_Site_UT263_20230209": None,
            "20230316_744558_1_background_0_FSD_Site_UT263_20230117": None,
        },
        "UT126": {
            "merge_UT126_det_v5a_20221116_det_ut126_ut263_v2_11v": {
                "sample_interval": 15
            },
            "20221020_712561_1_FSD_Site_UT126_20220927": {
                "sample_interval": 15
            },
            "20221017_711698_2_FSD_Site_UT126_20220807": {
                "sample_interval": 15
            },
            "merge_UT126_det_v5a_20221025_1031_UT126_11v": {
                "sample_interval": 15
            },
            "20221111_717960_1_FSD_Site_UT126_20221023": {
                "sample_interval": 15
            },
            "20221115_718929_1_FSD_Site_UT126_20221026": {
                "sample_interval": 15
            },
            "20221115_718929_1_FSD_Site_UT126_20221025": {
                "sample_interval": 15
            },
            "merge_UT126_det_v5a_20221019_20221103_UT126_11v": {
                "sample_interval": 15
            },
            "20221125_721390_1_FSD_Site_UT126_20221103": {
                "sample_interval": 15
            },
            "merge_UT126_det_v5a_20230108_20230213_UT126_11v": {
                "sample_interval": 15
            },
            "merge_UT126_det_v5a_20230108_20230213_UT126_val_11v": {
                "sample_interval": 15
            },
            "merge_UT126_det_v5a_20230201_20230207_UT126_11v": {
                "sample_interval": 15
            },
            "merge_UT126_det_v5a_20230201_20230207_UT126_val_11v": {
                "sample_interval": 15
            },
        },
        "UT1R3": {
            "merge_UT1R3_det_v5a_20221108_1109_UT1R3_11v": {
                "sample_interval": 20
            },
            "merge_UT1R3_det_v5a_20221028_20221111_UT1R3_11v": {
                "sample_interval": 20
            },
            "merge_UT1R3_det_v5a_20221113_20221130_UT1R3_11v": {
                "sample_interval": 20
            },
            "20221206_723762_1_FSD_Site_UT1R3_20221123": {
                "sample_interval": 20
            },
            "merge_UT1R3_det_v5a_20221213_20221215_UT1R3_11v": {
                "sample_interval": 20
            },
            "merge_UT1R3_det_v5a_20221217_20221220_UT1R3_11v": {
                "sample_interval": 20
            },
            "20230320_745267_1_background_0_FSD_Site_UT1R3_20230202": None,
        },
        "UTHS6": {
            "20221111_718174_1_FSD_Site_UTHS6_20221101": {
                "sample_interval": 15
            },
            "merge_UTHS6_det_v5a_20221028_1102_UTHS6_11v": {
                "sample_interval": 15
            },
            "merge_UTHS6_det_v5a_20221104_20221105_UTHS6_11v": {
                "sample_interval": 15
            },
            "merge_UTHS6_det_v5a_20221024_20221027_UTHS6_11v": {
                "sample_interval": 15
            },
            "merge_UTHS6_det_v5a_20221019_20221108_UTHS6_11v": {
                "sample_interval": 15
            },
            "merge_UTHS6_det_v5a_20221114_20221129_UTHS6_11v": {
                "sample_interval": 15
            },
            "20221206_723762_1_FSD_Site_UTHS6_20221114": {
                "sample_interval": 15
            },
            "merge_UTHS6_det_v5a_20221213_20221215_UTHS6_11v": {
                "sample_interval": 15
            },
            "20230320_745267_1_background_0_FSD_Site_UTHS6_20230203": None,
            "20230320_745267_1_background_0_FSD_Site_UTHS6_20230204": None,
            "20230320_745267_1_background_0_FSD_Site_UTHS6_20230205": None,
            "20230320_745267_1_background_0_FSD_Site_UTHS6_20230206": None,
        },
        "UT0Q9": {
            "merge_UT0Q9_det_v5a_20221016_20221020_UT0Q9_11v": {
                "sample_interval": 10
            },
            "merge_UT0Q9_det_v5a_20221216_20221218_UT0Q9_11v": {
                "sample_interval": 10
            },
            "merge_UT0Q9_det_v5a_20221026_20221027_UT0Q9_11v": {
                "sample_interval": 10
            },
            "merge_UT0Q9_det_v5a_20221024_20221024_UT0Q9_11v": {
                "sample_interval": 10
            },
            "merge_UT0Q9_det_v5a_20221026_20221027_UT0Q9_val_11v": {
                "sample_interval": 10
            },
            "merge_UT0Q9_det_v5a_20221216_20221218_UT0Q9_val_11v": {
                "sample_interval": 10
            },
            "20230316_744558_1_background_0_FSD_Site_UT0Q9_20230130": None,
            "20230316_744558_1_background_0_FSD_Site_UT0Q9_20230131": None,
            "20230320_745267_1_background_0_FSD_Site_UT0Q9_20230202": None,
            "20230320_745267_1_background_0_FSD_Site_UT0Q9_20230203": None,
            "merge_UT0Q9_det_v5a_20230219_20230221_UT0Q9_11v": None,
        },
        "LS912": {
            "merge_LS912_det_v5a_20221119_20221130_LS912_11v": {
                "sample_interval": 15
            },
            "20221215_725926_1_FSD_Site_LS912_20221126": {
                "sample_interval": 15
            },
            "merge_LS912_det_v5a_20221203_LS912_11v": {"sample_interval": 15},
            "merge_LS912_det_v5a_20221107_20221118_LS912_11v": {
                "sample_interval": 15
            },
            "20221202_723057_1_FSD_Site_LS912_20221115": {
                "sample_interval": 15
            },
            "merge_LS912_det_v5a_20221211_20221215_LS912_11v": {
                "sample_interval": 15
            },
            "20230316_744558_1_background_0_FSD_Site_LS912_20230116": None,
        },
        "NC110": {
            "merge_NC110_det_v5a_20221213_20221218_NC110_11v": {
                "sample_interval": 10
            },
            "merge_NC110_det_v5a_20221213_20221218_NC110_val_11v": {
                "sample_interval": 10
            },
            "merge_NC110_det_v5a_20230106_20230106_NC110_11v": {
                "sample_interval": 10
            },
            "merge_NC110_det_v5a_20230118_20230118_NC110_11v": {
                "sample_interval": 10
            },
            "merge_NC110_det_v5a_20230222_20230222_NC110_11v": None,
        },
        "UT3J5": {
            "merge_UT3J5_det_v5a_20221210_20221212_UT3J5_11v": {
                "sample_interval": 10
            },
            "merge_UT3J5_det_v5a_20221210_20221212_UT3J5_val_11v": {
                "sample_interval": 10
            },
            "20230316_744558_1_background_0_FSD_Site_UT3J5_20230131": None,
        },
    },
}
# update data
HDE_lidar_v1_2_6_small = update_item_in_dict(
    HDE_lidar_v1_2_6_small, v1_2_6_small_update, only_use_update_version
)

# wide vcs data of new version
# delete parking data in wide vcs range model
v1_2_6_wide_del = {
    "BJ": {
        "UT126": {
            "0005_UT126_20220825_ml_41map": None,
            "0006_UT126_20220830_ml_17map": None,
            "0008_UT126_20220906_ml_13map": None,
            "0010_UT126_20220909_ml_14map": None,
            "0011_UT126_20220916_ml_31map": None,
            "0012_UT126_20220920_ml_12map": None,
            "0017_UT126_20220930_ml_34map": None,
            "0019_UT126_20221010_ml_5map": None,
            "0021_UT126_20221013_ml_20map": None,
            "0025_UT126_20221021_ml_33map": None,
            "0028_UT126_20221027_ml_5map": None,
        },
        "UT3J5": {
            "0048_UT3J5_20221118_ml_20map": None,
            "0057_UT3J5_20221202_ml_11map": None,
            "0064_UT3J5_20221209_ml_59map": None,
        },
        "UTHS6": {
            "0036_UTHS6_20221104_ml_15map": None,
            "0039_UTHS6_20221111_ml_25map": None,
            "0044_UTHS6_20221118_ml_23map": None,
            "0054_UTHS6_20221128real_ml_2map": None,
            "0060_UTHS6_20221202_ml_13map": None,
            "0063_UTHS6_20221209_ml_30map": None,
        },
        "NC109": {
            "0035_NC109_20221104_ml_11map": None,
            "0040_NC109_20221111_ml_13map": None,
            "0045_NC109_20221118_ml_26map": None,
            "0058_NC109_20221202_ml_4map": None,
        },
        "NC110": {
            "0046_NC110_20221118_ml_19map": None,
            "0059_NC110_20221202_ml_4map": None,
        },
        "UT0Q9": {
            "0052_UT0Q9_20221128real_ml_12map": None,
            "0074_UT0Q9_20221216_ml_5map": None,
        },
        "UT1R3": {
            "0047_UT1R3_20221118_ml_14map": None,
            "0051_UT1R3_20221128real_ml_19map": None,
            "0061_UT1R3_20221202_ml_12map": None,
            "0062_UT1R3_20221209_ml_53map": None,
        },
    },
}
HDE_lidar_v1_2_6_wide = del_item_in_dict(
    HDE_lidar_v1_2_5_wide, v1_2_6_wide_del
)
v1_2_6_wide_update = {
    "11v_driving_data_wide": {
        "LS830": {
            "merge_LS830_det_v5a_20230221_20230221_LS830_11v": None,
        },
        # "LS912": {
        #     "20230316_744558_1_background_0_FSD_Site_LS912_20230116": None,
        # },
        "LX165": {
            "merge_LX165_det_v5a_20230404_20230408_LX165_11v": None,
            "merge_LX165_det_v5a_20230426_20230427_LX165_11v": None,
        },
        "NC110": {
            "merge_NC110_det_v5a_20230222_20230222_NC110_11v": None,
        },
        "UT0F3": {
            "20230316_744558_1_background_0_FSD_Site_UT0F3_20230128": None,
            "20230316_744558_1_background_0_FSD_Site_UT0F3_20230129": None,
            "20230316_744558_1_background_0_FSD_Site_UT0F3_20230130": None,
            "20230316_744558_1_background_0_FSD_Site_UT0F3_20230131": None,
            "merge_UT0F3_det_v5a_20230220_20230222_UT0F3_11v": None,
        },
        "UT0Q9": {
            "20230316_744558_1_background_0_FSD_Site_UT0Q9_20230130": None,
            "20230316_744558_1_background_0_FSD_Site_UT0Q9_20230131": None,
            "20230320_745267_1_background_0_FSD_Site_UT0Q9_20230202": None,
            "20230320_745267_1_background_0_FSD_Site_UT0Q9_20230203": None,
            "merge_UT0Q9_det_v5a_20230219_20230221_UT0Q9_11v": None,
        },
        "UT1R3": {
            "20230320_745267_1_background_0_FSD_Site_UT1R3_20230202": None,
        },
        "UT263": {
            "20230316_744495_1_background_0_FSD_Site_UT263_20230206": None,
            "20230316_744495_1_background_0_FSD_Site_UT263_20230209": None,
            "20230316_744558_1_background_0_FSD_Site_UT263_20230117": None,
        },
        "UT3J5": {
            "20230316_744558_1_background_0_FSD_Site_UT3J5_20230131": None,
        },
        "UTHS6": {
            "20230320_745267_1_background_0_FSD_Site_UTHS6_20230203": None,
            "20230320_745267_1_background_0_FSD_Site_UTHS6_20230204": None,
            "20230320_745267_1_background_0_FSD_Site_UTHS6_20230205": None,
            "20230320_745267_1_background_0_FSD_Site_UTHS6_20230206": None,
        },
    },
    "byd_7v_driving_data_wide": {
        "BYD19": {
            "20230331_747997_1_background_0_FSD_Site_BYD19_20230303": None,
            "20230331_747997_1_background_0_FSD_Site_BYD19_20230315": None,
            "20230331_747997_1_background_0_FSD_Site_BYD19_20230316": None,
            "20230331_747997_1_background_0_FSD_Site_BYD19_20230317": None,
        },
    },
}
# update data
HDE_lidar_v1_2_6_wide = update_item_in_dict(
    HDE_lidar_v1_2_6_wide, v1_2_6_wide_update, only_use_update_version
)

# update HDE_lidar_v1_2_7_small dataset
v1_2_7_small_del = {
    "BJ": {
        "UT126": {
            "0005_UT126_20220825_ml_41map": None,
            "0006_UT126_20220830_ml_17map": None,
            "0008_UT126_20220906_ml_13map": None,
            "0010_UT126_20220909_ml_14map": None,
            "0011_UT126_20220916_ml_31map": None,
            "0012_UT126_20220920_ml_12map": None,
            "0016_UT126_20220924_ml_18map": None,
            "0017_UT126_20220930_ml_34map": None,
            "0019_UT126_20221010_ml_5map": None,
            "0021_UT126_20221013_ml_20map": None,
            "0025_UT126_20221021_ml_33map": None,
            "0028_UT126_20221027_ml_5map": None,
        },
        "NC109": {
            "0035_NC109_20221104_ml_11map": None,
            "0040_NC109_20221111_ml_13map": None,
            "0045_NC109_20221118_ml_26map": None,
            "0058_NC109_20221202_ml_4map": None,
        },
        "NC110": {
            "0046_NC110_20221118_ml_19map": None,
            "0059_NC110_20221202_ml_4map": None,
        },
        "UT0Q9": {
            "0052_UT0Q9_20221128real_ml_12map": None,
            "0074_UT0Q9_20221216_ml_5map": None,
        },
        "UT1R3": {
            "0047_UT1R3_20221118_ml_14map": None,
            "0051_UT1R3_20221128real_ml_19map": None,
            "0061_UT1R3_20221202_ml_12map": None,
            "0062_UT1R3_20221209_ml_53map": None,
        },
        "UT3J5": {
            "0048_UT3J5_20221118_ml_20map": None,
            "0057_UT3J5_20221202_ml_11map": None,
            "0064_UT3J5_20221209_ml_59map": None,
        },
        "UTHS6": {
            "0036_UTHS6_20221104_ml_15map": None,
            "0039_UTHS6_20221111_ml_25map": None,
            "0044_UTHS6_20221118_ml_23map": None,
            "0054_UTHS6_20221128real_ml_2map": None,
            "0060_UTHS6_20221202_ml_13map": None,
            "0063_UTHS6_20221209_ml_30map": None,
        },
    },
}
HDE_lidar_v1_2_7_small = del_item_in_dict(
    HDE_lidar_v1_2_6_small,
    v1_2_7_small_del,
)
v1_2_7_small_update = {
    "BJ": {
        "LS830": {
            "merged__205_227__20230418_20230423": {
                "sample_interval": 1,
            },
            "merged__205_227__20230508_20230519": {
                "sample_interval": 1,
            },
        },
        "UT0F3": {
            "merged__205_227__20230418_20230513": {
                "sample_interval": 1,
            },
        },
        "UT0Q9": {
            "merged__205_227__20230417_20230508": {
                "sample_interval": 1,
            },
        },
        "UT126": {
            "merged__205_227__20230419": {
                "sample_interval": 1,
            },
        },
        "UT263": {
            "merged__205_227__20230421_20230423": {
                "sample_interval": 1,
            },
            "merged__205_227__20230427_20230516": {
                "sample_interval": 1,
            },
        },
        "UT1R3": {
            "merged__205_227__20230417_20230518": {
                "sample_interval": 1,
            },
        },
        "UT3J5": {
            "merged__205_227__20230425_20230520": {
                "sample_interval": 1,
            },
        },
    }
}
HDE_lidar_v1_2_7_small = update_item_in_dict(
    HDE_lidar_v1_2_7_small, v1_2_7_small_update, only_use_update_version
)

v1_2_7_wide_update = {
    "11v_driving_data_wide": {
        "LS830": {
            "merge_LS830_det_v5a_20230222_20230224_LS830_11v": None,
        },
        "UT3J5": {
            "merge_UT3J5_det_v5a_20230222_20230222_UT3J5_11v": None,
        },
        "UTHS6": {
            "merge_UTHS6_det_v5a_20230221_20230224_UTHS6_11v": None,
        },
        "LS912": {
            "merge_LS912_det_v5a_20230331_20230331_LS912_11v": None,
        },
        "NC109": {
            "merge_NC109_det_v5a_20230225_20230329_NC109_11v": None,
        },
        "UT0Q9": {
            "merge_UT0Q9_det_v5a_20230222_20230228_UT0Q9_11v": None,
        },
        "NC110": {
            "merge_NC110_det_v5a_20230225_20230228_NC110_11v": None,
            "merge_NC110_det_v5a_20230223_20230224_NC110_11v": None,
        },
        "UT0F3": {
            "merge_UT0F3_det_v5a_20230225_20230228_UT0F3_11v": None,
            "merge_UT0F3_det_v5a_20230223_20230224_UT0F3_11v": None,
        },
        "UT1R3": {
            "merge_UT1R3_det_v5a_20230220_20230225_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20230222_20230224_UT1R3_11v": None,
        },
    },
    "byd_7v_driving_data_wide": {
        "BYD18": {
            "merge_BYD18_det_v5a_20230401_20230415_BYD18_7v": None,
        },
        "BYD19": {
            "merge_BYD19_det_v5a_20230526_20230527_BYD19_7v": None,
            "merge_BYD19_det_v5a_20230407_20230412_BYD19_7v": None,
        },
    },
    "jira_badcase_data": {
        "word_background": {
            "MSD-17039": None,
            "MSD-18893": None,
            "MSD-16237": None,
            "MSD-18572": None,
            "MSD-18894": None,
            "MSD-16825": None,
            "MSD-19717": None,
            "MSD-21813": None,
            "MSD-17630": None,
            "MSD-21285": None,
            "MSD-21287": None,
            "MSD-20011": None,
            "MSD-21286": None,
        }
    },
}
# update data
HDE_lidar_v1_2_7_wide = update_item_in_dict(
    HDE_lidar_v1_2_6_wide, v1_2_7_wide_update, only_use_update_version
)

v1_2_8_small_update = {
    "BJ": {
        "LS830": {
            "merged__228_250__20230520_20230527": None,
        },
        "UT0F3": {
            "merged__228_250__20230514_20230605": None,
        },
        "UT0Q9": {
            "merged__228_250__20230519_20230601": None,
        },
        "UT1R3": {
            "merged__228_250__20230519_20230529": None,
        },
        "UT263": {
            "merged__228_250__20230519": None,
        },
        "UT370": {
            "merged__228_250__20230517_20230531": None,
        },
        "UT3J5": {
            "merged__228_250__20230516_20230605": None,
        },
    },
    "jira_badcase_data": {
        "word_background": {
            "MSD-17039": None,
            "MSD-18893": None,
            "MSD-16237": None,
            "MSD-18572": None,
            "MSD-18894": None,
            "MSD-16825": None,
            "MSD-19717": None,
            "MSD-21813": None,
            "MSD-17630": None,
            "MSD-21285": None,
            "MSD-21287": None,
            "MSD-20011": None,
            "MSD-21286": None,
        }
    },
}
# update data
HDE_lidar_v1_2_8_small = update_item_in_dict(
    HDE_lidar_v1_2_7_small, v1_2_8_small_update, only_use_update_version
)


# del data which does not contain stopline label
v1_2_8_wide_del = {
    "11v_driving_data_wide": {
        "UT263": {
            "merge_UT263_det_v5a_20221116_det_ut126_ut263_v2_11v": None,
            "20221018_711931_1_FSD_Site_UT263_20220925": None,
        },
        "UT126": {
            "merge_UT126_det_v5a_20221116_det_ut126_ut263_v2_11v": None,
            "20221020_712561_1_FSD_Site_UT126_20220927": None,
            "20221017_711698_2_FSD_Site_UT126_20220807": None,
        },
    }
}
# update data
HDE_lidar_v1_2_8_wide = del_item_in_dict(
    HDE_lidar_v1_2_7_wide, v1_2_8_wide_del
)
v1_2_8_wide_update = {
    "11v_driving_data_wide": {
        "UT263": {
            "merge_UT263_det_v5a_20230313_20230327_UT263_11v": None,
            "merge_UT263_det_v5a_20230328_20230331_UT263_11v": None,
        },
        "UT126": {
            "merge_UT126_det_v5a_20230325_20230325_UT126_11v": None,
            "merge_UT126_det_v5a_20230412_20230412_UT126_11v": None,
        },
        "LX165": {
            "merge_LX165_det_v5a_20230610_20230610_LX165_11v": None,
        },
        "LS830": {
            "merge_LS830_det_v5a_20230206_20230310_LS830_11v": None,
            "merge_LS830_det_v5a_20230227_20230228_LS830_11v": None,
            "merge_LS830_det_v5a_20230322_20230403_LS830_11v": None,
            "merge_LS830_det_v5a_20230316_20230321_LS830_11v": None,
            "merge_LS830_det_v5a_20230325_20230404_LS830_11v": None,
        },
        "UTHS6": {
            "merge_UTHS6_det_v5a_20230210_20230303_UTHS6_11v": None,
            "merge_UTHS6_det_v5a_20230228_20230329_UTHS6_11v": None,
            "merge_UTHS6_det_v5a_20230406_20230412_UTHS6_11v": None,
        },
        "LS912": {
            "merge_LS912_det_v5a_20230108_20230113_LS912_11v": None,
            "merge_LS912_det_v5a_20230401_20230404_LS912_11v": None,
            "merge_LS912_det_v5a_20230221_20230221_LS912_11v": None,
            "merge_LS912_det_v5a_20230126_20230220_LS912_11v": None,
            "merge_LS912_det_v5a_20230327_20230411_LS912_11v": None,
            "merge_LS912_det_v5a_20230407_20230407_LS912_11v": None,
            "merge_LS912_det_v5a_20230125_20230127_LS912_11v": None,
            "merge_LS912_det_v5a_20230406_20230408_LS912_11v": None,
        },
        "NC109": {
            "merge_NC109_det_v5a_20230224_20230224_NC109_11v": None,
        },
        "UT0Q9": {
            "merge_UT0Q9_det_v5a_20230207_20230211_UT0Q9_11v": None,
            "merge_UT0Q9_det_v5a_20230316_20230404_UT0Q9_11v": None,
            "merge_UT0Q9_det_v5a_20230407_20230409_UT0Q9_11v": None,
            "merge_UT0Q9_det_v5a_20230217_20230218_UT0Q9_11v": None,
            "merge_UT0Q9_det_v5a_20230411_20230411_UT0Q9_11v": None,
        },
        "NC110": {
            "merge_NC110_det_v5a_20230219_20230221_NC110_11v": None,
            "merge_NC110_det_v5a_20230412_20230412_NC110_11v": None,
        },
        "UT0F3": {
            "merge_UT0F3_det_v5a_20230207_20230212_UT0F3_11v": None,
            "merge_UT0F3_det_v5a_20230311_20230311_UT0F3_11v": None,
            "merge_UT0F3_det_v5a_20230216_20230218_UT0F3_11v": None,
        },
        "UT1R3": {
            "merge_UT1R3_det_v5a_20230107_20230107_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20230317_20230404_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20230228_20230228_UT1R3_11v": None,
            "merge_UT1R3_det_v5a_20230407_20230412_UT1R3_11v": None,
        },
    },
    "byd_7v_driving_data_wide": {
        "BYD18": {
            "merge_BYD18_det_v5a_20230402_20230417_BYD18_7v": None,
        },
        "BYD19": {
            "merge_BYD19_det_v5a_20230413_20230414_BYD19_7v": None,
        },
    },
}
# update data
HDE_lidar_v1_2_8_wide = update_item_in_dict(
    HDE_lidar_v1_2_8_wide, v1_2_8_wide_update, only_use_update_version
)

v1_2_9_small_update = {
    "BJ": {
        "LS830": {
            "merged__251_270__20230530_20230618": None,
            "merged__271_280__20230616": None,
        },
        "UT0F3": {
            "merged__251_270__20230530_20230621": None,
            "merged__271_280__20230614_20230628": None,
        },
        "UT0Q9": {
            "merged__251_270__20230602_20230607": None,
            "merged__251_270__20230608_20230614": None,
            "merged__251_270__20230616_20230617": None,
            "merged__271_280__20230611_20230613": None,
            "merged__271_280__20230619_20230625": None,
        },
        "UT1R3": {
            "merged__251_270__20230606_20230613": None,
            "merged__251_270__20230605": None,
            "merged__251_270__20230615_20230617": None,
            "merged__271_280__20230621_20230625": None,
        },
        "UT263": {
            "merged__251_270__20230612_20230619": None,
            "merged__271_280__20230616_20230621": None,
        },
        "UT370": {
            "merged__251_270__20230601_20230603": None,
            "merged__251_270__20230613_20230617": None,
        },
        "UT3J5": {
            "merged__251_270__20230601_20230613": None,
            "merged__251_270__20230617": None,
            "merged__271_280__20230615_20230620": None,
        },
    },
}
# update data
HDE_lidar_v1_2_9_small = update_item_in_dict(
    HDE_lidar_v1_2_8_small, v1_2_9_small_update, only_use_update_version
)
# Delete debased data. The images of fisheye left/right is reversed.
v1_2_9_small_del = {
    "BJ": {
        "NC109": {
            "merged__0_180__20221021_20221111": None,
        },
    },
}
HDE_lidar_v1_2_9_small = del_item_in_dict(
    HDE_lidar_v1_2_9_small,
    v1_2_9_small_del,
)


crossroad_wt_crosswalk_trainset_v1_0_0_wide_roadmarking = {
    "byd_7v_driving_data_wide": {
        "BYD19": {
            "crossroad_wt_crosswalk_trainset_merge_BYD19_20230531_20230531_single_posefilter_1p0": None,
            "crossroad_wt_crosswalk_trainset_merge_BYD19_20230609_20230611_single_posefilter_1p0": None,
            "crossroad_wt_crosswalk_trainset_merge_BYD19_20230629_20230629_single_posefilter_1p0": None,
        },
    },
    "11v_driving_data_wide": {
        "LS830": {
            "crossroad_wt_crosswalk_trainset_merge_LS830_20230518_20230526_single_posefilter_1p0": None,
            "crossroad_wt_crosswalk_trainset_merge_LS830_20230712_20230712_single_posefilter_1p0": None,
            "crossroad_wt_crosswalk_trainset_merge_LS830_20230719_20230719_single_posefilter_1p0": None,
        },
        "LS912": {
            "crossroad_wt_crosswalk_trainset_merge_LS912_20230423_20230721_single_posefilter_1p0": None,
        },
        "UT0F3": {
            "crossroad_wt_crosswalk_trainset_merge_UT0F3_20230425_20230515_single_posefilter_1p0": None,
        },
        "UT0Q9": {
            "crossroad_wt_crosswalk_trainset_merge_UT0Q9_20230423_20230423_single_posefilter_1p0": None,
            "crossroad_wt_crosswalk_trainset_merge_UT0Q9_20230619_20230704_single_posefilter_1p0": None,
            "crossroad_wt_crosswalk_trainset_merge_UT0Q9_20230706_20230721_single_posefilter_1p0": None,
        },
        "UT126": {
            "crossroad_wt_crosswalk_trainset_merge_UT126_20230523_20230523_single_posefilter_1p0": None,
            "crossroad_wt_crosswalk_trainset_merge_UT126_20230624_20230627_single_posefilter_1p0": None,
            "crossroad_wt_crosswalk_trainset_merge_UT126_20230629_20230717_single_posefilter_1p0": None,
        },
        "UT1R3": {
            "crossroad_wt_crosswalk_trainset_merge_UT1R3_20230628_20230630_single_posefilter_1p0": None,
            "crossroad_wt_crosswalk_trainset_merge_UT1R3_20230712_20230721_single_posefilter_1p0": None,
        },
        "UT263": {
            "crossroad_wt_crosswalk_trainset_merge_UT263_20230515_20230515_single_posefilter_1p0": None,
            "crossroad_wt_crosswalk_trainset_merge_UT263_20230519_20230520_single_posefilter_1p0": None,
            "crossroad_wt_crosswalk_trainset_merge_UT263_20230629_20230722_single_posefilter_1p0": None,
        },
        "UT370": {
            "crossroad_wt_crosswalk_trainset_merge_UT370_20230616_20230616_single_posefilter_1p0": None,
            "crossroad_wt_crosswalk_trainset_merge_UT370_20230625_20230722_single_posefilter_1p0": None,
        },
        "UT3J5": {
            "crossroad_wt_crosswalk_trainset_merge_UT3J5_20230714_20230720_single_posefilter_1p0": None,
        },
        "UTHS6": {
            "crossroad_wt_crosswalk_trainset_merge_UTHS6_20230514_20230714_single_posefilter_1p0": None,
            "crossroad_wt_crosswalk_trainset_merge_UTHS6_20230715_20230715_single_posefilter_1p0": None,
            "crossroad_wt_crosswalk_trainset_merge_UTHS6_20230722_20230722_single_posefilter_1p0": None,
        },
    },
}

crossroad_wo_crosswalk_trainset_v1_0_0_wide_roadmarking = {
    "byd_7v_driving_data_wide": {
        "BYD19": {
            "crossroad_wo_crosswalk_trainset_merge_BYD19_20230531_20230531_single_posefilter_1p0": None,
        },
    },
    "11v_driving_data_wide": {
        "LS830": {
            "crossroad_wo_crosswalk_trainset_merge_LS830_20230519_20230526_single_posefilter_1p0": None,
            "crossroad_wo_crosswalk_trainset_merge_LS830_20230719_20230719_single_posefilter_1p0": None,
        },
        "LS912": {
            "crossroad_wo_crosswalk_trainset_merge_LS912_20230423_20230721_single_posefilter_1p0": None
        },
        "UT0F3": {
            "crossroad_wo_crosswalk_trainset_merge_UT0F3_20230426_20230514_single_posefilter_1p0": None,
        },
        "UT0Q9": {
            "crossroad_wo_crosswalk_trainset_merge_UT0Q9_20230619_20230704_single_posefilter_1p0": None,
            "crossroad_wo_crosswalk_trainset_merge_UT0Q9_20230706_20230721_single_posefilter_1p0": None,
        },
        "UT126": {
            "crossroad_wo_crosswalk_trainset_merge_UT126_20230523_20230523_single_posefilter_1p0": None,
            "crossroad_wo_crosswalk_trainset_merge_UT126_20230624_20230627_single_posefilter_1p0": None,
            "crossroad_wo_crosswalk_trainset_merge_UT126_20230714_20230717_single_posefilter_1p0": None,
        },
        "UT1R3": {
            # "crossroad_wo_crosswalk_trainset_merge_UT1R3_20230627_20230627_single_posefilter_1p0": None,  # no data left after posefilter
            "crossroad_wo_crosswalk_trainset_merge_UT1R3_20230628_20230630_single_posefilter_1p0": None,
            "crossroad_wo_crosswalk_trainset_merge_UT1R3_20230712_20230721_single_posefilter_1p0": None,
        },
        "UT263": {
            "crossroad_wo_crosswalk_trainset_merge_UT263_20230504_20230515_single_posefilter_1p0": None,
            "crossroad_wo_crosswalk_trainset_merge_UT263_20230519_20230519_single_posefilter_1p0": None,
            "crossroad_wo_crosswalk_trainset_merge_UT263_20230629_20230722_single_posefilter_1p0": None,
        },
        "UT370": {
            "crossroad_wo_crosswalk_trainset_merge_UT370_20230616_20230616_single_posefilter_1p0": None,
            "crossroad_wo_crosswalk_trainset_merge_UT370_20230625_20230722_single_posefilter_1p0": None,
        },
        "UT3J5": {
            "crossroad_wo_crosswalk_trainset_merge_UT3J5_20230719_20230720_single_posefilter_1p0": None,
        },
        "UTHS6": {
            "crossroad_wo_crosswalk_trainset_merge_UTHS6_20230516_20230714_single_posefilter_1p0": None,
            "crossroad_wo_crosswalk_trainset_merge_UTHS6_20230715_20230717_single_posefilter_1p0": None,
        },
    },
}

junction_trainset_v1_0_0_wide_roadmarking = {
    "byd_7v_driving_data_wide": {
        "BYD19": {
            "junction_trainset_merge_BYD19_20230607_20230612_single_posefilter_1p0": None,
            "junction_trainset_merge_BYD19_20230629_20230629_single_posefilter_1p0": None,
        },
    },
    "11v_driving_data_wide": {
        "LS830": {
            "junction_trainset_merge_LS830_20230422_20230423_single_posefilter_1p0": None,
            "junction_trainset_merge_LS830_20230515_20230526_single_posefilter_1p0": None,
            "junction_trainset_merge_LS830_20230712_20230712_single_posefilter_1p0": None,
            "junction_trainset_merge_LS830_20230719_20230719_single_posefilter_1p0": None,
        },
        "LS912": {
            "junction_trainset_merge_LS912_20230422_20230721_single_posefilter_1p0": None,
        },
        "UT0F3": {
            "junction_trainset_merge_UT0F3_20230509_20230515_single_posefilter_1p0": None,
        },
        "UT0Q9": {
            "junction_trainset_merge_UT0Q9_20230624_20230704_single_posefilter_1p0": None,
            "junction_trainset_merge_UT0Q9_20230706_20230721_single_posefilter_1p0": None,
        },
        "UT126": {
            "junction_trainset_merge_UT126_20230624_20230627_single_posefilter_1p0": None,
            "junction_trainset_merge_UT126_20230629_20230717_single_posefilter_1p0": None,
        },
        "UT1R3": {
            "junction_trainset_merge_UT1R3_20230627_20230627_single_posefilter_1p0": None,
            "junction_trainset_merge_UT1R3_20230628_20230630_single_posefilter_1p0": None,
            "junction_trainset_merge_UT1R3_20230712_20230721_single_posefilter_1p0": None,
        },
        "UT263": {
            "junction_trainset_merge_UT263_20230504_20230515_single_posefilter_1p0": None,
            "junction_trainset_merge_UT263_20230519_20230519_single_posefilter_1p0": None,
            "junction_trainset_merge_UT263_20230629_20230722_single_posefilter_1p0": None,
        },
        "UT370": {
            "junction_trainset_merge_UT370_20230613_20230616_single_posefilter_1p0": None,
            "junction_trainset_merge_UT370_20230625_20230722_single_posefilter_1p0": None,
        },
        "UT3J5": {
            "junction_trainset_merge_UT3J5_20230710_20230720_single_posefilter_1p0": None,
        },
        "UTHS6": {
            "junction_trainset_merge_UTHS6_20230517_20230714_single_posefilter_1p0": None,
            "junction_trainset_merge_UTHS6_20230715_20230715_single_posefilter_1p0": None,
            "junction_trainset_merge_UTHS6_20230722_20230722_single_posefilter_1p0": None,
        },
    },
    "6v_driving_data_wide": {
        "LX066": {
            "junction_trainset_merge_LX066_20230728_20230728_single_posefilter_1p0": None,
        }
    },
}

# junction & crossroad data
junction_crossroad_trainset_v1_0_0_roadmarking_tmp = update_item_in_dict(
    junction_trainset_v1_0_0_wide_roadmarking,
    crossroad_wt_crosswalk_trainset_v1_0_0_wide_roadmarking,
    only_use_update_version,
)
junction_crossroad_trainset_v1_0_0_roadmarking = update_item_in_dict(
    junction_crossroad_trainset_v1_0_0_roadmarking_tmp,
    crossroad_wo_crosswalk_trainset_v1_0_0_wide_roadmarking,
    only_use_update_version,
)

# downsample SD data with scale=4
HDE_lidar_v1_2_8_wide = reduce_sample_interval(
    HDE_lidar_v1_2_8_wide, scale=4, ignore_tags=None
)
# update data
HDE_lidar_v1_3_0_wide_roadmarking = update_item_in_dict(
    HDE_lidar_v1_2_8_wide,
    junction_crossroad_trainset_v1_0_0_roadmarking,
    only_use_update_version,
)
