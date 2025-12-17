# flake8: noqa
from projects.pilot.configs.datasets.bev_discrete_obj.utils import (
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
pipeline_test = {"BJ_V3": {"LS912": {"0068_LS912_20221209_ml_3map": None}}}
v1_5_0 = {
    "BJ_V3": {
        "LS912": {"0068_LS912_20221209_ml_3map": None},
        "NC109": {
            "0032_NC109_20221028_ml_9map": None,
            "0040_NC109_20221111_ml_13map": None,
            "0045_NC109_20221118_ml_26map": None,
            "0058_NC109_20221202_ml_4map": None,
        },
        "NC110": {
            "0046_NC110_20221118_ml_19map": None,
            "0059_NC110_20221202_ml_4map": None,
            "0069_NC110_20221216_ml_17map": None,
            "0070_NC110_20221216_ml_19map": None,
            "0071_NC110_20221216real_ml_32map": None,
            "0075_NC110_20221223_ml_9map": None,
            "0084_NC110_20221213_ml_36map": None,
        },
        "UT0Q9": {
            "0052_UT0Q9_20221128real_ml_12map": None,
            "0066_UT0Q9_20221209_ml_11map": None,
            "0074_UT0Q9_20221216_ml_5map": None,
            "0079_UT0Q9_20221228_ml_28map": None,
        },
        "UT126": {
            "0005_UT126_20220825_ml_41map": None,
            "0008_UT126_20220906_ml_13map": None,
            "0010_UT126_20220909_ml_14map": None,
            "0011_UT126_20220916_ml_31map": None,
            "0012_UT126_20220920_ml_12map": None,
            "0016_UT126_20220924_ml_18map": None,
            "0017_UT126_20220930_ml_34map": None,
            "0021_UT126_20221013_ml_20map": None,
            "0023_UT126_20221018_ml_16map": None,
            "0025_UT126_20221021_ml_33map": None,
            "0028_UT126_20221027_ml_5map": None,
            "0031_UT126_20221028_ml_23map": None,
            "0033_UT126_20221104_ml_12map": None,
            "0041_UT126_20221118_ml_21map": None,
            "0042_UT126_20221118_ml_20map": None,
            "0049_UT126_20221128real_ml_7map": None,
            "0055_UT126_20221202_ml_26map": None,
            "0067_UT126_20221209_ml_5map": None,
            "0072_UT126_20221216_ml_15map": None,
            "0080_UT126_20221228_ml_40map": None,
        },
        "UT1R3": {
            "0047_UT1R3_20221118_ml_14map": None,
            "0051_UT1R3_20221128real_ml_19map": None,
            "0061_UT1R3_20221202_ml_12map": None,
            "0076_UT1R3_20221228_ml_70map": None,
            "0085_UT1R3_20221217_ml_64map": None,
        },
        "UT263": {
            "0007_UT263_20220902_ml_21map": None,
            "0009_UT263_20220906_ml_5map": None,
            "0018_UT263_20220930_ml_14map": None,
            "0022_UT263_20221013_ml_33map": None,
            "0024_UT263_20221018_ml_12map": None,
            "0026_UT263_20221021_ml_19map": None,
            "0029_UT263_20221027_ml_14map": None,
            "0038_UT263_20221111_ml_37map": None,
            "0043_UT263_20221118_ml_32map": None,
            "0050_UT263_20221128_ml_7map": None,
            "0056_UT263_20221202_ml_30map": None,
            "0065_UT263_20221209_ml_18map": None,
            "0073_UT263_20221216_ml_18map": None,
            "0078_UT263_20221228_ml_66map": None,
            "0083_UT263_20221218_ml_42map": None,
        },
        "UT3J5": {
            "0048_UT3J5_20221118_ml_20map": None,
            "0053_UT3J5_20221128real_ml_9map": None,
            "0057_UT3J5_20221202_ml_11map": None,
            "0064_UT3J5_20221209_ml_59map": None,
            "0081_UT3J5_20221228_ml_96map": None,
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
    "BACK_GROUND": {
        "UT263": {"FSD_Site_UT263_20221123": None},
        "UT126": {
            "FSD_Site_UT126_20221211": None,
            "FSD_Site_UT126_20221217": None,
            "FSD_Site_UT126_20221222": None,
        },
    },
}

v1_6_0_update = {
    "BJ_V3": {
        "UT263": {
            "0089_UT263_20221211_ml_51map": None,
        },
        "UTHS6": {
            "0090_UTHS6_20221212_ml_46map": None,
        },
        "UT0Q9": {
            "0091_UT0Q9_20221207_ml_66map": None,
        },
        "UT1R3": {
            "0092_UT1R3_20221214_ml_52map": None,
        },
        "UT126": {
            "0093_UT126_20230103_ml_32map": None,
        },
    }
}

v1_6_0 = update_item_in_dict(v1_5_0, v1_6_0_update, only_use_update_version)

v1_7_0_update = {
    "BJ_pipeline2.0": {
        "UT1R3": {
            "merged__180~180__20230323~20230327": {"sample_interval": 1},
        },
    }
}
v1_7_0 = update_item_in_dict(v1_6_0, v1_7_0_update, only_use_update_version)

v2_0_0 = {
    "BJ_pipeline2.0": {
        "LS830": {"merged__150_150__20230302": None},
        "LS912": {"merged__0_141__20221113": None},
        "NC109": {"merged__0_141__20221018_20221111": None},
        "NC110": {
            "merged__0_141__20221104_20221116": None,
            "merged__0_141__20221118_20221221": None,
            "merged__0_141__20221122_20221206": None,
            "merged__0_141__20221228_20230112": None,
            "merged__0_141__20230115_20230119": None,
        },
        "UT0F3": {"merged__0_141__20221211_20230206": None},
        "UT0Q9": {
            "merged__0_141__20221024_20221026": None,
            "merged__0_141__20221126_20221206": None,
            "merged__0_141__20221207_20230103": None,
            "merged__0_141__20230105_20230213": None,
        },
        "UT126": {
            "merged__0_141__20220815_20220829": None,
            "merged__0_141__20220830_20220914": None,
            "merged__0_141__20220920_20220922": None,
            "merged__0_141__20220923_20221206": None,
            "merged__0_141__20221207_20230214": None,
        },
        "UT1R3": {
            "merged__0_141__20221104_20221204": None,
            "merged__0_141__20221209_20230104": None,
            "merged__0_141__20230107_20230110": None,
            "merged__0_141__20230130_20230215": None,
            "merged__180_180__20230323_20230327": None,
        },
        "UT263": {
            "merged__0_141__20220921_20221018": None,
            "merged__0_141__20221028_20221110": None,
            "merged__0_141__20221111_20221121": None,
            "merged__0_141__20221122_20221123": None,
            "merged__0_141__20221126_20221206": None,
            "merged__0_141__20221207_20230103": None,
            "merged__0_141__20230108_20230212": None,
        },
        "UT3J5": {
            "merged__0_141__20221110_20221206": None,
            "merged__0_141__20221207_20221215": None,
            "merged__0_141__20230104": None,
            "merged__0_141__20230105_20230215": None,
        },
        "UTHS6": {
            "merged__0_141__20221021_20221031": None,
            "merged__0_141__20221027_20221030": None,
            "merged__0_141__20221101_20221206": None,
            "merged__0_141__20221208_20230118": None,
            "merged__0_141__20230210_20230215": None,
        },
    },
    "BACK_GROUND": {
        "UT263": {"FSD_Site_UT263_20221123": None},
        "UT126": {
            "FSD_Site_UT126_20221211": None,
            "FSD_Site_UT126_20221217": None,
            "FSD_Site_UT126_20221222": None,
        },
    },
}
v2_0_0_update = {
    "BJ_pipeline_2.0_parking": {
        "LS912": {"merged__0_141__20221113": None},
        "NC109": {"merged__0_141__20221018_20221111": None},
        "NC110": {
            "merged__0_141__20221104_20221116": None,
            "merged__0_141__20221118_20221221": None,
            "merged__0_141__20221122_20221206": None,
            "merged__0_141__20221228_20230112": None,
            "merged__0_141__20230117_20230119": None,
        },
        "UT0F3": {"merged__0_141__20221211_20230206": None},
        "UT0Q9": {
            "merged__0_141__20221024_20221026": None,
            "merged__0_141__20221126_20221206": None,
            "merged__0_141__20221207_20230103": None,
            "merged__0_141__20230105_20230213": None,
        },
        "UT126": {
            "merged__0_141__20220830_20220914": None,
            "merged__0_141__20220920_20220922": None,
            "merged__0_141__20220923_20221206": None,
            "merged__0_141__20221207_20230214": None,
        },
        "UT1R3": {
            "merged__0_141__20221104_20221204": None,
            "merged__0_141__20221209_20230104": None,
            "merged__0_141__20230107_20230110": None,
            "merged__0_141__20230130_20230215": None,
        },
        "UT263": {
            "merged__0_141__20220921_20221018": None,
            "merged__0_141__20221028_20221110": None,
            "merged__0_141__20221111_20221121": None,
            "merged__0_141__20221122_20221123": None,
            "merged__0_141__20221126_20221206": None,
            "merged__0_141__20221208_20230103": None,
            "merged__0_141__20230108_20230212": None,
        },
        "UT3J5": {
            "merged__0_141__20221110_20221206": None,
            "merged__0_141__20221207_20221215": None,
            "merged__0_141__20230104": None,
            "merged__0_141__20230105_20230215": None,
        },
        "UTHS6": {
            "merged__0_141__20221021_20221031": None,
            "merged__0_141__20221027_20221030": None,
            "merged__0_141__20221101_20221206": None,
            "merged__0_141__20221208_20230128": None,
            "merged__0_141__20230210_20230215": None,
        },
    }
}
v2_1_0 = update_item_in_dict(v2_0_0, v2_0_0_update, only_use_update_version)

v1_0_0_temporal = {
    "BJ_pipeline2.0_mult": {
        "LS912": {"merged__0_141__20221113": None},
        "NC109": {"merged__0_141__20221018_20221111": None},
        "NC110": {
            "merged__0_141__20221104_20221116": None,
            "merged__0_141__20221118_20221221": None,
            "merged__0_141__20221122_20221206": None,
            "merged__0_141__20221228_20230112": None,
            "merged__0_141__20230115_20230119": None,
        },
        "UT0F3": {"merged__0_141__20221211_20230206": None},
        "UT0Q9": {
            "merged__0_141__20221024_20221026": None,
            "merged__0_141__20221126_20221206": None,
            "merged__0_141__20221207_20230103": None,
            "merged__0_141__20230105_20230213": None,
        },
        "UT126": {
            "merged__0_141__20220803_20220829": None,
            "merged__0_141__20220830_20220914": None,
            "merged__0_141__20220920_20220922": None,
            "merged__0_141__20220923_20221206": None,
            "merged__0_141__20221207_20230214": None,
        },
        "UT1R3": {
            "merged__0_141__20221104_20221204": None,
            "merged__0_141__20221209_20230104": None,
            "merged__0_141__20230107_20230110": None,
            "merged__0_141__20230130_20230215": None,
        },
        "UT263": {
            "merged__0_141__20220921_20221018": None,
            "merged__0_141__20221028_20221110": None,
            "merged__0_141__20221111_20221121": None,
            "merged__0_141__20221122_20221123": None,
            "merged__0_141__20221126_20221206": None,
            "merged__0_141__20221207_20230103": None,
            "merged__0_141__20230108_20230212": None,
        },
        "UT3J5": {
            "merged__0_141__20221110_20221206": None,
            "merged__0_141__20221207_20221215": None,
            "merged__0_141__20230104": None,
            "merged__0_141__20230105_20230215": None,
        },
        "UTHS6": {
            "merged__0_141__20221021_20221031": None,
            "merged__0_141__20221027_20221030": None,
            "merged__0_141__20221101_20221206": None,
            "merged__0_141__20221208_20230115": None,
            "merged__0_141__20230210_20230215": None,
        },
    }
}
