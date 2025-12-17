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
pipeline_test = {
    "BJ_multi_frame": {
        "LS830": {
            "merged__145_179__20230302_20230304": None,
        },
    }
}
v1_6_0 = {
    "BJ": {
        "NC110": {
            "0070_NC110_20221216_ml_19map": None,
            "0071_NC110_20221216real_ml_32map": None,
        },
        "UT0Q9": {
            "0086_UT0Q9_20221218_ml_27map": None,
            "0079_UT0Q9_20221228_ml_28map": None,
            "0052_UT0Q9_20221128real_ml_12map": None,
            "0066_UT0Q9_20221209_ml_11map": None,
        },
        "UT126": {
            "0037_UT126_20221111_ml_20map": None,
            "0042_UT126_20221118_ml_20map": None,
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
            "0023_UT126_20221018_ml_16map": None,
            "0025_UT126_20221021_ml_33map": None,
            "0028_UT126_20221027_ml_5map": None,
            "0033_UT126_20221104_ml_12map": None,
            "0041_UT126_20221118_ml_21map": None,
            "0049_UT126_20221128real_ml_7map": None,
            "0055_UT126_20221202_ml_26map": None,
            "0072_UT126_20221216_ml_15map": None,
            "0080_UT126_20221228_ml_40map": None,
        },
        "UT1R3": {
            "0085_UT1R3_20221217_ml_64map": None,
            "0047_UT1R3_20221118_ml_14map": None,
            "0051_UT1R3_20221128real_ml_19map": None,
            "0061_UT1R3_20221202_ml_12map": None,
            "0062_UT1R3_20221209_ml_53map": None,
            "0076_UT1R3_20221228_ml_70map": None,
        },
        "UT263": {
            "0007_UT263_20220902_ml_21map": None,
            "0009_UT263_20220906_ml_5map": None,
            "0015_UT263_20220920_ml_3map": None,
            "0018_UT263_20220930_ml_14map": None,
            "0020_UT263_20221010_ml_5map": None,
            "0022_UT263_20221013_ml_33map": None,
            "0024_UT263_20221018_ml_12map": None,
            "0026_UT263_20221021_ml_19map": None,
            "0029_UT263_20221027_ml_14map": None,
            "0034_UT263_20221104_ml_23map": None,
            "0038_UT263_20221111_ml_37map": None,
            "0043_UT263_20221118_ml_32map": None,
            "0050_UT263_20221128_ml_7map": None,
            "0056_UT263_20221202_ml_30map": None,
            "0065_UT263_20221209_ml_18map": None,
            "0078_UT263_20221228_ml_66map": None,
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
            "0060_UTHS6_20221202_ml_13map": None,
            "0063_UTHS6_20221209_ml_30map": None,
        },
    },
    "11v_driving_data_small": {
        "UT263": {
            "20221018_711931_1_FSD_Site_UT263_20220925": None,
            "merge_UT263_det_v5a_20221116_det_ut126_ut263_v2_11v": {
                "sample_interval": 5
            },
        }
    },
}
v1_6_0_update = {
    "BJ": {
        "UT0F3": {
            "merged__154_179__20230331": None,
            "merged__154_179__20230228_20230329": None,
            "merged__146_149__20230223_20230224": None,
            "merged__142_142__20230214": None,
            "merged__104_141__20221211_20230213": None,
            "merged__103_103__20221210_20230104": None,
        },
        "NC110": {
            "merged__154_179__20230324_20230327": None,
            "merged__154_179__20230331": None,
            "merged__89_102__20221228": None,
            "merged__104_141__20230115": None,
            "merged__104_141__20221228": None,
        },
        "UT0Q9": {
            "merged__154_179__20230301": None,
            "merged__146_149__20230218_20230220": None,
            "merged__104_141__20230110_20230213": None,
            "merged__89_102__20221206": None,
            "merged__89_102__20221207_20221215": None,
            "merged__89_102__20230106_20230109": None,
        },
        "UT126": {
            "merged__104_141__20221231": None,
            "merged__89_102__20221214_20230106": None,
        },
        "UT1R3": {
            "merged__104_141__20221114_20221116": None,
            "merged__104_141__20230108": None,
            "merged__104_141__20230130_20230215": None,
            "merged__89_102__20221211_20221215": None,
            "merged__89_102__20230107_20230110": None,
        },
        "UT263": {
            "merged__154_179__20230301_20230321": None,
            "merged__146_149__20230227": None,
            "merged__146_149__20230214": None,
            "merged__104_141__20221115_20221118": None,
            "merged__104_141__20230204_20230212": None,
            "merged__89_102__20221210_20221215": None,
            "merged__89_102__20230108_20230112": None,
        },
        "UT3J5": {
            "merged__154_179__20230228_20230320": None,
            "merged__104_141__20221114_20221201": None,
            "merged__104_141__20221215": None,
            "merged__104_141__20230104": None,
            "merged__104_141__20230105_20230215": None,
            "merged__89_102__20230106_20230113": None,
        },
        "UTHS6": {
            "merged__154_179__20230301_20230306": None,
            "merged__146_149__20230222": None,
            "merged__104_141__20221114_20221117": None,
            "merged__104_141__20230210_20230215": None,
            "merged__89_102__20221210_20221214": None,
        },
        "LS830": {
            "merged__154_179__20230303": None,
            "merged__154_179__20230329_20230330": None,
        },
    }
}
v1_6_0 = update_item_in_dict(v1_6_0, v1_6_0_update, only_use_update_version)

v1_6_1_update = {
    "BJ": {
        "LS830": {
            "merged__180_250__20230403": None,
            "merged__180_250__20230418": None,
            "merged__180_250__20230512_20230529": None,
        },
        "NC110": {
            "merged__180_250__20230408": None,
        },
        "UT0F3": {
            "merged__180_250__20230313_20230330": None,
            "merged__180_250__20230418_20230605": None,
            "merged__180_250__20230413": None,
        },
        "UT0Q9": {
            "merged__180_250__20230318_20230327": None,
            "merged__180_250__20230408_20230414": None,
            "merged__180_250__20230415": None,
            "merged__180_250__20230417_20230601": None,
        },
        "UT126": {
            "merged__180_250__20230330_20230407": None,
            "merged__180_250__20230412_20230414": None,
            "merged__180_250__20230417_20230419": None,
        },
        "UT1R3": {
            "merged__180_250__20230322_20230410": None,
            "merged__180_250__20230411_20230414": None,
            "merged__180_250__20230415": None,
            "merged__180_250__20230417_20230529": None,
        },
        "UT263": {
            "merged__180_250__20230313_20230327": None,
            "merged__180_250__20230329_20230423": None,
            "merged__180_250__20230424_20230516": None,
            "merged__180_250__20230519": None,
        },
        "UT370": {
            "merged__180_250__20230517_20230531": None,
        },
        "UT3J5": {
            "merged__180_250__20230325_20230410": None,
            "merged__180_250__20230317": None,
            "merged__180_250__20230411": None,
            "merged__180_250__20230418_20230605": None,
        },
    }
}

v1_6_1 = update_item_in_dict(v1_6_0, v1_6_1_update, only_use_update_version)

v_1_7_0_update = {
    "BJ": {
        "UT263": {"merged__348_353__20230914_20230918": None},
        "UT370": {"merged__348_353__20230901_20230916": None},
    }
}
v_1_7_0 = update_item_in_dict(v1_6_1, v_1_7_0_update, only_use_update_version)

EK_v_1_0_update = {
    "BJ": {"HBEK342": {"merged__420_421__20240106_20240112": None}}
}

v_1_7_1 = update_item_in_dict(
    v_1_7_0, EK_v_1_0_update, only_use_update_version
)

v1_6_0_multi_frame = {
    "BJ_multi_frame": {
        "LS830": {
            "merged__145_179__20230302_20230304": None,
            "merged__145_179__20230329_20230330": None,
        },
        "UT0F3": {
            "merged__145_179__20230223_20230329": None,
            "merged__145_179__20230331": None,
            "merged__104_143__20221211_20230219": None,
        },
        "UT3J5": {
            "merged__145_179__20230228_20230321": None,
            "merged__104_143__20221114_20221201": None,
            "merged__104_143__20221215": None,
            "merged__104_143__20230104": None,
            "merged__104_143__20230105_20230215": None,
            "merged__43_102__20221110_20221206": None,
            "merged__43_102__20221207_20221208": None,
            "merged__43_102__20230106_20230113": None,
        },
        "UT1R3": {
            "merged__145_179__20230216_20230225": None,
            "merged__104_143__20221114_20221116": None,
            "merged__104_143__20230108": None,
            "merged__104_143__20230130_20230215": None,
            "merged__43_102__20221107_20221204": None,
            "merged__43_102__20221209_20221219": None,
            "merged__43_102__20230107_20230110": None,
        },
        "UT126": {
            "merged__104_143__20221231": None,
            "merged__43_102__20221207_20230106": None,
            "merged__5_36__20220803_20220829": None,
            "merged__5_36__20220830_20220914": None,
            "merged__5_36__20220920_20220922": None,
            "merged__5_36__20220923_20221027": None,
        },
        "UT263": {
            "merged__145_179__20230214": None,
            "merged__145_179__20230224_20230321": None,
            "merged__104_143__20230204_20230212": None,
            "merged__43_102__20221126_20221206": None,
            "merged__43_102__20221207_20221215": None,
            "merged__43_102__20230108_20230112": None,
            "merged__5_36__20220921_20221018": None,
        },
        "UTHS6": {
            "merged__145_179__20230221_20230310": None,
            "merged__104_143__20221114_20221117": None,
            "merged__104_143__20230210_20230215": None,
            "merged__43_102__20221104_20221206": None,
            "merged__43_102__20221208_20221214": None,
            "merged__38_41__20221027_20221029": None,
            "merged__38_41__20221031": None,
            "merged__38_41__20221101_20221102": None,
            "merged__5_36__20221021_20221024": None,
        },
        "NC110": {
            "merged__145_179__20230324_20230327": None,
            "merged__145_179__20230331": None,
            "merged__104_143__20221228": None,
            "merged__104_143__20230115_20230228": None,
            "merged__43_102__20221125_20221203": None,
            "merged__43_102__20221228": None,
        },
        "UT0Q9": {
            "merged__145_179__20230215_20230308": None,
            "merged__145_179__20230316_20230324": None,
            "merged__104_143__20230110_20230213": None,
            "merged__43_102__20221126_20221206": None,
            "merged__43_102__20221207_20221215": None,
            "merged__43_102__20230106_20230109": None,
        },
    },
}

v_1_6_1_multi_frame_update = {
    "BJ_multi_frame": {
        "LS830": {
            "merged__180_250__20230403": None,
            "merged__180_250__20230418": None,
            "merged__180_250__20230512_20230529": None,
        },
        "NC110": {
            "merged__180_250__20230408": None,
        },
        "UT0F3": {
            "merged__180_250__20230313_20230330": None,
            "merged__180_250__20230418_20230605": None,
            "merged__180_250__20230413": None,
        },
        "UT0Q9": {
            "merged__180_250__20230318_20230327": None,
            "merged__180_250__20230408_20230414": None,
            "merged__180_250__20230415": None,
            "merged__180_250__20230417_20230601": None,
        },
        "UT126": {
            "merged__180_250__20230330_20230407": None,
            "merged__180_250__20230412_20230414": None,
            "merged__180_250__20230417_20230419": None,
        },
        "UT1R3": {
            "merged__180_250__20230322_20230410": None,
            "merged__180_250__20230411_20230414": None,
            "merged__180_250__20230415": None,
            "merged__180_250__20230417_20230529": None,
        },
        "UT263": {
            "merged__180_250__20230313_20230327": None,
            "merged__180_250__20230329_20230423": None,
            "merged__180_250__20230424_20230516": None,
            "merged__180_250__20230519": None,
        },
        "UT370": {
            "merged__180_250__20230517_20230531": None,
        },
        "UT3J5": {
            "merged__180_250__20230325_20230410": None,
            "merged__180_250__20230317": None,
            "merged__180_250__20230411": None,
            "merged__180_250__20230418_20230605": None,
        },
    }
}

v_1_6_1_multi_frame = update_item_in_dict(
    v1_6_0_multi_frame, v_1_6_1_multi_frame_update
)
