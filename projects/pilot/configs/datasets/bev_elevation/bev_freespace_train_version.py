# flake8: noqa
import os

from hat.utils import Config
from projects.pilot.configs.datasets.bev_elevation.utils import (
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
    "pipeline_test": {
        "UT263": {
            "UT263_20230815_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
        }
    }
}
v3_6_0 = {
    "STATIC_ANNO_WIDE_TOTAL": {
        "UT126": {
            "FSD_Site_UT126_20220917_v5a_wide_11v": None,
            "FSD_Site_UT126_20220920_second_v5a_wide_11v": None,
            "FSD_Site_UT126_20220921_v5a_wide_11v": None,
            "FSD_Site_UT126_20220923_second_v5a_wide_11v": None,
            "FSD_Site_UT126_20220924_second_v5a_wide_11v": None,
            "FSD_Site_UT126_20220925_second_v5a_wide_11v": None,
            "FSD_Site_UT126_20220926_second_v5a_wide_11v": None,
            "FSD_Site_UT126_20220927_second_v5a_wide_11v": None,
            "FSD_Site_UT126_20220928_second_v5a_wide_11v": None,
            "FSD_Site_UT126_20221006_v5a_wide_11v": None,
            "FSD_Site_UT126_20221010_v5a_wide_11v": None,
            "FSD_Site_UT126_20221013_v5a_wide_11v": None,
            "FSD_Site_UT126_20221016_v5a_wide_11v": None,
            "FSD_Site_UT126_20221020_v5a_wide_11v": None,
            "FSD_Site_UT126_20221023_v5a_wide_11v": None,
            "FSD_Site_UT126_20221027_v5a_wide_11v": None,
            "FSD_Site_UT126_20221030_v5a_wide_11v": None,
        },
        "UTHS6": {
            "FSD_Site_UTHS6_20221018_v5a_wide_11v": None,
            "FSD_Site_UTHS6_20221020_v5a_wide_11v": None,
            "FSD_Site_UTHS6_20221022_v5a_wide_11v": None,
            "FSD_Site_UTHS6_20221025_v5a_wide_11v": None,
            "FSD_Site_UTHS6_20221028_v5a_wide_11v": None,
            "FSD_Site_UTHS6_20221030_v5a_wide_11v": None,
            "FSD_Site_UTHS6_20221102_v5a_wide_11v": None,
            "FSD_Site_UTHS6_20221105_v5a_wide_11v": None,
            "FSD_Site_UTHS6_20221107_v5a_wide_11v": None,
        },
        "UT0Q9": {
            "FSD_Site_UT0Q9_20221011_v5a_wide_11v": None,
            "FSD_Site_UT0Q9_20221012_v5a_wide_11v": None,
            "FSD_Site_UT0Q9_20221013_v5a_wide_11v": None,
            "FSD_Site_UT0Q9_20221015_v5a_wide_11v": None,
            "FSD_Site_UT0Q9_20221016_v5a_wide_11v": None,
            "FSD_Site_UT0Q9_20221017_v5a_wide_11v": None,
            "FSD_Site_UT0Q9_20221018_v5a_wide_11v": None,
            "FSD_Site_UT0Q9_20221019_v5a_wide_11v": None,
            "FSD_Site_UT0Q9_20221020_v5a_wide_11v": None,
        },
        "UT263": {
            "FSD_Site_UT263_20220930_v5a_wide_11v": None,
            "FSD_Site_UT263_20221005_v5a_wide_11v": None,
            "FSD_Site_UT263_20221007_v5a_wide_11v": None,
            "FSD_Site_UT263_20221008_v5a_wide_11v": None,
            "FSD_Site_UT263_20221010_v5a_wide_11v": None,
            "FSD_Site_UT263_20221011_v5a_wide_11v": None,
            "FSD_Site_UT263_20221013_v5a_wide_11v": None,
            "FSD_Site_UT263_20221015_v5a_wide_11v": None,
            "FSD_Site_UT263_20221026_v5a_wide_11v": None,
            "FSD_Site_UT263_20221027_v5a_wide_11v": None,
            "FSD_Site_UT263_20221029_v5a_wide_11v": None,
            "FSD_Site_UT263_20221031_v5a_wide_11v": None,
        },
        "UT3J5": {
            "FSD_Site_UT3J5_20221017_v5a_wide_11v": None,
            "FSD_Site_UT3J5_20221018_v5a_wide_11v": None,
            "FSD_Site_UT3J5_20221019_v5a_wide_11v": None,
            "FSD_Site_UT3J5_20221020_v5a_wide_11v": None,
            "FSD_Site_UT3J5_20221021_v5a_wide_11v": None,
            "FSD_Site_UT3J5_20221022_v5a_wide_11v": None,
            "FSD_Site_UT3J5_20221023_v5a_wide_11v": None,
        },
        "UT1R3": {
            "FSD_Site_UT1R3_20221028_v5a_wide_11v": None,
            "FSD_Site_UT1R3_20221029_v5a_wide_11v": None,
            "FSD_Site_UT1R3_20221030_v5a_wide_11v": None,
            "FSD_Site_UT1R3_20221107_v5a_wide_11v": None,
            "FSD_Site_UT1R3_20221108_v5a_wide_11v": None,
        },
    },
    "ZHANGAIWU_WIDE": {
        "UTHS6": {
            "FSD_Site_UTHS6_20220921_freespace_wide_artificial_train_resampling": None
        }
    },
    "BYD_WIDE_QUALITY_CHECK_V1": {
        "BYD18": {
            "20230203_FSD_Site_BYD18_20221120_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD18_20221121_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD18_20221122_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD18_20221123_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD18_20221124_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD18_20221205_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD18_20221206_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "byd_freespace_v1_FSD_Site_BYD18_20221201_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "byd_freespace_v1_FSD_Site_BYD18_20221202_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "byd_freespace_v2_FSD_Site_BYD18_20221201_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "byd_freespace_v2_FSD_Site_BYD18_20221202_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230316_FSD_Site_BYD18_20230119_v5a_autolabel_freespace_7v": None,
            "20230316_FSD_Site_BYD18_20230129_v5a_autolabel_freespace_7v": None,
            "20230316_FSD_Site_BYD18_20230116_v5a_autolabel_freespace_7v": None,
            "20230316_FSD_Site_BYD18_20230118_v5a_autolabel_freespace_7v": None,
        },
        "BYD19": {
            "20230203_FSD_Site_BYD19_20221107_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD19_20221110_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD19_20221112_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD19_20221113_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD19_20221114_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD19_20221115_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD19_20221116_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD19_20221117_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD19_20221118_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD19_20221119_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "byd_freespace_v1_FSD_Site_BYD19_20221210_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "byd_freespace_v2_FSD_Site_BYD19_20221210_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
        },
        "BYD72": {
            "byd_freespace_v1_FSD_Site_BYD72_20221129_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "byd_freespace_v1_FSD_Site_BYD72_20221130_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "byd_freespace_v2_FSD_Site_BYD72_20221128_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
        },
    },
    "PARKING_QUALITY_CHECK_V1": {
        "UT126": {
            "20230315_UT126_20220927_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20221008_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20221010_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20220914_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20221006_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20221009_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20220929_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20221004_D_v5a_autolabel_freespace_11v": None,
        }
    },
    "SHANGHAI_DEMO": {
        "LX165": {
            "20230316_FSD_Site_LX165_20230226_v5a_autolabel_freespace_11v": None,
            "20230316_FSD_Site_LX165_20230227_v5a_autolabel_freespace_11v": None,
            "20230316_FSD_Site_LX165_20230228_v5a_autolabel_freespace_11v": None,
            "20230316_FSD_Site_LX165_20230301_v5a_autolabel_freespace_11v": None,
            "20230327_FSD_Site_LX165_20230226_v5a_autolabel_freespace_11v": None,
            # "20230327_FSD_Site_LX165_20230227_v5a_autolabel_freespace_11v": None,  # 数据被错误覆盖，需要重新生成
            "20230327_FSD_Site_LX165_20230228_v5a_autolabel_freespace_11v": None,
            "20230327_FSD_Site_LX165_20230301_v5a_autolabel_freespace_11v": None,
            "20230327_FSD_Site_LX165_20230306_v5a_autolabel_freespace_11v": None,
            "20230327_FSD_Site_LX165_20230307_v5a_autolabel_freespace_11v": None,
            "20230327_FSD_Site_LX165_20230308_v5a_autolabel_freespace_11v": None,
        }
    },
}

v3_7_0_update = {
    "PARKING_QUALITY_CHECK_V1": {
        "UT126": {
            "20230315_UT126_20220915_D_v5a_autolabel_freespace_11v": None,
            "20230407_UT126_20230216_D_v5a_autolabel_freespace_11v": None,
            "20230407_UT126_20230210_D_v5a_autolabel_freespace_11v": None,
            "20230407_UT126_20230213_D_v5a_autolabel_freespace_11v": None,
        },
        "UT1R3": {
            "20230407_UT1R3_20230220_D_v5a_autolabel_freespace_11v": None,
            "20230407_UT1R3_20230222_D_v5a_autolabel_freespace_11v": None,
            "20230407_UT1R3_20230228_D_v5a_autolabel_freespace_11v": None,
        },
        "UT0F3": {
            "20230407_UT0F3_20230223_D_v5a_autolabel_freespace_11v": None,
        },
        "UT0Q9": {
            "20230407_UT0Q9_20230215_D_v5a_autolabel_freespace_11v": None,
            "20230407_UT0Q9_20230130_D_v5a_autolabel_freespace_11v": None,
        },
        "LS830": {
            "20230407_LS830_20230302_D_v5a_autolabel_freespace_11v": None,
        },
    }
}

v3_7_0 = update_item_in_dict(v3_6_0, v3_7_0_update, only_use_update_version)

# add BYD data ~27w, 其中新模组~18w
v3_8_0_update = {
    "BYD_WIDE_QUALITY_CHECK_V1": {
        "BYD18": {
            "20230326_FSD_Site_BYD18_20230130_v5a_autolabel_freespace_7v": None,
            # "20230326_FSD_Site_BYD18_20230131_v5a_autolabel_freespace_7v": None,
            "20230326_FSD_Site_BYD18_20230201_v5a_autolabel_freespace_7v": None,
            "20230402_FSD_Site_BYD18_20230207_v5a_autolabel_freespace_7v": None,
            "20230402_FSD_Site_BYD18_20230208_v5a_autolabel_freespace_7v": None,
            "20230402_FSD_Site_BYD18_20230209_v5a_autolabel_freespace_7v": None,
            "20230402_FSD_Site_BYD18_20230210_v5a_autolabel_freespace_7v": None,
        },
        "BYD19": {
            "20230410_FSD_Site_BYD19_20230211_v5a_autolabel_freespace_7v": None,
            "20230410_FSD_Site_BYD19_20230213_v5a_autolabel_freespace_7v": None,
            "20230410_FSD_Site_BYD19_20230214_v5a_autolabel_freespace_7v": None,
        },
    },
    "BYD_WIDE_QUALITY_CHECK_NEW_MODULE": {
        "BYD18": {
            "20230419_FSD_Site_BYD18_20230328_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230329_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230330_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230331_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230403_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230404_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230406_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230328_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230329_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230401_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230403_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230404_v5a_autolabel_freespace_7v": None,
        },
    },
}
v3_8_0 = update_item_in_dict(v3_7_0, v3_8_0_update, only_use_update_version)

v3_9_0_update = {
    "BYD_WIDE_QUALITY_CHECK_NEW_MODULE": {
        "BYD19": {
            "20230425_FSD_Site_BYD19_20230329_v5a_autolabel_freespace_7v": None,
            "20230425_FSD_Site_BYD19_20230331_v5a_autolabel_freespace_7v": None,
            "20230425_FSD_Site_BYD19_20230404_v5a_autolabel_freespace_7v": None,
            "20230425_FSD_Site_BYD19_20230406_v5a_autolabel_freespace_7v": None,
            "20230425_FSD_Site_BYD19_20230407_v5a_autolabel_freespace_7v": None,
        },
    },
}
v3_9_0_del = {
    "BYD_WIDE_QUALITY_CHECK_NEW_MODULE": {
        "BYD18": {
            "20230419_FSD_Site_BYD18_20230331_v5a_autolabel_freespace_7v": None,
        },
    }
}
v3_9_0 = update_item_in_dict(v3_8_0, v3_9_0_update, only_use_update_version)
v3_9_0 = del_item_in_dict(v3_9_0, v3_9_0_del)

v3_9_1_update = {
    "BYD_WIDE_QUALITY_CHECK_NEW_MODULE": {
        "BYD19": {
            "20230504_FSD_Site_BYD19_20230408_v5a_autolabel_freespace_7v": None,
            "20230504_FSD_Site_BYD19_20230410_v5a_autolabel_freespace_7v": None,
            "20230504_FSD_Site_BYD19_20230411_v5a_autolabel_freespace_7v": None,
            "20230504_FSD_Site_BYD19_20230412_v5a_autolabel_freespace_7v": None,
        },
        "BYD18": {
            "20230504_FSD_Site_BYD18_20230406_v5a_autolabel_freespace_7v": None,
            "20230504_FSD_Site_BYD18_20230412_v5a_autolabel_freespace_7v": None,
        },
    },
}
v3_9_1 = update_item_in_dict(v3_9_0, v3_9_1_update, only_use_update_version)

v3_9_2_update = {
    "PARKING_QUALITY_CHECK_V1": {
        "UT3J5": {
            "20230427_UT3J5_20230315_D_v5a_autolabel_freespace_11v": None,
            "20230427_UT3J5_20230314_D_v5a_autolabel_freespace_11v": None,
            "20230427_UT3J5_20230318_D_v5a_autolabel_freespace_11v": None,
            "20230427_UT3J5_20230313_D_v5a_autolabel_freespace_11v": None,
            "20230427_UT3J5_20230316_D_v5a_autolabel_freespace_11v": None,
            "20230414_UT3J5_20230214_D_v5a_autolabel_freespace_11v": None,
            "20230414_UT3J5_20230220_D_v5a_autolabel_freespace_11v": None,
            "20230414_UT3J5_20230210_D_v5a_autolabel_freespace_11v": None,
        },
        "LS830": {
            "20230427_LS830_20230307_D_v5a_autolabel_freespace_11v": None,
            "20230427_LS830_20230328_D_v5a_autolabel_freespace_11v": None,
            "20230427_LS830_20230329_D_v5a_autolabel_freespace_11v": None,
            "20230427_LS830_20230325_D_v5a_autolabel_freespace_11v": None,
            "20230427_LS830_20230323_D_v5a_autolabel_freespace_11v": None,
            "20230427_LS830_20230324_D_v5a_autolabel_freespace_11v": None,
            "20230427_LS830_20230306_D_v5a_autolabel_freespace_11v": None,
            "20230427_LS830_20230322_D_v5a_autolabel_freespace_11v": None,
            "20230427_LS830_20230303_D_v5a_autolabel_freespace_11v": None,
        },
        "UT0F3": {
            "20230423_UT0F3_20230316_D_v5a_autolabel_freespace_11v": None,
            "20230423_UT0F3_20230317_D_v5a_autolabel_freespace_11v": None,
            "20230423_UT0F3_20230320_D_v5a_autolabel_freespace_11v": None,
        },
        "UT263": {
            "20230414_UT263_20230214_D_v5a_autolabel_freespace_11v": None,
        },
        "UT126": {
            "20230414_UT126_20230211_D_v5a_autolabel_freespace_11v": None,
            "20230414_UT126_20230212_D_v5a_autolabel_freespace_11v": None,
            "20230414_UT126_20230214_D_v5a_autolabel_freespace_11v": None,
        },
        "UTHS6": {
            "20230414_UTHS6_20230228_D_v5a_autolabel_freespace_11v": None,
        },
        "UT1R3": {
            "20230414_UT1R3_20230323_D_v5a_autolabel_freespace_11v": None,
        },
    },
}
v3_9_2 = update_item_in_dict(v3_9_1, v3_9_2_update, only_use_update_version)

# add BYD ~32w, 主线模组13.8w, 泊车5.7w
v3_9_3_update = {
    "BYD_WIDE_QUALITY_CHECK_NEW_MODULE": {
        "BYD19": {
            "20230602_FSD_Site_BYD19_20230421_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230422_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230423_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230424_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230508_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230509_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230510_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230511_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230512_v5a_autolabel_freespace_7v": None,
        },
        "BYD18": {
            "20230602_FSD_Site_BYD18_20230414_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD18_20230415_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD18_20230416_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD18_20230417_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD18_20230418_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD18_20230420_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD18_20230422_v5a_autolabel_freespace_7v": None,
        },
    },
    "AUTO_LABEL_QUALITY_CHECK_ZHUXIAN": {
        "UT0Q9": {
            "20230602_FSD_Site_UT0Q9_20230223_v5a_autolabel_freespace_11v": None,
            "20230602_FSD_Site_UT0Q9_20230224_v5a_autolabel_freespace_11v": None,
        },
        "UTHS6": {
            "20230602_FSD_Site_UTHS6_20230222_v5a_autolabel_freespace_11v": None,
        },
        "UT1R3": {
            "20230602_FSD_Site_UT1R3_20230222_v5a_autolabel_freespace_11v": None,
            "20230602_FSD_Site_UT1R3_20230223_v5a_autolabel_freespace_11v": None,
        },
    },
    "PARKING_QUALITY_CHECK_V1": {
        "UT1R3": {
            "20230602_UT1R3_20230301_D_v5a_autolabel_freespace_11v": None,
            "20230602_UT1R3_20230315_D_v5a_autolabel_freespace_11v": None,
            "20230602_UT1R3_20230317_D_v5a_autolabel_freespace_11v": None,
            "20230602_UT1R3_20230318_D_v5a_autolabel_freespace_11v": None,
            "20230602_UT1R3_20230320_D_v5a_autolabel_freespace_11v": None,
            "20230602_UT1R3_20230321_D_v5a_autolabel_freespace_11v": None,
        },
        "UT263": {
            "20230602_UT263_20230317_D_v5a_autolabel_freespace_11v": None,
            "20230602_UT263_20230318_D_v5a_autolabel_freespace_11v": None,
            "20230602_UT263_20230320_D_v5a_autolabel_freespace_11v": None,
        },
        "UT0F3": {
            "20230602_UT0F3_20230323_D_v5a_autolabel_freespace_11v": None,
            "20230602_UT0F3_20230326_D_v5a_autolabel_freespace_11v": None,
        },
    },
}
v3_9_3 = update_item_in_dict(v3_9_2, v3_9_3_update, only_use_update_version)

v4_0_update = {
    "FILTER_V4.7_BAD_DATA_filter_ele_vismask": {
        "BYD18": {
            "20230203_FSD_Site_BYD18_20221120_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD18_20221121_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD18_20221122_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD18_20221123_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD18_20221124_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            # "20230203_FSD_Site_BYD18_20221205_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,   验证集
            "20230203_FSD_Site_BYD18_20221206_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "byd_freespace_v1_FSD_Site_BYD18_20221201_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "byd_freespace_v1_FSD_Site_BYD18_20221202_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "byd_freespace_v2_FSD_Site_BYD18_20221201_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "byd_freespace_v2_FSD_Site_BYD18_20221202_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230316_FSD_Site_BYD18_20230119_v5a_autolabel_freespace_7v": None,
            "20230316_FSD_Site_BYD18_20230129_v5a_autolabel_freespace_7v": None,
            "20230316_FSD_Site_BYD18_20230116_v5a_autolabel_freespace_7v": None,
            "20230316_FSD_Site_BYD18_20230118_v5a_autolabel_freespace_7v": None,
        },
        "BYD19": {
            "20230203_FSD_Site_BYD19_20221107_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            # "20230203_FSD_Site_BYD19_20221108_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309":      验证集
            "20230203_FSD_Site_BYD19_20221110_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD19_20221112_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD19_20221113_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD19_20221114_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD19_20221115_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD19_20221116_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD19_20221117_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD19_20221118_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD19_20221119_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "byd_freespace_v1_FSD_Site_BYD19_20221210_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "byd_freespace_v2_FSD_Site_BYD19_20221210_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
        },
        "BYD72": {
            # "FSD_Site_BYD72_20221128_v5a_v2_autolabel_freespace_7vscreen_iou_0309:": None,                     备用
            # "byd_freespace_v2_FSD_Site_BYD72_20221128_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309:": None,    # 验证集
            "byd_freespace_v1_FSD_Site_BYD72_20221129_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "byd_freespace_v1_FSD_Site_BYD72_20221130_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
        },
    },
    "PARKING_QUALITY_CHECK_V1_filter_ele_vismask": {
        "UT126": {
            "20230315_UT126_20220927_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20221008_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20220914_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20221006_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20221009_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20220929_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20221004_D_v5a_autolabel_freespace_11v": None,
            "20230407_UT126_20230216_D_v5a_autolabel_freespace_11v": None,
            # "20230407_UT126_20230210_D_v5a_autolabel_freespace_11v": None,  # 验证集
            "20230407_UT126_20230213_D_v5a_autolabel_freespace_11v": None,
        },
        "UT1R3": {
            # "20230407_UT1R3_20230220_D_v5a_autolabel_freespace_11v": None,  # 备用
            "20230407_UT1R3_20230222_D_v5a_autolabel_freespace_11v": None,
            "20230407_UT1R3_20230228_D_v5a_autolabel_freespace_11v": None,
        },
        "UT0F3": {
            "20230407_UT0F3_20230223_D_v5a_autolabel_freespace_11v": None,
        },
        "UT0Q9": {
            "20230407_UT0Q9_20230215_D_v5a_autolabel_freespace_11v": None,
            "20230407_UT0Q9_20230130_D_v5a_autolabel_freespace_11v": None,
        },
        "LS830": {
            "20230407_LS830_20230302_D_v5a_autolabel_freespace_11v": None,
        },
    },
    "BYD_WIDE_QUALITY_CHECK_NEW_MODULE_filter_ele_vismask": {
        "BYD18": {
            "20230326_FSD_Site_BYD18_20230130_v5a_autolabel_freespace_7v": None,
            "20230326_FSD_Site_BYD18_20230201_v5a_autolabel_freespace_7v": None,
            "20230402_FSD_Site_BYD18_20230207_v5a_autolabel_freespace_7v": None,
            "20230402_FSD_Site_BYD18_20230208_v5a_autolabel_freespace_7v": None,
            # "20230402_FSD_Site_BYD18_20230209_v5a_autolabel_freespace_7v": None,  # 备用
            "20230402_FSD_Site_BYD18_20230210_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230328_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230329_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230330_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230331_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230403_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230404_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230328_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230329_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230401_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230403_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230404_v5a_autolabel_freespace_7v": None,
        },
        "BYD19": {
            # "20230410_FSD_Site_BYD19_20230211_v5a_autolabel_freespace_7v": None,  # 备用
            "20230410_FSD_Site_BYD19_20230213_v5a_autolabel_freespace_7v": None,
            "20230410_FSD_Site_BYD19_20230214_v5a_autolabel_freespace_7v": None,
        },
    },
}

# add temporal dataset
v4_0 = update_item_in_dict(v3_9_3, v4_0_update, only_use_update_version)

v4_1_0_update = {
    # 时序2帧连续真值
    "AUTO_LABEL_QUALITY_CHECK_ZHUXIAN": {
        "LS830": {
            "FSD_Site_LS830_20230216_sequential_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LS830_20230217_sequential_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LS830_20230218_sequential_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LS830_20230219_sequential_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LS830_20230220_sequential_drive_zhuxian_v5a_11v": None,
        },
        "LS912": {
            "FSD_Site_LS912_20230220_sequential_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LS912_20230328_sequential_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LS912_20230329_sequential_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LS912_20230331_sequential_drive_zhuxian_v5a_11v": None,
        },
        "LX165": {
            "FSD_Site_LX165_20230426_sequential_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LX165_20230427_sequential_drive_zhuxian_v5a_11v": None,
        },
    },
    "PARKING_QUALITY_CHECK_V1": {
        "UT263": {
            "UT263_20230307_D_sequential_drive_zhuxian_v5a_11v": None,
            "UT263_20230311_D_sequential_drive_zhuxian_v5a_11v": None,
            "UT263_20230314_D_sequential_drive_zhuxian_v5a_11v": None,
            "UT263_20230315_D_sequential_drive_zhuxian_v5a_11v": None,
            "UT263_20230316_D_sequential_drive_zhuxian_v5a_11v": None,
            "UT263_20230321_D_sequential_drive_zhuxian_v5a_11v": None,
            "UT263_20230510_D_sequential_drive_zhuxian_v5a_11v": None,
            "UT263_20230512_D_sequential_drive_zhuxian_v5a_11v": None,
        },
    },
}
v4_1_0_del = {
    "BYD_WIDE_QUALITY_CHECK_V1": {
        "BYD18": {
            "20230203_FSD_Site_BYD18_20221120_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD18_20221121_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD18_20221122_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD18_20221123_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD18_20221124_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD18_20221206_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "byd_freespace_v1_FSD_Site_BYD18_20221201_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "byd_freespace_v1_FSD_Site_BYD18_20221202_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "byd_freespace_v2_FSD_Site_BYD18_20221201_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "byd_freespace_v2_FSD_Site_BYD18_20221202_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230316_FSD_Site_BYD18_20230119_v5a_autolabel_freespace_7v": None,
            "20230316_FSD_Site_BYD18_20230129_v5a_autolabel_freespace_7v": None,
            "20230316_FSD_Site_BYD18_20230116_v5a_autolabel_freespace_7v": None,
            "20230316_FSD_Site_BYD18_20230118_v5a_autolabel_freespace_7v": None,
            "20230326_FSD_Site_BYD18_20230130_v5a_autolabel_freespace_7v": None,
            "20230326_FSD_Site_BYD18_20230201_v5a_autolabel_freespace_7v": None,
            "20230402_FSD_Site_BYD18_20230207_v5a_autolabel_freespace_7v": None,
            "20230402_FSD_Site_BYD18_20230208_v5a_autolabel_freespace_7v": None,
            "20230402_FSD_Site_BYD18_20230210_v5a_autolabel_freespace_7v": None,
        },
        "BYD19": {
            "20230203_FSD_Site_BYD19_20221107_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD19_20221110_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD19_20221112_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD19_20221113_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD19_20221114_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD19_20221115_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD19_20221116_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD19_20221117_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD19_20221118_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230203_FSD_Site_BYD19_20221119_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "byd_freespace_v1_FSD_Site_BYD19_20221210_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "byd_freespace_v2_FSD_Site_BYD19_20221210_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230410_FSD_Site_BYD19_20230213_v5a_autolabel_freespace_7v": None,
            "20230410_FSD_Site_BYD19_20230214_v5a_autolabel_freespace_7v": None,
        },
        "BYD72": {
            "byd_freespace_v1_FSD_Site_BYD72_20221129_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "byd_freespace_v1_FSD_Site_BYD72_20221130_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
        },
    },
    "PARKING_QUALITY_CHECK_V1": {
        "UT126": {
            "20230315_UT126_20220927_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20221008_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20220914_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20221006_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20221009_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20220929_D_v5a_autolabel_freespace_11v": None,
            "20230315_UT126_20221004_D_v5a_autolabel_freespace_11v": None,
            "20230407_UT126_20230216_D_v5a_autolabel_freespace_11v": None,
            "20230407_UT126_20230213_D_v5a_autolabel_freespace_11v": None,
        },
        "UT1R3": {
            "20230407_UT1R3_20230222_D_v5a_autolabel_freespace_11v": None,
            "20230407_UT1R3_20230228_D_v5a_autolabel_freespace_11v": None,
        },
        "UT0F3": {
            "20230407_UT0F3_20230223_D_v5a_autolabel_freespace_11v": None,
        },
        "UT0Q9": {
            "20230407_UT0Q9_20230215_D_v5a_autolabel_freespace_11v": None,
            "20230407_UT0Q9_20230130_D_v5a_autolabel_freespace_11v": None,
        },
        "LS830": {
            "20230407_LS830_20230302_D_v5a_autolabel_freespace_11v": None,
        },
    },
    "BYD_WIDE_QUALITY_CHECK_NEW_MODULE": {
        "BYD18": {
            "20230419_FSD_Site_BYD18_20230328_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230329_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230330_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230403_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230404_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230328_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230329_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230401_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230403_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230404_v5a_autolabel_freespace_7v": None,
        },
    },
}
v4_1_0 = del_item_in_dict(v4_0, v4_1_0_del)
v4_1_0 = update_item_in_dict(v4_1_0, v4_1_0_update, only_use_update_version)
v4_1_0_del_no_fisheye = {
    "ZHANGAIWU_WIDE": {
        "UTHS6": {
            "FSD_Site_UTHS6_20220921_freespace_wide_artificial_train_resampling": None,
        }
    },
    "BYD_WIDE_QUALITY_CHECK_V1": {
        "BYD18": {
            "20230203_FSD_Site_BYD18_20221205_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
            "20230402_FSD_Site_BYD18_20230209_v5a_autolabel_freespace_7v": None,
        },
        "BYD19": {
            "20230410_FSD_Site_BYD19_20230211_v5a_autolabel_freespace_7v": None,
        },
        "BYD72": {
            "byd_freespace_v2_FSD_Site_BYD72_20221128_v5a_autolabel_freespace_drop_stillness_7v_after_reject": None,
        },
    },
    "FILTER_V4.7_BAD_DATA_filter_ele_vismask": {
        "BYD18": {
            "20230203_FSD_Site_BYD18_20221120_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD18_20221121_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD18_20221122_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD18_20221123_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD18_20221124_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD18_20221206_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "byd_freespace_v1_FSD_Site_BYD18_20221201_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "byd_freespace_v1_FSD_Site_BYD18_20221202_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "byd_freespace_v2_FSD_Site_BYD18_20221201_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "byd_freespace_v2_FSD_Site_BYD18_20221202_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230316_FSD_Site_BYD18_20230119_v5a_autolabel_freespace_7v": None,
            "20230316_FSD_Site_BYD18_20230129_v5a_autolabel_freespace_7v": None,
            "20230316_FSD_Site_BYD18_20230116_v5a_autolabel_freespace_7v": None,
            "20230316_FSD_Site_BYD18_20230118_v5a_autolabel_freespace_7v": None,
        },
        "BYD19": {
            "20230203_FSD_Site_BYD19_20221107_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD19_20221110_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD19_20221112_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD19_20221113_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD19_20221114_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD19_20221115_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD19_20221116_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD19_20221117_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD19_20221118_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD19_20221119_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "byd_freespace_v1_FSD_Site_BYD19_20221210_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "byd_freespace_v2_FSD_Site_BYD19_20221210_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
        },
        "BYD72": {
            "byd_freespace_v1_FSD_Site_BYD72_20221129_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "byd_freespace_v1_FSD_Site_BYD72_20221130_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
        },
    },
    "BYD_WIDE_QUALITY_CHECK_NEW_MODULE_filter_ele_vismask": {
        "BYD18": {
            "20230326_FSD_Site_BYD18_20230130_v5a_autolabel_freespace_7v": None,
            "20230326_FSD_Site_BYD18_20230201_v5a_autolabel_freespace_7v": None,
            "20230402_FSD_Site_BYD18_20230207_v5a_autolabel_freespace_7v": None,
            "20230402_FSD_Site_BYD18_20230208_v5a_autolabel_freespace_7v": None,
            "20230402_FSD_Site_BYD18_20230210_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230328_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230329_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230330_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230331_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230403_v5a_autolabel_freespace_7v": None,
            "20230419_FSD_Site_BYD18_20230404_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230328_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230329_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230401_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230403_v5a_autolabel_freespace_7v": None,
            "20230420_FSD_Site_BYD18_20230404_v5a_autolabel_freespace_7v": None,
        },
        "BYD19": {
            "20230410_FSD_Site_BYD19_20230213_v5a_autolabel_freespace_7v": None,
            "20230410_FSD_Site_BYD19_20230214_v5a_autolabel_freespace_7v": None,
        },
    },
    "BYD_WIDE_QUALITY_CHECK_NEW_MODULE": {
        "BYD18": {
            "20230419_FSD_Site_BYD18_20230406_v5a_autolabel_freespace_7v": None,
            "20230504_FSD_Site_BYD18_20230406_v5a_autolabel_freespace_7v": None,
            "20230504_FSD_Site_BYD18_20230412_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD18_20230414_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD18_20230415_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD18_20230416_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD18_20230417_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD18_20230418_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD18_20230420_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD18_20230422_v5a_autolabel_freespace_7v": None,
        },
        "BYD19": {
            "20230425_FSD_Site_BYD19_20230329_v5a_autolabel_freespace_7v": None,
            "20230425_FSD_Site_BYD19_20230331_v5a_autolabel_freespace_7v": None,
            "20230425_FSD_Site_BYD19_20230404_v5a_autolabel_freespace_7v": None,
            "20230425_FSD_Site_BYD19_20230406_v5a_autolabel_freespace_7v": None,
            "20230425_FSD_Site_BYD19_20230407_v5a_autolabel_freespace_7v": None,
            "20230504_FSD_Site_BYD19_20230408_v5a_autolabel_freespace_7v": None,
            "20230504_FSD_Site_BYD19_20230410_v5a_autolabel_freespace_7v": None,
            "20230504_FSD_Site_BYD19_20230411_v5a_autolabel_freespace_7v": None,
            "20230504_FSD_Site_BYD19_20230412_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230421_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230422_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230423_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230424_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230508_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230509_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230510_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230511_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230512_v5a_autolabel_freespace_7v": None,
        },
    },
}
v4_1_0_fisheye = del_item_in_dict(v4_1_0, v4_1_0_del_no_fisheye)

v5_0_0_temporal = {
    "BJ": {
        "UT126_densse_freespace": {
            "FSD_Site_UT126_20220909_v5a_11v_sequtential": None,
            "FSD_Site_UT126_20220812_v5a_11v_sequtential": None,
        },
        "UT263_densse_freespace": {
            "FSD_Site_UT263_20220924_v5a_11v_sequtential": None,
        },
    },
    "STATIC_ANNO_WIDE_TOTAL": {
        "UT126": {
            "FSD_Site_UT126_20220925_second_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221022_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220929_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220923_second_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221011_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220927_second_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221017_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221006_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221008_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220926_second_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221018_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221028_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221009_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220922_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221015_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221020_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221030_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221029_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220916_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221013_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220921_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221019_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221021_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220924_second_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221026_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221010_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220930_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221024_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221023_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221016_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221025_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221014_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221027_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221012_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220920_second_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220928_second_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220823_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220909_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220808_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220831_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220901_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220908_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220829_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220824_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220907_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220903_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220809_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220814_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220820_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220825_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220830_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220915_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220906_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220902_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220905_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220912_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220818_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220812_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220827_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220913_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220914_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220815_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220817_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220813_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220807_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220925_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220923_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220926_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220917_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220920_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220927_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220924_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220928_v5a_wide_11v_sequtential": None,
        },
        "UTHS6": {
            "FSD_Site_UTHS6_20221029_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221108_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221024_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221030_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221028_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221101_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221023_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221105_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221022_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221102_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221107_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221026_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221021_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221104_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221019_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221106_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221025_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221031_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221018_v5a_wide_11v_sequtential": None,
        },
        "UT0Q9": {
            "FSD_Site_UT0Q9_20221016_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT0Q9_20221017_v5a_wide_11v_sequtential": None,
        },
        "UT263": {
            "FSD_Site_UT263_20221012_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221026_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221010_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220930_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221007_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221028_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221005_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220929_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221009_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221031_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221006_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221023_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221030_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221024_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221011_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221027_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221013_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221008_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221025_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221016_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221029_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221015_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220831_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220922_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220924_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220926_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220830_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220928_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220925_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220906_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220923_v5a_wide_11v_sequtential": None,
        },
        "UT3J5": {
            "FSD_Site_UT3J5_20221022_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT3J5_20221023_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT3J5_20221021_v5a_wide_11v_sequtential": None,
        },
        "UT1R3": {
            "FSD_Site_UT1R3_20221030_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT1R3_20221028_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT1R3_20221107_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT1R3_20221108_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT1R3_20221029_v5a_wide_11v_sequtential": None,
        },
    },
    "FILTER_V4.7_BAD_DATA_filter_ele_vismask": {
        "BYD18": {
            "20230203_FSD_Site_BYD18_20221120_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "20230203_FSD_Site_BYD18_20221121_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "20230203_FSD_Site_BYD18_20221122_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "20230203_FSD_Site_BYD18_20221123_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "20230203_FSD_Site_BYD18_20221124_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            # "20230203_FSD_Site_BYD18_20221205_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,   验证集
            "20230203_FSD_Site_BYD18_20221206_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "byd_freespace_v1_FSD_Site_BYD18_20221201_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "byd_freespace_v1_FSD_Site_BYD18_20221202_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "byd_freespace_v2_FSD_Site_BYD18_20221201_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "byd_freespace_v2_FSD_Site_BYD18_20221202_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "20230316_FSD_Site_BYD18_20230119_v5a_autolabel_freespace_7v_total_gt": None,
            "20230316_FSD_Site_BYD18_20230129_v5a_autolabel_freespace_7v_total_gt": None,
            "20230316_FSD_Site_BYD18_20230116_v5a_autolabel_freespace_7v_total_gt": None,
            "20230316_FSD_Site_BYD18_20230118_v5a_autolabel_freespace_7v_total_gt": None,
        },
        "BYD19": {
            "20230203_FSD_Site_BYD19_20221107_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            # "20230203_FSD_Site_BYD19_20221108_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309":      验证集
            "20230203_FSD_Site_BYD19_20221110_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "20230203_FSD_Site_BYD19_20221112_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "20230203_FSD_Site_BYD19_20221113_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "20230203_FSD_Site_BYD19_20221114_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "20230203_FSD_Site_BYD19_20221115_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "20230203_FSD_Site_BYD19_20221116_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "20230203_FSD_Site_BYD19_20221117_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "20230203_FSD_Site_BYD19_20221118_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "20230203_FSD_Site_BYD19_20221119_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "byd_freespace_v1_FSD_Site_BYD19_20221210_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "byd_freespace_v2_FSD_Site_BYD19_20221210_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
        },
        "BYD72": {
            # "FSD_Site_BYD72_20221128_v5a_v2_autolabel_freespace_7vscreen_iou_0309:_total_gt": None,                     备用
            # "byd_freespace_v2_FSD_Site_BYD72_20221128_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309:_total_gt": None,    # 验证集
            "byd_freespace_v1_FSD_Site_BYD72_20221129_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "byd_freespace_v1_FSD_Site_BYD72_20221130_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
        },
    },
    "PARKING_QUALITY_CHECK_V1_filter_ele_vismask": {
        "UT126": {
            "20230315_UT126_20220927_D_v5a_autolabel_freespace_11v_total_gt": None,
            "20230315_UT126_20221008_D_v5a_autolabel_freespace_11v_total_gt": None,
            "20230315_UT126_20220914_D_v5a_autolabel_freespace_11v_total_gt": None,
            "20230315_UT126_20221006_D_v5a_autolabel_freespace_11v_total_gt": None,
            "20230315_UT126_20221009_D_v5a_autolabel_freespace_11v_total_gt": None,
            "20230315_UT126_20220929_D_v5a_autolabel_freespace_11v_total_gt": None,
            "20230315_UT126_20221004_D_v5a_autolabel_freespace_11v_total_gt": None,
            "20230407_UT126_20230216_D_v5a_autolabel_freespace_11v_total_gt": None,
            # "20230407_UT126_20230210_D_v5a_autolabel_freespace_11v_total_gt": None,  # 验证集
            "20230407_UT126_20230213_D_v5a_autolabel_freespace_11v_total_gt": None,
        },
        "UT1R3": {
            # "20230407_UT1R3_20230220_D_v5a_autolabel_freespace_11v_total_gt": None,  # 备用
            "20230407_UT1R3_20230222_D_v5a_autolabel_freespace_11v_total_gt": None,
            "20230407_UT1R3_20230228_D_v5a_autolabel_freespace_11v_total_gt": None,
        },
        "UT0F3": {
            "20230407_UT0F3_20230223_D_v5a_autolabel_freespace_11v_total_gt": None,
        },
        "UT0Q9": {
            "20230407_UT0Q9_20230215_D_v5a_autolabel_freespace_11v_total_gt": None,
            "20230407_UT0Q9_20230130_D_v5a_autolabel_freespace_11v_total_gt": None,
        },
        "LS830": {
            "20230407_LS830_20230302_D_v5a_autolabel_freespace_11v_total_gt": None,
        },
    },
    "BYD_WIDE_QUALITY_CHECK_NEW_MODULE_filter_ele_vismask": {
        "BYD18": {
            "20230326_FSD_Site_BYD18_20230130_v5a_autolabel_freespace_7v_total_gt": None,
            "20230326_FSD_Site_BYD18_20230201_v5a_autolabel_freespace_7v_total_gt": None,
            "20230402_FSD_Site_BYD18_20230207_v5a_autolabel_freespace_7v_total_gt": None,
            "20230402_FSD_Site_BYD18_20230208_v5a_autolabel_freespace_7v_total_gt": None,
            # "20230402_FSD_Site_BYD18_20230209_v5a_autolabel_freespace_7v_total_gt": None,  # 备用
            "20230402_FSD_Site_BYD18_20230210_v5a_autolabel_freespace_7v_total_gt": None,
            "20230419_FSD_Site_BYD18_20230328_v5a_autolabel_freespace_7v_total_gt": None,
            "20230419_FSD_Site_BYD18_20230329_v5a_autolabel_freespace_7v_total_gt": None,
            "20230419_FSD_Site_BYD18_20230330_v5a_autolabel_freespace_7v_total_gt": None,
            "20230419_FSD_Site_BYD18_20230331_v5a_autolabel_freespace_7v_total_gt": None,
            "20230419_FSD_Site_BYD18_20230403_v5a_autolabel_freespace_7v_total_gt": None,
            "20230419_FSD_Site_BYD18_20230404_v5a_autolabel_freespace_7v_total_gt": None,
            "20230420_FSD_Site_BYD18_20230328_v5a_autolabel_freespace_7v_total_gt": None,
            "20230420_FSD_Site_BYD18_20230329_v5a_autolabel_freespace_7v_total_gt": None,
            "20230420_FSD_Site_BYD18_20230401_v5a_autolabel_freespace_7v_total_gt": None,
            "20230420_FSD_Site_BYD18_20230403_v5a_autolabel_freespace_7v_total_gt": None,
            "20230420_FSD_Site_BYD18_20230404_v5a_autolabel_freespace_7v_total_gt": None,
        },
        "BYD19": {
            # "20230410_FSD_Site_BYD19_20230211_v5a_autolabel_freespace_7v_total_gt": None,  # 备用
            "20230410_FSD_Site_BYD19_20230213_v5a_autolabel_freespace_7v_total_gt": None,
            "20230410_FSD_Site_BYD19_20230214_v5a_autolabel_freespace_7v_total_gt": None,
        },
    },
}

# 去除备用的测试集
v4_1_1_del = {
    "BYD_WIDE_QUALITY_CHECK_NEW_MODULE": {
        "BYD19": {
            "20230425_FSD_Site_BYD19_20230329_v5a_autolabel_freespace_7v": None,
            "20230425_FSD_Site_BYD19_20230331_v5a_autolabel_freespace_7v": None,
            "20230425_FSD_Site_BYD19_20230404_v5a_autolabel_freespace_7v": None,
            "20230425_FSD_Site_BYD19_20230406_v5a_autolabel_freespace_7v": None,
            "20230425_FSD_Site_BYD19_20230407_v5a_autolabel_freespace_7v": None,
            "20230504_FSD_Site_BYD19_20230408_v5a_autolabel_freespace_7v": None,
            "20230504_FSD_Site_BYD19_20230410_v5a_autolabel_freespace_7v": None,
            "20230504_FSD_Site_BYD19_20230411_v5a_autolabel_freespace_7v": None,
            "20230504_FSD_Site_BYD19_20230412_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230421_v5a_autolabel_freespace_7v": None,
            "20230602_FSD_Site_BYD19_20230422_v5a_autolabel_freespace_7v": None,
        },
    },
    "AUTO_LABEL_QUALITY_CHECK_ZHUXIAN": {
        "LS912": {
            "FSD_Site_LS912_20230220_sequential_drive_zhuxian_v5a_11v": None,
        },
    },
}
v4_1_1_fisheye_del = {
    "AUTO_LABEL_QUALITY_CHECK_ZHUXIAN": {
        "LS912": {
            "FSD_Site_LS912_20230220_sequential_drive_zhuxian_v5a_11v": None,
        },
    },
}
v4_1_1 = del_item_in_dict(v4_1_0, v4_1_1_del)
v4_1_1_fisheye = del_item_in_dict(v4_1_0_fisheye, v4_1_1_fisheye_del)

v5_1_0_update = {
    "AUTO_LABEL_QUALITY_CHECK_ZHUXIAN": {
        "LS912": {
            "FSD_Site_LS912_20230403_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LS912_20230404_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LS912_20230327_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LS912_20230330_drive_zhuxian_v5a_11v": None,
        },
        "LX165": {
            "FSD_Site_LX165_20230313_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LX165_20230302_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LX165_20230309_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LX165_20230310_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LX165_20230311_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LX165_20230321_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LX165_20230322_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LX165_20230323_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LX165_20230406_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LX165_20230407_drive_zhuxian_v5a_11v": None,
            "FSD_Site_LX165_20230320_drive_zhuxian_v5a_11v": None,
        },
    },
    "PARKING_QUALITY_CHECK_V1_filter_ele_vismask": {
        "LS830": {
            "LS830_20230407_D_parking_zhuxian_v5a_11v": None,
            "LS830_20230327_D_parking_zhuxian_v5a_11v": None,
            # "LS830_20230411_D_parking_zhuxian_v5a_11v": None, 测试集
            "LS830_20230304_D_parking_zhuxian_v5a_11v": None,
            "LS830_20230321_D_parking_zhuxian_v5a_11v": None,
            "LS830_20230401_D_parking_zhuxian_v5a_11v": None,
            "LS830_20230330_D_parking_zhuxian_v5a_11v": None,
            "LS830_20230331_D_parking_zhuxian_v5a_11v": None,
            "LS830_20230410_D_parking_zhuxian_v5a_11v": None,
            "LS830_20230412_D_parking_zhuxian_v5a_11v": None,
            "LS830_20230413_D_parking_zhuxian_v5a_11v": None,
            "LS830_20230414_D_parking_zhuxian_v5a_11v": None,
        },
        "UT126": {
            "UT126_20230330_D_parking_zhuxian_v5a_11v": None,
            "UT126_20230328_D_parking_zhuxian_v5a_11v": None,
        },
        "UT263": {
            "UT263_20230301_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230303_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230304_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230306_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230325_D_parking_zhuxian_v5a_11v": None,
            # "UT263_20230327_D_parking_zhuxian_v5a_11v": None, 测试集
            "UT263_20230421_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230422_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230324_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230423_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230505_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230506_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230427_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230504_D_parking_zhuxian_v5a_11v": None,
        },
        "UT370": {
            "UT370_20230520_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230521_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230522_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230517_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230518_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230519_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230531_D_parking_zhuxian_v5a_11v": None,
            # "UT370_20230601_D_parking_zhuxian_v5a_11v": None, 测试集
            "UT370_20230524_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230527_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230523_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230528_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230530_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230526_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230529_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230602_D_parking_zhuxian_v5a_11v": None,
        },
    },
}
v5_1_0 = update_item_in_dict(v4_1_1, v5_1_0_update, only_use_update_version)
v5_1_0_temporal_update = {
    "AUTO_LABEL_QUALITY_CHECK_ZHUXIAN": {
        "LS912": {
            "FSD_Site_LS912_20230403_drive_zhuxian_v5a_sequential_11v": None,
            "FSD_Site_LS912_20230404_drive_zhuxian_v5a_sequential_11v": None,
            "FSD_Site_LS912_20230327_drive_zhuxian_v5a_sequential_11v": None,
            "FSD_Site_LS912_20230330_drive_zhuxian_v5a_sequential_11v": None,
        },
        "LX165": {
            "FSD_Site_LX165_20230313_drive_zhuxian_v5a_sequential_11v": None,
            "FSD_Site_LX165_20230302_drive_zhuxian_v5a_sequential_11v": None,
            "FSD_Site_LX165_20230309_drive_zhuxian_v5a_sequential_11v": None,
            "FSD_Site_LX165_20230310_drive_zhuxian_v5a_sequential_11v": None,
            "FSD_Site_LX165_20230311_drive_zhuxian_v5a_sequential_11v": None,
            "FSD_Site_LX165_20230321_drive_zhuxian_v5a_sequential_11v": None,
            "FSD_Site_LX165_20230322_drive_zhuxian_v5a_sequential_11v": None,
            "FSD_Site_LX165_20230323_drive_zhuxian_v5a_sequential_11v": None,
            "FSD_Site_LX165_20230406_drive_zhuxian_v5a_sequential_11v": None,
            "FSD_Site_LX165_20230407_drive_zhuxian_v5a_sequential_11v": None,
            "FSD_Site_LX165_20230320_drive_zhuxian_v5a_sequential_11v": None,
        },
    },
    "PARKING_QUALITY_CHECK_V1_filter_ele_vismask": {
        "LS830": {
            "LS830_20230407_D_parking_zhuxian_v5a_sequential_11v": None,
            "LS830_20230327_D_parking_zhuxian_v5a_sequential_11v": None,
            # "LS830_20230411_D_parking_zhuxian_v5a_sequential_11v": None, 测试集
            "LS830_20230304_D_parking_zhuxian_v5a_sequential_11v": None,
            "LS830_20230321_D_parking_zhuxian_v5a_sequential_11v": None,
            "LS830_20230401_D_parking_zhuxian_v5a_sequential_11v": None,
            "LS830_20230330_D_parking_zhuxian_v5a_sequential_11v": None,
            "LS830_20230331_D_parking_zhuxian_v5a_sequential_11v": None,
            "LS830_20230410_D_parking_zhuxian_v5a_sequential_11v": None,
            "LS830_20230412_D_parking_zhuxian_v5a_sequential_11v": None,
            "LS830_20230413_D_parking_zhuxian_v5a_sequential_11v": None,
            "LS830_20230414_D_parking_zhuxian_v5a_sequential_11v": None,
        },
        "UT126": {
            "UT126_20230330_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT126_20230328_D_parking_zhuxian_v5a_sequential_11v": None,
        },
        "UT263": {
            "UT263_20230301_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT263_20230303_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT263_20230304_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT263_20230306_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT263_20230325_D_parking_zhuxian_v5a_sequential_11v": None,
            # "UT263_20230327_D_parking_zhuxian_v5a_sequential_11v": None, 测试集
            "UT263_20230421_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT263_20230422_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT263_20230324_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT263_20230423_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT263_20230505_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT263_20230506_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT263_20230427_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT263_20230504_D_parking_zhuxian_v5a_sequential_11v": None,
        },
        "UT370": {
            "UT370_20230520_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT370_20230521_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT370_20230522_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT370_20230517_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT370_20230518_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT370_20230519_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT370_20230531_D_parking_zhuxian_v5a_sequential_11v": None,
            # "UT370_20230601_D_parking_zhuxian_v5a_sequential_11v": None, 测试集
            "UT370_20230524_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT370_20230527_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT370_20230523_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT370_20230528_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT370_20230530_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT370_20230526_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT370_20230529_D_parking_zhuxian_v5a_sequential_11v": None,
            "UT370_20230602_D_parking_zhuxian_v5a_sequential_11v": None,
        },
    },
}

v5_1_0_temporal = update_item_in_dict(
    v5_0_0_temporal, v5_1_0_temporal_update, only_use_update_version
)

v5_1_0_fisheye = update_item_in_dict(
    v4_1_1_fisheye, v5_1_0_update, only_use_update_version
)

v6_0_0_temporal = {
    "STATIC_ANNO_WIDE_TOTAL": {
        "UT126": {
            "FSD_Site_UT126_20220925_second_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221022_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220929_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220923_second_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221011_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220927_second_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221017_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221006_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221008_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220926_second_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221018_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221028_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221009_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220922_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221015_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221020_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221030_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221029_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220916_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221013_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220921_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221019_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221021_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220924_second_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221026_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221010_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220930_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221024_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221023_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221016_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221025_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221014_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221027_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20221012_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220920_second_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220928_second_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220823_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220909_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220808_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220831_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220901_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220908_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220829_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220824_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220907_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220903_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220809_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220814_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220820_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220825_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220830_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220915_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220906_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220902_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220905_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220912_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220818_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220812_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220827_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220913_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220914_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220815_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220817_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220813_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220807_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220925_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220923_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220926_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220917_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220920_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220927_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220924_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT126_20220928_v5a_wide_11v_sequtential": None,
        },
        "UTHS6": {
            "FSD_Site_UTHS6_20221029_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221108_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221024_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221030_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221028_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221101_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221023_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221105_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221022_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221102_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221107_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221026_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221021_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221104_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221019_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221106_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221025_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221031_v5a_wide_11v_sequtential": None,
            "FSD_Site_UTHS6_20221018_v5a_wide_11v_sequtential": None,
        },
        "UT0Q9": {
            "FSD_Site_UT0Q9_20221016_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT0Q9_20221017_v5a_wide_11v_sequtential": None,
        },
        "UT263": {
            "FSD_Site_UT263_20221012_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221026_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221010_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220930_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221007_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221028_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221005_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220929_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221009_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221031_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221006_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221023_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221030_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221024_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221011_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221027_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221013_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221008_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221025_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221016_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221029_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20221015_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220831_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220922_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220924_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220926_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220830_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220928_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220925_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220906_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT263_20220923_v5a_wide_11v_sequtential": None,
        },
        "UT3J5": {
            "FSD_Site_UT3J5_20221022_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT3J5_20221023_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT3J5_20221021_v5a_wide_11v_sequtential": None,
        },
        "UT1R3": {
            "FSD_Site_UT1R3_20221030_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT1R3_20221028_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT1R3_20221107_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT1R3_20221108_v5a_wide_11v_sequtential": None,
            "FSD_Site_UT1R3_20221029_v5a_wide_11v_sequtential": None,
        },
    },
    "AUTO_LABEL_QUALITY_CHECK_ZHUXIAN": {
        "LS830": {
            "FSD_Site_LS830_20230216_sequential_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LS830_20230217_sequential_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LS830_20230218_sequential_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LS830_20230219_sequential_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LS830_20230220_sequential_drive_zhuxian_v5a_11v_sequential_4frame": None,
        },
        "LS912": {
            "FSD_Site_LS912_20230327_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LS912_20230328_sequential_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LS912_20230329_sequential_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LS912_20230330_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LS912_20230331_sequential_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LS912_20230403_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LS912_20230404_drive_zhuxian_v5a_11v_sequential_4frame": None,
        },
        "LX165": {
            "FSD_Site_LX165_20230302_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LX165_20230309_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LX165_20230310_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LX165_20230311_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LX165_20230313_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LX165_20230320_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LX165_20230321_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LX165_20230322_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LX165_20230323_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LX165_20230406_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LX165_20230407_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LX165_20230426_sequential_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "FSD_Site_LX165_20230427_sequential_drive_zhuxian_v5a_11v_sequential_4frame": None,
        },
        "UT0Q9": {
            "20230602_FSD_Site_UT0Q9_20230223_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230602_FSD_Site_UT0Q9_20230224_v5a_autolabel_freespace_11v_sequential_4frame": None,
        },
        "UT1R3": {
            "20230602_FSD_Site_UT1R3_20230222_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230602_FSD_Site_UT1R3_20230223_v5a_autolabel_freespace_11v_sequential_4frame": None,
        },
        "UTHS6": {
            "20230602_FSD_Site_UTHS6_20230222_v5a_autolabel_freespace_11v_sequential_4frame": None,
        },
    },
    "BYD_WIDE_QUALITY_CHECK_NEW_MODULE": {
        "BYD18": {
            "20230419_FSD_Site_BYD18_20230406_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230504_FSD_Site_BYD18_20230406_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230504_FSD_Site_BYD18_20230412_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230602_FSD_Site_BYD18_20230414_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230602_FSD_Site_BYD18_20230415_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230602_FSD_Site_BYD18_20230416_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230602_FSD_Site_BYD18_20230417_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230602_FSD_Site_BYD18_20230418_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230602_FSD_Site_BYD18_20230420_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230602_FSD_Site_BYD18_20230422_v5a_autolabel_freespace_7v_sequential_4frame": None,
        },
        "BYD19": {
            "20230602_FSD_Site_BYD19_20230423_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230602_FSD_Site_BYD19_20230424_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230602_FSD_Site_BYD19_20230508_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230602_FSD_Site_BYD19_20230509_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230602_FSD_Site_BYD19_20230510_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230602_FSD_Site_BYD19_20230511_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230602_FSD_Site_BYD19_20230512_v5a_autolabel_freespace_7v_sequential_4frame": None,
        },
    },
    "BYD_WIDE_QUALITY_CHECK_NEW_MODULE_filter_ele_vismask": {
        "BYD18": {
            "20230326_FSD_Site_BYD18_20230130_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230326_FSD_Site_BYD18_20230201_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230402_FSD_Site_BYD18_20230207_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230402_FSD_Site_BYD18_20230208_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230402_FSD_Site_BYD18_20230210_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230419_FSD_Site_BYD18_20230328_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230419_FSD_Site_BYD18_20230329_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230419_FSD_Site_BYD18_20230330_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230419_FSD_Site_BYD18_20230331_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230419_FSD_Site_BYD18_20230403_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230419_FSD_Site_BYD18_20230404_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230420_FSD_Site_BYD18_20230328_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230420_FSD_Site_BYD18_20230329_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230420_FSD_Site_BYD18_20230401_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230420_FSD_Site_BYD18_20230403_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230420_FSD_Site_BYD18_20230404_v5a_autolabel_freespace_7v_sequential_4frame": None,
        },
        "BYD19": {
            "20230410_FSD_Site_BYD19_20230213_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230410_FSD_Site_BYD19_20230214_v5a_autolabel_freespace_7v_sequential_4frame": None,
        },
    },
    "BYD_WIDE_QUALITY_CHECK_V1": {
        "BYD18": {
            "20230203_FSD_Site_BYD18_20221205_v5a_autolabel_freespace_drop_stillness_7v_after_reject_sequential_4frame": None,
            "20230402_FSD_Site_BYD18_20230209_v5a_autolabel_freespace_7v_sequential_4frame": None,
        },
        "BYD19": {
            "20230410_FSD_Site_BYD19_20230211_v5a_autolabel_freespace_7v_sequential_4frame": None,
        },
        "BYD72": {
            "byd_freespace_v2_FSD_Site_BYD72_20221128_v5a_autolabel_freespace_drop_stillness_7v_after_reject_sequential_4frame": None,
        },
    },
    "FILTER_V4.7_BAD_DATA_filter_ele_vismask": {
        "BYD18": {
            "20230203_FSD_Site_BYD18_20221120_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "20230203_FSD_Site_BYD18_20221121_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "20230203_FSD_Site_BYD18_20221122_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "20230203_FSD_Site_BYD18_20221123_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "20230203_FSD_Site_BYD18_20221124_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "20230203_FSD_Site_BYD18_20221206_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "20230316_FSD_Site_BYD18_20230116_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230316_FSD_Site_BYD18_20230118_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230316_FSD_Site_BYD18_20230119_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "20230316_FSD_Site_BYD18_20230129_v5a_autolabel_freespace_7v_sequential_4frame": None,
            "byd_freespace_v1_FSD_Site_BYD18_20221201_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "byd_freespace_v1_FSD_Site_BYD18_20221202_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "byd_freespace_v2_FSD_Site_BYD18_20221201_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "byd_freespace_v2_FSD_Site_BYD18_20221202_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
        },
        "BYD19": {
            "20230203_FSD_Site_BYD19_20221107_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "20230203_FSD_Site_BYD19_20221110_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "20230203_FSD_Site_BYD19_20221112_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "20230203_FSD_Site_BYD19_20221113_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "20230203_FSD_Site_BYD19_20221114_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "20230203_FSD_Site_BYD19_20221115_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "20230203_FSD_Site_BYD19_20221116_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "20230203_FSD_Site_BYD19_20221117_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "20230203_FSD_Site_BYD19_20221118_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "20230203_FSD_Site_BYD19_20221119_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "byd_freespace_v1_FSD_Site_BYD19_20221210_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
            "byd_freespace_v2_FSD_Site_BYD19_20221210_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_sequential_4frame": None,
        },
    },
    "PARKING_QUALITY_CHECK_V1": {
        "LS830": {
            "20230427_LS830_20230303_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230427_LS830_20230306_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230427_LS830_20230307_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230427_LS830_20230322_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230427_LS830_20230323_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230427_LS830_20230324_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230427_LS830_20230325_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230427_LS830_20230328_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230427_LS830_20230329_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
        },
        "UT0F3": {
            "20230423_UT0F3_20230316_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230423_UT0F3_20230317_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230423_UT0F3_20230320_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230602_UT0F3_20230323_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230602_UT0F3_20230326_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
        },
        "UT126": {
            "20230315_UT126_20220915_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230315_UT126_20221010_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230407_UT126_20230210_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230414_UT126_20230211_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230414_UT126_20230212_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230414_UT126_20230214_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
        },
        "UT1R3": {
            "20230407_UT1R3_20230220_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230414_UT1R3_20230323_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230602_UT1R3_20230301_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230602_UT1R3_20230315_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230602_UT1R3_20230317_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230602_UT1R3_20230318_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230602_UT1R3_20230320_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230602_UT1R3_20230321_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
        },
        "UT263": {
            "20230414_UT263_20230214_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230602_UT263_20230317_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230602_UT263_20230318_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230602_UT263_20230320_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "UT263_20230307_D_sequential_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230311_D_sequential_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230314_D_sequential_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230315_D_sequential_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230316_D_sequential_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230321_D_sequential_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230510_D_sequential_drive_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230512_D_sequential_drive_zhuxian_v5a_11v_sequential_4frame": None,
        },
        "UT3J5": {
            "20230414_UT3J5_20230210_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230414_UT3J5_20230214_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230414_UT3J5_20230220_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230427_UT3J5_20230313_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230427_UT3J5_20230314_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230427_UT3J5_20230315_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230427_UT3J5_20230316_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230427_UT3J5_20230318_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
        },
        "UTHS6": {
            "20230414_UTHS6_20230228_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
        },
    },
    "PARKING_QUALITY_CHECK_V1_filter_ele_vismask": {
        "LS830": {
            "20230407_LS830_20230302_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "LS830_20230304_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "LS830_20230321_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "LS830_20230327_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "LS830_20230330_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "LS830_20230331_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "LS830_20230401_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "LS830_20230407_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "LS830_20230410_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "LS830_20230412_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "LS830_20230413_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "LS830_20230414_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
        },
        "UT0F3": {
            "20230407_UT0F3_20230223_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
        },
        "UT0Q9": {
            "20230407_UT0Q9_20230130_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230407_UT0Q9_20230215_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
        },
        "UT126": {
            "20230315_UT126_20220914_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230315_UT126_20220927_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230315_UT126_20220929_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230315_UT126_20221004_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230315_UT126_20221006_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230315_UT126_20221008_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230315_UT126_20221009_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230407_UT126_20230213_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230407_UT126_20230216_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "UT126_20230328_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT126_20230330_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
        },
        "UT1R3": {
            "20230407_UT1R3_20230222_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230407_UT1R3_20230228_D_v5a_autolabel_freespace_11v_sequential_4frame": None,
        },
        "UT263": {
            "UT263_20230301_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230303_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230304_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230306_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230324_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230325_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230421_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230422_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230423_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230427_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230504_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230505_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230506_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
        },
        "UT370": {
            "UT370_20230517_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230518_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230519_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230520_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230521_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230522_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230523_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230524_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230526_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230527_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230528_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230529_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230530_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230531_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230602_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
        },
    },
    "SHANGHAI_DEMO": {
        "LX165": {
            "20230316_FSD_Site_LX165_20230226_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230316_FSD_Site_LX165_20230227_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230316_FSD_Site_LX165_20230228_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230316_FSD_Site_LX165_20230301_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230327_FSD_Site_LX165_20230226_v5a_autolabel_freespace_11v_sequential_4frame": None,
            # "20230327_FSD_Site_LX165_20230227_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230327_FSD_Site_LX165_20230228_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230327_FSD_Site_LX165_20230301_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230327_FSD_Site_LX165_20230306_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230327_FSD_Site_LX165_20230307_v5a_autolabel_freespace_11v_sequential_4frame": None,
            "20230327_FSD_Site_LX165_20230308_v5a_autolabel_freespace_11v_sequential_4frame": None,
        }
    },
}

v5_2_0_update = {
    "PARKING_GT_FUSION_V1": {
        "UT263": {
            "UT263_20230514_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230516_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230517_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230507_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230614_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230617_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230619_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230620_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230621_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230612_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230627_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230630_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230706_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230628_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230701_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230703_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230704_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230629_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230702_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230705_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230711_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230712_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230710_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230713_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230708_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230714_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230725_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230718_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230720_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230721_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230722_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230724_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230715_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230805_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230727_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230728_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230803_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230719_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230731_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230801_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230804_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230807_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230802_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230726_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230815_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230813_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230809_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230812_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230810_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230814_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230808_D_parking_zhuxian_v5a_11v": None,
            "UT263_20230811_D_parking_zhuxian_v5a_11v": None,
        },
        "UT370": {
            "UT370_20230603_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230617_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230613_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230621_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230625_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230602_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230623_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230601_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230615_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230616_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230708_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230702_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230704_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230707_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230629_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230701_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230630_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230703_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230627_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230626_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230705_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230624_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230706_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230711_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230715_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230710_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230714_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230717_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230725_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230712_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230720_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230721_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230718_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230719_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230713_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230722_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230726_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230724_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230727_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230801_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230805_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230728_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230802_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230731_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230804_D_parking_zhuxian_v5a_11v": None,
            "UT370_20230803_D_parking_zhuxian_v5a_11v": None,
        },
        "UT0T1": {
            "UT0T1_20230610_D_parking_zhuxian_v5a_11v": None,
            "UT0T1_20230611_D_parking_zhuxian_v5a_11v": None,
            "UT0T1_20230620_D_parking_zhuxian_v5a_11v": None,
            "UT0T1_20230703_D_parking_zhuxian_v5a_11v": None,
            "UT0T1_20230624_D_parking_zhuxian_v5a_11v": None,
            "UT0T1_20230629_D_parking_zhuxian_v5a_11v": None,
            "UT0T1_20230701_D_parking_zhuxian_v5a_11v": None,
            "UT0T1_20230706_D_parking_zhuxian_v5a_11v": None,
            "UT0T1_20230626_D_parking_zhuxian_v5a_11v": None,
            "UT0T1_20230617_D_parking_zhuxian_v5a_11v": None,
            "UT0T1_20230615_D_parking_zhuxian_v5a_11v": None,
            "UT0T1_20230627_D_parking_zhuxian_v5a_11v": None,
            "UT0T1_20230704_D_parking_zhuxian_v5a_11v": None,
        },
    }
}

v5_2_0 = update_item_in_dict(v5_1_0, v5_2_0_update, only_use_update_version)

v5_2_1_del = {
    "PARKING_QUALITY_CHECK_V1_filter_ele_vismask": {
        "UT370": {
            "UT370_20230602_D_parking_zhuxian_v5a_11v",
        }
    }
}
v5_2_1 = del_item_in_dict(v5_2_0, v5_2_1_del)

v5_2_0_fisheye = update_item_in_dict(
    v5_1_0_fisheye, v5_2_0_update, only_use_update_version
)

v5_2_0_temporal_update = {
    "PARKING_GT_FUSION_V1": {
        "UT263": {
            "UT263_20230514_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230516_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230517_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230507_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230614_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230617_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230619_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230620_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230621_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230612_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230627_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230630_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230706_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230628_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230701_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230703_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230704_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230629_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230702_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230705_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230711_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230712_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230710_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230713_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230708_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230714_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230725_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230718_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230720_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230721_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230722_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230724_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230715_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230805_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230727_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230728_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230803_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230719_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230731_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230801_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230804_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230807_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230802_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230726_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
        },
        "UT370": {
            "UT370_20230603_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230617_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230613_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230621_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230602_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230623_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230601_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230615_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230616_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230708_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230702_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230704_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230707_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230629_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230701_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230630_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230703_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230627_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230626_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230705_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230624_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230706_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230711_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230715_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230710_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230714_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230717_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230725_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230712_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230720_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230721_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230718_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230719_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230713_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230722_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230726_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230724_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230727_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230801_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230805_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230728_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230802_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230731_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230804_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT370_20230803_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
        },
        "UT0T1": {
            "UT0T1_20230610_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT0T1_20230611_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
        },
        "UT263": {
            "UT263_20230815_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230813_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230809_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230812_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230810_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230814_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230808_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT263_20230811_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
        },
        "UT0T1": {
            "UT0T1_20230620_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT0T1_20230703_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT0T1_20230624_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT0T1_20230629_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT0T1_20230701_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT0T1_20230706_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT0T1_20230626_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT0T1_20230617_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT0T1_20230615_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT0T1_20230627_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
            "UT0T1_20230704_D_parking_zhuxian_v5a_11v_sequential_4frame": None,
        },
    }
}

v6_1_0_temporal = update_item_in_dict(
    v6_0_0_temporal, v5_2_0_temporal_update, only_use_update_version
)

v6_1_1_temporal_del = {
    "PARKING_QUALITY_CHECK_V1_filter_ele_vismask": {
        "UT370": {
            "UT370_20230602_D_parking_zhuxian_v5a_11v_sequential_4frame",
        }
    }
}

v6_1_1_temporal = del_item_in_dict(v6_1_0_temporal, v6_1_1_temporal_del)
