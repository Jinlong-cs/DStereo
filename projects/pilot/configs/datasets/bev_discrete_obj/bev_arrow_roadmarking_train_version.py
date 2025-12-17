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
    "11v_driving_data_wide_temporal": {
        "LX165": {
            "merge_LX165_det_v5a_20230318_20230323_LX165_11v": None,
        },
    }
}
union_arrow_v1_2_3_small_roadmarking_v1_2_1_small = {
    "11v_driving_data_small_union_small": {
        "LS912": {
            "20221202_723057_1_FSD_Site_LS912_20221115": None,
        },
    },
}
union_arrow_v1_2_3_wide_roadmarking_v1_2_1_wide = {
    "11v_driving_data_wide_union_wide": {
        "LS912": {
            "20221202_723057_1_FSD_Site_LS912_20221115": None,
        },
    },
}

# small vcs data of new version
union_small_update = {}
# update data
HDE_lidar_v1_2_5_small = update_item_in_dict(
    union_arrow_v1_2_3_small_roadmarking_v1_2_1_small,
    union_small_update,
    only_use_update_version,
)

# wide vcs data of new version
union_wide_update = {}
# update data
HDE_lidar_v1_2_5_wide = update_item_in_dict(
    union_arrow_v1_2_3_wide_roadmarking_v1_2_1_wide,
    union_wide_update,
    only_use_update_version,
)


# temporal version
temporal_v1_0_0_small = {
    "11v_driving_data_small_temporal": {
        "LX165": {
            "merge_LX165_det_v5a_20230318_20230323_LX165_11v": None,
            "merge_LX165_det_v5a_20230324_20230328_LX165_11v": None,
        },
        "NC109": {
            "merge_NC109_det_v5a_20221201_20221205_NC109_11v": None,
            "merge_NC109_det_v5a_20221223_20221231_NC109_11v": None,
            "merge_NC109_det_v5a_20230108_20230113_NC109_val_11v": None,
            "merge_NC109_det_v5a_20230108_20230113_NC109_11v": None,
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
            "merge_UT263_det_v5a_20230112_20230114_UT263_val_11v": None,
            "merge_UT263_det_v5a_20230112_20230114_UT263_11v": None,
        },
    },
}

temporal_v1_0_0_wide = {
    "11v_driving_data_wide_temporal": {
        "LX165": {
            "merge_LX165_det_v5a_20230318_20230323_LX165_11v": None,
            "merge_LX165_det_v5a_20230324_20230328_LX165_11v": None,
        },
        "NC109": {
            "merge_NC109_det_v5a_20221201_20221205_NC109_11v": None,
            "merge_NC109_det_v5a_20221223_20221231_NC109_11v": None,
            "merge_NC109_det_v5a_20230108_20230113_NC109_val_11v": None,
            "merge_NC109_det_v5a_20230108_20230113_NC109_11v": None,
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
            "merge_UT263_det_v5a_20230112_20230114_UT263_val_11v": None,
            "merge_UT263_det_v5a_20230112_20230114_UT263_11v": None,
        },
    },
}


# temporal version
temporal_v1_1_0_small_update = {
    "11v_parking_data_small_temporal": {
        "LS830": {
            "merged__0_204__20230302_20230309": None,
            "merged__0_204__20230322_20230408": None,
        },
        "NC109": {
            "merged__0_204__20221015_20221111": None,
        },
        "NC110": {
            "merged__0_204__20221104_20221116": None,
            "merged__0_204__20221118_20221221": None,
            "merged__0_204__20221122_20221206": None,
            "merged__0_204__20221228_20230112": None,
            "merged__0_204__20230115_20230330": None,
            "merged__0_204__20230331_20230414": None,
        },
        "UT0Q9": {
            "merged__0_204__20221024_20221026": None,
            "merged__0_204__20221126_20221206": None,
            "merged__0_204__20221207_20230103": None,
            "merged__0_204__20230105_20230308": None,
            "merged__0_204__20230316_20230327": None,
            "merged__0_204__20230408_20230414": None,
        },
        "UT126": {
            "merged__0_204__20220803_20220829": None,
            "merged__0_204__20220830_20220914": None,
            "merged__0_204__20220923_20221206": None,
            "merged__0_204__20221207_20230214": None,
            "merged__0_204__20230328_20230407": None,
            "merged__0_204__20230412_20230414": None,
            "merged__0_204__20230417": None,
        },
        "UT1R3": {
            "merged__0_204__20221104_20221204": None,
            "merged__0_204__20221209_20230104": None,
            "merged__0_204__20230107_20230110": None,
            "merged__0_204__20230130_20230225": None,
            "merged__0_204__20230322_20230410": None,
            "merged__0_204__20230411_20230414": None,
            "merged__0_204__20230420_20230422": None,
        },
        "UT263": {
            "merged__0_204__20220921_20221018": None,
            "merged__0_204__20221023_20221110": None,
            "merged__0_204__20221111_20221120": None,
            "merged__0_204__20221126_20221206": None,
            "merged__0_204__20221207_20230103": None,
            "merged__0_204__20230108_20230214": None,
            "merged__0_204__20230224_20230321": None,
            "merged__0_204__20230313_20230327": None,
            "merged__0_204__20230329_20230420": None,
            "merged__0_204__20230424": None,
        },
        "UTHS6": {
            "merged__0_204__20221021_20221031": None,
            "merged__0_204__20221027_20221030": None,
            "merged__0_204__20221101_20221206": None,
            "merged__0_204__20221208_20230128": None,
            "merged__0_204__20230210_20230310": None,
        },
    },
}

# update data
temporal_v1_1_0_small = update_item_in_dict(
    temporal_v1_0_0_small,
    temporal_v1_1_0_small_update,
    only_use_update_version,
)

temporal_v1_1_0_wide_update = {
    "11v_driving_data_wide_temporal": {
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
        "UT126": {
            "20230306_741986_1_background_0_FSD_Site_UT126_20230128": None,
            "20230306_741986_1_background_0_FSD_Site_UT126_20230129": None,
            "20230306_741986_1_background_0_FSD_Site_UT126_20230209": None,
            "20230308_742614_1_background_0_FSD_Site_UT126_20230208": None,
        },
        # "LS912": {
        #     "20230316_744558_1_background_0_FSD_Site_LS912_20230116": None,
        # },
        "LS830": {
            "merge_LS830_det_v5a_20230221_20230221_LS830_11v": None,
        },
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
            "20230306_741981_1_background_0_FSD_Site_UT1R3_20230210": None,
            "20230306_741981_1_background_0_FSD_Site_UT1R3_20230211": None,
            "20230423_753178_2_background_0_FSD_Site_UT1R3_20230221": None,
            "20230423_753189_2_background_0_FSD_Site_UT1R3_20230221": None,
            "20230320_745267_1_background_0_FSD_Site_UT1R3_20230202": None,
        },
        "UT263": {
            "20230422_752973_2_background_0_FSD_Site_UT263_20230228": None,
            "20230422_752974_2_background_0_FSD_Site_UT263_20230228": None,
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
}
# update data
temporal_v1_1_0_wide = update_item_in_dict(
    temporal_v1_0_0_wide, temporal_v1_1_0_wide_update, only_use_update_version
)

temporal_v1_2_0_wide_update = {
    "11v_driving_data_wide_temporal": {
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
}
# update data
temporal_v1_2_0_wide = update_item_in_dict(
    temporal_v1_1_0_wide, temporal_v1_2_0_wide_update, only_use_update_version
)

temporal_v1_3_0_wide_update = {
    "11v_driving_data_wide_temporal": {
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
            "merge_UT1R3_det_v5a_20230407_20230412_UT1R3_11v": None,
        },
    },
}
# update data
temporal_v1_3_0_wide = update_item_in_dict(
    temporal_v1_2_0_wide,
    temporal_v1_3_0_wide_update,
    only_use_update_version,
)
