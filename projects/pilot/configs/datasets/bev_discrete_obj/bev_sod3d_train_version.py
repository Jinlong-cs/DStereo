# flake8: noqa
from projects.pilot.configs.datasets.bev_discrete_obj.utils import (
    del_item_in_dict,
    remove_item_in_dict,
    replace_item_in_dict_with_postfix,
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
    "multi_tensors_4d_prelable_gt_v2.5.0_small_parking_sequence": {
        "UT3J5": {"parking_005_train_UT3J5_15_packs": None},
    }
}
train_auto_gt_v1_0_1_small = {
    "multi_tensors_4d_prelable_gt_v2.5.0_small_driving": {
        "UT126": {
            "driving_012_train_UT126_50_sites": None,
            "driving_013_train_UT126_56_sites": None,
            "driving_024_train_UT126_36_sites": None,
        },
        "UT263": {
            "driving_014_train_UT263_101_sites": None,
            "driving_016_train_UT263_1_sites": None,
            "driving_017_train_UT263_3_sites": None,
            "driving_022_train_UT263_52_sites": None,
            "driving_023_train_UT263_6_sites": None,
            "driving_025_train_UT263_2_sites": None,
            "driving_028_train_UT263_9_sites": None,
            "driving_029_train_UT263_13_sites": None,
            "driving_031_train_UT263_1_sites": None,
            "driving_032_train_UT263_50_sites": None,
            "driving_033_train_UT263_60_sites": None,
            "driving_038_train_UT263_4_sites": None,
            "driving_041_train_UT263_34_sites": None,
        },
        "UTHS6": {
            "driving_018_train_UTHS6_58_sites": None,
            "driving_019_train_UTHS6_28_sites": None,
            "driving_020_train_UTHS6_19_sites": None,
            "driving_021_train_UTHS6_72_sites": None,
            "driving_027_train_UTHS6_2_sites": None,
            "driving_030_train_UTHS6_2_sites": None,
            "driving_034_train_UTHS6_50_sites": None,
            "driving_035_train_UTHS6_33_sites": None,
            "driving_036_train_UTHS6_2_sites": None,
            "driving_037_train_UTHS6_1_sites": None,
        },
        "LS830": {
            "driving_039_train_LS830_62_sites": None,
            "driving_040_train_LS830_1_sites": None,
        },
        "UT1R3": {"driving_042_train_UT1R3_40_sites": None},
        "LX165": {"driving_043_train_LX165_2_sites": None},
    }
}
train_auto_gt_v1_0_1_small_sequence = {
    "multi_tensors_4d_prelable_gt_v2.5.0_small_driving_sequence": {
        "UT126": {
            "driving_012_train_UT126_50_sites": None,
            "driving_013_train_UT126_56_sites": None,
            "driving_024_train_UT126_36_sites": None,
        },
        "UT263": {
            "driving_014_train_UT263_101_sites": None,
            "driving_016_train_UT263_1_sites": None,
            "driving_017_train_UT263_3_sites": None,
            "driving_022_train_UT263_52_sites": None,
            "driving_023_train_UT263_6_sites": None,
            "driving_025_train_UT263_2_sites": None,
            "driving_028_train_UT263_9_sites": None,
            "driving_029_train_UT263_13_sites": None,
            "driving_031_train_UT263_1_sites": None,
            "driving_032_train_UT263_50_sites": None,
            "driving_033_train_UT263_60_sites": None,
            "driving_038_train_UT263_4_sites": None,
            "driving_041_train_UT263_34_sites": None,
        },
        "UTHS6": {
            "driving_018_train_UTHS6_58_sites": None,
            "driving_019_train_UTHS6_28_sites": None,
            "driving_020_train_UTHS6_19_sites": None,
            "driving_021_train_UTHS6_72_sites": None,
            "driving_027_train_UTHS6_2_sites": None,
            "driving_030_train_UTHS6_2_sites": None,
            "driving_034_train_UTHS6_50_sites": None,
            "driving_035_train_UTHS6_33_sites": None,
            "driving_036_train_UTHS6_2_sites": None,
            "driving_037_train_UTHS6_1_sites": None,
        },
        "LS830": {
            "driving_039_train_LS830_62_sites": None,
            "driving_040_train_LS830_1_sites": None,
        },
        "UT1R3": {"driving_042_train_UT1R3_40_sites": None},
        "LX165": {"driving_043_train_LX165_2_sites": None},
    }
}

v1_0_2_small_update = {
    "multi_tensors_4d_prelable_gt_v2.5.0_small_parking": {
        "UT3J5": {"parking_005_train_UT3J5_15_packs": None},
        "UT263": {
            "parking_006_train_UT263_53_packs": None,
            "parking_008_train_UT263_12_packs": None,
        },
        "UT1R3": {"parking_007_train_UT1R3_26_packs": None},
        "UT0Q9": {"parking_009_train_UT0Q9_68_packs": None},
        "UT0F3": {
            "parking_010_train_UT0F3_2_packs": None,
            "parking_012_train_UT0F3_44_packs": None,
        },
        "LS830": {"parking_011_train_LS830_8_packs": None},
    },
    "multi_tensors_4d_prelable_gt_v2.5.0_small_parking_sequence": {
        "UT3J5": {"parking_005_train_UT3J5_15_packs": None},
        "UT263": {
            "parking_006_train_UT263_53_packs": None,
            "parking_008_train_UT263_12_packs": None,
        },
        "UT1R3": {"parking_007_train_UT1R3_26_packs": None},
        "UT0Q9": {"parking_009_train_UT0Q9_68_packs": None},
        "UT0F3": {
            "parking_010_train_UT0F3_2_packs": None,
            "parking_012_train_UT0F3_44_packs": None,
        },
        "LS830": {"parking_011_train_LS830_8_packs": None},
    },
}
train_auto_gt_v1_0_2_small = update_item_in_dict(
    train_auto_gt_v1_0_1_small, v1_0_2_small_update, only_use_update_version
)

v1_0_2_small_update_sequence = {
    "multi_tensors_4d_prelable_gt_v2.5.0_small_parking_sequence": {
        "UT3J5": {"parking_005_train_UT3J5_15_packs": None},
        "UT263": {
            "parking_006_train_UT263_53_packs": None,
            "parking_008_train_UT263_12_packs": None,
        },
        "UT1R3": {"parking_007_train_UT1R3_26_packs": None},
        "UT0Q9": {"parking_009_train_UT0Q9_68_packs": None},
        "UT0F3": {
            "parking_010_train_UT0F3_2_packs": None,
            "parking_012_train_UT0F3_44_packs": None,
        },
        "LS830": {"parking_011_train_LS830_8_packs": None},
    },
    "multi_tensors_4d_prelable_gt_v2.5.0_small_parking_sequence": {
        "UT3J5": {"parking_005_train_UT3J5_15_packs": None},
        "UT263": {
            "parking_006_train_UT263_53_packs": None,
            "parking_008_train_UT263_12_packs": None,
        },
        "UT1R3": {"parking_007_train_UT1R3_26_packs": None},
        "UT0Q9": {"parking_009_train_UT0Q9_68_packs": None},
        "UT0F3": {
            "parking_010_train_UT0F3_2_packs": None,
            "parking_012_train_UT0F3_44_packs": None,
        },
        "LS830": {"parking_011_train_LS830_8_packs": None},
    },
}
train_auto_gt_v1_0_2_small_sequence = update_item_in_dict(
    train_auto_gt_v1_0_1_small_sequence,
    v1_0_2_small_update_sequence,
    only_use_update_version,
)

v1_0_3_small_update = {
    "multi_tensors_4d_prelable_gt_v2.5.0_small_driving": {
        "NC109": {
            "driving_045_train_NC109_53_sites": None,
            "driving_054_train_NC109_23_sites": None,
            "driving_063_train_NC109_30_sites": None,
            "driving_064_train_NC109_2_sites": None,
        },
        "UTHS6": {
            "driving_046_train_UTHS6_22_sites": None,
            "driving_060_train_UTHS6_44_sites": None,
        },
        "UT0F3": {
            "driving_047_train_UT0F3_70_sites": None,
            "driving_061_train_UT0F3_15_sites": None,
            "driving_071_train_UT0F3_14_sites": None,
        },
        "NC110": {
            "driving_048_train_NC110_9_sites": None,
            "driving_055_train_NC110_39_sites": None,
            "driving_062_train_NC110_12_sites": None,
        },
        "UT263": {
            "driving_049_train_UT263_54_sites": None,
            "driving_050_train_UT263_2_sites": None,
            "driving_057_train_UT263_5_sites": None,
            "driving_058_train_UT263_28_sites": None,
        },
        "UT3J5": {
            "driving_051_train_UT3J5_38_sites": None,
            "driving_059_train_UT3J5_65_sites": None,
        },
        "LS912": {
            "driving_052_train_LS912_23_sites": None,
            "driving_065_train_LS912_25_sites": None,
        },
        "LX165": {"driving_053_train_LX165_13_sites": None},
        "UT1R3": {"driving_056_train_UT1R3_40_sites": None},
        "UT0Q9": {"driving_066_train_UT0Q9_11_sites": None},
        "UT126": {"driving_067_train_UT126_34_sites": None},
        "LS830": {
            "driving_068_train_LS830_50_sites": None,
            "driving_069_train_LS830_65_sites": None,
            "driving_070_train_LS830_3_sites": None,
        },
    },
    "multi_tensors_4d_prelable_gt_v2.5.0_small_driving_sequence": {
        "NC109": {
            "driving_045_train_NC109_53_sites": None,
            "driving_054_train_NC109_23_sites": None,
            "driving_063_train_NC109_30_sites": None,
            "driving_064_train_NC109_2_sites": None,
        },
        "UTHS6": {
            "driving_046_train_UTHS6_22_sites": None,
            "driving_060_train_UTHS6_44_sites": None,
        },
        "UT0F3": {
            "driving_047_train_UT0F3_70_sites": None,
            "driving_061_train_UT0F3_15_sites": None,
            "driving_071_train_UT0F3_14_sites": None,
        },
        "NC110": {
            "driving_048_train_NC110_9_sites": None,
            "driving_055_train_NC110_39_sites": None,
            "driving_062_train_NC110_12_sites": None,
        },
        "UT263": {
            "driving_049_train_UT263_54_sites": None,
            "driving_050_train_UT263_2_sites": None,
            "driving_057_train_UT263_5_sites": None,
            "driving_058_train_UT263_28_sites": None,
        },
        "UT3J5": {
            "driving_051_train_UT3J5_38_sites": None,
            "driving_059_train_UT3J5_65_sites": None,
        },
        "LS912": {
            "driving_052_train_LS912_23_sites": None,
            "driving_065_train_LS912_25_sites": None,
        },
        "LX165": {"driving_053_train_LX165_13_sites": None},
        "UT1R3": {"driving_056_train_UT1R3_40_sites": None},
        "UT0Q9": {"driving_066_train_UT0Q9_11_sites": None},
        "UT126": {"driving_067_train_UT126_34_sites": None},
        "LS830": {
            "driving_068_train_LS830_50_sites": None,
            "driving_069_train_LS830_65_sites": None,
            "driving_070_train_LS830_3_sites": None,
        },
    },
    "multi_tensors_4d_prelable_gt_v2.5.0_small_parking": {
        "UT0Q9": {
            "parking_013_train_UT0Q9_6_packs": None,
            "parking_020_train_UT0Q9_33_packs": None,
        },
        "UT0F3": {"parking_014_train_UT0F3_69_packs": None},
        "UT1R3": {
            "parking_015_train_UT1R3_33_packs": None,
            "parking_018_train_UT1R3_9_packs": None,
        },
        "UT263": {
            "parking_016_train_UT263_25_packs": None,
            "parking_019_train_UT263_3_packs": None,
        },
        "UT3J5": {
            "parking_017_train_UT3J5_24_packs": None,
            "parking_022_train_UT3J5_6_packs": None,
        },
        "LS830": {"parking_021_train_LS830_9_packs": None},
    },
    "multi_tensors_4d_prelable_gt_v2.5.0_small_parking_sequence": {
        "UT0Q9": {
            "parking_013_train_UT0Q9_6_packs": None,
            "parking_020_train_UT0Q9_33_packs": None,
        },
        "UT0F3": {"parking_014_train_UT0F3_69_packs": None},
        "UT1R3": {
            "parking_015_train_UT1R3_33_packs": None,
            "parking_018_train_UT1R3_9_packs": None,
        },
        "UT263": {
            "parking_016_train_UT263_25_packs": None,
            "parking_019_train_UT263_3_packs": None,
        },
        "UT3J5": {
            "parking_017_train_UT3J5_24_packs": None,
            "parking_022_train_UT3J5_6_packs": None,
        },
        "LS830": {"parking_021_train_LS830_9_packs": None},
    },
    "multi_tensors_4d_prelable_gt_v2.6.0_small_parking": {
        "LS830": {"parking_023_train_LS830_6_packs": None},
        "UT0F3": {"parking_024_train_UT0F3_31_packs": None},
        "UT1R3": {"parking_025_train_UT1R3_8_packs": None},
        "UT263": {"parking_026_train_UT263_1_packs": None},
        "UT370": {"parking_027_train_UT370_55_packs": None},
        "UT3J5": {"parking_028_train_UT3J5_3_packs": None},
        "UT0Q9": {"parking_029_train_UT0Q9_6_packs": None},
    },
    "multi_tensors_4d_prelable_gt_v2.6.0_small_parking_sequence": {
        "LS830": {"parking_023_train_LS830_6_packs": None},
        "UT0F3": {"parking_024_train_UT0F3_31_packs": None},
        "UT1R3": {"parking_025_train_UT1R3_8_packs": None},
        "UT263": {"parking_026_train_UT263_1_packs": None},
        "UT370": {"parking_027_train_UT370_55_packs": None},
        "UT3J5": {"parking_028_train_UT3J5_3_packs": None},
        "UT0Q9": {"parking_029_train_UT0Q9_6_packs": None},
    },
}
train_auto_gt_v1_0_3_small = update_item_in_dict(
    train_auto_gt_v1_0_2_small, v1_0_3_small_update, only_use_update_version
)

train_auto_gt_v1_0_3_small_parking = remove_item_in_dict(
    train_auto_gt_v1_0_3_small, "driving"
)

v1_0_3_filter_update = {
    "multi_tensors_4d_prelable_gt_v2.5.0_small_driving": {
        "LS830": {
            "driving_068_train_LS830_50_sites_filtered": None,
            "driving_069_train_LS830_65_sites_filtered": None,
            "driving_070_train_LS830_3_sites_filtered": None,
        },
        "LS912": {
            "driving_052_train_LS912_23_sites_filtered": None,
            "driving_065_train_LS912_25_sites_filtered": None,
        },
        "LX165": {
            "driving_043_train_LX165_2_sites_filtered": None,
            "driving_053_train_LX165_13_sites_filtered": None,
        },
        "NC109": {
            "driving_045_train_NC109_53_sites_filtered": None,
            "driving_054_train_NC109_23_sites_filtered": None,
            "driving_063_train_NC109_30_sites_filtered": None,
            "driving_064_train_NC109_2_sites_filtered": None,
        },
        "NC110": {
            "driving_048_train_NC110_9_sites_filtered": None,
            "driving_055_train_NC110_39_sites_filtered": None,
            "driving_062_train_NC110_12_sites_filtered": None,
        },
        "UT0F3": {
            "driving_047_train_UT0F3_70_sites_filtered": None,
            "driving_061_train_UT0F3_15_sites_filtered": None,
            "driving_071_train_UT0F3_14_sites_filtered": None,
        },
        "UT0Q9": {"driving_066_train_UT0Q9_11_sites_filtered": None},
        "UT126": {
            "driving_012_train_UT126_50_sites_filtered": None,
            "driving_013_train_UT126_56_sites_filtered": None,
            "driving_024_train_UT126_36_sites_filtered": None,
            "driving_067_train_UT126_34_sites_filtered": None,
        },
        "UT1R3": {
            "driving_042_train_UT1R3_40_sites_filtered": None,
            "driving_056_train_UT1R3_40_sites_filtered": None,
        },
        "UT263": {
            "driving_014_train_UT263_101_sites_filtered": None,
            "driving_017_train_UT263_3_sites_filtered": None,
            "driving_022_train_UT263_52_sites_filtered": None,
            "driving_023_train_UT263_6_sites_filtered": None,
            "driving_025_train_UT263_2_sites_filtered": None,
            "driving_028_train_UT263_9_sites_filtered": None,
            "driving_029_train_UT263_13_sites_filtered": None,
            "driving_031_train_UT263_1_sites_filtered": None,
            "driving_032_train_UT263_50_sites_filtered": None,
            "driving_033_train_UT263_60_sites_filtered": None,
            "driving_038_train_UT263_4_sites_filtered": None,
            "driving_041_train_UT263_34_sites_filtered": None,
            "driving_049_train_UT263_54_sites_filtered": None,
            "driving_050_train_UT263_2_sites_filtered": None,
            "driving_057_train_UT263_5_sites_filtered": None,
            "driving_058_train_UT263_28_sites_filtered": None,
        },
        "UT3J5": {
            "driving_051_train_UT3J5_38_sites_filtered": None,
            "driving_059_train_UT3J5_65_sites_filtered": None,
        },
        "UTHS6": {
            "driving_018_train_UTHS6_58_sites_filtered": None,
            "driving_019_train_UTHS6_28_sites_filtered": None,
            "driving_020_train_UTHS6_19_sites_filtered": None,
            "driving_021_train_UTHS6_72_sites_filtered": None,
            "driving_027_train_UTHS6_2_sites_filtered": None,
            "driving_030_train_UTHS6_2_sites_filtered": None,
            "driving_034_train_UTHS6_50_sites_filtered": None,
            "driving_035_train_UTHS6_33_sites_filtered": None,
            "driving_036_train_UTHS6_2_sites_filtered": None,
            "driving_037_train_UTHS6_1_sites_filtered": None,
            "driving_046_train_UTHS6_22_sites_filtered": None,
            "driving_060_train_UTHS6_44_sites_filtered": None,
        },
    },
    "multi_tensors_4d_prelable_gt_v2.5.0_small_driving_sequence": {
        "LS830": {
            "driving_068_train_LS830_50_sites_filtered": None,
            "driving_069_train_LS830_65_sites_filtered": None,
            "driving_070_train_LS830_3_sites_filtered": None,
        },
        "LS912": {
            "driving_052_train_LS912_23_sites_filtered": None,
            "driving_065_train_LS912_25_sites_filtered": None,
        },
        "LX165": {"driving_053_train_LX165_13_sites_filtered": None},
        "NC109": {
            "driving_045_train_NC109_53_sites_filtered": None,
            "driving_054_train_NC109_23_sites_filtered": None,
            "driving_063_train_NC109_30_sites_filtered": None,
            "driving_064_train_NC109_2_sites_filtered": None,
        },
        "NC110": {
            "driving_048_train_NC110_9_sites_filtered": None,
            "driving_055_train_NC110_39_sites_filtered": None,
            "driving_062_train_NC110_12_sites_filtered": None,
        },
        "UT0F3": {
            "driving_047_train_UT0F3_70_sites_filtered": None,
            "driving_061_train_UT0F3_15_sites_filtered": None,
            "driving_071_train_UT0F3_14_sites_filtered": None,
        },
        "UT0Q9": {"driving_066_train_UT0Q9_11_sites_filtered": None},
        "UT126": {"driving_067_train_UT126_34_sites_filtered": None},
        "UT1R3": {"driving_056_train_UT1R3_40_sites_filtered": None},
        "UT263": {
            "driving_049_train_UT263_54_sites_filtered": None,
            "driving_050_train_UT263_2_sites_filtered": None,
            "driving_057_train_UT263_5_sites_filtered": None,
            "driving_058_train_UT263_28_sites_filtered": None,
        },
        "UT3J5": {
            "driving_051_train_UT3J5_38_sites_filtered": None,
            "driving_059_train_UT3J5_65_sites_filtered": None,
        },
        "UTHS6": {
            "driving_046_train_UTHS6_22_sites_filtered": None,
            "driving_060_train_UTHS6_44_sites_filtered": None,
        },
    },
    "multi_tensors_4d_prelable_gt_v2.5.0_small_parking": {
        "LS830": {
            "parking_001_val_LS830_4_packs_filtered": None,
            "parking_011_train_LS830_8_packs_filtered": None,
            "parking_021_train_LS830_9_packs_filtered": None,
        },
        "UT0F3": {
            "parking_010_train_UT0F3_2_packs_filtered": None,
            "parking_012_train_UT0F3_44_packs_filtered": None,
            "parking_014_train_UT0F3_69_packs_filtered": None,
        },
        "UT0Q9": {
            "parking_009_train_UT0Q9_68_packs_filtered": None,
            "parking_013_train_UT0Q9_6_packs_filtered": None,
            "parking_020_train_UT0Q9_33_packs_filtered": None,
        },
        "UT1R3": {
            "parking_002_val_UT1R3_5_packs_filtered": None,
            "parking_007_train_UT1R3_26_packs_filtered": None,
            "parking_015_train_UT1R3_33_packs_filtered": None,
            "parking_018_train_UT1R3_9_packs_filtered": None,
        },
        "UT263": {
            "parking_003_val_UT263_9_packs_filtered": None,
            "parking_006_train_UT263_53_packs_filtered": None,
            "parking_008_train_UT263_12_packs_filtered": None,
            "parking_016_train_UT263_25_packs_filtered": None,
        },
        "UT3J5": {
            "parking_004_val_UT3J5_6_packs_filtered": None,
            "parking_005_train_UT3J5_15_packs_filtered": None,
            "parking_017_train_UT3J5_24_packs_filtered": None,
            "parking_022_train_UT3J5_6_packs_filtered": None,
        },
    },
    "multi_tensors_4d_prelable_gt_v2.5.0_small_parking_sequence": {
        "LS830": {
            "parking_011_train_LS830_8_packs_filtered": None,
            "parking_021_train_LS830_9_packs_filtered": None,
        },
        "UT0F3": {
            "parking_010_train_UT0F3_2_packs_filtered": None,
            "parking_012_train_UT0F3_44_packs_filtered": None,
            "parking_014_train_UT0F3_69_packs_filtered": None,
        },
        "UT0Q9": {
            "parking_009_train_UT0Q9_68_packs_filtered": None,
            "parking_013_train_UT0Q9_6_packs_filtered": None,
            "parking_020_train_UT0Q9_33_packs_filtered": None,
        },
        "UT1R3": {
            "parking_007_train_UT1R3_26_packs_filtered": None,
            "parking_015_train_UT1R3_33_packs_filtered": None,
            "parking_018_train_UT1R3_9_packs_filtered": None,
        },
        "UT263": {
            "parking_006_train_UT263_53_packs_filtered": None,
            "parking_008_train_UT263_12_packs_filtered": None,
            "parking_016_train_UT263_25_packs_filtered": None,
        },
        "UT3J5": {
            "parking_005_train_UT3J5_15_packs_filtered": None,
            "parking_017_train_UT3J5_24_packs_filtered": None,
            "parking_022_train_UT3J5_6_packs_filtered": None,
        },
    },
    "multi_tensors_4d_prelable_gt_v2.6.0_small_parking": {
        "LS830": {"parking_023_train_LS830_6_packs_filtered": None},
        "UT0F3": {"parking_024_train_UT0F3_31_packs_filtered": None},
        "UT0Q9": {"parking_029_train_UT0Q9_6_packs_filtered": None},
        "UT1R3": {"parking_025_train_UT1R3_8_packs_filtered": None},
        "UT370": {"parking_027_train_UT370_55_packs_filtered": None},
    },
    "multi_tensors_4d_prelable_gt_v2.6.0_small_parking_sequence": {
        "LS830": {"parking_023_train_LS830_6_packs_filtered": None},
        "UT0F3": {"parking_024_train_UT0F3_31_packs_filtered": None},
        "UT0Q9": {"parking_029_train_UT0Q9_6_packs_filtered": None},
        "UT1R3": {"parking_025_train_UT1R3_8_packs_filtered": None},
        "UT370": {"parking_027_train_UT370_55_packs_filtered": None},
    },
}

train_auto_gt_v1_0_3_small_filtered = replace_item_in_dict_with_postfix(
    train_auto_gt_v1_0_3_small, v1_0_3_filter_update, "_filtered"
)

print("train_auto_gt_v1_0_3_small_filtered")
v1_0_3_small_update_sequence = {
    "multi_tensors_4d_prelable_gt_v2.5.0_small_driving_sequence": {
        "NC109": {
            "driving_045_train_NC109_53_sites": None,
            "driving_054_train_NC109_23_sites": None,
            "driving_063_train_NC109_30_sites": None,
            "driving_064_train_NC109_2_sites": None,
        },
        "UTHS6": {
            "driving_046_train_UTHS6_22_sites": None,
            "driving_060_train_UTHS6_44_sites": None,
        },
        "UT0F3": {
            "driving_047_train_UT0F3_70_sites": None,
            "driving_061_train_UT0F3_15_sites": None,
            "driving_071_train_UT0F3_14_sites": None,
        },
        "NC110": {
            "driving_048_train_NC110_9_sites": None,
            "driving_055_train_NC110_39_sites": None,
            "driving_062_train_NC110_12_sites": None,
        },
        "UT263": {
            "driving_049_train_UT263_54_sites": None,
            "driving_050_train_UT263_2_sites": None,
            "driving_057_train_UT263_5_sites": None,
            "driving_058_train_UT263_28_sites": None,
        },
        "UT3J5": {
            "driving_051_train_UT3J5_38_sites": None,
            "driving_059_train_UT3J5_65_sites": None,
        },
        "LS912": {
            "driving_052_train_LS912_23_sites": None,
            "driving_065_train_LS912_25_sites": None,
        },
        "LX165": {"driving_053_train_LX165_13_sites": None},
        "UT1R3": {"driving_056_train_UT1R3_40_sites": None},
        "UT0Q9": {"driving_066_train_UT0Q9_11_sites": None},
        "UT126": {"driving_067_train_UT126_34_sites": None},
        "LS830": {
            "driving_068_train_LS830_50_sites": None,
            "driving_069_train_LS830_65_sites": None,
            "driving_070_train_LS830_3_sites": None,
        },
    },
    "multi_tensors_4d_prelable_gt_v2.5.0_small_driving_sequence": {
        "NC109": {
            "driving_045_train_NC109_53_sites": None,
            "driving_054_train_NC109_23_sites": None,
            "driving_063_train_NC109_30_sites": None,
            "driving_064_train_NC109_2_sites": None,
        },
        "UTHS6": {
            "driving_046_train_UTHS6_22_sites": None,
            "driving_060_train_UTHS6_44_sites": None,
        },
        "UT0F3": {
            "driving_047_train_UT0F3_70_sites": None,
            "driving_061_train_UT0F3_15_sites": None,
            "driving_071_train_UT0F3_14_sites": None,
        },
        "NC110": {
            "driving_048_train_NC110_9_sites": None,
            "driving_055_train_NC110_39_sites": None,
            "driving_062_train_NC110_12_sites": None,
        },
        "UT263": {
            "driving_049_train_UT263_54_sites": None,
            "driving_050_train_UT263_2_sites": None,
            "driving_057_train_UT263_5_sites": None,
            "driving_058_train_UT263_28_sites": None,
        },
        "UT3J5": {
            "driving_051_train_UT3J5_38_sites": None,
            "driving_059_train_UT3J5_65_sites": None,
        },
        "LS912": {
            "driving_052_train_LS912_23_sites": None,
            "driving_065_train_LS912_25_sites": None,
        },
        "LX165": {"driving_053_train_LX165_13_sites": None},
        "UT1R3": {"driving_056_train_UT1R3_40_sites": None},
        "UT0Q9": {"driving_066_train_UT0Q9_11_sites": None},
        "UT126": {"driving_067_train_UT126_34_sites": None},
        "LS830": {
            "driving_068_train_LS830_50_sites": None,
            "driving_069_train_LS830_65_sites": None,
            "driving_070_train_LS830_3_sites": None,
        },
    },
    "multi_tensors_4d_prelable_gt_v2.5.0_small_parking_sequence": {
        "UT0Q9": {
            "parking_013_train_UT0Q9_6_packs": None,
            "parking_020_train_UT0Q9_33_packs": None,
        },
        "UT0F3": {"parking_014_train_UT0F3_69_packs": None},
        "UT1R3": {
            "parking_015_train_UT1R3_33_packs": None,
            "parking_018_train_UT1R3_9_packs": None,
        },
        "UT263": {
            "parking_016_train_UT263_25_packs": None,
            "parking_019_train_UT263_3_packs": None,
        },
        "UT3J5": {
            "parking_017_train_UT3J5_24_packs": None,
            "parking_022_train_UT3J5_6_packs": None,
        },
        "LS830": {"parking_021_train_LS830_9_packs": None},
    },
    "multi_tensors_4d_prelable_gt_v2.5.0_small_parking_sequence": {
        "UT0Q9": {
            "parking_013_train_UT0Q9_6_packs": None,
            "parking_020_train_UT0Q9_33_packs": None,
        },
        "UT0F3": {"parking_014_train_UT0F3_69_packs": None},
        "UT1R3": {
            "parking_015_train_UT1R3_33_packs": None,
            "parking_018_train_UT1R3_9_packs": None,
        },
        "UT263": {
            "parking_016_train_UT263_25_packs": None,
            "parking_019_train_UT263_3_packs": None,
        },
        "UT3J5": {
            "parking_017_train_UT3J5_24_packs": None,
            "parking_022_train_UT3J5_6_packs": None,
        },
        "LS830": {"parking_021_train_LS830_9_packs": None},
    },
    "multi_tensors_4d_prelable_gt_v2.6.0_small_parking_sequence": {
        "LS830": {"parking_023_train_LS830_6_packs": None},
        "UT0F3": {"parking_024_train_UT0F3_31_packs": None},
        "UT1R3": {"parking_025_train_UT1R3_8_packs": None},
        "UT263": {"parking_026_train_UT263_1_packs": None},
        "UT370": {"parking_027_train_UT370_55_packs": None},
        "UT3J5": {"parking_028_train_UT3J5_3_packs": None},
        "UT0Q9": {"parking_029_train_UT0Q9_6_packs": None},
    },
    "multi_tensors_4d_prelable_gt_v2.6.0_small_parking_sequence": {
        "LS830": {"parking_023_train_LS830_6_packs": None},
        "UT0F3": {"parking_024_train_UT0F3_31_packs": None},
        "UT1R3": {"parking_025_train_UT1R3_8_packs": None},
        "UT263": {"parking_026_train_UT263_1_packs": None},
        "UT370": {"parking_027_train_UT370_55_packs": None},
        "UT3J5": {"parking_028_train_UT3J5_3_packs": None},
        "UT0Q9": {"parking_029_train_UT0Q9_6_packs": None},
    },
}
train_auto_gt_v1_0_3_small_sequence = update_item_in_dict(
    train_auto_gt_v1_0_2_small_sequence,
    v1_0_3_small_update_sequence,
    only_use_update_version,
)
