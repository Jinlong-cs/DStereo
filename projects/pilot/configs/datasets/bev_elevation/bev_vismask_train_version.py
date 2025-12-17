# flake8: noqa
import os

from hat.utils import Config
from projects.pilot.configs.datasets.bev_elevation.utils import (
    del_item_in_dict,
    update_item_in_dict,
)

cfg_dir = os.path.dirname(__file__)
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
        "LX165": {
            "20230327_FSD_Site_LX165_20230308_v5a_autolabel_freespace_11v_filter_vismask_sequential_val": None,
        }
    }
}
v4_9 = {
    "FILTER_V4.7_BAD_DATA": {
        "4GD36": {
            "FSD_Site_4GD36_20220506_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_4GD36_20220516_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_4GD36_20220517_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_4GD36_20220518_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_4GD36_20220519_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_4GD36_20220520_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_4GD36_20220521_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_4GD36_20220523_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_4GD36_20220608_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_4GD36_20220609_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_4GD36_20220611_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_4GD36_20220613_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_4GD36_20220614_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_4GD36_20220615_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_4GD36_20220617_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_4GD36_20220618_v5a_autolabel_freespacescreen_iou_0309": None,
        },
        "BYD18": {
            "20230203_FSD_Site_BYD18_20221120_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD18_20221121_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD18_20221122_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD18_20221123_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            "20230203_FSD_Site_BYD18_20221124_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,
            # "20230203_FSD_Site_BYD18_20221205_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309": None,  验证集
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
        # "H3165": {
        #     "FSD_Site_H3165_20220513_v5a_autolabel_freespacescreen_iou_0309": None,
        #     "FSD_Site_H3165_20220514_v5a_autolabel_freespacescreen_iou_0309": None,
        #     "FSD_Site_H3165_20220516_v5a_autolabel_freespacescreen_iou_0309": None,
        # },  # 标定错误
        "UT0Q9": {
            "FSD_Site_UT0Q9_20220517_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_UT0Q9_20220518_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_UT0Q9_20220519_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_UT0Q9_20220521_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_UT0Q9_20220524_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_UT0Q9_20220528_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_UT0Q9_20220602_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_UT0Q9_20220604_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_UT0Q9_20220605_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_UT0Q9_20220609_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_UT0Q9_20220611_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_UT0Q9_20220615_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_UT0Q9_20220618_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_UT0Q9_20220620_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_UT0Q9_20220621_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_UT0Q9_20220629_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_UT0Q9_20220630_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_UT0Q9_20220701_v5a_autolabel_freespacescreen_iou_0309": None,
            "FSD_Site_UT0Q9_20220702_v5a_autolabel_freespacescreen_iou_0309": None,
        },
        "UT1R3": {
            "FSD_Site_UT1R3_20220505_v5a_autolabel_freespacescreen_iou_0309": None
        },
    }
}

# 11v数据用于鱼眼模型
v5_0_0_fisheye = {
    "FILTER_FSV4.0_BAD_DATA_11v": {
        "UT126": {
            "20230315_UT126_20220915_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230315_UT126_20221010_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230414_UT126_20230211_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230414_UT126_20230214_D_v5a_autolabel_freespace_11v_filter_vismask": None,
        },
        "UT1R3": {
            "20230602_UT1R3_20230301_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230602_UT1R3_20230315_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230602_UT1R3_20230317_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230602_UT1R3_20230318_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230602_UT1R3_20230320_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230602_UT1R3_20230321_D_v5a_autolabel_freespace_11v_filter_vismask": None,
        },
        "UT0F3": {
            "20230423_UT0F3_20230316_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230423_UT0F3_20230317_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230602_UT0F3_20230323_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230602_UT0F3_20230326_D_v5a_autolabel_freespace_11v_filter_vismask": None,
        },
        "LS830": {
            "20230427_LS830_20230306_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230427_LS830_20230307_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230427_LS830_20230323_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230427_LS830_20230324_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230427_LS830_20230325_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230427_LS830_20230328_D_v5a_autolabel_freespace_11v_filter_vismask": None,
        },
        "UT3J5": {
            "20230427_UT3J5_20230313_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230427_UT3J5_20230315_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230427_UT3J5_20230318_D_v5a_autolabel_freespace_11v_filter_vismask": None,
        },
        "UT263": {
            "20230414_UT263_20230214_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230602_UT263_20230317_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230602_UT263_20230318_D_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230602_UT263_20230320_D_v5a_autolabel_freespace_11v_filter_vismask": None,
        },
        "LX165": {
            "20230316_FSD_Site_LX165_20230226_v5a_autolabel_freespace_11v_filter_vismask": None,
            # "20230316_FSD_Site_LX165_20230227_v5a_autolabel_freespace_11v_filter_vismask": None,  # 测试集备用
            "20230316_FSD_Site_LX165_20230228_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230316_FSD_Site_LX165_20230301_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230327_FSD_Site_LX165_20230226_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230327_FSD_Site_LX165_20230228_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230327_FSD_Site_LX165_20230301_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230327_FSD_Site_LX165_20230306_v5a_autolabel_freespace_11v_filter_vismask": None,
            "20230327_FSD_Site_LX165_20230307_v5a_autolabel_freespace_11v_filter_vismask": None,
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
            "20230407_UT126_20230213_D_v5a_autolabel_freespace_11v": None,
        },
        "UT1R3": {
            "20230407_UT1R3_20230222_D_v5a_autolabel_freespace_11v": None,
        },
        "UT0F3": {
            "20230407_UT0F3_20230223_D_v5a_autolabel_freespace_11v": None,
        },
    },
    "FILTER_FSV4.1_BAD_DATA_11v": {
        "LS830": {
            "FSD_Site_LS830_20230216_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,
            "FSD_Site_LS830_20230217_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,
            "FSD_Site_LS830_20230218_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,
            "FSD_Site_LS830_20230219_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,
            "FSD_Site_LS830_20230220_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,
        },
        "LS912": {
            "FSD_Site_LS912_20230220_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,  # 测试集备用
            # "FSD_Site_LS912_20230328_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,  # 测试集备用
            "FSD_Site_LS912_20230329_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,
            "FSD_Site_LS912_20230331_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,
        },
        "LX165": {
            "FSD_Site_LX165_20230426_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,
            "FSD_Site_LX165_20230427_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,
        },
        "UT263": {
            "UT263_20230307_D_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,
            "UT263_20230308_D_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,
            "UT263_20230314_D_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,
            "UT263_20230315_D_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,
            "UT263_20230321_D_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,
            "UT263_20230510_D_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,
            "UT263_20230512_D_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,
        },
    },
}
v5_0_0 = update_item_in_dict(v5_0_0_fisheye, v4_9)
v5_0_0_update = {
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
v5_0_0 = update_item_in_dict(v5_0_0, v5_0_0_update)

# 16帧连续真值
v1_0_temporal = {
    "FILTER_V4.7_BAD_DATA_filter_ele_vismask": {
        "BYD18": {
            "20230203_FSD_Site_BYD18_20221120_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "20230203_FSD_Site_BYD18_20221121_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "20230203_FSD_Site_BYD18_20221122_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "20230203_FSD_Site_BYD18_20221123_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
            "20230203_FSD_Site_BYD18_20221124_v5a_autolabel_freespace_drop_stillness_7v_after_rejectscreen_iou_0309_total_gt": None,
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
            "20230407_UT126_20230213_D_v5a_autolabel_freespace_11v_total_gt": None,
        },
        "UT0F3": {
            "20230407_UT0F3_20230223_D_v5a_autolabel_freespace_11v_total_gt": None,
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
            "20230410_FSD_Site_BYD19_20230213_v5a_autolabel_freespace_7v_total_gt": None,
            "20230410_FSD_Site_BYD19_20230214_v5a_autolabel_freespace_7v_total_gt": None,
        },
    },
}
# 16帧gt
v1_1_temporal_update = {
    "SOURCE_VISMASK_V5.0.0_FISHEYE_DATA": {
        "UT126": {
            "20230315_UT126_20220915_D_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "20230315_UT126_20221010_D_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "20230414_UT126_20230211_D_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "20230414_UT126_20230214_D_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
        },
        "UT1R3": {
            "20230602_UT1R3_20230318_D_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
        },
        "UT0F3": {
            "20230423_UT0F3_20230316_D_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "20230423_UT0F3_20230317_D_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "20230602_UT0F3_20230323_D_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "20230602_UT0F3_20230326_D_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
        },
        "LS830": {
            "20230427_LS830_20230307_D_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "FSD_Site_LS830_20230216_sequential_drive_zhuxian_v5a_11v_filter_vismask_sequential": None,
            "FSD_Site_LS830_20230217_sequential_drive_zhuxian_v5a_11v_filter_vismask_sequential": None,
            "FSD_Site_LS830_20230218_sequential_drive_zhuxian_v5a_11v_filter_vismask_sequential": None,
            "FSD_Site_LS830_20230219_sequential_drive_zhuxian_v5a_11v_filter_vismask_sequential": None,
            "FSD_Site_LS830_20230220_sequential_drive_zhuxian_v5a_11v_filter_vismask_sequential": None,
        },
        "UT3J5": {
            "20230427_UT3J5_20230313_D_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "20230427_UT3J5_20230315_D_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "20230427_UT3J5_20230318_D_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
        },
        "UT263": {
            "20230602_UT263_20230317_D_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "20230602_UT263_20230318_D_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "20230602_UT263_20230320_D_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "UT263_20230307_D_sequential_drive_zhuxian_v5a_11v_filter_vismask_sequential": None,
            "UT263_20230308_D_sequential_drive_zhuxian_v5a_11v_filter_vismask_sequential": None,
            "UT263_20230314_D_sequential_drive_zhuxian_v5a_11v_filter_vismask_sequential": None,
            "UT263_20230315_D_sequential_drive_zhuxian_v5a_11v_filter_vismask_sequential": None,
            "UT263_20230321_D_sequential_drive_zhuxian_v5a_11v_filter_vismask_sequential": None,
            "UT263_20230512_D_sequential_drive_zhuxian_v5a_11v_filter_vismask_sequential": None,
        },
        "LX165": {
            "20230316_FSD_Site_LX165_20230226_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "20230316_FSD_Site_LX165_20230228_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "20230316_FSD_Site_LX165_20230301_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "20230327_FSD_Site_LX165_20230226_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "20230327_FSD_Site_LX165_20230228_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "20230327_FSD_Site_LX165_20230301_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "20230327_FSD_Site_LX165_20230306_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "20230327_FSD_Site_LX165_20230307_v5a_autolabel_freespace_11v_filter_vismask_sequential": None,
            "FSD_Site_LX165_20230426_sequential_drive_zhuxian_v5a_11v_filter_vismask_sequential": None,
            "FSD_Site_LX165_20230427_sequential_drive_zhuxian_v5a_11v_filter_vismask_sequential": None,
        },
        "LS912": {
            "FSD_Site_LS912_20230220_sequential_drive_zhuxian_v5a_11v_filter_vismask_sequential": None,
            "FSD_Site_LS912_20230329_sequential_drive_zhuxian_v5a_11v_filter_vismask_sequential": None,
            "FSD_Site_LS912_20230331_sequential_drive_zhuxian_v5a_11v_filter_vismask_sequential": None,
        },
    },
}
v1_1_temporal = update_item_in_dict(
    v1_0_temporal, v1_1_temporal_update, only_use_update_version
)

# 去除备用的测试集
v5_0_1_del = {
    "FILTER_FSV4.1_BAD_DATA_11v": {
        "LS912": {
            "FSD_Site_LS912_20230220_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,
        },
    },
}
v5_0_1_fisheye_del = {
    "FILTER_FSV4.1_BAD_DATA_11v": {
        "LS912": {
            "FSD_Site_LS912_20230220_sequential_drive_zhuxian_v5a_11v_filter_vismask": None,
        },
    },
}
v1_2_temporal_del = {
    "SOURCE_VISMASK_V5.0.0_FISHEYE_DATA": {
        "LS912": {
            "FSD_Site_LS912_20230220_sequential_drive_zhuxian_v5a_11v_filter_vismask_sequential": None,
        },
    },
}
v5_0_1 = del_item_in_dict(v5_0_0, v5_0_1_del)
v5_0_1_fisheye = del_item_in_dict(v5_0_0_fisheye, v5_0_1_fisheye_del)
v1_2_temporal = del_item_in_dict(v1_1_temporal, v1_2_temporal_del)

# 基于freespace v5.0.1_update数据质检vismask
v5_0_2_fisheye_update = {
    "FILTER_FSV5.1.0_UPDATE_BAD_DATA_11v": {
        "LS912": {
            "FSD_Site_LS912_20230330_drive_zhuxian_v5a_11v_fliter_vismask": None,
            "FSD_Site_LS912_20230403_drive_zhuxian_v5a_11v_fliter_vismask": None,
        },
        # "UT126": {
        #     "UT126_20230328_D_parking_zhuxian_v5a_11v_fliter_vismask": None,  # 全部不可见错误
        #     "UT126_20230330_D_parking_zhuxian_v5a_11v_fliter_vismask": None,  # 全部不可见错误
        # },
        "UT263": {
            "UT263_20230303_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT263_20230304_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT263_20230306_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            # "UT263_20230324_D_parking_zhuxian_v5a_11v_fliter_vismask": None,  # 全部不可见错误
            # "UT263_20230325_D_parking_zhuxian_v5a_11v_fliter_vismask": None,  # 全部不可见错误
            # "UT263_20230327_D_parking_zhuxian_v5a_11v_fliter_vismask": None,  # 全部不可见错误
            "UT263_20230421_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT263_20230422_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT263_20230423_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT263_20230427_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT263_20230504_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT263_20230505_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT263_20230506_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
        },
        "LX165": {
            "FSD_Site_LX165_20230309_drive_zhuxian_v5a_11v_fliter_vismask": None,
            "FSD_Site_LX165_20230310_drive_zhuxian_v5a_11v_fliter_vismask": None,
            "FSD_Site_LX165_20230311_drive_zhuxian_v5a_11v_fliter_vismask": None,
            "FSD_Site_LX165_20230313_drive_zhuxian_v5a_11v_fliter_vismask": None,
            "FSD_Site_LX165_20230320_drive_zhuxian_v5a_11v_fliter_vismask": None,
            "FSD_Site_LX165_20230321_drive_zhuxian_v5a_11v_fliter_vismask": None,
            "FSD_Site_LX165_20230322_drive_zhuxian_v5a_11v_fliter_vismask": None,
            "FSD_Site_LX165_20230323_drive_zhuxian_v5a_11v_fliter_vismask": None,
            "FSD_Site_LX165_20230406_drive_zhuxian_v5a_11v_fliter_vismask": None,
            "FSD_Site_LX165_20230407_drive_zhuxian_v5a_11v_fliter_vismask": None,
        },
        "UT370": {
            "UT370_20230517_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT370_20230519_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT370_20230520_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT370_20230521_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT370_20230522_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT370_20230523_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT370_20230524_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT370_20230526_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT370_20230527_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT370_20230528_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT370_20230529_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT370_20230530_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT370_20230531_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT370_20230601_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "UT370_20230602_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
        },
        "LS830": {
            "LS830_20230304_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "LS830_20230321_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "LS830_20230327_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "LS830_20230330_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "LS830_20230331_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "LS830_20230401_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "LS830_20230407_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "LS830_20230410_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "LS830_20230411_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "LS830_20230412_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
            "LS830_20230413_D_parking_zhuxian_v5a_11v_fliter_vismask": None,
        },
    },
}
v1_3_temporal_update = {
    "FILTER_FSV5.1.0_UPDATE_BAD_DATA_11v": {
        "LS912": {
            "FSD_Site_LS912_20230330_drive_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "FSD_Site_LS912_20230403_drive_zhuxian_v5a_11v_fliter_vismask_sequential": None,
        },
        # "UT126": {
        #     "UT126_20230328_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,  # 全部不可见错误
        #     "UT126_20230330_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,  # 全部不可见错误
        # },
        "UT263": {
            "UT263_20230303_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT263_20230304_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT263_20230306_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            # "UT263_20230324_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,  # 全部不可见错误
            # "UT263_20230325_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,  # 全部不可见错误
            "UT263_20230421_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT263_20230422_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT263_20230423_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT263_20230427_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT263_20230504_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT263_20230505_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT263_20230506_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
        },
        "LX165": {
            "FSD_Site_LX165_20230309_drive_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "FSD_Site_LX165_20230310_drive_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "FSD_Site_LX165_20230311_drive_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "FSD_Site_LX165_20230313_drive_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "FSD_Site_LX165_20230320_drive_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "FSD_Site_LX165_20230321_drive_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "FSD_Site_LX165_20230322_drive_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "FSD_Site_LX165_20230323_drive_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "FSD_Site_LX165_20230406_drive_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "FSD_Site_LX165_20230407_drive_zhuxian_v5a_11v_fliter_vismask_sequential": None,
        },
        "UT370": {
            "UT370_20230517_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT370_20230519_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT370_20230520_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT370_20230521_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT370_20230522_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT370_20230523_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT370_20230524_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT370_20230526_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT370_20230528_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT370_20230529_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT370_20230530_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT370_20230531_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT370_20230601_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "UT370_20230602_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
        },
        "LS830": {
            "LS830_20230304_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "LS830_20230321_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "LS830_20230327_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "LS830_20230330_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "LS830_20230331_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "LS830_20230401_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "LS830_20230407_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "LS830_20230410_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "LS830_20230411_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "LS830_20230412_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
            "LS830_20230413_D_parking_zhuxian_v5a_11v_fliter_vismask_sequential": None,
        },
    },
}
v5_0_2_fisheye = update_item_in_dict(v5_0_1_fisheye, v5_0_2_fisheye_update)
v5_0_2 = update_item_in_dict(v5_0_1, v5_0_2_fisheye_update)
v1_3_temporal = update_item_in_dict(v1_2_temporal, v1_3_temporal_update)
