import os

from base_2d_parking_datasets import datapaths
from easydict import EasyDict

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
root = os.path.join(bucket_root, "matrix2")


# ----- Required -----
datapaths.update(
    dict(
        vehicle_3d_detection=dict(
            train_batch_size_per_ctx=16,
            train_data_paths=[
                dict(
                    rec_path=[
                        # UV135
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220506_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220507_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220507_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220508_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220509_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220509_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220510_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220510_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220511_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220511_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220512_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220512_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220513_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/UV135/data_UV135_20220516_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/UV135/data_UV135_20220516_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/UV135/data_UV135_20220517_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/UV135/data_UV135_20220517_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/UV135/data_UV135_20220518_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/UV135/data_UV135_20220518_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220520_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220521_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220523_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220525_v05__no_front__.rec",  # noqa
                        # CA110
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220609_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220610_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220611_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220613_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220614_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220615_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220616_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220617_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220618_v05__no_front__.rec",  # noqa
                        # V2
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220620_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220622_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220623_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220629_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220630_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220701_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220702_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220703_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220704_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220705_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220706_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220707_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220708_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220709_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220710_v05__no_front__.rec",  # noqa
                        # V3
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220810_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220811_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220812_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220814_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220817_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220818_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220819_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220820_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220821_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220810_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220811_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220812_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220814_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220818_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220819_v05__no_front__.rec",  # noqa
                        # V4
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220824_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220825_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220826_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220827_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220828_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220905_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220906_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220907_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220908_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220909_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220912_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220913_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220914_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220915_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220916_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220917_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220918_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220919_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220920_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220921_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220824_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220825_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220826_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220827_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220828_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220905_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220906_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220907_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220908_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220909_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220912_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220913_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220914_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220915_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220916_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220917_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220918_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220919_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220920_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220921_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220922_v05.1__no_front__.rec",  # noqa
                        # v5
                        # f"{root}/data/real_3d/v05/CA138/data_CA138_20221019_v05__no_front__.rec", #noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221025_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/CA138/data_CA138_20221028_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221029_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221030_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221031_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/CA138/data_CA138_20221101_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221102_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221103_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221104_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221105_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221106_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221107_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/CA138/data_CA138_20221108_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221109_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221110_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221111_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221027_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/CA138/data_CA138_20221028_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221029_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221030_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221031_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/CA138/data_CA138_20221101_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/CA138/data_CA138_20221102_v05__no_front__.rec", #noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221103_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221104_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/CA138/data_CA138_20221105_v05__no_front__.rec", #noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221106_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221107_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/CA138/data_CA138_20221108_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/CA138/data_CA138_20221109_v05__no_front__.rec", #noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221110_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221111_v05__no_front__.rec",  # noqa
                        # v6 137534
                        f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221217_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221218_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221219_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221220_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221221_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221222_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221223_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221224_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221225_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221226_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221227_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221228_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221229_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221201_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221202_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221203_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221204_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221205_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221206_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221207_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221209_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221211_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221212_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221213_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221214_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221216_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221217_v05__no_front__day_filtered.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220506_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220507_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220507_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220508_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220509_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220509_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220510_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220510_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220511_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220511_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220512_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220512_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220513_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220516_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220516_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220517_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220517_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220518_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220518_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220520_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220521_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220523_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/UV135/with_calib_all/data_UV135_20220525_v05__no_front__day.anno.pb_rec",  # noqa
                        # CA110
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220609_v05__no_front__day.anno.pb_rec",
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220610_v05__no_front__day.anno.pb_rec",
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220611_v05__no_front__day.anno.pb_rec",
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220613_v05__no_front__day.anno.pb_rec",
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220614_v05__no_front__day.anno.pb_rec",
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220615_v05__no_front__day.anno.pb_rec",
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220616_v05__no_front__day.anno.pb_rec",
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220617_v05__no_front__day.anno.pb_rec",
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220618_v05__no_front__day.anno.pb_rec",
                        # V2
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220620_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220622_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220623_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220629_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220630_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220701_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220702_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220703_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220704_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220705_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220706_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220707_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220708_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220709_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220710_v05__no_front__day.anno.pb_rec",  # noqa
                        # V3
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/with_calib_all/data_CA138_20220810_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/with_calib_all/data_CA138_20220811_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/with_calib_all/data_CA138_20220812_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/with_calib_all/data_CA138_20220814_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/with_calib_all/data_CA138_20220817_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/with_calib_all/data_CA138_20220818_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/with_calib_all/data_CA138_20220819_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/with_calib_all/data_CA138_20220820_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/with_calib_all/data_CA138_20220821_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/with_calib_all/data_CA138_20220810_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/with_calib_all/data_CA138_20220811_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/with_calib_all/data_CA138_20220812_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/with_calib_all/data_CA138_20220814_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/with_calib_all/data_CA138_20220818_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/with_calib_all/data_CA138_20220819_v05__no_front__night.anno.pb_rec",  # noqa
                        # V4
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220824_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220825_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220826_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220827_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220828_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220905_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220906_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220907_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220908_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220909_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220912_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220913_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220914_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220915_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220916_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220917_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220918_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220919_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220920_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220921_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220824_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220825_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220826_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220827_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220828_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220905_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220906_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220907_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220908_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220909_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220912_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220913_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220914_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220915_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220916_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220917_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220918_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220919_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220920_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220921_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05.1/CA138/with_calib_all/data_CA138_20220922_v05.1__no_front__day.anno.pb_rec",  # noqa
                        # v5
                        # f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221019_v05__no_front__day.anno.pb_rec", #noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221025_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221028_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221029_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221030_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221031_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221101_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221102_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221103_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221104_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221105_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221106_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221107_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221108_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221109_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221110_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221111_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221027_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221028_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221029_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221030_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221031_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221101_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221102_v05__no_front__night.anno.pb_rec", #noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221103_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221104_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221105_v05__no_front__night.anno.pb_rec", #noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221106_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221107_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221108_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221109_v05__no_front__night.anno.pb_rec", #noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221110_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA138/data_CA138_20221111_v05__no_front__night.anno.pb_rec",  # noqa
                        # v6
                        f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221217_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221218_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221219_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221220_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221221_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221222_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221223_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221224_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221225_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221226_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221227_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221228_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/CA138/data_CA138_20221229_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221201_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221202_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221203_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221204_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221205_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221206_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221207_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221209_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221211_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221212_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221213_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221214_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221216_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221217_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                    ],
                    sample_weight=16,
                ),
            ],
            val_batch_size_per_ctx=10,
            val_data_paths=[
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/v05/UV051/data_UV051_20220507_v05__no_front__.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/UV051/with_calib_all//data_UV051_20220507_v05__no_front__day.anno.pb_rec",  # noqa
                    ],
                    sample_weight=5,
                ),
            ],
        ),
        person_3d_detection=dict(
            train_batch_size_per_ctx=16,
            train_data_paths=[
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220609_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220610_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220611_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220613_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220506_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220508_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220511_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220511_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220512_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220512_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220513_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220516_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220516_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220517_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220517_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220518_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220518_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220520_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220521_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220523_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/UV135/data_UV135_20220525_v05__no_front__.rec",  # noqa
                        # V2
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220609_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220610_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220611_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220613_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220614_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220615_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220616_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220617_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220618_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220620_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220622_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220623_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220629_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220630_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220701_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220702_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220703_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220704_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220705_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220706_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220707_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220708_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220709_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220710_v05__no_front__.rec",  # noqa
                        # V3
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220810_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220811_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220812_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220814_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220818_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220819_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220820_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20220821_v05__no_front__.rec",  # noqa
                        # V4
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220824_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220825_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220826_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220827_v05.1__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220828_v05.1__no_front__.rec", #noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220905_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220906_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220907_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220908_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220909_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220912_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220913_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220914_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220915_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220916_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220917_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220918_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220919_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220920_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220921_v05.1__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220922_v05.1__no_front__.rec", #noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220824_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220825_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220826_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220827_v05.1__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220828_v05.1__no_front__.rec", #noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220905_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220906_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220907_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220908_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220909_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220912_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220913_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220914_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220915_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220916_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220917_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220918_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220919_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220920_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA138/data_CA138_20220921_v05.1__no_front__.rec",  # noqa
                        # v5
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221019_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221025_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221028_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221029_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221030_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221031_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221101_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221102_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221103_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221104_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221105_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221106_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221107_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221108_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221109_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221110_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221111_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221027_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221028_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221029_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221030_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221031_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221101_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221103_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221104_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221105_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221106_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221107_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221108_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221109_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221110_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA138/data_CA138_20221111_v05__no_front__.rec",  # noqa
                        # v6 22284
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221217_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221219_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221220_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221221_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221222_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221225_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221226_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221228_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221201_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221202_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221203_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221204_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221205_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221206_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221207_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221209_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221211_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221212_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221213_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221214_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221216_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221217_v05__no_front__day.rec",
                    ],
                    anno_path=[
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/CA110/with_calib_all/person_data_CA110_20220609_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/CA110/with_calib_all/person_data_CA110_20220610_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/CA110/with_calib_all/person_data_CA110_20220611_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/CA110/with_calib_all/person_data_CA110_20220613_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/UV135/with_calib_all/person_data_UV135_20220506_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/UV135/with_calib_all/person_data_UV135_20220508_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/UV135/with_calib_all/person_data_UV135_20220511_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/UV135/with_calib_all/person_data_UV135_20220511_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/UV135/with_calib_all/person_data_UV135_20220512_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/UV135/with_calib_all/person_data_UV135_20220512_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/UV135/with_calib_all/person_data_UV135_20220513_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/UV135/with_calib_all/person_data_UV135_20220516_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/UV135/with_calib_all/person_data_UV135_20220516_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/UV135/with_calib_all/person_data_UV135_20220517_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/UV135/with_calib_all/person_data_UV135_20220517_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/UV135/with_calib_all/person_data_UV135_20220518_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/UV135/with_calib_all/person_data_UV135_20220518_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/UV135/with_calib_all/person_data_UV135_20220520_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/UV135/with_calib_all/person_data_UV135_20220521_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/UV135/with_calib_all/person_data_UV135_20220523_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/UV135/with_calib_all/person_data_UV135_20220525_v05__no_front__day.anno.pb_rec",  # noqa
                        # V2
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220609_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220610_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220611_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220613_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220614_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220615_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220616_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220617_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220618_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220620_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220622_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220623_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220629_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220630_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220701_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220702_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220703_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220704_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220705_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220706_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220707_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220708_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220709_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V2/CA110/with_calib_all/person_data_CA110_20220710_v05__no_front__day.anno.pb_rec",  # noqa
                        # V3
                        f"{root}/data/real_3d/person3d/v05/CA138/with_calib_all/data_CA138_20220810_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/with_calib_all/data_CA138_20220811_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/with_calib_all/data_CA138_20220812_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/with_calib_all/data_CA138_20220814_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/with_calib_all/data_CA138_20220818_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/with_calib_all/data_CA138_20220819_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/with_calib_all/data_CA138_20220820_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/with_calib_all/data_CA138_20220821_v05__no_front__day.anno.pb_rec",  # noqa
                        # V4
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220824_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220825_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220826_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220827_v05.1__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220828_v05.1__no_front__day.anno.pb_rec", #noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220905_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220906_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220907_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220908_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220909_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220912_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220913_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220914_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220915_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220916_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220917_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220918_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220919_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220920_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220921_v05.1__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220922_v05.1__no_front__day.anno.pb_rec", #noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220824_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220825_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220826_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220827_v05.1__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220828_v05.1__no_front__night.anno.pb_rec", #noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220905_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220906_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220907_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220908_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220909_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220912_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220913_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220914_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220915_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220916_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220917_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220918_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220919_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220920_v05.1__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA138/with_calib_all/data_CA138_20220921_v05.1__no_front__night.anno.pb_rec",  # noqa
                        # v5
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221019_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221025_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221028_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221029_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221030_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221031_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221101_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221102_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221103_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221104_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221105_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221106_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221107_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221108_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221109_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221110_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221111_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221027_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221028_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221029_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221030_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221031_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221101_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221103_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221104_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221105_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221106_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221107_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221108_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221109_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221110_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221111_v05__no_front__night.anno.pb_rec",  # noqa
                        # v6 22284
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221217_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221219_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221220_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221221_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221222_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221225_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221226_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA138/data_CA138_20221228_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221201_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221202_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221203_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221204_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221205_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221206_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221207_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221209_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221211_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221212_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221213_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221214_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221216_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221217_v05__no_front__day.anno.pb_rec",
                    ],
                    sample_weight=9,
                ),
            ],
            val_batch_size_per_ctx=10,
            val_data_paths=[
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/v05/UV135/data_UV135_20220509_v05__no_front__.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/C385V1/UV135/with_calib_all/person_data_UV135_20220509_v05__no_front__day.anno.pb_rec",  # noqa
                    ],
                    sample_weight=5,
                ),
            ],
        ),
    )
)


datapaths = EasyDict(datapaths)
buckets = [
    "matrix2",
]
