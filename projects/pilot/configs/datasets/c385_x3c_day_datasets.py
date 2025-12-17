import os

from base_2d_day_datasets import datapaths
from easydict import EasyDict

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
root = os.path.join(bucket_root, "matrix2")


# ----- Required -----
datapaths.update(
    dict(
        vehicle_3d_detection=dict(
            train_batch_size_per_ctx=12,
            train_data_paths=[
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220729_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220730_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220731_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220801_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220802_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220803_v05__no_front__.rec",  # noqa
                        # V3
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220804_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220805_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220807_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220808_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220809_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220810_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220811_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220812_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220813_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220814_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220815_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220816_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220817_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220818_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220821_v05__no_front__day_filtered.rec",  # noqa
                        # CA038
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20220819_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20220820_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20220821_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20220822_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20220823_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20220824_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20220825_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20220826_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20220827_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20220828_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20220830_v05__no_front__day_filtered.rec",  # noqa
                        # V4
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20220901_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20220902_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20220903_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05.1/CA038/data_CA038_20220905_v05.1__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05.1/CA038/data_CA038_20220906_v05.1__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05.1/CA038/data_CA038_20220907_v05.1__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05.1/CA038/data_CA038_20220908_v05.1__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05.1/CA038/data_CA038_20220909_v05.1__no_front__day_filtered.rec",  # noqa
                        # v5
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220822_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220823_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220824_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220825_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220826_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220827_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220828_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220829_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220830_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220901_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220902_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220903_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220904_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220905_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05.3/CA038/data_CA038_20220916_v05.3__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05.3/CA038/data_CA038_20220917_v05.3__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05.3/CA038/data_CA038_20220920_v05.3__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05.3/CA038/data_CA038_20220921_v05.3__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05.3/CA038/data_CA038_20220922_v05.3__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05.3/CA038/data_CA038_20221019_v05.3__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221009_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221010_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221025_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221026_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221028_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221030_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221031_v05__no_front__day_filtered.rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221101_v05__no_front__day_filtered.rec", #noqa
                        #     f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221102_v05__no_front__day_filtered.rec", #noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221103_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221104_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221105_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221106_v05__no_front__day_filtered.rec",  # noqa
                        #    # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221107_v05__no_front__day_filtered.rec", #noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221108_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221109_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221110_v05__no_front__day_filtered.rec",  # noqa
                        # v6 321393
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220822_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220823_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220824_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220825_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220826_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220827_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220828_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220829_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220830_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220901_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220902_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220903_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20220904_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20221221_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20221224_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20221225_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20221226_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20221227_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221025_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221026_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221028_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221030_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221031_v05__no_front__day_filtered.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220729_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220730_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220731_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220801_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220802_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220803_v05__no_front__day.anno.pb_rec",  # noqa
                        # V3
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220804_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220805_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220807_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220808_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220809_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220810_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220811_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220812_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220813_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220814_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220815_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220816_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220817_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220818_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220821_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # CA038
                        f"{root}/data/real_3d/filtered/v05/CA038/with_calib_all/data_CA038_20220819_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/with_calib_all/data_CA038_20220820_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/with_calib_all/data_CA038_20220821_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/with_calib_all/data_CA038_20220822_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/with_calib_all/data_CA038_20220823_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/with_calib_all/data_CA038_20220824_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/with_calib_all/data_CA038_20220825_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/with_calib_all/data_CA038_20220826_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/with_calib_all/data_CA038_20220827_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/with_calib_all/data_CA038_20220828_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/with_calib_all/data_CA038_20220830_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # V4
                        f"{root}/data/real_3d/filtered/v05/CA038/with_calib_all/data_CA038_20220901_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/with_calib_all/data_CA038_20220902_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/with_calib_all/data_CA038_20220903_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05.1/CA038/with_calib_all/data_CA038_20220905_v05.1__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05.1/CA038/with_calib_all/data_CA038_20220906_v05.1__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05.1/CA038/with_calib_all/data_CA038_20220907_v05.1__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05.1/CA038/with_calib_all/data_CA038_20220908_v05.1__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05.1/CA038/with_calib_all/data_CA038_20220909_v05.1__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05.1/CA038/with_calib_all/data_CA038_20220913_v05.1__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05.1/CA038/with_calib_all/data_CA038_20220915_v05.1__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05.1/CA038/with_calib_all/data_CA038_20220918_v05.1__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05.1/CA038/with_calib_all/data_CA038_20220919_v05.1__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05.1/CA038/with_calib_all/data_CA038_20220920_v05.1__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05.1/CA038/with_calib_all/data_CA038_20220921_v05.1__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05.1/CA038/with_calib_all/data_CA038_20220922_v05.1__no_front__day_filtered.anno.pb_rec",  # noqa
                        # v5
                        # f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220822_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220823_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220824_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220825_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220826_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220827_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220828_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220829_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220830_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220901_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220902_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220903_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220904_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220905_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05.3/CA038/data_CA038_20220916_v05.3__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05.3/CA038/data_CA038_20220917_v05.3__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05.3/CA038/data_CA038_20220920_v05.3__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05.3/CA038/data_CA038_20220921_v05.3__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05.3/CA038/data_CA038_20220922_v05.3__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05.3/CA038/data_CA038_20221019_v05.3__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221009_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221010_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221025_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221026_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221028_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221030_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221031_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221101_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        #     f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221102_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221103_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221104_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221105_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221106_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        #    # f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221107_v05__no_front__day_filtered.anno.pb_rec", #noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221108_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221109_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/data_CA110_20221110_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # v6
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220822_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220823_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220824_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220825_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220826_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220827_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220828_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220829_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220830_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220901_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220902_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220903_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20220904_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20221221_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20221224_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20221225_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20221226_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA038/data_CA038_20221227_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20221025_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20221026_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20221028_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20221030_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/CA110/with_calib_all/data_CA110_20221031_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                    ],
                    sample_weight=16,
                ),
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/data_CA038_20220819_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/data_CA038_20220820_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/data_CA038_20220821_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/data_CA038_20220822_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/data_CA038_20220823_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/data_CA038_20220824_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/data_CA038_20220825_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/data_CA038_20220826_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/data_CA038_20220827_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/data_CA038_20220828_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/data_CA038_20220829_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/data_CA038_20220830_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220723_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220724_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220725_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220726_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220801_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220802_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220803_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220804_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220805_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220807_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220808_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220809_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220810_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220811_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220812_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220813_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220814_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220815_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220816_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220817_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220818_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220821_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220822_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220823_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220824_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220825_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220826_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220827_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220828_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220829_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/data_CA110_20220830_v05__rotation_badcase__day_.rec",  # noqa
                        # v6
                        f"{root}/data/real_3d/badcase/v05/CA038/vehicle/rotation_badcase/rotation_badcase_data_CA038_20221221_v05__nofront_day__.rec",  # noqa
                        f"{root}/data/real_3d/badcase/v05/CA038/vehicle/rotation_badcase/rotation_badcase_data_CA038_20221224_v05__nofront_day__.rec",  # noqa
                        f"{root}/data/real_3d/badcase/v05/CA038/vehicle/rotation_badcase/rotation_badcase_data_CA038_20221225_v05__nofront_day__.rec",  # noqa
                        f"{root}/data/real_3d/badcase/v05/CA038/vehicle/rotation_badcase/rotation_badcase_data_CA038_20221226_v05__nofront_day__.rec",  # noqa
                        f"{root}/data/real_3d/badcase/v05/CA038/vehicle/rotation_badcase/rotation_badcase_data_CA038_20221227_v05__nofront_day__.rec",  # noqa
                        f"{root}/data/real_3d/badcase/v05/CA038/vehicle/rotation_badcase/rotation_badcase_data_CA038_20221228_v05__nofront_day__.rec",  # noqa
                        f"{root}/data/real_3d/badcase/v05/CA038/vehicle/rotation_badcase/rotation_badcase_data_CA038_20221229_v05__nofront_day__.rec",  # noqa
                        f"{root}/data/real_3d/badcase/v05/CA038/vehicle/rotation_badcase/rotation_badcase_data_CA038_20221230_v05__nofront_day__.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/with_calib_all/data_CA038_20220819_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/with_calib_all/data_CA038_20220820_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/with_calib_all/data_CA038_20220821_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/with_calib_all/data_CA038_20220822_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/with_calib_all/data_CA038_20220823_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/with_calib_all/data_CA038_20220824_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/with_calib_all/data_CA038_20220825_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/with_calib_all/data_CA038_20220826_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/with_calib_all/data_CA038_20220827_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/with_calib_all/data_CA038_20220828_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/with_calib_all/data_CA038_20220829_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA038/with_calib_all/data_CA038_20220830_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220723_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220724_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220725_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220726_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220801_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220802_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220803_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220804_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220805_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220807_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220808_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220809_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220810_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220811_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220812_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220813_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220814_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220815_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220816_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220817_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220818_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220821_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220822_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220823_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220824_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220825_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220826_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220827_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220828_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220829_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/rotation_badcase/CA110/with_calib_all/data_CA110_20220830_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        # v6 add
                        f"{root}/data/real_3d/badcase/v05/CA038/vehicle/rotation_badcase/rotation_badcase_data_CA038_20221221_v05__nofront_day__.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/v05/CA038/vehicle/rotation_badcase/rotation_badcase_data_CA038_20221224_v05__nofront_day__.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/v05/CA038/vehicle/rotation_badcase/rotation_badcase_data_CA038_20221225_v05__nofront_day__.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/v05/CA038/vehicle/rotation_badcase/rotation_badcase_data_CA038_20221226_v05__nofront_day__.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/v05/CA038/vehicle/rotation_badcase/rotation_badcase_data_CA038_20221227_v05__nofront_day__.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/v05/CA038/vehicle/rotation_badcase/rotation_badcase_data_CA038_20221228_v05__nofront_day__.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/v05/CA038/vehicle/rotation_badcase/rotation_badcase_data_CA038_20221229_v05__nofront_day__.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/v05/CA038/vehicle/rotation_badcase/rotation_badcase_data_CA038_20221230_v05__nofront_day__.anno.pb_rec",  # noqa
                    ],
                    sample_weight=0.7,
                ),
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/data_vehicle_CA038_20220819_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/data_vehicle_CA038_20220820_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/data_vehicle_CA038_20220821_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/data_vehicle_CA038_20220822_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/data_vehicle_CA038_20220823_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/data_vehicle_CA038_20220824_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/data_vehicle_CA038_20220825_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/data_vehicle_CA038_20220826_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/data_vehicle_CA038_20220827_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/data_vehicle_CA038_20220828_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/data_vehicle_CA038_20220829_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/data_vehicle_CA038_20220830_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220723_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220724_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220725_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220726_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220801_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220802_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220803_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220804_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220805_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220807_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220808_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220809_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220810_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220811_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220812_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220813_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220814_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220815_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220816_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220817_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220818_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220821_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220822_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220823_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220824_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220825_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220826_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220827_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220828_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220829_v05__truncation__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/data_vehicle_CA110_20220830_v05__truncation__day_.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/with_calib_all/data_vehicle_CA038_20220819_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/with_calib_all/data_vehicle_CA038_20220820_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/with_calib_all/data_vehicle_CA038_20220821_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/with_calib_all/data_vehicle_CA038_20220822_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/with_calib_all/data_vehicle_CA038_20220823_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/with_calib_all/data_vehicle_CA038_20220824_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/with_calib_all/data_vehicle_CA038_20220825_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/with_calib_all/data_vehicle_CA038_20220826_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/with_calib_all/data_vehicle_CA038_20220827_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/with_calib_all/data_vehicle_CA038_20220828_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/with_calib_all/data_vehicle_CA038_20220829_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA038/with_calib_all/data_vehicle_CA038_20220830_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220723_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220724_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220725_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220726_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220801_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220802_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220803_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220804_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220805_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220807_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220808_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220809_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220810_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220811_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220812_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220813_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220814_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220815_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220816_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220817_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220818_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220821_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220822_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220823_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220824_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220825_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220826_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220827_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220828_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220829_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/truncation_badcase/CA110/with_calib_all/data_vehicle_CA110_20220830_v05__truncation__day_.anno.pb_rec",  # noqa
                    ],
                    sample_weight=0.3,
                ),
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/badcase/specialCar/CA038/data_CA038_20220819_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/data_CA038_20220820_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/data_CA038_20220821_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/data_CA038_20220822_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/data_CA038_20220823_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/data_CA038_20220824_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/data_CA038_20220825_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/data_CA038_20220826_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/data_CA038_20220827_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/data_CA038_20220828_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/data_CA038_20220829_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/data_CA038_20220830_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220801_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220802_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220803_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220804_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220805_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220807_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220808_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220809_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220810_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220811_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220812_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220813_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220814_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220815_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220816_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220817_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220818_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220821_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220822_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220823_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220824_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220825_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220826_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220827_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220828_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220829_v05__special_vehicle__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/data_CA110_20220830_v05__special_vehicle__day_.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/data/real_3d/badcase/specialCar/CA038/with_calib_all/data_CA038_20220819_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/with_calib_all/data_CA038_20220820_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/with_calib_all/data_CA038_20220821_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/with_calib_all/data_CA038_20220822_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/with_calib_all/data_CA038_20220823_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/with_calib_all/data_CA038_20220824_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/with_calib_all/data_CA038_20220825_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/with_calib_all/data_CA038_20220826_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/with_calib_all/data_CA038_20220827_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/with_calib_all/data_CA038_20220828_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/with_calib_all/data_CA038_20220829_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA038/with_calib_all/data_CA038_20220830_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220801_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220802_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220803_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220804_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220805_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220807_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220808_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220809_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220810_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220811_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220812_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220813_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220814_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220815_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220816_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220817_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220818_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220821_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220822_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220823_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220824_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220825_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220826_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220827_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220828_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220829_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/specialCar/CA110/with_calib_all/data_CA110_20220830_v05__special_vehicle__day_.anno.pb_rec",  # noqa
                    ],
                    sample_weight=0.3,
                ),
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20220801_v05__occlusion_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20220802_v05__occlusion_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20220803_v05__occlusion_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20220804_v05__occlusion_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20220805_v05__occlusion_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20220807_v05__occlusion_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20220808_v05__occlusion_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20220809_v05__occlusion_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20220810_v05__occlusion_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20220811_v05__occlusion_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20220812_v05__occlusion_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20220813_v05__occlusion_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20220814_v05__occlusion_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20220815_v05__occlusion_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20220816_v05__occlusion_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20220817_v05__occlusion_badcase__day_.rec",  # noqa
                        # f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20220818_v05__occlusion_badcase__day_.rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221010_v05__occlusion_badcase__day_.rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221101_v05__occlusion_badcase__day_.rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221102_v05__occlusion_badcase__day_.rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221103_v05__occlusion_badcase__day_.rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221104_v05__occlusion_badcase__day_.rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221105_v05__occlusion_badcase__day_.rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221106_v05__occlusion_badcase__day_.rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221107_v05__occlusion_badcase__day_.rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221108_v05__occlusion_badcase__day_.rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221109_v05__occlusion_badcase__day_.rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221110_v05__occlusion_badcase__day_.rec", #noqa
                    ],
                    anno_path=[
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/with_calib_all/data_CA110_20220801_v05__occlusion_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/with_calib_all/data_CA110_20220802_v05__occlusion_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/with_calib_all/data_CA110_20220803_v05__occlusion_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/with_calib_all/data_CA110_20220804_v05__occlusion_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/with_calib_all/data_CA110_20220805_v05__occlusion_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/with_calib_all/data_CA110_20220807_v05__occlusion_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/with_calib_all/data_CA110_20220808_v05__occlusion_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/with_calib_all/data_CA110_20220809_v05__occlusion_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/with_calib_all/data_CA110_20220810_v05__occlusion_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/with_calib_all/data_CA110_20220811_v05__occlusion_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/with_calib_all/data_CA110_20220812_v05__occlusion_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/with_calib_all/data_CA110_20220813_v05__occlusion_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/with_calib_all/data_CA110_20220814_v05__occlusion_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/with_calib_all/data_CA110_20220815_v05__occlusion_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/with_calib_all/data_CA110_20220816_v05__occlusion_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/with_calib_all/data_CA110_20220817_v05__occlusion_badcase__day_.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/with_calib_all/data_CA110_20220818_v05__occlusion_badcase__day_.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221010_v05__occlusion_badcase__day_.anno.pb_rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221101_v05__occlusion_badcase__day_.anno.pb_rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221102_v05__occlusion_badcase__day_.anno.pb_rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221103_v05__occlusion_badcase__day_.anno.pb_rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221104_v05__occlusion_badcase__day_.anno.pb_rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221105_v05__occlusion_badcase__day_.anno.pb_rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221106_v05__occlusion_badcase__day_.anno.pb_rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221107_v05__occlusion_badcase__day_.anno.pb_rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221108_v05__occlusion_badcase__day_.anno.pb_rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221109_v05__occlusion_badcase__day_.anno.pb_rec", #noqa
                        #         f"{root}/data/real_3d/badcase/occlusion_badcase/CA110/data_CA110_20221110_v05__occlusion_badcase__day_.anno.pb_rec", #noqa
                    ],
                    sample_weight=0.3,
                ),
            ],
            val_batch_size_per_ctx=10,
            val_data_paths=[
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220725_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220736_v05__no_front__.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220725_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/jianan.jiang/data/3d/split_data/v05/CA110/with_calib_all/data_CA110_20220736_v05__no_front__day.anno.pb_rec",  # noqa
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
                        # V2
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220729_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220730_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220731_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220801_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220802_v05__no_front__.rec",  # noqa
                        # V3
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220803_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220804_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220805_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220810_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220811_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220812_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220813_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220814_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220815_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220816_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220817_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220818_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220821_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA038/data_CA038_20220817_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA038/data_CA038_20220818_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA038/data_CA038_20220819_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA038/data_CA038_20220820_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA038/data_CA038_20220821_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA038/data_CA038_20220822_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA038/data_CA038_20220823_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA038/data_CA038_20220824_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA038/data_CA038_20220825_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA038/data_CA038_20220826_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA038/data_CA038_20220827_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA038/data_CA038_20220828_v05__no_front__.rec",  # noqa
                        # V4
                        f"{root}/data/real_3d/v05/CA038/data_CA038_20220829_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA038/data_CA038_20220830_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA038/data_CA038_20220901_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA038/data_CA038_20220902_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA038/data_CA038_20220903_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA038/data_CA038_20220904_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA038/data_CA038_20220905_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA038/data_CA038_20220906_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA038/data_CA038_20220907_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA038/data_CA038_20220908_v05.1__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05.1/CA038/data_CA038_20220909_v05.1__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05.1/CA038/data_CA038_20220913_v05.1__no_front__.rec", #noqa
                        # f"{root}/data/real_3d/v05.1/CA038/data_CA038_20220915_v05.1__no_front__.rec", #noqa
                        # f"{root}/data/real_3d/v05.1/CA038/data_CA038_20220918_v05.1__no_front__.rec", #noqa
                        # f"{root}/data/real_3d/v05.1/CA038/data_CA038_20220919_v05.1__no_front__.rec", #noqa
                        # f"{root}/data/real_3d/v05.1/CA038/data_CA038_20220920_v05.1__no_front__.rec", #noqa
                        # f"{root}/data/real_3d/v05.1/CA038/data_CA038_20220921_v05.1__no_front__.rec", #noqa
                        # f"{root}/data/real_3d/v05.1/CA038/data_CA038_20220922_v05.1__no_front__.rec", #noqa
                        #
                        # f"{root}/data/real_3d/v05/CA110/data_CA110_20220810_v05__no_front__.rec", #noqa
                        # f"{root}/data/real_3d/v05/CA110/data_CA110_20220811_v05__no_front__.rec", #noqa
                        # f"{root}/data/real_3d/v05/CA110/data_CA110_20220812_v05__no_front__.rec", #noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220813_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220814_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220815_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220816_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220817_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220818_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220821_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220822_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220823_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220824_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220825_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220826_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220827_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220828_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220829_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220830_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220831_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220901_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220902_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220903_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220904_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220905_v05__no_front__.rec",  # noqa
                        # v5
                        # f"{root}/data/real_3d/v05.3/CA038/data_CA038_20220915_v05.3__no_front__.rec", #noqa
                        # f"{root}/data/real_3d/v05.3/CA038/data_CA038_20220916_v05.3__no_front__.rec", #noqa
                        # f"{root}/data/real_3d/v05.3/CA038/data_CA038_20220917_v05.3__no_front__.rec", #noqa
                        # f"{root}/data/real_3d/v05.3/CA038/data_CA038_20220918_v05.3__no_front__.rec", #noqa
                        # f"{root}/data/real_3d/v05.3/CA038/data_CA038_20220919_v05.3__no_front__.rec", #noqa
                        # f"{root}/data/real_3d/v05.3/CA038/data_CA038_20220920_v05.3__no_front__.rec", #noqa
                        # f"{root}/data/real_3d/v05.3/CA038/data_CA038_20220921_v05.3__no_front__.rec", #noqa
                        # f"{root}/data/real_3d/v05.3/CA038/data_CA038_20220922_v05.3__no_front__.rec", #noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20221009_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20221010_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20221025_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20221026_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20221028_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20221030_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20221031_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20221101_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20221102_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20221103_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20221104_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20221105_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20221106_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20221107_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20221108_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20221109_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/CA110/data_CA110_20221110_v05__no_front__.rec",  # noqa
                        # v6
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220822_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220823_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220824_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220825_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220826_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220827_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220828_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220829_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220830_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220901_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220902_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220903_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220904_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/data_CA038_20221221_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/data_CA038_20221224_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/data_CA038_20221225_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/data_CA038_20221226_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/data_CA038_20221227_v05__no_front__day.rec",  # noqa
                    ],
                    anno_path=[
                        # V2
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/CA110/training_v1/with_calib_all/person_data_CA110_20220729_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/CA110/training_v1/with_calib_all/person_data_CA110_20220730_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/CA110/training_v1/with_calib_all/person_data_CA110_20220731_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/CA110/training_v1/with_calib_all/person_data_CA110_20220801_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/CA110/training_v1/with_calib_all/person_data_CA110_20220802_v05__no_front__day.anno.pb_rec",  # noqa
                        # V3
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/CA110/training_v1/with_calib_all/person_data_CA110_20220803_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/CA110/training_v1/with_calib_all/person_data_CA110_20220804_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/CA110/training_v1/with_calib_all/person_data_CA110_20220805_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220810_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220811_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220812_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220813_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220814_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220815_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220816_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220817_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220818_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220821_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/with_calib_all/data_CA038_20220817_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/with_calib_all/data_CA038_20220818_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/with_calib_all/data_CA038_20220819_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/with_calib_all/data_CA038_20220820_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/with_calib_all/data_CA038_20220821_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/with_calib_all/data_CA038_20220822_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/with_calib_all/data_CA038_20220823_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/with_calib_all/data_CA038_20220824_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/with_calib_all/data_CA038_20220825_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/with_calib_all/data_CA038_20220826_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/with_calib_all/data_CA038_20220827_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/with_calib_all/data_CA038_20220828_v05__no_front__day.anno.pb_rec",  # noqa
                        # V4
                        f"{root}/data/real_3d/person3d/v05/CA038/with_calib_all/data_CA038_20220829_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/with_calib_all/data_CA038_20220830_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/with_calib_all/data_CA038_20220901_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/with_calib_all/data_CA038_20220902_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/with_calib_all/data_CA038_20220903_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/with_calib_all/data_CA038_20220904_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA038/with_calib_all/data_CA038_20220905_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA038/with_calib_all/data_CA038_20220906_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA038/with_calib_all/data_CA038_20220907_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA038/with_calib_all/data_CA038_20220908_v05.1__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05.1/CA038/with_calib_all/data_CA038_20220909_v05.1__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/person3d/v05.1/CA038/with_calib_all/data_CA038_20220913_v05.1__no_front__day.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/person3d/v05.1/CA038/with_calib_all/data_CA038_20220915_v05.1__no_front__day.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/person3d/v05.1/CA038/with_calib_all/data_CA038_20220918_v05.1__no_front__day.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/person3d/v05.1/CA038/with_calib_all/data_CA038_20220919_v05.1__no_front__day.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/person3d/v05.1/CA038/with_calib_all/data_CA038_20220920_v05.1__no_front__day.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/person3d/v05.1/CA038/with_calib_all/data_CA038_20220921_v05.1__no_front__day.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/person3d/v05.1/CA038/with_calib_all/data_CA038_20220922_v05.1__no_front__day.anno.pb_rec", #noqa
                        #
                        # f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220810_v05__no_front__day.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220811_v05__no_front__day.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220812_v05__no_front__day.anno.pb_rec", #noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220813_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220814_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220815_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220816_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220817_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220818_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220821_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220822_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220823_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220824_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220825_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220826_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220827_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220828_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220829_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220830_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220831_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220901_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220902_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220903_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220904_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220905_v05__no_front__day.anno.pb_rec",  # noqa
                        # v5
                        # f"{root}/data/real_3d/person3d/v05.3/CA038/data_CA038_20220915_v05.3__no_front__day.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/person3d/v05.3/CA038/data_CA038_20220916_v05.3__no_front__day.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/person3d/v05.3/CA038/data_CA038_20220917_v05.3__no_front__day.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/person3d/v05.3/CA038/data_CA038_20220918_v05.3__no_front__day.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/person3d/v05.3/CA038/data_CA038_20220919_v05.3__no_front__day.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/person3d/v05.3/CA038/data_CA038_20220920_v05.3__no_front__day.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/person3d/v05.3/CA038/data_CA038_20220921_v05.3__no_front__day.anno.pb_rec", #noqa
                        # f"{root}/data/real_3d/person3d/v05.3/CA038/data_CA038_20220922_v05.3__no_front__day.anno.pb_rec", #noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221009_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221010_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221025_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221026_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221028_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221030_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221031_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221101_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221102_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221103_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221104_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221105_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221106_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221107_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221108_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221109_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/person3d/v05/CA110/data_CA110_20221110_v05__no_front__day.anno.pb_rec",  # noqa
                        # v6
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220822_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220823_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220824_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220825_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220826_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220827_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220828_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220829_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220830_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220901_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220902_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220903_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA110/with_calib_all/data_CA110_20220904_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/data_CA038_20221221_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/data_CA038_20221224_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/data_CA038_20221225_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/data_CA038_20221226_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/v05/CA038/data_CA038_20221227_v05__no_front__day.anno.pb_rec",  # noqa
                    ],
                    sample_weight=9,
                ),
            ],
            val_batch_size_per_ctx=10,
            val_data_paths=[
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/v05/CA110/data_CA110_20220725_v05__no_front__.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/CA110/training_v1/with_calib_all/person_with_calib_all/data_CA110_20220725_v05__no_front__day.anno.pb_rec",  # noqa
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
