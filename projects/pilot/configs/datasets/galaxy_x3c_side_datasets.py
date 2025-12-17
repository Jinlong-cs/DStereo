import os

from base_2d_galaxy_datasets import datapaths
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
                # x02 常规数据
                dict(
                    rec_path=[
                        # LX513 4.0 day
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220723_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220724_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220725_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220726_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220816_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220817_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220818_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220819_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220820_v05__no_front__day_filtered.rec",  # noqa
                        # LX513 4.0 night
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220723_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220724_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220725_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220726_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220814_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220816_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220817_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220818_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220819_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220820_v05__no_front__night_filtered.rec",  # noqa
                        # LX513 5.0 day
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220822_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220823_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220824_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220825_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220826_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220901_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220902_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220903_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220904_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220905_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220906_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220907_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220908_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220909_v05__no_front__day_filtered.rec",  # noqa
                        # LX513 5.0 night
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220821_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220822_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220823_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220824_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220825_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220826_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220901_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220902_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220905_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220906_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220907_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220908_v05__no_front__night_filtered.rec",  # noqa
                        # LX553 4.0 day
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220811_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220812_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220813_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220815_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220816_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220817_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220818_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220819_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220820_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220821_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220822_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220824_v05__no_front__day_filtered.rec",  # noqa
                        # LX553 4.0 night
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220811_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220812_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220813_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220814_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220815_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220816_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220817_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220818_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220819_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220820_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220821_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220822_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220824_v05__no_front__night_filtered.rec",  # noqa
                        # LX553 5.0 day
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220802_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220803_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220804_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220805_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220806_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220807_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220808_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220825_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220826_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220827_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220829_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220830_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220831_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220901_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220902_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220903_v05__no_front__day_filtered.rec",  # noqa
                        # LX553 5.0 night
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220802_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220803_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220804_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220805_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220806_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220807_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220808_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220825_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220826_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220827_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220828_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220829_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220830_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220831_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220901_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220902_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220903_v05__no_front__night_filtered.rec",  # noqa
                        # LX513 6.0 day
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220910_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220914_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220915_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220917_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220919_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220920_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220921_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220922_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220925_v05__no_front__day_filtered.rec",  # noqa
                        # LX513 6.0 night
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220915_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220921_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220922_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20220925_v05__no_front__night_filtered.rec",  # noqa
                        # LX553 6.0 day
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220905_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220906_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220916_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220917_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220919_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220920_v05__no_front__day_filtered.rec",  # noqa
                        # LX553 6.0 night
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220905_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20220906_v05__no_front__night_filtered.rec",  # noqa
                        # LX513 7.0 day
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221009_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221011_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221013_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221015_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221016_v05__no_front__day_filtered.rec",  # noqa
                        # LX513 7.0 night
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221009_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221011_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221013_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221015_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221016_v05__no_front__night_filtered.rec",  # noqa
                        # LX553 7.0 day
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221013_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221014_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221015_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221016_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221017_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221018_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221019_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221020_v05__no_front__day_filtered.rec",  # noqa
                        # LX553 7.0 night
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221013_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221014_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221015_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221016_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221017_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221018_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221019_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221020_v05__no_front__night_filtered.rec",  # noqa
                        # LX513 8.0 day
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221023_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221024_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221025_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221027_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221029_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221031_v05__no_front__day_filtered.rec",  # noqa
                        # LX513 8.0 night
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221023_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221024_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221025_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/data_LX513_20221029_v05__no_front__night_filtered.rec",  # noqa
                        # LX553 8.0 day
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221021_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221022_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221023_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221024_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221025_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221027_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221028_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221029_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221031_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221102_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221103_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221104_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221105_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221107_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221109_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221110_v05__no_front__day_filtered.rec",  # noqa
                        # LX553 8.0 night
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221021_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221022_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221023_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221024_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221025_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221027_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/data_LX553_20221029_v05__no_front__night_filtered.rec",  # noqa
                    ],
                    anno_path=[
                        # LX513 4.0 day
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220723_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220724_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220725_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220726_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220816_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220817_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220818_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220819_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220820_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # LX513 4.0 night
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220723_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220724_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220725_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220726_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220814_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220816_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220817_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220818_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220819_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220820_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # LX513 5.0 day
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220822_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220823_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220824_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220825_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220826_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220901_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220902_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220903_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220904_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220905_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220906_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220907_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220908_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220909_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # LX513 5.0 night
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220821_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220822_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220823_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220824_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220825_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220826_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220901_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220902_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220905_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220906_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220907_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220908_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # LX553 4.0 day
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220811_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220812_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220813_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220815_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220816_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220817_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220818_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220819_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220820_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220821_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220822_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220824_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # LX553 4.0 night
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220811_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220812_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220813_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220814_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220815_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220816_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220817_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220818_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220819_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220820_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220821_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220822_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220824_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # LX553 5.0 day
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220802_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220803_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220804_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220805_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220806_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220807_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220808_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220825_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220826_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220827_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220829_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220830_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220831_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220901_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220902_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220903_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # LX553 5.0 night
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220802_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220803_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220804_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220805_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220806_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220807_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220808_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220825_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220826_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220827_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220828_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220829_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220830_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220831_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220901_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220902_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220903_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # LX513 6.0 day
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220910_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220914_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220915_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220917_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220919_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220920_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220921_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220922_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220925_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # LX513 6.0 night
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220915_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220921_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220922_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220925_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # LX553 6.0 day
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220905_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220906_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220916_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220917_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220919_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220920_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # LX553 6.0 night
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220905_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220906_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # LX513 7.0 day
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221009_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221011_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221013_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221015_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221016_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # LX513 7.0 night
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221009_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221011_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221013_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221015_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221016_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # LX553 7.0 day
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221013_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221014_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221015_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221016_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221017_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221018_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221019_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221020_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # LX553 7.0 night
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221013_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221014_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221015_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221016_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221017_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221018_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221019_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221020_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # LX513 8.0 day
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221023_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221024_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221025_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221027_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221029_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221031_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # LX513 8.0 night
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221023_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221024_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221025_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221029_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # LX553 8.0 day
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221021_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221022_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221023_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221024_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221025_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221027_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221028_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221029_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221031_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221102_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221103_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221104_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221105_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221107_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221109_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221110_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # LX553 8.0 night
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221021_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221022_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221023_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221024_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221025_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221027_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221029_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                    ],
                    sample_weight=12.8,
                ),
                # x02 角度badcase
                dict(
                    rec_path=[
                        f"{root}/users/wenyuan.zeng/galaxy/rotation_badcase_4v/data_LX513_v05_rotation_badcase_day.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/rotation_badcase_4v/data_LX513_v05_rotation_badcase_night.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/rotation_badcase_4v/data_LX553_v05_rotation_badcase_day.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/rotation_badcase_4v/data_LX553_v05_rotation_badcase_night.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/rotation_badcase_4v/data_LX513_v05_rotation_badcase_10_11_day.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/rotation_badcase_4v/data_LX513_v05_rotation_badcase_10_11_night.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/rotation_badcase_4v/data_LX553_v05_rotation_badcase_10_11_day.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/rotation_badcase_4v/data_LX553_v05_rotation_badcase_10_11_night.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/users/wenyuan.zeng/galaxy/rotation_badcase_4v/data_LX513_v05_rotation_badcase_day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/rotation_badcase_4v/data_LX513_v05_rotation_badcase_night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/rotation_badcase_4v/data_LX553_v05_rotation_badcase_day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/rotation_badcase_4v/data_LX553_v05_rotation_badcase_night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/rotation_badcase_4v/data_LX513_v05_rotation_badcase_10_11_day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/rotation_badcase_4v/data_LX513_v05_rotation_badcase_10_11_night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/rotation_badcase_4v/data_LX553_v05_rotation_badcase_10_11_day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/rotation_badcase_4v/data_LX553_v05_rotation_badcase_10_11_night.anno.pb_rec",  # noqa
                    ],
                    sample_weight=0.63,
                ),
                # x02 异型车badcase
                dict(
                    rec_path=[
                        f"{root}/users/wenyuan.zeng/galaxy/special_car_4v/data_LX513_v05_special_car_day.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/special_car_4v/data_LX513_v05_special_car_night.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/special_car_4v/data_LX553_v05_special_car_day.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/special_car_4v/data_LX553_v05_special_car_night.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/users/wenyuan.zeng/galaxy/special_car_4v/data_LX513_v05_special_car_day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/special_car_4v/data_LX513_v05_special_car_night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/special_car_4v/data_LX553_v05_special_car_day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy/special_car_4v/data_LX553_v05_special_car_night.anno.pb_rec",  # noqa
                    ],
                    sample_weight=0.1,
                ),
                # x02 截断数据
                dict(
                    rec_path=[
                        # LX513 4.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220723_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220724_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220725_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220726_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220816_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220817_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220818_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220819_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220820_v05__truncation__day_.rec",  # noqa
                        # LX513 4.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220723_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220724_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220725_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220726_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220814_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220816_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220817_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220818_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220819_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220820_v05__truncation__night_.rec",  # noqa
                        # LX513 5.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220822_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220823_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220824_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220825_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220826_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220901_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220902_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220903_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220904_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220905_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220906_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220907_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220908_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220909_v05__truncation__day_.rec",  # noqa
                        # LX513 5.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220821_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220822_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220823_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220824_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220825_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220826_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220901_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220902_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220905_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220906_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220907_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220908_v05__truncation__night_.rec",  # noqa
                        # LX553 4.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220811_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220812_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220813_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220815_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220816_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220817_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220818_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220819_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220820_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220821_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220822_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220824_v05__truncation__day_.rec",  # noqa
                        # LX553 4.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220811_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220812_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220813_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220814_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220815_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220816_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220817_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220818_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220819_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220820_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220821_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220822_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220824_v05__truncation__night_.rec",  # noqa
                        # LX553 5.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220802_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220803_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220804_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220805_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220806_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220807_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220808_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220825_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220826_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220827_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220829_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220830_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220831_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220901_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220902_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220903_v05__truncation__day_.rec",  # noqa
                        # LX553 5.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220802_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220803_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220804_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220805_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220806_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220807_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220808_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220825_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220826_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220827_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220828_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220829_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220830_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220831_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220901_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220902_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220903_v05__truncation__night_.rec",  # noqa
                        # LX513 6.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220910_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220914_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220915_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220917_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220919_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220920_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220921_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220922_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220925_v05__truncation__day_.rec",  # noqa
                        # LX513 6.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220915_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220921_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220922_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220925_v05__truncation__night_.rec",  # noqa
                        # LX553 6.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220905_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220906_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220916_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220917_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220919_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220920_v05__truncation__day_.rec",  # noqa
                        # LX553 6.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220905_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220906_v05__truncation__night_.rec",  # noqa
                        # LX513 7.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221009_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221011_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221013_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221015_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221016_v05__truncation__day_.rec",  # noqa
                        # LX513 7.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221009_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221011_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221013_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221015_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221016_v05__truncation__night_.rec",  # noqa
                        # LX553 7.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221013_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221014_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221015_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221016_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221017_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221018_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221019_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221020_v05__truncation__day_.rec",  # noqa
                        # LX553 7.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221013_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221014_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221015_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221016_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221017_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221018_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221019_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221020_v05__truncation__night_.rec",  # noqa
                        # LX513 8.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221024_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221025_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221027_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221029_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221031_v05__truncation__day_.rec",  # noqa
                        # LX513 8.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221023_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221024_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221025_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221029_v05__truncation__night_.rec",  # noqa
                        # LX553 8.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221021_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221022_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221023_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221024_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221025_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221027_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221028_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221029_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221031_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221102_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221103_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221104_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221105_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221107_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221109_v05__truncation__day_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221110_v05__truncation__day_.rec",  # noqa
                        # LX553 8.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221021_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221022_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221023_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221024_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221025_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221027_v05__truncation__night_.rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221029_v05__truncation__night_.rec",  # noqa
                    ],
                    anno_path=[
                        # LX513 4.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220723_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220724_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220725_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220726_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220816_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220817_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220818_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220819_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220820_v05__truncation__day_.anno.pb_rec",  # noqa
                        # LX513 4.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220723_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220724_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220725_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220726_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220814_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220816_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220817_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220818_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220819_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220820_v05__truncation__night_.anno.pb_rec",  # noqa
                        # LX513 5.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220822_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220823_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220824_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220825_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220826_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220901_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220902_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220903_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220904_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220905_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220906_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220907_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220908_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220909_v05__truncation__day_.anno.pb_rec",  # noqa
                        # LX513 5.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220821_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220822_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220823_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220824_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220825_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220826_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220901_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220902_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220905_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220906_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220907_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220908_v05__truncation__night_.anno.pb_rec",  # noqa
                        # LX553 4.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220811_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220812_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220813_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220815_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220816_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220817_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220818_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220819_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220820_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220821_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220822_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220824_v05__truncation__day_.anno.pb_rec",  # noqa
                        # LX553 4.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220811_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220812_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220813_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220814_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220815_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220816_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220817_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220818_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220819_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220820_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220821_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220822_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220824_v05__truncation__night_.anno.pb_rec",  # noqa
                        # LX553 5.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220802_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220803_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220804_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220805_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220806_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220807_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220808_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220825_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220826_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220827_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220829_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220830_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220831_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220901_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220902_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220903_v05__truncation__day_.anno.pb_rec",  # noqa
                        # LX553 5.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220802_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220803_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220804_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220805_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220806_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220807_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220808_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220825_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220826_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220827_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220828_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220829_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220830_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220831_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220901_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220902_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220903_v05__truncation__night_.anno.pb_rec",  # noqa
                        # LX513 6.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220910_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220914_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220915_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220917_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220919_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220920_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220921_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220922_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220925_v05__truncation__day_.anno.pb_rec",  # noqa
                        # LX513 6.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220915_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220921_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220922_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220925_v05__truncation__night_.anno.pb_rec",  # noqa
                        # LX553 6.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220905_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220906_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220916_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220917_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220919_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220920_v05__truncation__day_.anno.pb_rec",  # noqa
                        # LX553 6.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220905_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220906_v05__truncation__night_.anno.pb_rec",  # noqa
                        # LX513 7.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221009_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221011_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221013_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221015_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221016_v05__truncation__day_.anno.pb_rec",  # noqa
                        # LX513 7.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221009_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221011_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221013_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221015_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221016_v05__truncation__night_.anno.pb_rec",  # noqa
                        # LX553 7.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221013_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221014_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221015_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221016_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221017_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221018_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221019_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221020_v05__truncation__day_.anno.pb_rec",  # noqa
                        # LX553 7.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221013_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221014_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221015_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221016_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221017_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221018_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221019_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221020_v05__truncation__night_.anno.pb_rec",  # noqa
                        # LX513 8.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221024_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221025_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221027_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221029_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221031_v05__truncation__day_.anno.pb_rec",  # noqa
                        # LX513 8.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221023_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221024_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221025_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221029_v05__truncation__night_.anno.pb_rec",  # noqa
                        # LX553 8.0 day
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221021_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221022_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221023_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221024_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221025_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221027_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221028_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221029_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221031_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221102_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221103_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221104_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221105_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221107_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221109_v05__truncation__day_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221110_v05__truncation__day_.anno.pb_rec",  # noqa
                        # LX553 8.0 night
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221021_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221022_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221023_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221024_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221025_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221027_v05__truncation__night_.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221029_v05__truncation__night_.anno.pb_rec",  # noqa
                    ],
                    sample_weight=1,
                ),
                # x03 常规数据
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221119_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221119_v05__no_front__night_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221120_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221120_v05__no_front__night_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221121_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221121_v05__no_front__night_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221122_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221123_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221123_v05__no_front__night_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221124_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221124_v05__no_front__night_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221125_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221126_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221127_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221128_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221129_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221130_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221124_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221124_v05__no_front__night_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221125_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221125_v05__no_front__night_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221126_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221126_v05__no_front__night_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221127_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221127_v05__no_front__night_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221128_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221128_v05__no_front__night_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221129_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221129_v05__no_front__night_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221130_v05__no_front__day_filtered.rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221130_v05__no_front__night_filtered.rec",
                    ],
                    anno_path=[
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221119_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221119_v05__no_front__night_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221120_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221120_v05__no_front__night_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221121_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221121_v05__no_front__night_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221122_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221123_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221123_v05__no_front__night_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221124_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221124_v05__no_front__night_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221125_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221126_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221127_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221128_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221129_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221130_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221124_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221124_v05__no_front__night_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221125_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221125_v05__no_front__night_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221126_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221126_v05__no_front__night_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221127_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221127_v05__no_front__night_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221128_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221128_v05__no_front__night_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221129_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221129_v05__no_front__night_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221130_v05__no_front__day_filtered.anno.pb_rec",
                        f"{root}/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221130_v05__no_front__night_filtered.anno.pb_rec",
                    ],
                    sample_weight=3.2,
                ),
            ],
            val_batch_size_per_ctx=10,
            val_data_paths=[
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220323_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220324_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220325_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220326_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220329_v05__no_front__.rec",  # noqa
                    ],
                    sample_weight=5,
                ),
            ],
        ),
        person_3d_detection=dict(
            train_batch_size_per_ctx=16,
            train_data_paths=[
                # x02
                dict(
                    rec_path=[
                        # LX513 4.0 day
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220723_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220724_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220725_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220726_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220816_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220817_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220818_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220819_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220820_v05__no_front__.rec",  # noqa
                        # LX513 4.0 night
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220723_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220724_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220725_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220726_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220814_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220816_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220817_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220818_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220819_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220820_v05__no_front__.rec",  # noqa
                        # LX513 5.0 day
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220822_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220823_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220824_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220825_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220826_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220901_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220902_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220903_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220904_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220905_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220906_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220907_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220908_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220909_v05__no_front__.rec",  # noqa
                        # LX513 5.0 night
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220821_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220822_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220823_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220824_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220825_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220826_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220901_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220902_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220905_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220906_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220907_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220908_v05__no_front__.rec",  # noqa
                        # LX553 4.0 day
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220811_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220812_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220813_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220815_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220816_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220817_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220818_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220819_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220820_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220821_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220822_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220824_v05__no_front__.rec",  # noqa
                        # LX553 4.0 night
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220811_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220812_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220813_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220815_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220816_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220817_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220818_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220819_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220820_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220821_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220822_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220824_v05__no_front__.rec",  # noqa
                        # LX553 5.0 day
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220802_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220803_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220804_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220805_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220806_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220807_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220808_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220825_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220826_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220827_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220829_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220830_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220901_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220902_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220903_v05__no_front__.rec",  # noqa
                        # LX553 5.0 night
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220802_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220803_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220804_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220805_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220806_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220807_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220808_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220825_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220826_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220827_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220828_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220829_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220830_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220831_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220901_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220902_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220903_v05__no_front__.rec",  # noqa
                        # LX513 6.0 day
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220910_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220912_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220914_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220915_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220917_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220919_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220921_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220922_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220925_v05__no_front__.rec",  # noqa
                        # LX513 6.0 night
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220912_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220915_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220921_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220922_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20220925_v05__no_front__.rec",  # noqa
                        # LX553 6.0 day
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220905_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220906_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220916_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220917_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220919_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220920_v05__no_front__.rec",  # noqa
                        # LX553 6.0 night
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220905_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20220906_v05__no_front__.rec",  # noqa
                        # LX513 7.0 day
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221009_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221011_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221013_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221015_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221016_v05__no_front__.rec",  # noqa
                        # LX513 7.0 night
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221009_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221011_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221013_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221015_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221016_v05__no_front__.rec",  # noqa
                        # LX553 7.0 day
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221013_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221014_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221015_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221017_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221018_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221019_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221020_v05__no_front__.rec",  # noqa
                        # LX553 7.0 night
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221013_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221014_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221015_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221017_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221018_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221019_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221020_v05__no_front__.rec",  # noqa
                        # LX513 8.0 day
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221023_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221024_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221025_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221027_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221029_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221031_v05__no_front__.rec",  # noqa
                        # LX513 8.0 night
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221023_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221024_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221025_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX513/data_LX513_20221029_v05__no_front__.rec",  # noqa
                        # LX553 8.0 day
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221021_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221022_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221023_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221024_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221025_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221027_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221028_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221029_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221031_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221102_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221103_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221104_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221105_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221107_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221109_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221110_v05__no_front__.rec",  # noqa
                        # LX553 8.0 night
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221021_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221022_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221023_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221024_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221025_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221027_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/LX553/data_LX553_20221029_v05__no_front__.rec",  # noqa
                        # x03
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221119_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221119_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221120_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221120_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221121_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221121_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221122_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221123_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221123_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221124_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221124_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221125_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221126_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221127_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221128_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221129_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221130_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221201_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221202_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221203_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221203_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221204_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221204_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221205_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221205_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221206_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221206_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221207_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221207_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221208_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221208_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221209_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221209_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221210_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221210_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221211_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221211_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221212_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221212_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221213_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221213_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221214_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221214_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221215_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221215_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221216_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221216_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221217_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221217_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221222_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221222_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221223_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221223_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221225_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221225_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221226_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221226_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221227_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221227_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221228_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221228_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221229_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221229_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221230_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221230_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221231_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221124_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221124_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221125_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221125_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221126_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221126_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221127_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221127_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221128_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221128_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221129_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221129_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221130_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221130_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221201_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221201_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221202_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221202_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221203_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221203_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221204_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221204_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221205_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221205_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221206_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221206_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221207_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221207_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221208_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221208_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221209_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221209_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221210_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221210_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221211_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221211_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221212_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221212_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221213_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221213_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221215_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221215_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221216_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221216_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221217_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221218_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221218_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221219_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221219_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221220_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221220_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221221_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221221_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221222_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221222_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221223_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221223_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221224_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221224_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221225_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221225_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221226_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221226_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221227_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221227_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221228_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221228_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221229_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221229_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221230_v05__no_front__day.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221230_v05__no_front__night.rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221231_v05__no_front__night.rec",
                    ],
                    anno_path=[
                        # LX513 4.0 day
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220723_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220724_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220725_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220726_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220816_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220817_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220818_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220819_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220820_v05__no_front__day.anno.pb_rec",  # noqa
                        # LX513 4.0 night
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220723_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220724_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220725_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220726_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220814_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220816_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220817_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220818_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220819_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220820_v05__no_front__night.anno.pb_rec",  # noqa
                        # LX513 5.0 day
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220822_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220823_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220824_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220825_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220826_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220901_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220902_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220903_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220904_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220905_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220906_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220907_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220908_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220909_v05__no_front__day.anno.pb_rec",  # noqa
                        # LX513 5.0 night
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220821_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220822_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220823_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220824_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220825_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220826_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220901_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220902_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220905_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220906_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220907_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220908_v05__no_front__night.anno.pb_rec",  # noqa
                        # LX553 4.0 day
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220811_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220812_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220813_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220815_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220816_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220817_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220818_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220819_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220820_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220821_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220822_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220824_v05__no_front__day.anno.pb_rec",  # noqa
                        # LX553 4.0 night
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220811_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220812_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220813_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220815_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220816_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220817_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220818_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220819_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220820_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220821_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220822_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220824_v05__no_front__night.anno.pb_rec",  # noqa
                        # LX553 5.0 day
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220802_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220803_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220804_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220805_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220806_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220807_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220808_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220825_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220826_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220827_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220829_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220830_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220901_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220902_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220903_v05__no_front__day.anno.pb_rec",  # noqa
                        # LX553 5.0 night
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220802_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220803_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220804_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220805_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220806_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220807_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220808_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220825_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220826_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220827_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220828_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220829_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220830_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220831_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220901_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220902_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220903_v05__no_front__night.anno.pb_rec",  # noqa
                        # LX513 6.0 day
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220910_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220912_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220914_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220915_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220917_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220919_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220921_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220922_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220925_v05__no_front__day.anno.pb_rec",  # noqa
                        # LX513 6.0 night
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220912_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220915_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220921_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220922_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20220925_v05__no_front__night.anno.pb_rec",  # noqa
                        # LX553 6.0 day
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220905_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220906_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220916_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220917_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220919_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220920_v05__no_front__day.anno.pb_rec",  # noqa
                        # LX553 6.0 night
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220905_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20220906_v05__no_front__night.anno.pb_rec",  # noqa
                        # LX513 7.0 day
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221009_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221011_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221013_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221015_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221016_v05__no_front__day.anno.pb_rec",  # noqa
                        # LX513 7.0 night
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221009_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221011_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221013_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221015_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221016_v05__no_front__night.anno.pb_rec",  # noqa
                        # LX553 7.0 day
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221013_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221014_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221015_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221017_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221018_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221019_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221020_v05__no_front__day.anno.pb_rec",  # noqa
                        # LX553 7.0 night
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221013_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221014_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221015_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221017_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221018_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221019_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221020_v05__no_front__night.anno.pb_rec",  # noqa
                        # LX513 8.0 day
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221023_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221024_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221025_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221027_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221029_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221031_v05__no_front__day.anno.pb_rec",  # noqa
                        # LX513 8.0 night
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221023_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221024_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221025_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX513/with_calib_all/data_person_LX513_20221029_v05__no_front__night.anno.pb_rec",  # noqa
                        # LX553 8.0 day
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221021_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221022_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221023_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221024_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221025_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221027_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221028_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221029_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221031_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221102_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221103_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221104_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221105_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221107_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221109_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221110_v05__no_front__day.anno.pb_rec",  # noqa
                        # LX553 8.0 night
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221021_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221022_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221023_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221024_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221025_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221027_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/users/wenyuan.zeng/data/3d/split_data_person/v05/galaxy/4v/LX553/with_calib_all/data_person_LX553_20221029_v05__no_front__night.anno.pb_rec",  # noqa
                        # x03
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221119_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221119_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221120_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221120_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221121_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221121_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221122_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221123_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221123_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221124_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221124_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221125_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221126_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221127_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221128_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221129_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221130_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221201_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221202_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221203_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221203_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221204_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221204_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221205_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221205_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221206_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221206_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221207_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221207_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221208_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221208_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221209_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221209_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221210_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221210_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221211_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221211_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221212_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221212_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221213_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221213_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221214_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221214_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221215_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221215_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221216_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221216_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221217_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221217_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221222_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221222_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221223_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221223_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221225_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221225_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221226_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221226_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221227_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221227_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221228_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221228_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221229_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221229_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221230_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221230_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX066/side/data_LX066_20221231_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221124_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221124_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221125_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221125_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221126_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221126_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221127_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221127_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221128_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221128_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221129_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221129_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221130_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221130_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221201_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221201_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221202_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221202_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221203_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221203_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221204_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221204_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221205_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221205_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221206_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221206_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221207_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221207_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221208_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221208_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221209_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221209_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221210_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221210_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221211_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221211_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221212_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221212_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221213_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221213_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221215_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221215_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221216_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221216_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221217_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221218_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221218_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221219_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221219_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221220_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221220_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221221_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221221_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221222_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221222_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221223_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221223_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221224_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221224_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221225_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221225_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221226_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221226_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221227_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221227_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221228_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221228_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221229_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221229_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221230_v05__no_front__day.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221230_v05__no_front__night.anno.pb_rec",
                        f"{root}/data/real_3d/person3d/v05/LX038/side/data_LX038_20221231_v05__no_front__night.anno.pb_rec",
                    ],
                    sample_weight=9,
                ),
            ],
            val_batch_size_per_ctx=10,
            val_data_paths=[
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210503_v04__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210504_v04__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210505_v04__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210515_v04__no_front__.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_ignore_false_day/v04/65U3D/data_65U3D_20210503_v04__no_front__.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_ignore_false_day/v04/65U3D/data_65U3D_20210504_v04__no_front__.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_ignore_false_day/v04/65U3D/data_65U3D_20210505_v04__no_front__.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_ignore_false_day/v04/65U3D/data_65U3D_20210515_v04__no_front__.anno.pb_rec",  # noqa
                    ],
                    sample_weight=5,
                ),
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210504_v04__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210505_v04__no_front__.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_ignore_false_day/v04/30C1G/data_30C1G_20210504_v04__no_front__.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_ignore_false_day/v04/30C1G/data_30C1G_20210505_v04__no_front__.anno.pb_rec",  # noqa
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
