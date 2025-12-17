import os

from easydict import EasyDict

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
root = os.path.join(bucket_root, "matrix")

datapaths = dict(
    multiview_dynamic_3d_detection=dict(
        train_data_paths=[
            dict(
                rec_path=[
                    # f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220726_v05__no_front__.rec",  # noqa
                    # f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220727_v05__no_front__.rec",  # noqa
                    # f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220728_v05__no_front__.rec",  # noqa
                    # f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220729_v05__no_front__.rec",  # noqa
                    # f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220730_v05__no_front__.rec",  # noqa
                    # f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220731_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220801_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220802_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220803_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220804_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220805_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220806_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220807_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220808_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220809_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220810_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220811_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220812_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220813_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220814_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220815_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220816_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220817_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220818_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220819_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220820_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220821_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220822_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220823_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220824_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220825_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220826_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220827_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220828_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220829_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220830_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220831_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220901_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220902_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220903_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220905_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220906_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220916_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220917_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220919_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220920_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220922_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220924_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220925_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220926_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20220928_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221013_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221014_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221015_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221016_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221017_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221018_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221019_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221020_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221021_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221022_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221023_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221024_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221025_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221027_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221028_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221029_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221031_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221102_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221103_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221104_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221105_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221107_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221109_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX553/data_LX553_20221110_v05__no_front__.rec",  # noqa
                ]
            ),
            dict(
                rec_path=[
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220723_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220724_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220725_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220726_v05__no_front__.rec",  # noqa
                    # f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220803_v05__no_front__.rec",  # noqa
                    # f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220805_v05__no_front__.rec",  # noqa
                    # f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220806_v05__no_front__.rec",  # noqa
                    # f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220807_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220808_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220809_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220810_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220811_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220814_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220815_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220816_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220817_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220818_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220819_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220820_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220821_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220822_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220823_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220824_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220825_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220826_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220827_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220901_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220902_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220903_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220904_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220905_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220906_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220907_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220908_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220909_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220910_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220912_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220914_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220915_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220916_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220917_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220919_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220920_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220921_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220922_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220925_v05__no_front__.rec",  # noqa
                    # f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220926_v05__no_front__.rec",  # noqa
                    # f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220927_v05__no_front__.rec",  # noqa
                    # f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220928_v05__no_front__.rec",  # noqa
                    # f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20220929_v05__no_front__.rec",  # noqa
                    # f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20221008_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20221009_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20221011_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20221013_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20221015_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20221016_v05__no_front__.rec",  # noqa
                    # f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20221022_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20221023_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20221024_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20221025_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20221027_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20221029_v05__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/LX513/data_LX513_20221031_v05__no_front__.rec",  # noqa
                ]
            ),
        ],
    )
)

datapaths = EasyDict(datapaths)
