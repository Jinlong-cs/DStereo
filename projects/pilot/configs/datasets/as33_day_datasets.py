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
                # normal dataset 1887840
                dict(
                    # RX060
                    rec_path=[
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220811_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220812_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220815_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220816_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220817_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220818_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220819_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220821_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220822_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220823_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220824_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220825_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220826_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220827_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220829_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220830_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220831_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220901_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220902_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220903_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220904_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220905_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220906_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220908_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220910_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220911_v05__no_front__day_filtered.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220811_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220812_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220815_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220816_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220817_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220818_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220819_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220821_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220822_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220823_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220824_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220825_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220826_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220827_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220829_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220830_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220831_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220901_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220902_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220903_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220904_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220905_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220906_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220908_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220910_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220911_v05__side__day.anno.pb_rec",  # noqa
                    ],
                    sample_weight=3.5,
                ),
                dict(
                    rec_path=[
                        # 5.13 add RX333 & RX633 460236
                        # RX333
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220312_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220313_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220314_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220315_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220316_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220318_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220321_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220412_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220413_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220414_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220415_v05__no_front__day_filtered.rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220314_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220316_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220317_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220318_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220319_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220320_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220322_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220323_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220324_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220325_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220401_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220402_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220403_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220404_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220407_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220408_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220412_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220414_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220415_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220416_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220417_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220424_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220425_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220426_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220427_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220428_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220429_v05__no_front__day_filtered.rec",  # noqa
                        # 5.26 add RX141 & RX333 & RX633 726829 689839
                        # RX141
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220502_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220503_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220504_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220505_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220506_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220507_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220508_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220509_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220510_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220512_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220513_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220514_v05__no_front__day_filtered.rec",  # noqa
                        # RX333
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220507_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220508_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220509_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220510_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220511_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220512_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220513_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220514_v05__no_front__day_filtered.rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220502_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220503_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220504_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220505_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220506_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220507_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220508_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220509_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220510_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220512_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220513_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220514_v05__no_front__day_filtered.rec",  # noqa
                        # 6.29 add RX141 & RX333 & RX633
                        # RX141
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220516_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220517_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220519_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220520_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220521_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220522_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220523_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220524_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220525_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220526_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220527_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220528_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220530_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220601_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220602_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220604_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220607_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220608_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220609_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220610_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220611_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220612_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220613_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220614_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220615_v05__no_front__day_filtered.rec",  # noqa
                        # RX333
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220515_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220516_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220517_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220518_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220519_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220520_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220521_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220522_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220523_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220524_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220525_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220526_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220527_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220528_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220529_v05__no_front__day_filtered.rec",  # noqa
                        # 8.10
                        # RX333
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220713_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220718_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220719_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220720_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220721_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220722_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220723_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220724_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220725_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220726_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220728_v05__no_front__day_filtered.rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220709_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220710_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220711_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220712_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220713_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220714_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220715_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220716_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220718_v05__no_front__day_filtered.rec",  # noqa
                    ],
                    anno_path=[
                        # 5.13 add
                        # RX333
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220312_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220313_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220314_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220315_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220316_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220318_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220321_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220412_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220413_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220414_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220415_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220314_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220316_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220317_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220318_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220319_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220320_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220322_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220323_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220324_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220325_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220401_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220402_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220403_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220404_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220407_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220408_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220412_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220414_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220415_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220416_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220417_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220424_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220425_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220426_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220427_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220428_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220429_v05__side__day.anno.pb_rec",  # noqa
                        # 5.26 add
                        # RX141
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220502_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220503_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220504_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220505_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220506_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220507_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220508_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220509_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220510_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220512_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220513_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220514_v05__side__day.anno.pb_rec",  # noqa
                        # RX333
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220507_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220508_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220509_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220510_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220511_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220512_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220513_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220514_v05__side__day.anno.pb_rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220502_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220503_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220504_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220505_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220506_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220507_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220508_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220509_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220510_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220512_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220513_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220514_v05__side__day.anno.pb_rec",  # noqa
                        # 6.29 add
                        # RX141
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220516_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220517_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220519_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220520_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220521_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220522_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220523_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220524_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220525_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220526_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220527_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220528_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220530_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220601_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220602_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220604_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220607_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220608_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220609_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220610_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220611_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220612_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220613_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220614_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220615_v05__side__day.anno.pb_rec",  # noqa
                        # RX333
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220515_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220516_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220517_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220518_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220519_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220520_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220521_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220522_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220523_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220524_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220525_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220526_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220527_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220528_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220529_v05__side__day.anno.pb_rec",  # noqa
                        # 8.10
                        # RX333
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220713_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220718_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220719_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220720_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220721_v05__side__day.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220722_v05__side__day.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220723_v05__side__day.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220724_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220725_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220726_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220728_v05__side__day.anno.pb_rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220709_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220710_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220711_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220712_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220713_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220714_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220715_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220716_v05__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220718_v05__side__day.anno.pb_rec",  # noqa
                    ],
                    sample_weight=15.0,
                ),
                # badcase dataset 90699
                dict(
                    rec_path=[
                        # 5.19 add RX333 & RX633 21466
                        # RX333
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220312_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220313_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220314_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220315_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220316_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220318_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220321_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220412_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220413_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220414_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220415_v05__rotation_badcase__day_.rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220314_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220316_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220317_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220318_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220319_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220320_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220322_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220323_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220324_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220325_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220401_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220402_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220403_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220404_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220407_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220408_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220412_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220414_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220415_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220416_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220417_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220424_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220425_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220426_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220427_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220428_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220429_v05__rotation_badcase__day_.rec",  # noqa
                        # 6.29 add RX141 RX333 & RX633
                        # RX141
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220502_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220503_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220504_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220505_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220506_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220507_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220508_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220509_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220510_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220512_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220513_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220514_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220516_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220517_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220519_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220520_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220521_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220522_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220523_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220524_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220525_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220526_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220527_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220528_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220530_v05__rotation_badcase__day_.rec",  # noqa
                        # RX333
                        # f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220507_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220508_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220509_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220510_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220511_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220512_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220513_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220514_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220515_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220516_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220517_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220518_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220519_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220520_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220521_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220522_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220523_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220524_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220525_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220526_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220527_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220528_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220529_v05__rotation_badcase__day_.rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220502_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220503_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220504_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220505_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220506_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220507_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220508_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220509_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220510_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220512_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220513_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220514_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220515_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220516_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220517_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220518_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220519_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220520_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220523_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220524_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220525_v05__rotation_badcase__day_.rec",  # noqa
                        # 30C1G
                        f"{root}/data/real_3d/v05/30C1G/hard_case/data_30C1G_202104_202108__rotbadcase_v2.rec",  # noqa
                    ],
                    anno_path=[
                        # 5.19 add
                        # RX333
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220312_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220313_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220314_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220315_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220316_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220318_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220321_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220412_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220413_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220414_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220415_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220314_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220316_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220317_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220318_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220319_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220320_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220322_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220323_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220324_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220325_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220401_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220402_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220403_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220404_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220407_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220408_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220412_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220414_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220415_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220416_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220417_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220424_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220425_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220426_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220427_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220428_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220429_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        # 6.29 add
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220502_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220503_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220504_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220505_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220506_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220507_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220508_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220509_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220510_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220512_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220513_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220514_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220516_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220517_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220519_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220520_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220521_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220522_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220523_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220524_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220525_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220526_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220527_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220528_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220530_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        # RX333
                        # f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220507_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220508_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220509_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220510_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220511_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220512_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220513_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220514_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220515_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220516_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220517_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220518_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220519_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220520_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220521_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220522_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220523_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220524_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220525_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220526_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220527_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220528_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220529_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220502_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220503_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220504_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220505_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220506_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220507_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220508_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220509_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220510_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220512_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220513_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220514_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220515_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220516_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220517_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220518_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220519_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220520_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220523_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220524_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220525_v05__rotation_badcase__side__day.anno.pb_rec",  # noqa
                        # 30C1G
                        f"{root}/data/real_3d/v05/30C1G/hard_case/with_image_source/data_30C1G_202104_202108__rotbadcase_v2.anno.pb_rec",  # noqa
                    ],
                    sample_weight=1.0,
                ),
                # special_vehicle dataset 32958
                dict(
                    rec_path=[
                        # 6.29 add RX141 RX333 & RX633
                        # RX141
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_2022-05-06_v05__occlusion_badcase__day.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_2022-05-0615_v05__special_vehicle__day.rec",  # noqa
                        # RX333
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_2022-03-04-05_v05__special_vehicle__day.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_2022-03-04-05_v05__special_vehicle__day.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_2022-07_v05__special_vehicle__day.rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_2022-03-07_v05__occlusion_badcase__day.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_2022-03-04-05-0615_v05__special_vehicle__day.rec",  # noqa
                        f"{root}/multicam_pilot/data/train/vehicle/vehicle_3D_detection/hard_case/30C1G_filtered_10m_contain_bus_truck_specialcar_weisen_v04.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_2022-07_v05__special_vehicle__day.rec",  # noqa
                    ],
                    anno_path=[
                        # RX141
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_2022-05-06_v05__occlusion_badcase__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_2022-05-0615_v05__special_vehicle__side__day.anno.pb_rec",  # noqa
                        # RX333
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_2022-03-04-05_v05__special_vehicle__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_2022-03-04-05_v05__special_vehicle__side__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_2022-07_v05__special_vehicle__side__day.anno.pb_rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_2022-03-07_v05__occlusion_badcase__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_2022-03-04-05-0615_v05__special_vehicle__side__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/hard/with_image_source/30C1G_filtered_10m_contain_bus_truck_specialcar_weisen_v04_day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_2022-07_v05__special_vehicle__side__day.anno.pb_rec",  # noqa
                    ],
                    sample_weight=1.0,
                ),
                dict(
                    # 30C1G
                    # 731340
                    rec_path=[
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210423_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210425_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210426_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210427_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210428_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210429_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210430_v04__no_front__day_filtered.rec",  # noqa
                        # f'{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210501_v04__no_front__day_filtered.rec',  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210506_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210507_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210508_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210509_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210510_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210511_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210512_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210513_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210514_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210515_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210517_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210518_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210519_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210520_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210521_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210522_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210523_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210524_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210525_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/data_30C1G_20210527_v04__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210528_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210531_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210601_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210602_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210603_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210604_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210605_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210609_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210610_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210611_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210612_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210613_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210614_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210616_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210617_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210618_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210619_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210620_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210621_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210625_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210626_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210627_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210628_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210701_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210702_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210703_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210704_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210705_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210706_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210707_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210708_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210709_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210710_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210711_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210712_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210713_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210714_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210715_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210716_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210717_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210718_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210719_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210720_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210721_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210722_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210723_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210725_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210726_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210729_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210801_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210802_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210808_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210819_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210820_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210821_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210823_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210825_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210826_v05__no_front__day_filtered.rec",  # noqa
                        # f'{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210827_v05__no_front__day_filtered.rec',  # noqa
                        # f'{root}/data/real_3d/filtered/v05/30C1G/data_30C1G_20210828_v05__no_front__day_filtered.rec',  # noqa
                    ],
                    anno_path=[
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210423_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210425_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210426_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210427_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210428_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210429_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210430_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f'{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210501_v04__no_front__day_filtered.anno.pb_rec',  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210506_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210507_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210508_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210509_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210510_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210511_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210512_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210513_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210514_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210515_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210517_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210518_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210519_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210520_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210521_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210522_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210523_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210524_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210525_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v04/30C1G/with_image_source/data_30C1G_20210527_v04__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210528_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210531_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210601_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210602_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210603_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210604_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210605_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210609_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210610_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210611_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210612_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210613_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210614_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210616_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210617_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210618_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210619_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210620_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210621_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210625_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210626_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210627_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210628_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210701_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210702_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210703_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210704_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210705_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210706_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210707_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210708_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210709_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210710_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210711_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210712_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210713_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210714_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210715_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210716_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210717_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210718_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210719_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210720_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210721_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210722_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210723_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210725_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210726_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210729_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210801_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210802_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210808_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210819_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210820_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210821_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210823_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210825_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210826_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        # f'{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210827_v05__no_front__day_filtered.anno.pb_rec',  # noqa
                        # f'{root}/data/real_3d/filtered/v05/30C1G/with_image_source/data_30C1G_20210828_v05__no_front__day_filtered.anno.pb_rec',  # noqa
                    ],
                    sample_weight=5.0,
                ),
            ],
            val_batch_size_per_ctx=10,
            val_data_paths=[
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220326_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220328_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220329_v05__no_front__day_filtered.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220326_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220328_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220329_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                    ],
                    sample_weight=1,
                )
            ],
        ),
        person_3d_detection=dict(
            train_batch_size_per_ctx=16,
            train_data_paths=[
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/person3d/as33/day/data_RX060_20220811_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/data_RX060_20220812_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/data_RX060_20220815_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/data_RX060_20220816_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/data_RX060_20220817_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220502_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220503_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220504_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220505_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220506_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220507_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220508_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220509_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220510_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220512_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220513_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220514_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220516_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220517_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220519_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220520_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220521_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220522_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220523_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220524_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220525_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220526_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220527_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220528_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220530_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220312_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220313_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220314_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220315_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220316_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220408_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220412_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220413_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220414_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220415_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220507_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220508_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220509_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220510_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220511_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220512_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220513_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220514_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220515_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220516_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220517_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220518_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220519_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220520_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220521_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220522_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220523_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220524_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220525_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220526_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220527_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220528_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220529_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220311_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220313_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220314_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220315_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220316_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220317_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220318_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220319_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220320_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220322_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220323_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220324_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220325_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220401_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220402_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220403_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220404_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220406_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220407_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220408_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220412_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220414_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220415_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220416_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220417_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220424_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220425_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220426_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220427_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220428_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220429_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220502_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220503_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220504_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220505_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220506_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220507_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220508_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220509_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220510_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220512_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220513_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220514_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220515_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220516_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220517_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220518_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220523_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220524_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220525_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220602_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220604_v05__no_front__day.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220605_v05__no_front__day.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/data/real_3d/person3d/as33/day/data_RX060_20220811_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/data_RX060_20220812_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/data_RX060_20220815_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/data_RX060_20220816_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/data_RX060_20220817_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220502_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220503_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220504_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220505_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220506_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220507_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220508_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220509_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220510_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220512_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220513_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220514_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220516_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220517_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220519_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220520_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220521_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220522_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220523_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220524_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220525_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220526_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220527_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220528_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX141_20220530_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220312_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220313_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220314_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220315_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220316_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220408_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220412_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220413_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220414_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220415_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220507_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220508_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220509_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220510_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220511_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220512_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220513_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220514_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220515_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220516_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220517_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220518_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220519_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220520_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220521_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220522_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220523_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220524_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220525_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220526_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220527_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220528_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX333_20220529_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220311_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220313_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220314_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220315_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220316_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220317_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220318_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220319_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220320_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220322_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220323_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220324_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220325_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220401_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220402_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220403_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220404_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220406_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220407_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220408_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220412_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220414_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220415_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220416_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220417_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220424_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220425_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220426_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220427_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220428_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220429_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220502_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220503_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220504_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220505_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220506_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220507_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220508_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220509_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220510_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220512_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220513_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220514_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220515_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220516_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220517_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220518_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220523_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220524_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220525_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220602_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220604_v05__no_front__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/day/person_data_RX633_20220605_v05__no_front__day.anno.pb_rec",  # noqa
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
buckets = ["matrix2"]
