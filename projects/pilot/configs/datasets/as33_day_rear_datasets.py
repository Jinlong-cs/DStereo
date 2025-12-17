import os

from as33_day_datasets import datapaths
from easydict import EasyDict

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
root = os.path.join(bucket_root, "matrix2")


# ----- Required -----

datapaths.update(
    dict(
        # day rear 540802 = 425206 90401 25195 (12.58 2.67 0.745) 12.6 2.7 0.7
        vehicle_3d_detection=dict(
            train_batch_size_per_ctx=12,
            train_data_paths=[
                # normal dataset 425206
                dict(
                    rec_path=[
                        # 5.13 add RX333 & RX633 460236
                        # RX333
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220312_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220313_v05__no_front__day_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220314_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220315_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220316_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220318_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220321_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220412_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220413_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220414_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220415_v05__no_front__day_filtered.rec",  # noqa
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
                        # 5.26 add RX141 & RX333 & RX633 726829
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
                        # 7.13 add RX141 & RX333
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
                        # 818
                        # RX333
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220713_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220718_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220719_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220720_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220721_v05__no_front__day_filtered.rec",  # noqa
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
                        # 0825
                        # RX060
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220811_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220812_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220815_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220816_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220817_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220818_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220819_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220821_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220822_v05__no_front__.rec",  # noqa
                    ],
                    anno_path=[
                        # 5.13
                        # RX333
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220312_v05__rear__day.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220313_v05__rear__day.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220314_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220315_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220316_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220318_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220321_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220412_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220413_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220414_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220415_v05__rear__day.anno.pb_rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220314_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220316_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220317_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220318_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220319_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220320_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220322_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220323_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220324_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220325_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220401_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220402_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220403_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220404_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220407_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220408_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220412_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220414_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220415_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220416_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220417_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220424_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220425_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220426_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220427_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220428_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220429_v05__rear__day.anno.pb_rec",  # noqa
                        # 5.26 add
                        # RX141
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220502_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220503_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220504_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220505_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220506_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220507_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220508_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220509_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220510_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220512_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220513_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220514_v05__rear__day.anno.pb_rec",  # noqa
                        # RX333
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220508_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220509_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220510_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220511_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220512_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220513_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220514_v05__rear__day.anno.pb_rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220502_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220503_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220504_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220505_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220506_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220507_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220508_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220509_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220510_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220512_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220513_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220514_v05__rear__day.anno.pb_rec",  # noqa
                        # 7.13 add
                        # RX141
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220516_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220517_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220519_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220520_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220521_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220522_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220523_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220524_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220525_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220526_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220527_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220528_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220530_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220601_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220602_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220604_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220607_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220608_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220609_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220610_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220611_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220612_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220613_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220614_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220615_v05__rear__day.anno.pb_rec",  # noqa
                        # RX333
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220515_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220516_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220517_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220518_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220519_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220520_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220521_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220522_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220523_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220524_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220525_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220526_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220527_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220528_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220529_v05__rear__day.anno.pb_rec",  # noqa
                        # 818
                        # RX333
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220713_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220718_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220719_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220720_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220721_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220725_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220726_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220728_v05__rear__day.anno.pb_rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220709_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220710_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220711_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220712_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220713_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220714_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220715_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220716_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220718_v05__rear__day.anno.pb_rec",  # noqa
                        # 0825
                        # RX060
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220811_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220812_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220815_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220816_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220817_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220818_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220819_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220821_v05__rear__day.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220822_v05__rear__day.anno.pb_rec",  # noqa
                    ],
                    # sample_weight=12.6,
                    sample_weight=13.2,
                ),
                # badcase dataset 90699 90401
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
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220515_v05__rotation_badcase__day_.rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220516_v05__rotation_badcase__day_.rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220517_v05__rotation_badcase__day_.rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220518_v05__rotation_badcase__day_.rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220519_v05__rotation_badcase__day_.rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220520_v05__rotation_badcase__day_.rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220523_v05__rotation_badcase__day_.rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220524_v05__rotation_badcase__day_.rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220525_v05__rotation_badcase__day_.rec",  # noqa
                        # 7.12 add RX141
                        # RX141
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220601_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220602_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220604_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220607_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220608_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220609_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220610_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220611_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220612_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220613_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220614_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220615_v05__rotation_badcase__day_.rec",  # noqa
                        # 818
                        # RX333
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220713_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220718_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220719_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220720_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220721_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220725_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220726_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220728_v05__rotation_badcase__day_.rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220709_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220710_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220711_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220712_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220713_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220714_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220715_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220716_v05__rotation_badcase__day_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220718_v05__rotation_badcase__day_.rec",  # noqa
                    ],
                    anno_path=[
                        # 5.19 add
                        # RX333
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220312_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220313_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220314_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220315_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220316_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220318_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220321_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220412_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220413_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220414_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220415_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220314_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220316_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220317_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220318_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220319_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220320_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220322_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220323_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220324_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220325_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220401_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220402_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220403_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220404_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220407_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220408_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220412_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220414_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220415_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220416_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220417_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220424_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220425_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220426_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220427_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220428_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220429_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        # 6.29 add
                        # RX141
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220502_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220503_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220504_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220505_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220506_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220507_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220508_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220509_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220510_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220512_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220513_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220514_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220516_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220517_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220519_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220520_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220521_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220522_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220523_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220524_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220525_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220526_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220527_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220528_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220530_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        # RX333
                        # f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220507_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220508_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220509_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220510_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220511_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220512_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220513_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220514_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220515_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220516_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220517_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220518_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220519_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220520_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220521_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220522_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220523_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220524_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220525_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220526_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220527_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220528_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220529_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220502_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220503_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220504_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220505_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220506_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220507_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220508_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220509_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220510_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220512_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220513_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220514_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220515_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220516_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220517_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220518_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220519_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220520_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220523_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220524_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220525_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        # 7.12 add RX141
                        # RX141
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220601_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220602_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220604_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220607_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220608_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220609_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220610_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220611_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220612_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220613_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220614_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_20220615_v05__rotation_badcase__day_.anno.pb_rec",  # noqa
                        # 818
                        # RX333
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220713_v05__rotation_badcase__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220718_v05__rotation_badcase__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220719_v05__rotation_badcase__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220720_v05__rotation_badcase__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220721_v05__rotation_badcase__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220725_v05__rotation_badcase__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220726_v05__rotation_badcase__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_20220728_v05__rotation_badcase__rear__day.anno.pb_rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220709_v05__rotation_badcase__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220710_v05__rotation_badcase__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220711_v05__rotation_badcase__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220712_v05__rotation_badcase__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220713_v05__rotation_badcase__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220714_v05__rotation_badcase__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220715_v05__rotation_badcase__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220716_v05__rotation_badcase__rear__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_20220718_v05__rotation_badcase__rear__day.anno.pb_rec",  # noqa
                    ],
                    sample_weight=2.2,
                ),
                # special_vehicle dataset 32958 25195
                dict(
                    rec_path=[
                        # 6.29 add RX141 RX333 & RX633
                        # RX141
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_2022-05-0615_v05__special_vehicle__day.rec",  # noqa
                        # RX333
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_2022-03-04-05_v05__special_vehicle__day.rec",  # noqa
                        # RX633
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_2022-03-04-05-0615_v05__special_vehicle__day.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_2022-03-04-0514_v05__special_vehicle__day.rec",  # noqa
                    ],
                    anno_path=[
                        # RX141
                        f"{root}/data/real_3d/v05/RX141/day_badcase/data_RX141_2022-05-0615_v05__special_vehicle__day.anno.pb_rec",  # noqa
                        # RX333
                        f"{root}/data/real_3d/v05/RX333/day_badcase/data_RX333_2022-03-04-05_v05__special_vehicle__day.anno.pb_rec",  # noqa
                        # RX633
                        # f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_2022-03-04-05-0615_v05__special_vehicle__day.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/day_badcase/data_RX633_2022-03-04-0514_v05__special_vehicle__day.anno.pb_rec",  # noqa
                    ],
                    sample_weight=0.6,
                ),
            ],
            val_batch_size_per_ctx=10,
            val_data_paths=[
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220326_v05__no_front__day_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220329_v05__no_front__day_filtered.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220326_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220329_v05__no_front__day_filtered.anno.pb_rec",  # noqa
                    ],
                    sample_weight=5,
                ),
            ],
        ),
        lane_parsing=dict(
            train_batch_size_per_ctx=16,
            train_data_paths=[
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_3500_20210419_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_3500_20210419_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=1.5,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_3902_20210524_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_3902_20210524_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=1.5,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_6291_20210711_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_6291_20210711_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=3,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_1899_20210711_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_1899_20210711_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=0.5,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_1721_20210803_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_1721_20210803_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=1,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_5598_20210803_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_5598_20210803_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=3,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_20735_20210825_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_20735_20210825_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=8,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_3512_20210825_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_3512_20210825_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=2,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_13475_20211018_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_13475_20211018_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=5,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_6000_20211018_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_6000_20211018_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=3,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_4387_20211129_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_4387_20211129_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=8,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_wissen_num_653_20211129_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_wissen_num_653_20211129_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=0.1,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_wissen_num_15689_20211129_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_wissen_num_15689_20211129_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=8,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_12116_20211220_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_12116_20211220_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=12,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_x3c_num_2802_20211220_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_x3c_num_2802_20211220_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=2,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_16876_20220118_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_16876_20220118_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=12,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_x3c_num_6284_20220118_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_x3c_num_6284_20220118_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=3,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_2909_20220214_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_2909_20220214_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=1,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_x3c_num_673_20220214_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_x3c_num_673_20220214_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=0.1,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_day_x3c_num_6785_20220328_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_day_x3c_num_6785_20220328_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=4,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_night_x3c_num_5356_20220328_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_night_x3c_num_5356_20220328_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=0.2,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_day_x3c_D100L_num_5286_20220424_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_day_x3c_D100L_num_5286_20220424_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=4,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_day_x3c_CO_OX3GB_A100L_num_2629_20220505_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_day_x3c_CO_OX3GB_A100L_num_2629_20220505_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=8,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_night_x3c_CO_OX3GB_A100L_num_166_20220505_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_night_x3c_CO_OX3GB_A100L_num_166_20220505_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=1,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_0233_CW_A23GB_A114L_num_1777_20220525_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_0233_CW_A23GB_A114L_num_1777_20220525_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=1.0,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_0233_CW_A23GB_A114L_num_2461_20220525_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_0233_CW_A23GB_A114L_num_2461_20220525_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=1.0,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_0233_CW_A23GB_A114L_num_2193_20220525_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_0233_CW_A23GB_A114L_num_2193_20220525_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=1.1,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_0233_CW_A23GB_A114L_num_2058_20220608_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_0233_CW_A23GB_A114L_num_2058_20220608_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=1,
                ),
                dict(
                    rec_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_0233_CW_A23GB_A114L_num_576_20220621_5_2classes.rec",  # noqa
                    anno_path=f"{root}/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_0233_CW_A23GB_A114L_num_576_20220621_5_2classes.anno.pb_rec",  # noqa
                    sample_weight=0.5,
                ),
            ],
        ),
    )
)

datapaths = EasyDict(datapaths)
buckets = ["matrix2"]
