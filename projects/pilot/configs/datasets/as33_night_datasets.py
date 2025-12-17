import os

from base_2d_night_datasets import datapaths
from easydict import EasyDict

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
root = os.path.join(bucket_root, "matrix2")


# ----- Required -----

datapaths.update(
    dict(
        # 0825 total 1008686 = 962442 + 46244 (15.3 0.7) focal_length_default = 1114.3466796875 focal_length_default = 1114.3466796875
        # 0818 total 896180 = 849936 + 46244 (15.2 0.8) focal_length_default = 1114.3466796875
        # total 7.13 704831 + 46244 = 751075 (15 1)
        vehicle_3d_detection=dict(
            train_batch_size_per_ctx=12,
            train_data_paths=[
                # normal dataset 704831(7.13)
                dict(
                    rec_path=[
                        # 5.13 add RX333 & RX633 226961
                        # RX333
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220311_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220312_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220313_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220314_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220315_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220316_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220318_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220321_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220322_v05__no_front__night_filtered.rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220311_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220313_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220314_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220315_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220316_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220317_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220318_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220319_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220320_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220322_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220323_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220324_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220325_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220402_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220403_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220404_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220405_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220407_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220408_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220412_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220414_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220416_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220417_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220418_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220419_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220420_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220421_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220422_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220423_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220424_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220425_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220426_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220427_v05__no_front__night_filtered.rec",  # noqa
                        # 5.26 add RX141 & RX333 & RX633 212940  179816
                        # RX141
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220502_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220503_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220504_v05__no_front__night_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220506_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220507_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220508_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220509_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220510_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220512_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220513_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220514_v05__no_front__night_filtered.rec",  # noqa
                        # RX333
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220507_v05__no_front__night_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220508_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220509_v05__no_front__night_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220510_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220511_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220512_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220513_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220514_v05__no_front__night_filtered.rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220504_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220505_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220506_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220507_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220508_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220509_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220510_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220512_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220513_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220514_v05__no_front__night_filtered.rec",  # noqa
                        # 7.13 add
                        # RX141
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220516_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220517_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220519_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220520_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220521_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220522_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220523_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220524_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220525_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220526_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220527_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220528_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220530_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220601_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220602_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220603_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220604_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220607_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220608_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220609_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220610_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220611_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220613_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220614_v05__no_front__night_filtered.rec",  # noqa
                        # RX333
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220515_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220516_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220517_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220518_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220519_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220520_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220521_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220522_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220523_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220524_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220525_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220526_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220527_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220528_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220529_v05__no_front__night_filtered.rec",  # noqa
                        # # RX633
                        # f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220515_v05__no_front__night_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220516_v05__no_front__night_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220517_v05__no_front__night_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220518_v05__no_front__night_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220519_v05__no_front__night_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220520_v05__no_front__night_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220524_v05__no_front__night_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220525_v05__no_front__night_filtered.rec",  # noqa
                        # 818
                        # RX141
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220617_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220618_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220619_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220620_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220621_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220622_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220623_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220624_v05__no_front__night_filtered.rec",  # noqa
                        # RX333
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220713_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220718_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220719_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220720_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220721_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220725_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220726_v05__no_front__night_filtered.rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220707_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220708_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220709_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220710_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220711_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220712_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220713_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220714_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220715_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220716_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220717_v05__no_front__night_filtered.rec",  # noqa
                        # 0825
                        # RX060
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220811_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220812_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220814_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220815_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220816_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220817_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220818_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220819_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220821_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220822_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220823_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220824_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220825_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220903_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220904_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220908_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX060/data_RX060_20220918_v05__no_front__.rec",  # noqa
                    ],
                    anno_path=[
                        # 5.13 add
                        # RX333
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220311_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220312_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220313_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220314_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220315_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220316_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220318_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220321_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220322_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220311_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220313_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220314_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220315_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220316_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220317_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220318_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220319_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220320_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220322_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220323_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220324_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220325_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220402_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220403_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220404_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220405_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220407_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220408_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220412_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220414_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220416_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220417_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220418_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220419_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220420_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220421_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220422_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220423_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220424_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220425_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220426_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220427_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # 5.26 add
                        # RX141
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220502_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220503_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220504_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220506_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220507_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220508_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220509_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220510_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220512_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220513_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220514_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # RX333
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220507_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220508_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220509_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220510_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220511_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220512_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220513_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220514_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220504_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220505_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220506_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220507_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220508_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220509_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220510_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220512_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220513_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220514_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # 7.13 add
                        # RX141
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220516_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220517_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220519_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220520_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220521_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220522_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220523_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220524_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220525_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220526_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220527_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220528_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220530_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220601_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220602_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220603_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220604_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220607_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220608_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220609_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220610_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220611_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220613_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220614_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # RX333
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220515_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220516_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220517_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220518_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220519_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220520_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220521_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220522_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220523_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220524_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220525_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220526_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220527_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220528_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220529_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # RX141
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220617_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220618_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220619_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220620_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220621_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220622_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220623_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX141/data_RX141_20220624_v05__side__night.anno.pb_rec",  # noqa
                        # RX333
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220713_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220718_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220719_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220720_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220721_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220725_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX333/data_RX333_20220726_v05__side__night.anno.pb_rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220707_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220708_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220709_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220710_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220711_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220712_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220713_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220714_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220715_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220716_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220717_v05__side__night.anno.pb_rec",  # noqa
                        # 0825
                        # RX060
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220811_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220812_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220814_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220815_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220816_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220817_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX060/data_RX060_20220818_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220819_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220821_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220822_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220823_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220824_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220825_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220903_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220904_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220908_v05__side__night.anno.pb_rec",  # noqa
                        f"{root}/users/jun01.guan/data/3d/split_data/v05/RX060/data_RX060_20220918_v05__side__night.anno.pb_rec",  # noqa
                    ],
                    sample_weight=15.3,
                ),
                # badcase dataset 37992(7.13)+8252 = 46244
                dict(
                    rec_path=[
                        # 5.19 add RX333 & RX633 11529
                        # RX333
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220311_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220312_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220313_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220314_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220315_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220316_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220318_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220321_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220322_v05__rotation_badcase__night_.rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220311_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220313_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220314_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220315_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220316_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220317_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220318_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220319_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220320_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220322_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220323_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220324_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220325_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220402_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220403_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220404_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220405_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220407_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220408_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220412_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220416_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220417_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220418_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220419_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220420_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220421_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220422_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220423_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220424_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220425_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220426_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220427_v05__rotation_badcase__night_.rec",  # noqa
                        # 7.13 add RX141 RX333 & RX633
                        # RX141
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220502_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220503_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220504_v05__rotation_badcase__night_.rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220506_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220507_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220508_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220509_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220510_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220512_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220513_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220514_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220516_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220517_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220519_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220520_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220521_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220522_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220523_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220524_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220525_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220526_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220527_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220528_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220530_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220601_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220602_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220603_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220604_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220607_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220608_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220609_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220610_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220611_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220613_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220614_v05__rotation_badcase__night_.rec",  # noqa
                        # RX333
                        # f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220507_v05__rotation_badcase__night_.rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220508_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220509_v05__rotation_badcase__night_.rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220510_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220511_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220512_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220513_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220514_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220515_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220516_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220517_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220518_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220519_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220520_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220521_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220522_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220523_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220524_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220525_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220526_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220527_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220528_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220529_v05__rotation_badcase__night_.rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220504_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220505_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220506_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220507_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220508_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220509_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220510_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220512_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220513_v05__rotation_badcase__night_.rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220514_v05__rotation_badcase__night_.rec",  # noqa
                        # 7.13 add special_vehicle 8252
                        # RX141
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_2022-05-0615_v05__special_vehicle__night.rec",  # noqa
                        # RX333
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_2022-03-04-05_v05__special_vehicle__night.rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_2022-03-04-0514_v05__special_vehicle__night.rec",  # noqa
                    ],
                    anno_path=[
                        # 5.19 add
                        # RX333
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220311_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220312_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220313_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220314_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220315_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220316_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220318_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220321_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220322_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220311_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220313_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220314_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220315_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220316_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220317_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220318_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220319_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220320_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220322_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220323_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220324_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220325_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220402_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220403_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220404_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220405_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220407_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220408_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220412_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220416_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220417_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220418_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220419_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220420_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220421_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220422_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220423_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220424_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220425_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220426_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220427_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        # 7.13 add RX141 RX333 & RX633
                        # RX141
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220502_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220503_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220504_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220506_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220507_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220508_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220509_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220510_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220512_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220513_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220514_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220516_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220517_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220519_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220520_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220521_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220522_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220523_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220524_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220525_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220526_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220527_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220528_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220530_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220601_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220602_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220603_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220604_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220607_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220608_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220609_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220610_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220611_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220613_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_20220614_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        # RX333
                        # f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220507_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220508_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220509_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220510_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220511_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220512_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220513_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220514_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220515_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220516_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220517_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220518_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220519_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220520_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220521_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220522_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220523_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220524_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220525_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220526_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220527_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220528_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_20220529_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220504_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220505_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220506_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220507_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220508_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220509_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220510_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220512_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220513_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_20220514_v05__rotation_badcase__night_.anno.pb_rec",  # noqa
                        # RX141
                        f"{root}/data/real_3d/v05/RX141/night_badcase/data_RX141_2022-05-0615_v05__special_vehicle__night.anno.pb_rec",  # noqa
                        # RX333
                        f"{root}/data/real_3d/v05/RX333/night_badcase/data_RX333_2022-03-04-05_v05__special_vehicle__night.anno.pb_rec",  # noqa
                        # RX633
                        f"{root}/data/real_3d/v05/RX633/night_badcase/data_RX633_2022-03-04-0514_v05__special_vehicle__night.anno.pb_rec",  # noqa
                    ],
                    sample_weight=0.7,
                ),
            ],
            val_batch_size_per_ctx=10,
            val_data_paths=[
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220326_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220328_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220329_v05__no_front__night_filtered.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220326_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220328_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/RX633/data_RX633_20220329_v05__no_front__night_filtered.anno.pb_rec",  # noqa
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
                        f"{root}/data/real_3d/person3d/as33/night/data_RX060_20220811_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/data_RX060_20220812_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/data_RX060_20220814_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/data_RX060_20220815_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/data_RX060_20220816_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/data_RX060_20220817_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220502_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220503_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220504_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220506_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220507_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220508_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220509_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220510_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220512_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220513_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220514_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220516_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220517_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220519_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220520_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220521_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220522_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220523_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220524_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220525_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220526_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220527_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220528_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220530_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220311_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220313_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220314_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220315_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220316_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220322_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220412_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220507_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220508_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220509_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220510_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220511_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220512_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220513_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220514_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220515_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220516_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220517_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220518_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220519_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220520_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220521_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220522_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220523_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220524_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220525_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220526_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220527_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220528_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220529_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220315_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220316_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220317_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220319_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220322_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220323_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220324_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220325_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220404_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220405_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220406_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220407_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220408_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220411_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220412_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220414_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220416_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220417_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220418_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220419_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220420_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220421_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220423_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220424_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220425_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220426_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220427_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220504_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220505_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220506_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220507_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220508_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220509_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220510_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220512_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220513_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220514_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220515_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220516_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220517_v05__no_front__night.rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220518_v05__no_front__night.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/data/real_3d/person3d/as33/night/data_RX060_20220811_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/data_RX060_20220812_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/data_RX060_20220814_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/data_RX060_20220815_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/data_RX060_20220816_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/data_RX060_20220817_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220502_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220503_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220504_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220506_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220507_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220508_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220509_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220510_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220512_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220513_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220514_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220516_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220517_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220519_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220520_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220521_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220522_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220523_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220524_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220525_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220526_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220527_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220528_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX141_20220530_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220311_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220313_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220314_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220315_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220316_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220322_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220412_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220507_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220508_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220509_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220510_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220511_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220512_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220513_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220514_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220515_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220516_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220517_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220518_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220519_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220520_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220521_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220522_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220523_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220524_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220525_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220526_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220527_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220528_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX333_20220529_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220315_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220316_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220317_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220319_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220322_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220323_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220324_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220325_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220404_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220405_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220406_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220407_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220408_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220411_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220412_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220414_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220416_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220417_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220418_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220419_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220420_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220421_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220423_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220424_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220425_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220426_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220427_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220504_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220505_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220506_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220507_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220508_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220509_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220510_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220512_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220513_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220514_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220515_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220516_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220517_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/person3d/as33/night/person_data_RX633_20220518_v05__no_front__night.anno.pb_rec",  # noqa
                    ],
                    sample_weight=9,
                )
            ],
            val_batch_size_per_ctx=10,
            val_data_paths=[
                dict(
                    rec_path=[
                        f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210503_v04__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210504_v04__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210505_v04__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v04/65U3D/data_65U3D_20210515_v04__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210504_v04__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v04/30C1G/data_30C1G_20210505_v04__no_front__.rec",  # noqa
                    ],
                    anno_path=[
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210503_v04__no_front__.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210504_v04__no_front__.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210505_v04__no_front__.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/65U3D/data_65U3D_20210515_v04__no_front__.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210504_v04__no_front__.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v04_ignore_false/v04/30C1G/data_30C1G_20210505_v04__no_front__.anno.pb_rec",  # noqa
                    ],
                    sample_weight=5,
                )
            ],
        ),
    )
)

datapaths = EasyDict(datapaths)
buckets = ["matrix2"]
