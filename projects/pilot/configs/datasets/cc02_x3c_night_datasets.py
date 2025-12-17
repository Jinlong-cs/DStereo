import os

from base_2d_night_datasets import datapaths
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
                        # JL082
                        # f'{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220311_v05__no_front__night_filtered.rec',  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220313_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220314_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220318_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220320_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220321_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220322_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220323_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220324_v05__no_front__night_filtered.rec",  # noqa
                        # f'{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220326_v05__no_front__night_filtered.rec',  # noqa
                        # 4月
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220401_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220402_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220403_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220404_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220406_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220407_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220408_v05__no_front__night_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220409_v05__no_front__night_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220410_v05__no_front__night_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220411_v05__no_front__night_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220412_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220413_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220414_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220415_v05__no_front__night_filtered.rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220416_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220417_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220418_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220419_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220420_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220421_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220422_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220423_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220424_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220425_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220427_v05__no_front__night_filtered.rec",  # noqa
                        # JL515
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220311_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220312_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220313_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220315_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220316_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220318_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220319_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220320_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220321_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220322_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220324_v05__no_front__night_filtered.rec",  # noqa
                        # f'{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220326_v05__no_front__night_filtered.rec',  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220327_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220328_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220329_v05__no_front__night_filtered.rec",  # noqa
                        # 202204
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220402_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220403_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220405_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220407_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220408_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220409_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220412_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220417_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220421_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220422_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220426_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220427_v05__no_front__night_filtered.rec",  # noqa
                        # JL773
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220312_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220313_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220314_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220315_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220316_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220317_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220318_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220319_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220321_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220322_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220324_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220325_v05__no_front__night_filtered.rec",  # noqa
                        # 202204
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220401_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220402_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220403_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220404_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220406_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220407_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220412_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220413_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220414_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220415_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220416_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220417_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220418_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220419_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220420_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220421_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220422_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220423_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220424_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220425_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220426_v05__no_front__night_filtered.rec",  # noqa
                        # JL918
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220311_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220312_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220313_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220315_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220316_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220318_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220319_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220320_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220322_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220323_v05__no_front__night_filtered.rec",  # noqa
                        # 202204
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220402_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220403_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220404_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220405_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220406_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220407_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220412_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220413_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220415_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220416_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220417_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220418_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220421_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220422_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220424_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220425_v05__no_front__night_filtered.rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220426_v05__no_front__night_filtered.rec",  # noqa
                    ],
                    anno_path=[
                        # f'{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220311_v05__no_front__night_filtered.anno.pb_rec',  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220313_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220314_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220318_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220320_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220321_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220322_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220323_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220324_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # f'{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220326_v05__no_front__night_filtered.anno.pb_rec',  # noqa
                        # 4月
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220401_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220402_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220403_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220404_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220406_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220407_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220408_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220409_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220410_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220411_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220412_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220413_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220414_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220415_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220416_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220417_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220418_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220419_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220420_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220421_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220422_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220423_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220424_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220425_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL082/data_JL082_20220427_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # JL515
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220311_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220312_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220313_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220315_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220316_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220318_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220319_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220320_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220321_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220322_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220324_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # f'{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220326_v05__no_front__night_filtered.anno.pb_rec',  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220327_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220328_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220329_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # 202204
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220402_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220403_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220405_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220407_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220408_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220409_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220412_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220417_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220421_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220422_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220426_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL515/data_JL515_20220427_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # JL773
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220312_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220313_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220314_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220315_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220316_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220317_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220318_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220319_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220321_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220322_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220324_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220325_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # 202204
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220401_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220402_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220403_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220404_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220406_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220407_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220412_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220413_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220414_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220415_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220416_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220417_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220418_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220419_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220420_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220421_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220422_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220423_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220424_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220425_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL773/data_JL773_20220426_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # JL918
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220311_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220312_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220313_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220315_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220316_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220318_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220319_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220320_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220322_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220323_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        # 202204
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220402_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220403_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220404_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220405_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220406_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220407_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220412_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220413_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220415_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220416_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220417_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220418_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220421_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220422_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220424_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220425_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                        f"{root}/data/real_3d/filtered/v05/JL918/data_JL918_20220426_v05__no_front__night_filtered.anno.pb_rec",  # noqa
                    ],
                    sample_weight=16,
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
                dict(
                    # DG 101,102,201,202; x3c
                    rec_path=[
                        # f"{root}/data/real_3d/v05/DG101/data_DG101_20211015_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG101/data_DG101_20211022_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG101/data_DG101_20211023_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG101/data_DG101_20211110_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG101/data_DG101_20211111_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG102/data_DG102_20211013_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG102/data_DG102_20211014_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG102/data_DG102_20211015_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG102/data_DG102_20211101_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG102/data_DG102_20211102_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG102/data_DG102_20211103_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG102/data_DG102_20211104_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG102/data_DG102_20211105_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG102/data_DG102_20211106_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG102/data_DG102_20211109_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG102/data_DG102_20211110_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG102/data_DG102_20211111_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG102/data_DG102_20211112_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG201/data_DG201_20211020_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG201/data_DG201_20211021_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG201/data_DG201_20211024_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG201/data_DG201_20211026_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG201/data_DG201_20211101_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG201/data_DG201_20211102_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG201/data_DG201_20211106_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG201/data_DG201_20211110_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG201/data_DG201_20211120_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG201/data_DG201_20211122_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG201/data_DG201_20211123_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG201/data_DG201_20211124_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG201/data_DG201_20211125_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG201/data_DG201_20211126_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG201/data_DG201_20211127_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG201/data_DG201_20211128_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG201/data_DG201_20211129_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211016_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211017_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211020_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211021_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211022_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211023_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211024_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211025_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211116_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211120_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211121_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211122_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211123_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211124_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211125_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211126_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211127_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211128_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211129_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG101/data_DG101_20211016_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG101/data_DG101_20211018_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG101/data_DG101_20211019_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG101/data_DG101_20211020_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG101/data_DG101_20211022_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG101/data_DG101_20211023_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG101/data_DG101_20211110_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG102/data_DG102_20211013_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG102/data_DG102_20211015_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG102/data_DG102_20211101_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211017_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211020_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211022_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/DG202/data_DG202_20211024_v05__no_front__.rec",  # noqa
                        # JL
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220311_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220313_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220314_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220318_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220320_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220321_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220323_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220324_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220311_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220312_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220313_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220315_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220316_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220318_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220319_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220320_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220321_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220322_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220324_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220328_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220312_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220313_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220314_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220315_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220316_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220317_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220318_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220319_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220321_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220322_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220325_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220311_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220312_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220313_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220315_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220318_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220319_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220320_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220322_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220323_v05__no_front__.rec",  # noqa
                        # f"{root}/data/real_3d/v05/JL918/data_JL918_20220324_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220401_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220402_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220403_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220404_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220406_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220407_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220408_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220409_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220410_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220413_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220414_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220415_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220416_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220417_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220418_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220419_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220420_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220421_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220422_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220423_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220424_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220425_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL082/data_JL082_20220427_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220402_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220403_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220405_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220407_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220408_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220409_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220412_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220416_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220417_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220418_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220421_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220422_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220426_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL515/data_JL515_20220427_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220401_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220402_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220403_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220404_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220406_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220407_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220412_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220413_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220414_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220415_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220416_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220417_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220418_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220419_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220421_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220422_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220423_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220424_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220425_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL773/data_JL773_20220426_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220402_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220403_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220404_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220405_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220406_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220407_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220412_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220413_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220415_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220416_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220417_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220418_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220421_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220422_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220424_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220425_v05__no_front__.rec",  # noqa
                        f"{root}/data/real_3d/v05/JL918/data_JL918_20220426_v05__no_front__.rec",  # noqa
                    ],
                    anno_path=[
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/person_data_DG101_20211015_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/person_data_DG101_20211022_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/person_data_DG101_20211023_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/person_data_DG101_20211110_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/person_data_DG101_20211111_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/person_data_DG102_20211013_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/person_data_DG102_20211014_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/person_data_DG102_20211015_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/person_data_DG102_20211101_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/person_data_DG102_20211102_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/person_data_DG102_20211103_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/person_data_DG102_20211104_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/person_data_DG102_20211105_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/person_data_DG102_20211106_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/person_data_DG102_20211109_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/person_data_DG102_20211110_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/person_data_DG102_20211111_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/person_data_DG102_20211112_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211020_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211021_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211024_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211026_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211101_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211102_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211106_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211110_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211120_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211122_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211123_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211124_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211125_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211126_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211127_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211128_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211129_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211016_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211017_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211020_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211021_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211022_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211023_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211024_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211025_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211116_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211120_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211121_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211122_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211123_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211124_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211125_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211126_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211127_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211128_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211129_v05__no_front__day.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/person_data_DG101_20211016_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/person_data_DG101_20211018_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/person_data_DG101_20211019_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/person_data_DG101_20211020_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/person_data_DG101_20211022_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/person_data_DG101_20211023_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/person_data_DG101_20211110_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/person_data_DG102_20211013_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/person_data_DG102_20211015_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/person_data_DG102_20211101_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211017_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211020_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211022_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/person_data_DG202_20211024_v05__no_front__night.anno.pb_rec",  # noqa
                        # JL
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL082/person_data_JL082_20220311_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL082/person_data_JL082_20220313_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL082/person_data_JL082_20220314_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL082/person_data_JL082_20220318_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL082/person_data_JL082_20220320_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL082/person_data_JL082_20220321_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL082/person_data_JL082_20220323_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL082/person_data_JL082_20220324_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL515/person_data_JL515_20220311_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL515/person_data_JL515_20220312_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL515/person_data_JL515_20220313_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL515/person_data_JL515_20220315_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL515/person_data_JL515_20220316_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL515/person_data_JL515_20220318_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL515/person_data_JL515_20220319_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL515/person_data_JL515_20220320_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL515/person_data_JL515_20220321_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL515/person_data_JL515_20220322_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL515/person_data_JL515_20220324_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL515/person_data_JL515_20220328_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL773/person_data_JL773_20220312_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL773/person_data_JL773_20220313_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL773/person_data_JL773_20220314_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL773/person_data_JL773_20220315_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL773/person_data_JL773_20220316_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL773/person_data_JL773_20220317_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL773/person_data_JL773_20220318_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL773/person_data_JL773_20220319_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL773/person_data_JL773_20220321_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL773/person_data_JL773_20220322_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL773/person_data_JL773_20220325_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL918/person_data_JL918_20220311_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL918/person_data_JL918_20220312_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL918/person_data_JL918_20220313_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL918/person_data_JL918_20220315_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL918/person_data_JL918_20220318_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL918/person_data_JL918_20220319_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL918/person_data_JL918_20220320_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL918/person_data_JL918_20220322_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL918/person_data_JL918_20220323_v05__no_front__night.anno.pb_rec",  # noqa
                        # f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/JL918/person_data_JL918_20220324_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220401_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220402_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220403_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220404_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220406_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220407_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220408_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220409_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220410_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220413_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220414_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220415_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220416_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220417_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220418_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220419_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220420_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220421_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220422_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220423_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220424_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220425_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL082/person_data_JL082_20220427_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL515/person_data_JL515_20220402_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL515/person_data_JL515_20220403_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL515/person_data_JL515_20220405_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL515/person_data_JL515_20220407_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL515/person_data_JL515_20220408_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL515/person_data_JL515_20220409_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL515/person_data_JL515_20220412_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL515/person_data_JL515_20220416_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL515/person_data_JL515_20220417_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL515/person_data_JL515_20220418_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL515/person_data_JL515_20220421_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL515/person_data_JL515_20220422_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL515/person_data_JL515_20220426_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL515/person_data_JL515_20220427_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220401_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220402_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220403_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220404_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220406_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220407_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220412_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220413_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220414_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220415_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220416_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220417_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220418_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220419_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220421_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220422_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220423_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220424_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220425_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL773/person_data_JL773_20220426_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL918/person_data_JL918_20220402_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL918/person_data_JL918_20220403_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL918/person_data_JL918_20220404_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL918/person_data_JL918_20220405_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL918/person_data_JL918_20220406_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL918/person_data_JL918_20220407_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL918/person_data_JL918_20220412_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL918/person_data_JL918_20220413_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL918/person_data_JL918_20220415_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL918/person_data_JL918_20220416_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL918/person_data_JL918_20220417_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL918/person_data_JL918_20220418_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL918/person_data_JL918_20220421_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL918/person_data_JL918_20220422_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL918/person_data_JL918_20220424_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL918/person_data_JL918_20220425_v05__no_front__night.anno.pb_rec",  # noqa
                        f"{root}/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/20220511/JL918/person_data_JL918_20220426_v05__no_front__night.anno.pb_rec",  # noqa
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
