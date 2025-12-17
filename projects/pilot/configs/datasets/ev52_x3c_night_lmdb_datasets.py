from easydict import EasyDict
from maxfa_x3c_night_lmdb_datasets import datapaths

root = "dmpv2://matrix2"

# ----- Required -----

append_data = dict(
    vehicle_3d_detection=dict(
        train_batch_size_per_ctx=12,
        train_data_paths=[
            dict(
                data_path=[
                    # EV073
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/EV073/data_EV073_20230129_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/EV073/data_EV073_20230305_v05__no_front__night",  # noqa
                    # EV035
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/EV035/data_EV035_20230301_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/EV035/data_EV035_20230302_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/EV035/data_EV035_20230303_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/EV035/data_EV035_20230304_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/EV035/data_EV035_20230306_v05__no_front__night",  # noqa
                ],
                sample_weight=20.4,
            ),
            dict(
                data_path=[
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230403_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230405_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230406_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230407_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230408_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230409_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230410_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230411_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230412_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230413_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230414_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230415_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230416_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230308_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230310_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230311_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230312_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230313_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230314_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230316_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230317_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230318_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230319_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230320_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230321_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/EV035/data_EV035_20230322_v05__no_front__night",  # noqa
                ],
                sample_weight=102,
            ),
            dict(
                data_path=[
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230308_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230310_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230311_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230312_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230313_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230314_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230316_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230317_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230318_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230319_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230321_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230322_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230403_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230406_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230407_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230408_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230409_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230410_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230411_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230412_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/rotation_badcase/rotation_badcase_data_EV035_20230413_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230308_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230310_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230311_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230312_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230313_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230314_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230316_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230317_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230318_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230319_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230321_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230322_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230403_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230406_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230407_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230408_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230409_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230410_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230411_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230412_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0504/vehicle/special_vehicle_badcase/special_vehicle_badcase_data_EV035_20230413_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/night_glare_badcase/night_glare_badcase_data_EV035_20230317_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/night_glare_badcase/night_glare_badcase_data_EV035_20230318_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/night_glare_badcase/night_glare_badcase_data_EV035_20230319_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/night_glare_badcase/night_glare_badcase_data_EV035_20230321_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/night_glare_badcase/night_glare_badcase_data_EV035_20230322_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/night_glare_badcase/night_glare_badcase_data_EV035_20230403_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/night_glare_badcase/night_glare_badcase_data_EV035_20230406_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/night_glare_badcase/night_glare_badcase_data_EV035_20230407_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/night_glare_badcase/night_glare_badcase_data_EV035_20230408_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/night_glare_badcase/night_glare_badcase_data_EV035_20230409_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/night_glare_badcase/night_glare_badcase_data_EV035_20230410_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/night_glare_badcase/night_glare_badcase_data_EV035_20230411_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/night_glare_badcase/night_glare_badcase_data_EV035_20230412_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/night_glare_badcase/night_glare_badcase_data_EV035_20230413_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/truncation_badcase/truncation_badcase_data_EV035_20230317_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/truncation_badcase/truncation_badcase_data_EV035_20230318_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/truncation_badcase/truncation_badcase_data_EV035_20230319_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/truncation_badcase/truncation_badcase_data_EV035_20230320_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/truncation_badcase/truncation_badcase_data_EV035_20230321_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/truncation_badcase/truncation_badcase_data_EV035_20230322_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/truncation_badcase/truncation_badcase_data_EV035_20230403_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/truncation_badcase/truncation_badcase_data_EV035_20230405_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/truncation_badcase/truncation_badcase_data_EV035_20230406_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/truncation_badcase/truncation_badcase_data_EV035_20230407_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/truncation_badcase/truncation_badcase_data_EV035_20230408_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/truncation_badcase/truncation_badcase_data_EV035_20230409_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/truncation_badcase/truncation_badcase_data_EV035_20230410_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/truncation_badcase/truncation_badcase_data_EV035_20230411_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/truncation_badcase/truncation_badcase_data_EV035_20230412_v05__nofront_night__",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/badcase/v05/EV035_0530/vehicle/truncation_badcase/truncation_badcase_data_EV035_20230413_v05__nofront_night__",  # noqa
                ],
                sample_weight=17,
            ),
        ],
    ),
    person_3d_detection=dict(
        train_batch_size_per_ctx=12,
        train_data_paths=[
            dict(
                data_path=[
                    # EV073
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV073/data_EV073_20230129_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV073/data_EV073_20230305_v05__no_front__night",  # noqa
                    # EV035
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230301_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230302_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230303_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230304_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230306_v05__no_front__night",  # noqa
                ],
                sample_weight=10.8,
            ),
            dict(
                data_path=[
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230308_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230310_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230311_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230312_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230313_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230314_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230316_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230317_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230318_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230319_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230320_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230321_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230322_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230403_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230405_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230406_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230407_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230408_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230409_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230410_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230411_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230412_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230413_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230414_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230415_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV035/data_EV035_20230416_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV073/data_EV073_20230306_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV073/data_EV073_20230307_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV073/data_EV073_20230308_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV073/data_EV073_20230310_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV073/data_EV073_20230311_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV073/data_EV073_20230312_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV073/data_EV073_20230413_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV073/data_EV073_20230415_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV073/data_EV073_20230416_v05__no_front__night",  # noqa
                    f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/person3d/v05/EV073/data_EV073_20230418_v05__no_front__night",  # noqa
                ],
                sample_weight=54,
            ),
        ],
    ),
)

for task, task_detail in append_data.items():
    if task not in datapaths:
        datapaths[task] = task_detail
        continue
    datapaths[task]["train_data_paths"].extend(task_detail["train_data_paths"])


datapaths = EasyDict(datapaths)
buckets = [
    "matrix2",
]
