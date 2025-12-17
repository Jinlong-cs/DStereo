from easydict import EasyDict

root = "dmpv2://matrix2"
datapaths = dict()
datapaths.update(
    {
        "vehicle_3d_detection": {
            "train_data_paths": [
                # x03 常规数据
                {
                    "data_path": [
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221119_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221119_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221120_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221120_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221121_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221121_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221122_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221123_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221123_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221124_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221124_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221125_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221126_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221127_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221128_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221129_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/side_vehicle_v2/data_LX066_20221130_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221124_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221124_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221125_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221125_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221126_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221126_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221127_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221127_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221128_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221128_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221129_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221129_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221130_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/side_vehicle_v2/data_LX038_20221130_v05__no_front__night_filtered.anno",
                    ],
                    "sample_weight": 14.604,
                    "partition": "normal",
                },
            ]
        },
    }
)

datapaths = EasyDict(datapaths)
