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
                        # x03 1.0+2.0+3.0
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221119_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221120_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221121_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221122_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221123_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221124_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221125_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221126_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221127_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221128_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221129_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221130_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221201_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221202_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221203_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221204_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221205_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221206_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221207_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221208_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221209_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221210_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221211_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221212_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221215_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221216_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221217_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221222_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221223_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221225_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221226_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221227_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221228_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221229_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221230_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221119_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221120_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221121_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221123_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221124_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221203_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221204_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221205_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221206_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221207_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221208_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221209_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221210_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221211_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221215_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221216_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221217_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221222_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221223_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221225_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221226_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221227_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221228_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221229_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221230_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX066/rear_vehicle_v2/data_LX066_20221231_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221124_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221125_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221126_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221127_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221128_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221129_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221130_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221201_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221202_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221203_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221204_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221205_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221206_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221207_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221208_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221209_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221210_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221211_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221212_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221213_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221215_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221216_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221218_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221219_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221220_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221221_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221222_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221223_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221224_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221225_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221226_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221227_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221228_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221229_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221230_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221124_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221125_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221126_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221127_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221128_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221129_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221130_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221201_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221202_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221203_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221204_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221205_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221206_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221207_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221208_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221209_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221210_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221211_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221212_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221213_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221215_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221216_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221217_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221218_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221219_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221220_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221221_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221222_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221223_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221224_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221225_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221226_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221227_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221228_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221229_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221230_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX038/rear_vehicle_v2/data_LX038_20221231_v05__no_front__night_filtered.anno",  # noqa
                    ],
                    "sample_weight": 4,
                    "partition": "normal",
                },
                # x03 event链路 角度badcase
                {
                    "data_path": [
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX038_20221216_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX038_20221217_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX038_20221218_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX038_20221221_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX038_20221222_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX038_20221223_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX038_20221224_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX038_20221225_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX038_20221226_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX038_20221227_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX038_20221228_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX038_20221229_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX038_20221230_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX066_20221202_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX066_20221205_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX066_20221209_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX066_20221222_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX066_20221223_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX066_20221224_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX066_20221225_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX066_20221226_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX066_20221227_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX066_20221228_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX066_20221229_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DGDLMismatch/data_LX066_20221230_VehicleDet3DGDLMismatch__no_front__.anno",  # noqa
                    ],
                    "sample_weight": 0.1,
                    "partition": "rotation",
                },
                # x03 event链路 角度badcase
                {
                    "data_path": [
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX038_20221216_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX038_20221217_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX038_20221218_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX038_20221220_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX038_20221221_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX038_20221222_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX038_20221223_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX038_20221224_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX038_20221225_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX038_20221226_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX038_20221227_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX038_20221228_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX038_20221229_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX038_20221230_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX066_20221202_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX066_20221205_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX066_20221208_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX066_20221222_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX066_20221223_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX066_20221224_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX066_20221225_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX066_20221226_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX066_20221227_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX066_20221229_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy/x03_badcase_VehicleDet3DRearpartMismatch/data_LX066_20221230_VehicleDet3DRearpartMismatch__no_front__.anno",  # noqa
                    ],
                    "sample_weight": 0.1,
                    "partition": "rotation",
                },
            ]
        },
    }
)
datapaths = EasyDict(datapaths)
