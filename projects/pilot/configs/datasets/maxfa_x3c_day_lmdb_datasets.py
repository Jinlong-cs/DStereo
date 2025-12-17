from base_2d_day_lmdb_datasets import datapaths
from easydict import EasyDict

root = "dmpv2://matrix2"

# ----- Required -----

datapaths.update(
    dict(
        vehicle_3d_detection=dict(
            train_batch_size_per_ctx=12,
            train_data_paths=[
                # common 907050
                dict(
                    data_path=[
                        # DG102
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211011_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211012_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211013_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211014_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211015_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211016_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211028_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211029_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211101_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211102_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211103_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211104_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211105_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211106_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211109_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211110_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211111_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211112_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211113_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211114_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211117_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211118_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211119_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211121_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211122_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211123_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211124_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211125_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211126_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211129_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211204_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211205_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20220211_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20220212_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20220214_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20220217_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20220218_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20220219_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20220223_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20220224_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20220225_v05__no_front__day_filtered.anno",  # noqa
                        # 4.27
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211211_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211212_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211213_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211217_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211218_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211219_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211221_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211222_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211225_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211226_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211230_v05__no_front__day_filtered.anno",  # noqa
                        # DG101
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211015_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211021_v05__no_front__day_filtered.anno",  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211022_v05__no_front__day_filtered.anno',  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211023_v05__no_front__day_filtered.anno',  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211110_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211111_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211112_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211115_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211116_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211117_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211118_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211119_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211120_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211121_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211201_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211202_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211203_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211204_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211206_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211207_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211209_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211210_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211211_v05__no_front__day_filtered.anno",  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211212_v05__no_front__day_filtered.anno',  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211213_v05__no_front__day_filtered.anno",  # noqa
                        # 4.27 add
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211214_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211215_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211217_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211219_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211220_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211221_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211222_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211223_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211224_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211226_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211227_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211228_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211230_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211231_v05__no_front__day_filtered.anno",  # noqa
                        # DG201
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211020_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211021_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211024_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211026_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211101_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211102_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211106_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211110_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211111_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211112_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211113_v05__no_front__day_filtered.anno",  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211114_v05__no_front__day_filtered.anno',  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211117_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211118_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211119_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211120_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211122_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211123_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211124_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211125_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211126_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211127_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211128_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211129_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211130_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211201_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211202_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211203_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211205_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211206_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211208_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211209_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211210_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211211_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211212_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211214_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211215_v05__no_front__day_filtered.anno",  # noqa
                        # 3.15 adda
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220210_v05__no_front__day_filtered.anno',  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220212_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220214_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220215_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220216_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220217_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220218_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220219_v05__no_front__day_filtered.anno",  # noqa
                        # 4.27
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211216_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211218_v05__no_front__day_filtered.anno",  # noqa
                        # 7.11 add
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220622_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220623_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220624_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220625_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220626_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220627_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220628_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220629_v05__no_front__day_filtered.anno",  # noqa
                        # DG202
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211016_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211017_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211020_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211021_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211022_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211023_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211024_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211025_v05__no_front__day_filtered.anno",  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211116_v05__no_front__day_filtered.anno',  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211117_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211118_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211119_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211120_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211121_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211122_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211123_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211124_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211125_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211126_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211127_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211128_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211129_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211130_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211201_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211202_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211203_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211204_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211205_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211206_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211207_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211208_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211209_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211210_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211211_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211212_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211213_v05__no_front__day_filtered.anno",  # noqa
                        # 4.27
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211220_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211221_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211222_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211225_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211231_v05__no_front__day_filtered.anno",  # noqa
                    ],
                    sample_weight=15,
                ),
                # rotbadcase 39586
                dict(
                    data_path=[
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG102/day_badcase/with_calib_all_split/rear/data_DG102_2021-10-11-12__day_rotbadcase.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG101/day_badcase/with_calib_all_split/rear/data_DG101_2021-10-11-12__day_rotbadcase.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/day_badcase/with_calib_all_split/rear/data_DG201_2021-10-11-12__day_rotbadcase.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG202/day_badcase/with_calib_all_split/rear/data_DG202_2021-10-11-12__day_rotbadcase.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/day_badcase/with_calib_all_split/rear/data_DG201_20220622_v05__rotation_badcase__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/day_badcase/with_calib_all_split/rear/data_DG201_20220623_v05__rotation_badcase__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/day_badcase/with_calib_all_split/rear/data_DG201_20220624_v05__rotation_badcase__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/day_badcase/with_calib_all_split/rear/data_DG201_20220625_v05__rotation_badcase__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/day_badcase/with_calib_all_split/rear/data_DG201_20220626_v05__rotation_badcase__day_.anno",  # noqa
                    ],
                    sample_weight=0.5,
                ),
                # special vehicle 7423
                dict(
                    data_path=[
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG102/day_badcase/with_calib_all_split/rear/data_DG102_202110-202202_special_vehicle__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG101/day_badcase/with_calib_all_split/rear/data_DG101_202110-202202_special_vehicle__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/day_badcase/with_calib_all_split/rear/data_DG201_202110-202202_special_vehicle__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG202/day_badcase/with_calib_all_split/rear/data_DG202_202110-202202_special_vehicle__day.anno",  # noqa
                    ],
                    sample_weight=0.2,
                ),
                # ========================== side =======================
                # common 2881102
                dict(
                    data_path=[
                        # DG102
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211011_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211012_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211013_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211014_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211015_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211016_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211028_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211029_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211101_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211102_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211103_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211104_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211105_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211106_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211109_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211110_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211111_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211112_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211113_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211114_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211117_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211118_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211119_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211121_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211122_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211123_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211124_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211125_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211126_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211129_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211204_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211205_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20220211_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20220212_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20220214_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20220217_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20220218_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20220219_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20220223_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20220224_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20220225_v05__no_front__day_filtered.anno",  # noqa
                        # 4.27
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211211_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211212_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211213_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211217_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211218_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211219_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211221_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211222_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211225_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211226_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211230_v05__no_front__day_filtered.anno",  # noqa
                        # DG101
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211015_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211021_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211022_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211023_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211110_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211111_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211112_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211115_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211116_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211117_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211118_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211119_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211120_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211121_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211201_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211202_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211203_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211204_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211206_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211207_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211209_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211210_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211211_v05__no_front__day_filtered.anno",  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211212_v05__no_front__day_filtered.anno',  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211213_v05__no_front__day_filtered.anno",  # noqa
                        # 4.27 add
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211214_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211215_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211217_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211219_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211220_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211221_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211222_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211223_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211224_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211226_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211227_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211228_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211230_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211231_v05__no_front__day_filtered.anno",  # noqa
                        # DG201
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211020_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211021_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211024_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211026_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211101_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211102_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211106_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211110_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211111_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211112_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211113_v05__no_front__day_filtered.anno",  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211114_v05__no_front__day_filtered.anno',  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211117_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211118_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211119_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211120_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211122_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211123_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211124_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211125_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211126_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211127_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211128_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211129_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211130_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211201_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211202_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211203_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211205_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211206_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211208_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211209_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211210_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211211_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211212_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211214_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211215_v05__no_front__day_filtered.anno",  # noqa
                        # 3.15 adda
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220210_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220212_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220214_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220215_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220216_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220217_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220218_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220219_v05__no_front__day_filtered.anno",  # noqa
                        # 4.27
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211216_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211218_v05__no_front__day_filtered.anno",  # noqa
                        # 7.11 add
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220622_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220623_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220624_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220625_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220626_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220627_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220628_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220629_v05__no_front__day_filtered.anno",  # noqa
                        # DG202
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211016_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211017_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211020_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211021_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211022_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211023_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211024_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211025_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211116_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211117_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211118_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211119_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211120_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211121_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211122_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211123_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211124_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211125_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211126_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211127_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211128_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211129_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211130_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211201_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211202_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211203_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211204_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211205_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211206_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211207_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211208_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211209_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211210_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211211_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211212_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211213_v05__no_front__day_filtered.anno",  # noqa
                        # 4.27
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211220_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211221_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211222_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211225_v05__no_front__day_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211231_v05__no_front__day_filtered.anno",  # noqa
                    ],
                    sample_weight=15,
                ),
                # rotbadcase: 120419
                dict(
                    data_path=[
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG102/day_badcase/with_calib_all_split/side/data_DG102_2021-10-11-12__day_rotbadcase.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG101/day_badcase/with_calib_all_split/side/data_DG101_2021-10-11-12__day_rotbadcase.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/day_badcase/with_calib_all_split/side/data_DG201_2021-10-11-12__day_rotbadcase.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG202/day_badcase/with_calib_all_split/side/data_DG202_2021-10-11-12__day_rotbadcase.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/day_badcase/with_calib_all_split/side/data_DG201_20220622_v05__rotation_badcase__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/day_badcase/with_calib_all_split/side/data_DG201_20220623_v05__rotation_badcase__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/day_badcase/with_calib_all_split/side/data_DG201_20220624_v05__rotation_badcase__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/day_badcase/with_calib_all_split/side/data_DG201_20220625_v05__rotation_badcase__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/day_badcase/with_calib_all_split/side/data_DG201_20220626_v05__rotation_badcase__day_.anno",  # noqa
                    ],
                    sample_weight=0.5,
                ),
                # special vehicle: 29089
                dict(
                    data_path=[
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG102/day_badcase/with_calib_all_split/side/data_DG102_202110-202202_special_vehicle__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG101/day_badcase/with_calib_all_split/side/data_DG101_202110-202202_special_vehicle__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/day_badcase/with_calib_all_split/side/data_DG201_202110-202202_special_vehicle__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG202/day_badcase/with_calib_all_split/side/data_DG202_202110-202202_special_vehicle__day.anno",  # noqa
                    ],
                    sample_weight=0.2,
                ),
            ],
            val_batch_size_per_ctx=4,
            val_data_paths=[
                dict(
                    rec_path=[],
                    data_path=[],
                    sample_weight=5,
                ),
            ],
        ),
        person_3d_detection=dict(
            train_batch_size_per_ctx=16,
            train_data_paths=[
                # common 265737
                dict(
                    # DG 101,102,201,202; x3c
                    data_path=[
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211015_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211022_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211023_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211110_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211111_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211112_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211115_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211116_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211117_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211118_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211119_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211120_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211121_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211122_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211123_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211125_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211126_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211128_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211129_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211130_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211201_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211202_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211203_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211204_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211205_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211206_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211207_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211209_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211210_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211211_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211212_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211213_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211214_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211013_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211014_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211015_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211101_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211102_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211103_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211104_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211105_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211106_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211109_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211110_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211111_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211112_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211113_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211114_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211115_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211117_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211118_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211119_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211120_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211121_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211122_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211123_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211124_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211125_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211126_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211129_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211204_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211205_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211020_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211021_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211024_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211026_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211101_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211102_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211103_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211104_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211105_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211106_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211108_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211109_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211110_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211111_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211112_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211113_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211115_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211116_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211117_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211118_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211119_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211120_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211121_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211122_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211123_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211124_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211125_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211126_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211127_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211128_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211129_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211130_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211201_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211202_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211203_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211205_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211206_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211208_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211209_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211210_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211211_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211212_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211214_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211215_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211016_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211017_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211020_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211021_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211022_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211023_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211024_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211025_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211110_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211111_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211112_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211113_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211114_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211115_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211116_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211117_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211118_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211119_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211120_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211121_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211122_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211123_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211124_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211125_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211126_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211127_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211128_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211129_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211130_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211201_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211202_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211203_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211204_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211205_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211206_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211207_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211208_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211209_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211210_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211211_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211212_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211213_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220316_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220317_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220319_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220321_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220322_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220323_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220324_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220326_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220402_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220403_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220405_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220406_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220407_v05__no_front__day.anno",  # noqa
                        # 11
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220620_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220621_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220622_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220623_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220624_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220625_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220626_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220627_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220628_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220629_v05__no_front__day.anno",  # noqa
                    ],
                    sample_weight=9,
                ),
                # ============================ side =========================
                # common 823581
                dict(
                    # DG 101,102,201,202; x3c
                    data_path=[
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211015_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211022_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211023_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211110_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211111_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211112_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211115_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211116_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211117_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211118_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211119_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211120_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211121_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211122_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211123_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211125_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211126_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211128_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211129_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211130_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211201_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211202_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211203_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211204_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211205_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211206_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211207_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211209_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211210_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211211_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211212_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211213_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211214_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211013_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211014_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211015_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211101_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211102_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211103_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211104_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211105_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211106_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211109_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211110_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211111_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211112_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211113_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211114_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211115_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211117_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211118_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211119_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211120_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211121_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211122_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211123_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211124_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211125_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211126_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211129_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211204_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211205_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211020_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211021_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211024_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211026_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211101_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/person_data_DG201_20211102_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211103_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211104_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211105_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211106_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211108_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211109_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211110_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211111_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211112_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211113_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211115_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211116_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211117_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211118_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211119_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211120_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211121_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211122_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211123_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211124_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211125_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211126_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211127_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211128_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211129_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211130_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211201_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211202_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211203_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211205_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211206_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211208_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211209_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211210_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211211_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211212_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211214_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211215_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211016_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211017_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211020_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211021_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211022_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211023_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211024_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211025_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211110_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211111_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211112_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211113_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211114_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211115_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211116_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211117_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211118_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211119_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211120_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211121_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211122_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211123_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211124_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211125_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211126_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211127_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211128_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211129_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211130_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211201_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211202_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211203_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211204_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211205_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211206_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211207_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211208_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211209_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211210_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211211_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211212_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211213_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220316_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220317_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220319_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220321_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220322_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220323_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220324_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220326_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220402_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220403_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220405_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220406_v05__no_front__day.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220407_v05__no_front__day.anno",  # noqa
                        # 11
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220620_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220621_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220622_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220623_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220624_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220625_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220626_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220627_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220628_v05__no_front__day.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220629_v05__no_front__day.anno",  # noqa
                    ],
                    sample_weight=9,
                ),
            ],
            val_batch_size_per_ctx=10,
            val_data_paths=[
                dict(
                    rec_path=[],
                    data_path=[],
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
