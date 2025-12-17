from base_2d_night_lmdb_datasets import datapaths
from easydict import EasyDict

root = "dmpv2://matrix2"


# ----- Required -----
datapaths.update(
    dict(
        vehicle_3d_detection=dict(
            train_batch_size_per_ctx=12,
            train_data_paths=[
                # common 174236
                dict(
                    data_path=[
                        # DG102
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211011_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211012_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211013_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211015_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211016_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211028_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211029_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211101_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211113_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211114_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211116_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211117_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211118_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211119_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211125_v05__no_front__night_filtered.anno",  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211129_v05__no_front__night_filtered.anno',  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211204_v05__no_front__night_filtered.anno',  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20211205_v05__no_front__night_filtered.anno",  # noqa
                        # 4.2 add
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20220207_v05__no_front__night_filtered.anno",  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20220210_v05__no_front__night_filtered.anno',  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20220212_v05__no_front__night_filtered.anno',  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20220214_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20220216_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/rear/data_DG102_20220218_v05__no_front__night_filtered.anno",  # noqa
                        # DG101
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211016_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211018_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211019_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211020_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211021_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211022_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211023_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211110_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211112_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211113_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211114_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211115_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211201_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211202_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211203_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211204_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211206_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211207_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211210_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211211_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211212_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/rear/data_DG101_20211213_v05__no_front__night_filtered.anno",  # noqa
                        # DG201
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211112_v05__no_front__night_filtered.anno',  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211113_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211120_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211122_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211123_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211126_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211127_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211128_v05__no_front__night_filtered.anno",  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211129_v05__no_front__night_filtered.anno',  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211130_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211201_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211202_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211203_v05__no_front__night_filtered.anno",  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211205_v05__no_front__night_filtered.anno',  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211206_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211208_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211209_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211210_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211211_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211212_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211214_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20211215_v05__no_front__night_filtered.anno",  # noqa
                        # 4.2 add
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220207_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220208_v05__no_front__night_filtered.anno",  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220210_v05__no_front__night_filtered.anno',  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220211_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220212_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220214_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220215_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220217_v05__no_front__night_filtered.anno",  # noqa
                        # 7.11 add
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220622_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220623_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220624_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220625_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/rear/data_DG201_20220626_v05__no_front__night_filtered.anno",  # noqa
                        # DG202
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211017_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211020_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211022_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211024_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211117_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211120_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211121_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211123_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211124_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211125_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211126_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211127_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211128_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211129_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211130_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211202_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211203_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211206_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211208_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211209_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211210_v05__no_front__night_filtered.anno",  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/rear/data_DG202_20211212_v05__no_front__night_filtered.anno',  # noqa
                    ],
                    sample_weight=15.6,
                ),
                # rotbadcase: 4748
                dict(
                    data_path=[
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG102/night_badcase/with_calib_all_split/rear/data_DG102_2021-10-11-12__night_rotbadcase.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG101/night_badcase/with_calib_all_split/rear/data_DG101_2021-10-11-12__night_rotbadcase.anno",  # noqa
                        # 7.11 add
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/night_badcase/with_calib_all_split/rear/data_DG201_20220622_v05__rotation_badcase__night_.anno',  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/night_badcase/with_calib_all_split/rear/data_DG201_20220623_v05__rotation_badcase__night_.anno',  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/night_badcase/with_calib_all_split/rear/data_DG201_20220624_v05__rotation_badcase__night_.anno',  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/night_badcase/with_calib_all_split/rear/data_DG201_20220625_v05__rotation_badcase__night_.anno',  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/night_badcase/with_calib_all_split/rear/data_DG201_20220626_v05__rotation_badcase__night_.anno',  # noqa
                    ],
                    sample_weight=1.0,
                ),
                # special vehicle: 1308
                dict(
                    data_path=[
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG101/night_badcase/with_calib_all_split/rear/data_DG101_202110-202202_special_vehicle__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/night_badcase/with_calib_all_split/rear/data_DG201_202110-202202_special_vehicle__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG202/night_badcase/with_calib_all_split/rear/data_DG202_202110-202202_special_vehicle__night.anno",  # noqa
                    ],
                    sample_weight=0.3,
                ),
                # ================ side =======================
                # common 565515
                dict(
                    data_path=[
                        # DG102
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211011_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211012_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211013_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211015_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211016_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211028_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211029_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211101_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211113_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211114_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211116_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211117_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211118_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211119_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211125_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211129_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211204_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20211205_v05__no_front__night_filtered.anno",  # noqa
                        # 4.2 add
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20220207_v05__no_front__night_filtered.anno",  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20220210_v05__no_front__night_filtered.anno',  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20220212_v05__no_front__night_filtered.anno',  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20220214_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20220216_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG102/with_calib_all_split/side/data_DG102_20220218_v05__no_front__night_filtered.anno",  # noqa
                        # DG101
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211016_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211018_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211019_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211020_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211021_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211022_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211023_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211110_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211112_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211113_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211114_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211115_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211201_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211202_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211203_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211204_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211206_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211207_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211210_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211211_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211212_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG101/with_calib_all_split/side/data_DG101_20211213_v05__no_front__night_filtered.anno",  # noqa
                        # DG201
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211112_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211113_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211120_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211122_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211123_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211126_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211127_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211128_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211129_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211130_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211201_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211202_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211203_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211205_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211206_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211208_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211209_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211210_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211211_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211212_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211214_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20211215_v05__no_front__night_filtered.anno",  # noqa
                        # 4.2 add
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220207_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220208_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220210_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220211_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220212_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220214_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220215_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220217_v05__no_front__night_filtered.anno",  # noqa
                        # 7.11 add
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220622_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220623_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220624_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220625_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG201/with_calib_all_split/side/data_DG201_20220626_v05__no_front__night_filtered.anno",  # noqa
                        # DG202
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211017_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211020_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211022_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211024_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211117_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211120_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211121_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211123_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211124_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211125_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211126_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211127_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211128_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211129_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211130_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211202_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211203_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211206_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211208_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211209_v05__no_front__night_filtered.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211210_v05__no_front__night_filtered.anno",  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/DG202/with_calib_all_split/side/data_DG202_20211212_v05__no_front__night_filtered.anno',  # noqa
                    ],
                    sample_weight=15.6,
                ),
                # rotbadcase 14089
                dict(
                    data_path=[
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG102/night_badcase/with_calib_all_split/side/data_DG102_2021-10-11-12__night_rotbadcase.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG101/night_badcase/with_calib_all_split/side/data_DG101_2021-10-11-12__night_rotbadcase.anno",  # noqa
                        # 7.11 add
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/night_badcase/with_calib_all_split/side/data_DG201_20220622_v05__rotation_badcase__night_.anno",  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/night_badcase/with_calib_all_split/side/data_DG201_20220623_v05__rotation_badcase__night_.anno',  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/night_badcase/with_calib_all_split/side/data_DG201_20220624_v05__rotation_badcase__night_.anno',  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/night_badcase/with_calib_all_split/side/data_DG201_20220625_v05__rotation_badcase__night_.anno",  # noqa
                        # f'{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/night_badcase/with_calib_all_split/side/data_DG201_20220626_v05__rotation_badcase__night_.anno',  # noqa
                    ],
                    sample_weight=1.0,
                ),
                # special vehicle: 5564
                dict(
                    data_path=[
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG101/night_badcase/with_calib_all_split/side/data_DG101_202110-202202_special_vehicle__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG201/night_badcase/with_calib_all_split/side/data_DG201_202110-202202_special_vehicle__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/v05/DG202/night_badcase/with_calib_all_split/side/data_DG202_202110-202202_special_vehicle__night.anno",  # noqa
                    ],
                    sample_weight=0.3,
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
                # common 30878
                dict(
                    # DG 101,102,201,202; x3c
                    data_path=[
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211016_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211018_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211019_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211020_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211022_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211023_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211110_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211112_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211113_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211114_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211115_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211122_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211126_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211128_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211129_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211130_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211201_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211202_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211203_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211204_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211205_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211206_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211207_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211210_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211211_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211212_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/rear/person_data_DG101_20211213_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211013_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211015_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211101_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211113_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211114_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211115_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211116_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211117_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211118_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211119_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211120_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211125_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211129_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211204_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/rear/person_data_DG102_20211205_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211103_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211104_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211105_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211108_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211109_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211112_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211113_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211120_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211122_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211123_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211126_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211127_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211128_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211129_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211130_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211201_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211202_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211203_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211205_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211206_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211208_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211209_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211210_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211211_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211212_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211214_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/rear/person_data_DG201_20211215_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211017_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211020_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211022_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211024_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211110_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211112_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211114_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211117_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211120_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211121_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211123_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211124_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211125_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211126_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211127_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211128_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211129_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211130_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211202_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211203_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211206_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211208_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211209_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211210_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/rear/person_data_DG202_20211212_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220321_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220322_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220323_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220324_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220326_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220402_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220405_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/rear/person_data_DG102_20220406_v05__no_front__night.anno",  # noqa
                        # 11
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220622_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220623_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220624_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220625_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220626_v05__no_front__night.anno",  # noqa
                    ],
                    sample_weight=9,
                ),
                # =============== side ======================
                # common 112077
                dict(
                    # DG 101,102,201,202; x3c
                    data_path=[
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211016_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211018_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211019_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211020_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211022_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211023_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211110_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211112_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211113_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211114_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211115_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211122_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211126_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211128_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211129_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211130_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211201_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211202_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211203_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211204_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211205_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211206_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211207_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211210_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211211_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211212_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG101/with_calib_all_split/side/person_data_DG101_20211213_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211013_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211015_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211101_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211113_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211114_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211115_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211116_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211117_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211118_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211119_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211120_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211125_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211129_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211204_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG102/with_calib_all_split/side/person_data_DG102_20211205_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211103_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211104_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211105_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211108_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211109_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211112_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211113_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211120_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211122_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211123_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211126_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211127_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211128_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211129_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211130_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211201_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211202_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211203_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211205_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211206_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211208_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211209_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211210_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211211_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211212_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211214_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG201/with_calib_all_split/side/person_data_DG201_20211215_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211017_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211020_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211022_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211024_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211110_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211112_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211114_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211117_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211120_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211121_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211123_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211124_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211125_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211126_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211127_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211128_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211129_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211130_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211202_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211203_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211206_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211208_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211209_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211210_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_x3c_remove_all_ignore/DG202/with_calib_all_split/side/person_data_DG202_20211212_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220321_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220322_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220323_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220324_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220326_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220402_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220405_v05__no_front__night.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/DG102/with_calib_all_split/side/person_data_DG102_20220406_v05__no_front__night.anno",  # noqa
                        # 11
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220622_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220623_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220624_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220625_v05__no_front__night.anno",  # noqa
                        # f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/person_real_3d/v05_remove_all_ignore_jianan/maxfa_10.1/DG201/person_data_DG201_20220626_v05__no_front__night.anno",  # noqa
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
