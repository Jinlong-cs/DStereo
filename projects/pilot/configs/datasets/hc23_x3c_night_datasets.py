import os

from base_2d_day_datasets import datapaths
from easydict import EasyDict

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
root = os.path.join(bucket_root, "matrix")


datapaths.update(
    dict(
        multiview_vehicle_3d_detection=dict(
            train_data_paths=[
                dict(
                    rec_path=[
                        # V2.1
                        f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221109_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221110_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221111_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221113_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221105_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221106_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221107_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221108_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221109_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221110_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221111_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221113_v03__no_front__.rec",
                        # V3.0
                        f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221112_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221114_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221115_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221116_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221117_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221118_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221119_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221120_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221121_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221122_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221123_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221125_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221126_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221128_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221112_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221117_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221118_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221119_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221120_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221121_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221122_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221123_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221125_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD18/data_BYD18_20221120_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD18/data_BYD18_20221121_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD18/data_BYD18_20221122_v03__no_front__.rec",
                        f"{root}/data/real_3d/multi_view/v03/BYD18/data_BYD18_20221124_v03__no_front__.rec",
                    ],
                    sample_weight=9,
                ),
            ],
            val_data_paths=[
                dict(
                    json_file=f"{root}/users/zixiang.pei/data/3d/val/v221031/ut126/UT126_20220801_20220802_20220810/extract_annos/UT126_20220801_20220802_20220810_vehicle_sampled_duplicated.json",  # noqa
                    img_dir=f"{root}/users/zixiang.pei/data/3d/val/v221031/ut126/UT126_20220801_20220802_20220810/extract_annos/vehicle/imgs",  # noqa
                    sample_weight=16,
                ),
            ],
        ),
    )
)


datapaths["multiview_pedestrian_3d_detection"] = dict(
    train_data_paths=datapaths["multiview_vehicle_3d_detection"][
        "train_data_paths"
    ],
    val_data_paths=[
        dict(
            json_file=f"{root}/users/zixiang.pei/data/3d/val/v221031/ut126/UT126_20220801_20220802_20220810/extract_annos/UT126_20220801_20220802_20220810_person_sampled_duplicated.json",  # noqa
            img_dir=f"{root}/users/zixiang.pei/data/3d/val/v221031/ut126/UT126_20220801_20220802_20220810/extract_annos/person/imgs",  # noqa
        )
    ],
)

datapaths["multiview_cyclist_3d_detection"] = dict(
    train_data_paths=datapaths["multiview_vehicle_3d_detection"][
        "train_data_paths"
    ],
    val_data_paths=[
        dict(
            json_file=f"{root}/users/zixiang.pei/data/3d/val/v221031/ut126/UT126_20220801_20220802_20220810/extract_annos/UT126_20220801_20220802_20220810_cyclist_sampled_duplicated.json",  # noqa
            img_dir=f"{root}/users/zixiang.pei/data/3d/val/v221031/ut126/UT126_20220801_20220802_20220810/extract_annos/cyclist/imgs",  # noqa
        )
    ],
)

datapaths = EasyDict(datapaths)
buckets = [
    "matrix",
]
