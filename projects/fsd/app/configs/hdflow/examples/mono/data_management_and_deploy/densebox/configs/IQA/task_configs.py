import os

from hdflow.plugins.mono.data_management_and_deploy.base.structure import (  # noqa
    DataDeployTasksCfgs,
)

__all__ = [
    "TASK_NAME_TO_TS_CFG",
]


root_dir = os.path.join(os.path.dirname(os.path.relpath(__file__)))


def anno_transform_list(default, **kwargs):
    sensor_map = {
        "10635": [(1280, 720)],
        "820": [(3840, 2160)],
        "390": [(1920, 1080)],
        "0323": [(2280, 1080), (2048, 1080), (1920, 908), (1920, 910)],  # noqa
        "0220": [(1824, 940)],
    }
    sensor_crop = {  # define by [up, bottom, left, right]
        (1920, 1080): [0, 144, 48, 48],  # 390
        (3840, 2160): [0, 288, 96, 96],  # 820 (96, 0), (3744, 1872)
        (2048, 1080): [0, 144, 112, 112],  # 323 (112, 0), (1936, 936)
        (2280, 1080): [0, 144, 228, 228],  # 323
        (1920, 908): [0, 121, 193, 193],  # 323
        (1920, 910): [0, 121, 192, 192],  # 323
        (1824, 940): [0, 0, 0, 0],  # 220
        (1280, 720): [0, 0, 0, 0],  # 10635
    }
    keep_sensor = ["0323", "10635", "390", "0220", "820"]
    keep_shape = []
    for item in keep_sensor:
        keep_shape += sensor_map[item]

    anno_transformer = dict(
        type="Compose",
        transformer=[
            dict(
                type="DenseBoxWorkConditionAnnoTs",
                one_class=True,
                verbose=False,
                keep_shape=tuple(keep_shape),
                sensor_crop=sensor_crop,
                tasks=["image_label"],
            ),
            default,
        ],
    )
    return anno_transformer


# eval set configs
eval_setting_config_default = {
    "cls16.2_add_freespace": "hdfs://hobot-bigdata/user/meng01.wang/setting/cls16.2_add_freespace.yaml"  # noqa
}

TASK_NAME_TO_TS_CFG = {
    # ----------------------------------------
    # block/blur/glare
    # ----------------------------------------
    "block_classification": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/block_classification_config.yaml",
        ),
        class_name=["normal", "mild", "medium", "severe"],
        task_owners_email=["jing.li@horizon.ai"],
        anno_transformer_update_fn=anno_transform_list,
        eval_setting_config=eval_setting_config_default,
        evalset_config=dict(
            task_name="ADAS Classification V2",
            dataset_name="block_classification-default-test",
            name_prefix="block_classification",
            eval_creator_cfg=dict(
                type="CommonClassification",
            ),
        ),
    ),
    "blur_classification": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/blur_classification_config.yaml",
        ),
        class_name=["normal", "mild", "medium", "severe"],
        task_owners_email=["jing.li@horizon.ai"],
        anno_transformer_update_fn=anno_transform_list,
        eval_setting_config=eval_setting_config_default,
        evalset_config=dict(
            task_name="ADAS Classification V2",
            dataset_name="blur_classification-default-test",
            name_prefix="blur_classification",
            eval_creator_cfg=dict(
                type="CommonClassification",
            ),
        ),
    ),
    "glare_classification": DataDeployTasksCfgs(
        task_ts_cfg=os.path.join(
            root_dir,
            "packing/glare_classification_config.yaml",
        ),
        class_name=["normal", "mild", "medium", "severe"],
        task_owners_email=["jing.li@horizon.ai"],
        anno_transformer_update_fn=anno_transform_list,
        eval_setting_config=eval_setting_config_default,
        evalset_config=dict(
            task_name="ADAS Classification V2",
            dataset_name="glare_classification-default-test",
            name_prefix="glare_classification",
            eval_creator_cfg=dict(
                type="CommonClassification",
            ),
        ),
    ),
}
