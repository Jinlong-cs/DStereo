import os

from hdflow.plugins.mono.data_management_and_deploy.base.structure import (  # noqa
    DataDeployTasksCfgs,
)

__all__ = [
    "TASK_NAME_TO_TS_CFG",
]

root_dir = os.path.join(os.path.dirname(os.path.relpath(__file__)))

# eval set configs
eval_setting_config_default = {
    "lane_parsing_3_0_20": "hdfs://hobot-bigdata/user/shun.wang/eval_setting_lane_parsing/setting_us_4cls_3_0-20.yaml",  # noqa
    "lane_parsing_3_20_50": "hdfs://hobot-bigdata/user/shun.wang/eval_setting_lane_parsing/setting_us_4cls_3_20-50.yaml",  # noqa
    "lane_parsing_3_50_100": "hdfs://hobot-bigdata/user/shun.wang/eval_setting_lane_parsing/setting_us_4cls_3_50-100.yaml",  # noqa
    "lane_parsing_3_100_200": "hdfs://hobot-bigdata/user/shun.wang/eval_setting_lane_parsing/setting_us_4cls_3_100-200.yaml",  # noqa
    "lane_parsing_8_0_20": "hdfs://hobot-bigdata/user/shun.wang/eval_setting_lane_parsing/setting_us_4cls_8_0-20.yaml",  # noqa
    "lane_parsing_8_20_50": "hdfs://hobot-bigdata/user/shun.wang/eval_setting_lane_parsing/setting_us_4cls_8_20-50.yaml",  # noqa
    "lane_parsing_8_50_100": "hdfs://hobot-bigdata/user/shun.wang/eval_setting_lane_parsing/setting_us_4cls_8_50-100.yaml",  # noqa
    "lane_parsing_8_100_200": "hdfs://hobot-bigdata/user/shun.wang/eval_setting_lane_parsing/setting_us_4cls_8_100-200.yaml",  # noqa
    "lane_parsing_all": "hdfs://hobot-bigdata/user/shun.wang/eval_setting_lane_parsing/setting_us_4cls_all.yaml",  # noqa
}
evalset_config_default = dict(
    task_name="Semantic Segmentation",
    dataset_name="lane-parsing-test",
    name_prefix="lane-parsing-default",
    eval_creator_cfg=dict(
        type="CommonLaneParsing",
    ),
)

TASK_NAME_TO_TS_CFG = {
    # ----------------------------------------
    # Parsing
    # ----------------------------------------
    "lane_parsing_cn": DataDeployTasksCfgs(
        # packing
        task_ts_cfg=None,
        class_name=None,
        task_owners_email=["yingqian.zhao@horizon.ai"],
        parsing_label_map=os.path.join(
            root_dir,
            "packing/lane_parsing_labelmap_6cls.py",  # noqa
        ),
        keep_duplicate=False,
        # evalset
        eval_setting_config=eval_setting_config_default,
        evalset_config=evalset_config_default,
    ),
    "lane_parsing_us": DataDeployTasksCfgs(
        # packing
        task_ts_cfg=None,
        class_name=None,
        task_owners_email=["shun.wang@horizon.ai"],
        parsing_label_map=os.path.join(
            root_dir,
            "packing/lane_parsing_us_labelmap_4cls.py",  # noqa
        ),
        keep_duplicate=False,
        # evalset
        eval_setting_config=eval_setting_config_default,
        evalset_config=evalset_config_default,
    ),
}
