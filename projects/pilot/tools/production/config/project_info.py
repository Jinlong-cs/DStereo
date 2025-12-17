import os.path as osp
import re
from typing import List

from projects.pilot.configs.project_utils.enum import BEVModelSetting

dirname = osp.relpath(osp.dirname(__file__))

supported_project = [
    "c385",
    "as33",
    "cc02",
    "galaxy",
    "hc23",
    "ev52",
    "byd",
    "nio",
    "test",
] + list(BEVModelSetting.values())
# AutoThreshold config file of project
# (eval_data_setting, model_type): filename

_project_threshold_config_map_by_eval_data_setting = {
    # AS33
    ("as33_day", "resize_2"): osp.join(
        dirname, "auto_threshold/as33_day_resize_2.py"
    ),
    ("as33_night", "resize_2"): osp.join(
        dirname, "auto_threshold/as33_night_resize_2.py"
    ),
    ("as33_day", "resize_4"): osp.join(
        dirname, "auto_threshold/as33_day_resize_4.py"
    ),
    ("as33_night", "resize_4"): osp.join(
        dirname, "auto_threshold/as33_night_resize_4.py"
    ),
    ("as33_day", "crop"): osp.join(dirname, "auto_threshold/as33_day_crop.py"),
    ("as33_night", "crop"): osp.join(
        dirname, "auto_threshold/as33_night_crop.py"
    ),
    # C385
    ("c385_x3c_parking", "resize_2"): osp.join(
        dirname, "auto_threshold/sedan_x3c_parking_resize_2.py"
    ),
    ("c385_x3c_parking", "resize_4"): osp.join(
        dirname, "auto_threshold/sedan_x3c_parking_resize_4.py"
    ),
    ("c385_x3c_day", "resize_2"): osp.join(
        dirname, "auto_threshold/sedan_x3c_day_resize_2.py"
    ),
    ("c385_x3c_night", "resize_2"): osp.join(
        dirname, "auto_threshold/sedan_x3c_night_resize_2.py"
    ),
    ("c385_x3c_day", "resize_4"): osp.join(
        dirname, "auto_threshold/sedan_x3c_day_resize_4.py"
    ),
    ("c385_x3c_night", "resize_4"): osp.join(
        dirname, "auto_threshold/sedan_x3c_night_resize_4.py"
    ),
    ("c385_x3c_day", "crop"): osp.join(
        dirname, "auto_threshold/sedan_x3c_day_crop.py"
    ),
    ("c385_x3c_night", "crop"): osp.join(
        dirname, "auto_threshold/sedan_x3c_night_crop.py"
    ),
    # CC02
    ("cc02_day", "resize_2"): osp.join(
        dirname, "auto_threshold/cc02_day_resize_2.py"
    ),
    ("cc02_night", "resize_2"): osp.join(
        dirname, "auto_threshold/cc02_night_resize_2.py"
    ),
    ("cc02_day", "resize_4"): osp.join(
        dirname, "auto_threshold/cc02_day_resize_4.py"
    ),
    ("cc02_night", "resize_4"): osp.join(
        dirname, "auto_threshold/cc02_night_resize_4.py"
    ),
    ("cc02_day", "crop"): osp.join(dirname, "auto_threshold/cc02_day_crop.py"),
    ("cc02_night", "crop"): osp.join(
        dirname, "auto_threshold/cc02_night_crop.py"
    ),
    # Galaxy x02 resize
    ("galaxy_x02_side_day", "resize_2_side_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_x3c_side_day_resize.py"
    ),
    ("galaxy_x02_side_night", "resize_2_side_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_x3c_side_night_resize.py"
    ),
    ("galaxy_x02_rear_day", "resize_2_rear_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_0233_rear_day_resize.py"
    ),
    ("galaxy_x02_rear_night", "resize_2_rear_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_0233_rear_night_resize.py"
    ),
    # Galaxy x02 crop
    ("galaxy_x02_rear_day", "crop_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_0233_rear_day_crop.py"
    ),
    ("galaxy_x02_rear_night", "crop_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_0233_rear_night_crop.py"
    ),
    ("galaxy_x02_side_right_day", "crop_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_x3c_side_right_day_crop.py"
    ),
    ("galaxy_x02_side_right_night", "crop_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_x3c_side_right_night_crop.py"
    ),
    ("galaxy_x02_side_left_day", "crop_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_x3c_side_left_day_crop.py"
    ),
    ("galaxy_x02_side_left_night", "crop_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_x3c_side_left_night_crop.py"
    ),
    # Galaxy x03 resize
    ("galaxy_x03_side_day", "resize_2_side_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_x3c_side_day_resize.py"
    ),
    ("galaxy_x03_side_night", "resize_2_side_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_x3c_side_night_resize.py"
    ),
    ("galaxy_x03_rear_day", "resize_2_rear_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_0233_rear_day_resize.py"
    ),
    ("galaxy_x03_rear_night", "resize_2_rear_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_0233_rear_night_resize.py"
    ),
    # Galaxy x03 crop
    ("galaxy_x03_rear_day", "crop_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_0233_rear_day_crop.py"
    ),
    ("galaxy_x03_rear_night", "crop_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_0233_rear_night_crop.py"
    ),
    ("galaxy_x03_side_right_day", "crop_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_x3c_side_right_day_crop.py"
    ),
    ("galaxy_x03_side_right_night", "crop_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_x3c_side_right_night_crop.py"
    ),
    ("galaxy_x03_side_left_day", "crop_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_x3c_side_left_day_crop.py"
    ),
    ("galaxy_x03_side_left_night", "crop_bayes"): osp.join(
        dirname, "auto_threshold/galaxy_x3c_side_left_night_crop.py"
    ),
    # EV52
    ("ev52_x3c_day", "resize_2"): osp.join(
        dirname, "auto_threshold/suv_x3c_day_resize_2.py"
    ),
    ("ev52_x3c_night", "resize_2"): osp.join(
        dirname, "auto_threshold/suv_x3c_night_resize_2.py"
    ),
    ("ev52_x3c_day", "resize_4"): osp.join(
        dirname, "auto_threshold/suv_x3c_day_resize_4.py"
    ),
    ("ev52_x3c_night", "resize_4"): osp.join(
        dirname, "auto_threshold/suv_x3c_night_resize_4.py"
    ),
    ("ev52_x3c_day", "crop"): osp.join(
        dirname, "auto_threshold/suv_x3c_day_crop.py"
    ),
    ("ev52_x3c_night", "crop"): osp.join(
        dirname, "auto_threshold/suv_x3c_night_crop.py"
    ),
    # NIOFY_CN
    ("niofy_cn_x3c_side_day", "resize_2_side_bayes"): osp.join(
        dirname, "auto_threshold/niofy_cn_x3c_side_day_resize.py"
    ),
    ("niofy_cn_x3c_side_night", "resize_2_side_bayes"): osp.join(
        dirname, "auto_threshold/niofy_cn_x3c_side_night_resize.py"
    ),
    ("niofy_cn_x3c_rear_day", "resize_2_rear_bayes"): osp.join(
        dirname, "auto_threshold/niofy_cn_x3c_rear_day_resize.py"
    ),
    ("niofy_cn_x3c_rear_night", "resize_2_rear_bayes"): osp.join(
        dirname, "auto_threshold/niofy_cn_x3c_rear_night_resize.py"
    ),
    ("niofy_cn_x3c_rear_night", "crop_bayes"): osp.join(
        dirname, "auto_threshold/niofy_cn_x3c_rear_night_crop.py"
    ),
    ("niofy_cn_x3c_rear_day", "crop_bayes"): osp.join(
        dirname, "auto_threshold/niofy_cn_x3c_rear_day_crop.py"
    ),
    # test
    ("test", "resize_2"): osp.join(dirname, "auto_threshold/test_resize.py"),
    ("test", "resize_4"): osp.join(dirname, "auto_threshold/test_resize.py"),
    ("test", "resize_2_rear_bayes"): osp.join(
        dirname, "auto_threshold/test_resize.py"
    ),
    ("test", "resize_2_side_bayes"): osp.join(
        dirname, "auto_threshold/test_resize.py"
    ),
    ("test", "crop"): osp.join(dirname, "auto_threshold/test_crop.py"),
    ("test", "crop_bayes"): osp.join(dirname, "auto_threshold/test_crop.py"),
}


def get_project_thresh_config(eval_data_setting: str, model_type: str):
    return _project_threshold_config_map_by_eval_data_setting[
        (eval_data_setting, model_type)
    ]


def get_project_name(multi_data_setting: List[List[str]]):
    dag_project = set()
    for p in supported_project:
        for eval_data_settings in multi_data_setting:
            for eval_data_setting in eval_data_settings:
                if p in eval_data_setting:
                    dag_project.add(p)
    assert len(dag_project) > 0, "Please check the ltc in supported_project !!"
    return list(dag_project)


def standardized_model_version(model_version: str):
    # format check
    assert re.match(
        "v\d+\.\d+\.\d+$", model_version  # noqa
    ), "model_version must match 'v\d+\.\d+\.\d+$', such as v1.0.0."  # noqa
    # remove unused zero, such as v1.0.01 -> v1.0.1
    split_version = model_version[1:].split(".")
    standard_split_version = list(map(lambda x: str(int(x)), split_version))
    model_version = "v" + ".".join(standard_split_version)
    return model_version
