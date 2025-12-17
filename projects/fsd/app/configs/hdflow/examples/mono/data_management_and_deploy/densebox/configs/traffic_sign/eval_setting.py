import logging
import os.path
from dataclasses import fields

import yaml
from hdflow.plugins.mono.data_management_and_deploy.eval_set.eval_setting.parameter_def import (  # noqa
    AttrDataEntry,
    ClassificationSettingConfig,
    DetectionSettingConfig,
    InstAttr,
)
from hdflow.plugins.mono.data_management_and_deploy.eval_set.eval_setting.setting import (  # noqa
    need_cls_resize_crop_merge_tasks,
)
from hdflow.plugins.mono.data_management_and_deploy.eval_set.eval_setting.setting_entry import (  # noqa
    default_combination_tags,
    default_independent_tags,
    get_default_settings,
    taskname2combination,
    taskname2independent,
    taskname2length,
)
from hdflow.plugins.mono.data_management_and_deploy.eval_set.eval_setting.task_name2setting_info import (  # noqa
    task_name2setting_info,
)

logger = logging.getLogger(__name__)

# 使用 yaml 更新 标准setting, 并注册到对应的数据结构中。
# 标志牌有一些默认的设置，都放在 hdflow/plugins/mono/data_management_and_deploy/eval_set/eval_setting/setting_template 中  # noqa
# setting 的写法首先阅读文档：https://horizonrobotics.feishu.cn/wiki/wikcnaI6RKbTZY2VbuZGZwROJXf?table=tblSKGyNT9kZ4vOb&view=vewP2B92zv  # noqa

current_dir = os.path.dirname(os.path.realpath(__file__))

# yaml 格式的模板
classname2yaml = {
    "TrafficSign_4pe_det_cls": f"{current_dir}/eval_setting_templates/cnts-eval-profile-det-cls-all-sep-template.yaml",  # noqa
    "TrafficSignGL_4pe_det": f"{current_dir}/eval_setting_templates/glts-eval-profile-det-template.yaml",  # noqa
    "TrafficSignGL_4pe_det_cls": f"{current_dir}/eval_setting_templates/glts-eval-profile-det-cls-all-sep-template.yaml",  # noqa
}

# 最短边，用于内参测距时的参考长度
sign_classname2short_side = {
    "TrafficSign_4pe_det": 0.8,
    "TrafficSign_4pe_det_cls": 0.8,
    "TrafficSignRamp_2pe_det": 0.8,
    "TrafficSignType_2pe_cls": 0.8,
    "TrafficSignTypeSpeedLimit_2pe_cls": 0.8,
    "TrafficSignOcclusion_2pe_cls": 0.8,
    "TrafficSignGL_4pe_det": 0.8,
    "TrafficSignGL_4pe_det_cls": 0.8,
}
# setting 中的roi 设置
sign_classname2width_height = {
    "TrafficSign_4pe_det": "all",
    "TrafficSign_4pe_det_cls": "all",
    "TrafficSignRamp_2pe_det": "all",
    "TrafficSignType_2pe_cls": "all",
    "TrafficSignTypeSpeedLimit_2pe_cls": "all",
    "TrafficSignOcclusion_2pe_cls": "all",
    "TrafficSignGL_4pe_det": "all",
    "TrafficSignGL_4pe_det_cls": "all",
}

# 需要多任务 resize/crop/merge评测的任务
need_cls_resize_crop_merge_tasks += [
    "TrafficSign_4pe_det_cls",
    "TrafficSignGL_4pe_det",
    "TrafficSignGL_4pe_det_cls",
]

# 评测距离段划分
taskname2length.update(
    {
        # traffic sign
        "TrafficSign_4pe_det_cls": [
            {
                "tag_type": "length",
                "tag_list": [
                    dict(digest="0_25", val=[0, 25]),
                    dict(digest="25_50", val=[25, 50]),
                    dict(digest="50_70", val=[50, 70]),
                    dict(digest="70_90", val=[70, 90]),
                    dict(digest="0_50", val=[0, 50]),
                    dict(digest="0_70", val=[0, 70]),
                    dict(digest="0_90", val=[0, 90]),
                ],
                "tag_filter": None,
                "extra_tags": None,
                "is_combination": True,
            },
        ],
        "TrafficSignGL_4pe_det": [
            {
                "tag_type": "length",
                "tag_list": [
                    dict(digest="0_25", val=[0, 25]),
                    dict(digest="25_50", val=[25, 50]),
                    dict(digest="50_70", val=[50, 70]),
                    dict(digest="70_90", val=[70, 90]),
                    dict(digest="0_50", val=[0, 50]),
                    dict(digest="0_70", val=[0, 70]),
                    dict(digest="0_90", val=[0, 90]),
                ],
                "tag_filter": None,
                "extra_tags": None,
                "is_combination": True,
            },
        ],
        "TrafficSignGL_4pe_det_cls": [
            {
                "tag_type": "length",
                "tag_list": [
                    dict(digest="0_25", val=[0, 25]),
                    dict(digest="25_50", val=[25, 50]),
                    dict(digest="50_70", val=[50, 70]),
                    dict(digest="70_90", val=[70, 90]),
                    dict(digest="0_50", val=[0, 50]),
                    dict(digest="0_70", val=[0, 70]),
                    dict(digest="0_90", val=[0, 90]),
                ],
                "tag_filter": None,
                "extra_tags": None,
                "is_combination": True,
            },
        ],
    }
)


# 一组组合标签
# 例如： [[tag1, tag2], [tag1, tag2, tag3]， 表示两组组合标签，
# 第一组是tag1, tag2 组合，第二组是tag1, tag2, tag3 组合
# 参考 hdflow.plugins.mono.data_management_and_deploy.eval_set.eval_setting.setting_entry中的写法  # noqa
# 以下几个task 写在底层，如有更新，可以在此覆盖
# TrafficSign_4pe_det, TrafficSignType_2pe_cls,
# TrafficSignOcclusion_2pe_cls, TrafficSignTypeSpeedLimit_2pe_cls
taskname2combination.update(
    {
        "TrafficSign_4pe_det_cls": default_combination_tags,
        "TrafficSignGL_4pe_det": [],
        "TrafficSignGL_4pe_det_cls": [],
    }
)


# 一组独立标签，
# 例如：[[], [tag1], [tag2], [tag3]], 表示有三组独立标签
# 第一组 [] 表示不加任何tag；第二组[tag1]表示tag1 的独立标签，以此类推
# 参考 hdflow.plugins.mono.data_management_and_deploy.eval_set.eval_setting.setting_entry中的写法  # noqa
# 以下几个task 写在底层，如有更新，可以在此覆盖
# TrafficSign_4pe_det, TrafficSignType_2pe_cls,
# TrafficSignOcclusion_2pe_cls, TrafficSignTypeSpeedLimit_2pe_cls
taskname2independent.update(
    {
        "TrafficSign_4pe_det_cls": default_independent_tags,
        "TrafficSignGL_4pe_det": [[]],
        "TrafficSignGL_4pe_det_cls": [[]],
    }
)


def setting_config_parser(cfg_dict, setting_type):
    cfg_setting_fileds = [it.name for it in fields(setting_type)]
    for key, value in cfg_dict.items():
        if key not in cfg_setting_fileds:
            logging.warning(
                f'Invalid key in "{setting_type.__name__}", delete it!'
            )
            cfg_dict.pop(key)

        if key == "attrs":
            for attrs_key, attrs_value in value.items():
                data_entry_list = []
                for data_entry in attrs_value:
                    data_entry_list.append(
                        setting_config_parser(data_entry, AttrDataEntry)
                    )
                cfg_dict[key][attrs_key] = data_entry_list
            cfg_dict[key] = setting_config_parser(cfg_dict[key], InstAttr)

    return setting_type(**cfg_dict)


def update_classname2template(
    task_name2setting_info,
):
    for classname, yaml_setting in classname2yaml.items():
        cfg = yaml.load(open(yaml_setting), Loader=yaml.SafeLoader)
        if "det" in classname.lower():
            setting_type = DetectionSettingConfig
        else:
            setting_type = ClassificationSettingConfig

        setting_config = setting_config_parser(cfg, setting_type)
        task_name2setting_info[classname] = dict()
        task_name2setting_info[classname]["template"] = setting_config

    return task_name2setting_info


def update_classname2width_height(task_name2setting_info):
    for key_i, val_i in sign_classname2width_height.items():
        task_name2setting_info[key_i]["width_or_height"] = val_i
    return task_name2setting_info


def update_classname2short_side(task_name2setting_info):
    for key_i, val_i in sign_classname2short_side.items():
        task_name2setting_info[key_i]["hort_side"] = val_i
    return task_name2setting_info


task_name2setting_info = update_classname2template(task_name2setting_info)
task_name2setting_info = update_classname2width_height(task_name2setting_info)
task_name2setting_info = update_classname2short_side(task_name2setting_info)
