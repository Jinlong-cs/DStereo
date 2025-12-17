import os
import re
import time
from tempfile import NamedTemporaryFile
from typing import Dict, List

import torch

from hat.utils import Config

__all__ = [
    "update_by_compile_model_type",
    "update_by_val_only",
    "remove_bev",
    "init_task_config",
    "get_time_now",
    "gpu_infinite_loop",
]


def update_by_compile_model_type(
    configs: List[Dict],
    compile_model_type: str,
    model_type: str,
) -> List[Dict]:
    """
    Update task config by `compile_model_type`.
    """

    def _only_one_bev_stage1(cfgs):
        """
        For bev-type task, only one stage1 is needed.
        """
        new_cfgs = []
        has_bev = False
        for cfg in cfgs:
            if "BEV_BASE_CONFIG" in cfg:
                if has_bev:
                    continue
                else:
                    has_bev = True
            new_cfgs.append(cfg)
        return new_cfgs

    if compile_model_type == "fisheye":
        assert model_type == "mt-fisheye"
        return configs

    elif compile_model_type == "6v_fov120":
        assert model_type == "mt-6v" or model_type == "bev-mt-6v"
        new_configs = [c for c in configs if "fov100" not in c.task_name]
        print(
            f"{[c.task_name for c in configs]} => {[c.task_name for c in new_configs]}"  # noqa
        )  # noqa
        new_configs = _only_one_bev_stage1(new_configs)
        return new_configs

    elif compile_model_type == "6v_fov100":
        assert model_type == "mt-6v" or model_type == "bev-mt-6v"
        # remove fov120, depth_pose_resflow, road_arrow
        new_configs = [
            c
            for c in configs
            if ("fov120" not in c.task_name)
            and ("depth_pose_resflow" not in c.task_name)
            and ("road_arrow" not in c.task_name)
        ]
        # keep stride4 output for fov100 in lane/parsing, to improve fps.
        # already done in task config.
        print(
            f"{[c.task_name for c in configs]} => {[c.task_name for c in new_configs]}"  # noqa
        )  # noqa
        new_configs = _only_one_bev_stage1(new_configs)
        return new_configs
    else:
        # `compile_model_type` is None or others, do nothing
        return configs


def update_by_val_only(configs: List[Dict], val_only: bool) -> List[Dict]:
    """
    Remove 2d task config when `val_only`, since 2d task don't support
    `val_only`.
    """
    if not val_only:
        return configs
    new_configs = [c for c in configs if ("auto_2d_v2" not in c._filename)]
    print(
        f"{[c.task_name for c in configs]} => {[c.task_name for c in new_configs]}"  # noqa
    )  # noqa
    return new_configs


def remove_bev(configs: List[Dict]) -> List[Dict]:
    """
    Remove bev task.
    """
    new_configs = [c for c in configs if ("bev" not in c._filename)]
    print(
        f"{[c.task_name for c in configs]} => {[c.task_name for c in new_configs]}"  # noqa
    )  # noqa
    return new_configs


_CCONFIG = None


def init_task_config(configs: List[str]) -> List[Config]:
    """
    Initialize config, support template.
    """
    assert configs
    actual_configs = list()
    for config_path in configs:
        if os.path.exists(config_path):
            config = Config.fromfile(config_path)
        else:
            assert "__" in config_path
            src_config_path = config_path.split("__")[0]
            assert os.path.exists(src_config_path)
            kv_pairs = config_path.split("__")[1:]
            dst_config_path = update_config_by_var(src_config_path, kv_pairs)
            config = Config.fromfile(dst_config_path)
            if os.path.exists(dst_config_path):
                os.remove(dst_config_path)
        actual_configs.append(config)
    return actual_configs


def update_config_by_var(config_path, kv_pairs):
    """
    Update config var in config_path by `kv_pairs`, and return updated config
    file saved in temp file path.

    """
    assert os.path.exists(config_path)
    dst_fid = NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=os.path.dirname(config_path)
    )
    kv_vars = {}
    for kv_pair in kv_pairs:
        assert kv_pair.startswith("$") and "=" in kv_pair
        key = kv_pair.split("=")[0][1:].strip()
        value = kv_pair.split("=")[1].strip()
        kv_vars[key] = value

    with open(config_path, "r") as src_fid:
        for line in src_fid:
            for k, v in kv_vars.items():
                m = re.search(f"^{k} = .*", line)
                if m is not None:
                    line = f"{k} = '{v}'\n"
                    break
            dst_fid.write(line)
    return dst_fid.name


def get_time_now():
    timeArray = time.localtime(int(time.time()))
    time_now = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    time_now = time_now.replace(" ", "").replace("-", "").replace(":", "")
    return time_now


def gpu_infinite_loop():
    """
    Do some calculations in gpu to make sure the utilization of gpu is not 0.
    Always used in some cases that you want run your cpu program
    in gpu cluster. Such as compiling model in gpu cluster.
    NOTE: This function will cost about 900M memory in gpu 0 until
    it`s parent process exits.
    """
    while True:
        a = torch.ones((100, 100, 100)).cuda()
        b = torch.ones((100, 100, 100)).cuda()
        a = a + b


def regroup_head_function(regroup_task, org_cfg, reg_cfg):
    class_name_list = []
    task_name_list = []
    for task_name in regroup_task.values():
        if "_" in task_name:
            task_list = task_name.split("_")
            for task in task_list:
                if "tl" in task:
                    class_name_list.append("traffic_light")
                elif "full" in task:
                    class_name_list.append("vehicle_full")
                elif "ts" in task:
                    class_name_list.appen("traffic_sign")
                else:
                    class_name_list.append(task)
                task_name_list.append(task)
        else:
            class_name_list.append(task_name)
            task_name_list.append(task_name)
    for task, abbr_task in zip(class_name_list, task_name_list):
        if isinstance(org_cfg, dict):
            for t in org_cfg.keys():
                if abbr_task in t or task in t:
                    org_cfg.pop(t)
                    break
        elif isinstance(org_cfg, list):
            for t in org_cfg:
                if abbr_task in t or task in t:
                    org_cfg.remove(t)
                    break
    if isinstance(org_cfg, dict):
        org_cfg.update(reg_cfg)
    elif isinstance(org_cfg, list):
        org_cfg.extend(reg_cfg)
    return org_cfg


def get_list(key, task_configs, merge_list=False, wrap_fn=None):
    result = []
    for config in task_configs:
        assert config
        val = getattr(config, key)
        if wrap_fn:
            task_name = config.task_name
            if isinstance(val, list):
                for i in range(len(val)):
                    val[i] = wrap_fn(task_name, val[i])
            else:
                val = wrap_fn(task_name, val)
        if isinstance(val, list) and merge_list:
            result.extend(val)
        else:
            result.append(val)
    return result
