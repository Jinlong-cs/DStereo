import copy
import json
import os
import re
import time
from tempfile import NamedTemporaryFile
from typing import Dict, List

import numpy as np
import torch

from hat.utils import Config

__all__ = [
    "update_by_compile_model_type",
    "remove_bev",
    "init_task_config",
    "get_inference_info",
    "get_time_now",
    "gpu_infinite_loop",
    "del_item_in_dict",
    "update_item_in_dict",
    "remove_item_in_dict",
    "save_data_to_npy",
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
            and ("traffic_cone" not in c.task_name)
            and ("traffic_light" not in c.task_name)
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


def remove_bev(configs: List[Dict]) -> List[Dict]:
    """
    Remove bev task.
    """
    new_configs = [c for c in configs if ("bev" not in c._filename)]
    print(
        f"{[c.task_name for c in configs]} => {[c.task_name for c in new_configs]}"  # noqa
    )  # noqa
    return new_configs


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


def get_inference_info(
    key: str,
    inference_pack: Dict,
    bucket_root: str,
) -> Dict:
    info = dict(
        test_image_dir=None,
        test_attribte_json_path=None,
        test_image_calibration=None,
        test_image_dist_coeffs=None,
        test_homo_path=None,
        test_homo_offset_path=None,
        test_calib_path=None,
        test_camera_module_type=None,
    )
    if key is None:
        return info
    assert key in inference_pack, f"Unknown pack: {key}"
    inference_info = inference_pack[key]
    pack_id = inference_info["pack_id"]
    image_dir = inference_info["image_dir"]

    # default single view image path
    test_image_dir = os.path.join(bucket_root, pack_id, image_dir)

    # attribute info about this pack
    test_attribte_json_path = os.path.join(
        bucket_root, pack_id, "attribute.json"
    )
    att_file = json.load(open(test_attribte_json_path))

    # set calibration, used by real3d task
    test_image_calibration = np.zeros((3, 4), dtype=np.float32)
    test_image_calibration[:3, :3] = np.array(
        att_file["calibration"][image_dir]["K"], dtype=np.float32
    )
    # set dist_coeff, used by real3d task
    test_image_dist_coeffs = np.array(
        att_file["calibration"][image_dir]["d"], dtype=np.float32
    )

    test_homo_path = (
        os.path.join(bucket_root, inference_info["test_homo_path"])
        if inference_info.get("test_homo_path", None)
        else None
    )
    test_homo_offset_path = (
        os.path.join(bucket_root, inference_info["test_homo_offset_path"])
        if inference_info.get("test_homo_offset_path", None)
        else None
    )

    test_calib_path = (
        os.path.join(bucket_root, inference_info["test_calib_path"])
        if inference_info.get("test_calib_path", None)
        else None
    )

    test_camera_module_type = (
        os.path.join(bucket_root, inference_info["camera_module_type"])
        if inference_info.get("camera_module_type", None)
        else None
    )

    info["test_image_dir"] = test_image_dir
    info["test_attribte_json_path"] = test_attribte_json_path
    info["test_image_calibration"] = test_image_calibration
    info["test_image_dist_coeffs"] = test_image_dist_coeffs
    info["test_homo_path"] = test_homo_path
    info["test_homo_offset_path"] = test_homo_offset_path
    info["test_calib_path"] = test_calib_path
    info["test_camera_module_type"] = test_camera_module_type
    return info


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


def check_name(value, re_exp="\W"):  # noqa W605
    res = re.findall(re_exp, value)
    if res:
        print(
            f"Task name must match ^[a-zA-Z]+[a-zA-Z0-9_]*$, "
            f"but get {res} in your task name, "
            f"which will be replaced by `_`."
        )
        for c in res:
            value = value.replace(c, "_")
    return value


def get_rle_padding(size, multiplier_base=256):
    """Get padding size for width align to least multiple multiplier_base.

    Args:
        size (tuple): ordered (h, w).
        multiplier_base (int, optional): align width to the least mutiple
            of multipiler_base.
    """
    w = size[1]
    align_w = w
    if w % multiplier_base != 0:
        align_w = (w // multiplier_base + 1) * multiplier_base
    padding = [0, align_w - w, 0, 0]  # (left, right, top, bottom)
    use_rle_pad = align_w > w
    return use_rle_pad, padding, align_w


def del_item_in_dict(base_dict, del_dict):
    """
    Delete item in base_dict based on del_dict.
    Example:
        base_dict = {
            "BJ":
                "LS912": {
                    "LS912_1": None,
                    "LS912_2": None,
                },
            }
        del_dict = {
            "BJ":
                "LS912": {
                    "LS912_1": None,
                },
            }
        return :
            {
            "BJ":
                "LS912": {
                    "LS912_2": None,
                },
            }
    """
    base_dict = copy.deepcopy(base_dict)
    for location_name in del_dict.keys():
        for plat_name in del_dict[location_name]:
            for dataset_name in del_dict[location_name][plat_name]:
                assert (
                    dataset_name in base_dict[location_name][plat_name]
                ), f"delete {dataset_name} not in base_dict,please check again"  # noqa
                base_dict[location_name][plat_name].pop(dataset_name)
    return base_dict


def update_item_in_dict(base_dict, update_dict, only_use_update_version=False):
    """
    Update item in base_dict based on del_dict..
    Args:
        only_use_update_version: whther to return update datasets only.
                if Ture,set sample_interval =1 and only return update_dict,
                usually used in data check
    Example:
        base_dict = {
            "BJ":
                "LS912": {
                    "LS912_1": None,
                    "LS912_2": None,
                },
            }
        update_dict = {
            "BJ":
                "LS912": {
                    "LS912_1": {sample:1},
                    "LS912_3": None,
                },
            }
        return :
            {
            "BJ":
                "LS912": {
                    "LS912_1": {sample:1},
                    "LS912_2": None,
                    "LS912_3": None,
                },
            }
    """

    if only_use_update_version:
        for location_name in update_dict.keys():
            for plat_name in update_dict[location_name]:
                for dataset_name, dataset_info in update_dict[location_name][
                    plat_name
                ].items():
                    if dataset_info is None:
                        dataset_info = {}
                    dataset_info.update({"sample_interval": 1})
                    update_dict[location_name][plat_name][
                        dataset_name
                    ] = dataset_info
        return update_dict
    base_dict = copy.deepcopy(base_dict)
    for location_name in update_dict.keys():
        if location_name not in base_dict.keys():
            base_dict[location_name] = {}
        for plat_name in update_dict[location_name]:
            if plat_name not in base_dict[location_name].keys():
                base_dict[location_name][plat_name] = {}
            base_dict[location_name][plat_name].update(
                update_dict[location_name][plat_name]
            )
    return base_dict


# if tuple in data , save in json would raise TypeError:
# keys must be str, int, float, bool or None, not tuple
# but can be saved in numpy
def save_data_to_npy(data, save_path):
    """
    if tuple in data,should not save in json,but can save in numpy
    """
    save_dir, _ = os.path.split(save_path)
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_path, data)

    # if data is dict, should read like that:
    # np.load(save_path, allow_pickle=True).item()


def reduce_sample_interval(data_dict, scale=1, ignore_tags=None):
    """
    Reduce old dataset sample interval.
    """
    assert isinstance(scale, int) and scale > 1
    if ignore_tags is None:
        ignore_tags = []
    for location_name in data_dict.keys():
        for plat_name in data_dict[location_name]:
            for dataset_name, dataset_info in data_dict[location_name][
                plat_name
            ].items():
                for ignore_tag in ignore_tags:
                    if ignore_tag in dataset_name:
                        continue
                if dataset_info is None:
                    dataset_info = {}
                    dataset_info.update({"sample_interval": scale})
                else:
                    dataset_info["sample_interval"] *= scale
                data_dict[location_name][plat_name][
                    dataset_name
                ] = dataset_info
    return data_dict


def remove_item_in_dict(dataset_dict, remove_key):
    """
    Remove item in dataset_dict based on remove_key..
    Args:
        dataset_dict: the dataset needed to remove items.
        remove_key: data_names that contain the remove_key
            will be removed from the dataset_dict.

    Example:
        dataset_dict = {
            "BJ":
                "LS912": {
                    "data_name1_abc": None,
                    "data_name2_def": None,
                },
            }
        remove_key = "abc"
        return :
            {
            "BJ":
                "LS912": {
                    "data_name2_def": None,
                },
            }
    """
    rt_dataset_dict = copy.deepcopy(dataset_dict)
    for site, plate_dict in dataset_dict.items():
        for plate, date_dict in plate_dict.items():
            for data_name in date_dict.keys():
                if remove_key not in data_name:
                    rt_dataset_dict[site][plate].pop(data_name)
            if len(rt_dataset_dict[site][plate]) == 0:
                rt_dataset_dict[site].pop(plate)
        if len(rt_dataset_dict[site]) == 0:
            rt_dataset_dict.pop(site)
    return rt_dataset_dict


def replace_item_in_dict_with_postfix(base_dict, update_dict, postfix):
    """
    Replace dataset_name in base_dict based on update_dict..
    Args:
        dataset_dict: the dataset needed to remove items.
        remove_key: data_names that contain the remove_key
            will be removed from the dataset_dict.

    Example:
        base_dict = {
            "BJ":
                "LS912": {
                    "data_name1_abc": None,
                    "data_name2_def": None,
                },
            }
        update_dict = {
            "BJ":
                "LS912": {
                    "data_name1_abc_filtered": None,
                    "data_name1_efg_filtered": None,
                    "data_name2_def": None,
                },
            }
        postfix = "_filtered"
        return :
            {
            "BJ":
                "LS912": {
                    "data_name1_abc_filtered": None,
                    "data_name2_def": None,
                },
            }
    """
    rt_dataset_dict = copy.deepcopy(base_dict)
    for site, plate_dict in base_dict.items():
        if site not in update_dict.keys():
            continue
        for plate, date_dict in plate_dict.items():
            if plate not in update_dict[site].keys():
                continue
            for data_name in date_dict.keys():
                interested_key = f"{data_name}{postfix}"
                if interested_key in update_dict[site][plate].keys():
                    rt_dataset_dict[site][plate].pop(data_name)
                    rt_dataset_dict[site][plate][interested_key] = update_dict[
                        site
                    ][plate][interested_key]
            if len(rt_dataset_dict[site][plate]) == 0:
                rt_dataset_dict[site].pop(plate)
        if len(rt_dataset_dict[site]) == 0:
            rt_dataset_dict.pop(site)
    return rt_dataset_dict
