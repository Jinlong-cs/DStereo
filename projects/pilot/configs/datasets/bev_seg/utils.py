import copy


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
