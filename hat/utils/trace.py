# Copyright (c) Horizon Robotics. All rights reserved.

from collections import ChainMap, OrderedDict
from collections.abc import Mapping
from typing import List

try:
    from hatbc.workflow.trace import make_traceable
except ImportError:
    make_traceable = property

__all__ = [
    "combine_dict",
    "get_item_from_dict",
    "get_part_dict",
    "get_item_from_list",
]


@make_traceable
def combine_dict(*dict_list):
    """
    Traceable function for combining multi dict to a single dict.

    usually used for building a graph model.
    """
    return OrderedDict(ChainMap(*dict_list))


@make_traceable
def get_item_from_list(data_list, index):
    """
    Traceable function return an item from a list.

    usually used for building a graph model.
    """
    return data_list[index]


@make_traceable
def get_item_from_dict(data_dict, name):
    """
    Traceable function to return an item from a dict.

    usually used for building a graph model.
    """
    return data_dict.get(name, None)


@make_traceable
def get_part_dict(data_dict: Mapping, names: List):
    """
    Traceable function to return a subsetion of of original dict.

    usually used for building a graph model.
    """
    result = OrderedDict()
    for name in names:
        if name in data_dict:
            result[name] = data_dict[name]
    return result
