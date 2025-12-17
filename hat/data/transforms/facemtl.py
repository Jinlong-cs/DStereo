# Copyright (c) Horizon Robotics. All rights reserved.
from hat.registry import OBJECT_REGISTRY

__all__ = ["FaceMtlTransformLabel"]


@OBJECT_REGISTRY.register
class FaceMtlTransformLabel(object):
    """Transform label to fit multitaskgraphmodel.

    Args:
        name: task name.
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, data):
        tmp_dict = {}
        pop_keys = []
        for key, value in data.items():
            if key == "img":
                continue
            pop_keys.append(key)
            tmp_dict.update({key: value})
        data[self.name] = tmp_dict
        for key in pop_keys:
            data.pop(key)
        return data
