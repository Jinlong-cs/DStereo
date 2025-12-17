# Copyright (c) Horizon Robotics. All rights reserved.
# This file is used to manage all global variables in configs.

from typing import Any

global global_config
global_config = {}


def set_config(key: str, value: Any):
    global_config[key] = value


def get_config(key: str):
    try:
        return global_config[key]
    except KeyError:
        return None
