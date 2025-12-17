# -*- coding:utf-8 -*-
# Copyright (c) Horizon Robotics, All rights reserved.

import pytest

from hat.utils.data_search import find_closest_element

sorted = sorted = [6, 10, 13, 16, 20]
find = [
    0,
    1,
    2,
    5,
    6,
    7,
    8,
    9,
    11.5,
    15,
    20,
    24,
    25,
    30,
]
bound_treshold = 5


def test_find_closet_element_default():
    finded = [find_closest_element(f, sorted) for f in find]
    target = [
        (6, 0),  # 0 -> 6
        (6, 0),  # 1 -> 6
        (6, 0),  # 2 -> 6
        (6, 0),  # 5 -> 6
        (6, 0),  # 6 -> 6
        (6, 0),  # 7 -> 6
        (6, 0),  # 8 -> 6
        (10, 1),  # 9 -> 10
        (10, 1),  # 11.5 -> 10
        (16, 3),  # 15 -> 16
        (20, 4),  # 20 -> 20
        (20, 4),  # 24 -> 20
        (20, 4),  # 25 -> 20
        (20, 4),  # 30 -> 20
    ]
    assert all([f == t for f, t in zip(finded, target)])


def test_find_cloest_element_cloest_left():
    finded = [find_closest_element(f, sorted, closest_left=True) for f in find]
    target = [
        (6, 0),  # 0 -> 6
        (6, 0),  # 1 -> 6
        (6, 0),  # 2 -> 6
        (6, 0),  # 5 -> 6
        (6, 0),  # 6 -> 6
        (6, 0),  # 7 -> 6
        (6, 0),  # 8 -> 6
        (6, 0),  # 9 -> 6
        (10, 1),  # 11.5 -> 10
        (13, 2),  # 15 -> 13
        (20, 4),  # 20 -> 20
        (20, 4),  # 24 -> 20
        (20, 4),  # 25 -> 20
        (20, 4),  # 30 -> 20
    ]
    assert all([f == t for f, t in zip(finded, target)])


def test_find_closet_element_closest_right():
    finded = [
        find_closest_element(f, sorted, closest_right=True) for f in find
    ]
    target = [
        (6, 0),  # 0 -> 6
        (6, 0),  # 1 -> 6
        (6, 0),  # 2 -> 6
        (6, 0),  # 5 -> 6
        (6, 0),  # 6 -> 6
        (10, 1),  # 7 -> 10
        (10, 1),  # 8 -> 10
        (10, 1),  # 9 -> 10
        (13, 2),  # 11.5 -> 13
        (16, 3),  # 15 -> 16
        (20, 4),  # 20 -> 20
        (20, 4),  # 24 -> 20
        (20, 4),  # 25 -> 20
        (20, 4),  # 30 -> 20
    ]
    assert all([f == t for f, t in zip(finded, target)])


def test_find_closet_element_bt():
    finded = [
        find_closest_element(f, sorted, out_of_bound_threshold=bound_treshold)
        for f in find
    ]
    target = [
        (None, None),  # 0 -> None
        (None, None),  # 1 -> None
        (6, 0),  # 2 -> 6
        (6, 0),  # 5 -> 6
        (6, 0),  # 6 -> 6
        (6, 0),  # 7 -> 6
        (6, 0),  # 8 -> 6
        (10, 1),  # 9 -> 10
        (10, 1),  # 11.5 -> 10
        (16, 3),  # 15 -> 16
        (20, 4),  # 20 -> 20
        (20, 4),  # 24 -> 20
        (None, None),  # 25 -> None
        (None, None),  # 30 -> None
    ]
    assert all([f == t for f, t in zip(finded, target)])


def test_find_closet_element_bt_left():
    finded = [
        find_closest_element(
            f, sorted, out_of_bound_threshold=bound_treshold, closest_left=True
        )
        for f in find
    ]
    target = [
        (None, None),  # 0 -> None
        (None, None),  # 1 -> None
        (6, 0),  # 2 -> None
        (6, 0),  # 5 -> None
        (6, 0),  # 6 -> 6
        (6, 0),  # 7 -> 6
        (6, 0),  # 8 -> 6
        (6, 0),  # 9 -> 6
        (10, 1),  # 11.5 -> 10
        (13, 2),  # 15 -> 13
        (20, 4),  # 20 -> 20
        (20, 4),  # 24 -> 20
        (None, None),  # 25 -> None
        (None, None),  # 30 -> None
    ]
    assert all([f == t for f, t in zip(finded, target)])


def test_find_closet_element_bt_right():
    finded = [
        find_closest_element(
            f,
            sorted,
            out_of_bound_threshold=bound_treshold,
            closest_right=True,
        )
        for f in find
    ]
    target = [
        (None, None),  # 0 -> None
        (None, None),  # 1 -> None
        (6, 0),  # 2 -> 6
        (6, 0),  # 5 -> 6
        (6, 0),  # 6 -> 6
        (10, 1),  # 7 -> 10
        (10, 1),  # 8 -> 10
        (10, 1),  # 9 -> 10
        (13, 2),  # 11.5 -> 13
        (16, 3),  # 15 -> 16
        (20, 4),  # 20 -> 20
        (20, 4),  # 24 -> None
        (None, None),  # 25 -> None
        (None, None),  # 30 -> None
    ]
    assert all([f == t for f, t in zip(finded, target)])


if __name__ == "__main__":
    pytest.main(["-s", __file__])
