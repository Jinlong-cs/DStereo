from collections import OrderedDict
from typing import Dict

import torch

from hat.models.task_modules.workcondition import AttrDecoder, WkDecoder


def test_wk_decoder():

    task_type = "classification"

    wk_decoder = WkDecoder(
        task_descs=OrderedDict(
            [
                ("scene_classification", ("pred_cls", task_type)),
                ("weather_classification", ("pred_cls", task_type)),
            ]
        ),
        transforms=None,
        name_dict={
            "scene_classification": [
                "Highway",
                "Urban",
                "Rural",
                "Tunnel",
                "Charge_Station",
                "Underground_Parking_Lot",
            ],
            "weather_classification": [
                "Sunny",
                "Cloudy",
                "Rainy",
                "Snowy",
                "Heavy_Rain",
                "Other",
            ],
            "illumination_classification": [
                "natural_light",
                "lamplight",
                "Hard_Light",
                "dark",
            ],
            "time_classification": ["Day", "Night", "Other"],
        },
    )

    fake_res_data = [
        dict(pred_cls=torch.randn([1, 6])),
        dict(pred_cls=torch.randn([1, 6])),
    ]

    pred_ret = wk_decoder(*fake_res_data)
    assert isinstance(pred_ret[0], Dict)
    assert len(pred_ret[0].keys()) == len(fake_res_data) + 1


def test_attr_decoder():

    task_name = "vehicle_attribute"
    task_type = "classification"

    attr_decoder = AttrDecoder(
        task_descs=OrderedDict([(task_name, ("pred_cls", task_type))]),
        name_dict={
            "category": [
                "Sedan_Car",
                "SUV",
                "Bus",
                "BigTruck",
                "Lorry",
                "Bike",
                "MiniVan",
                "Special_vehicle",
                "Motorcycle_electrombile",
                "Tricycle",
                "Motor-Tricycle",
                "Vehicle_others",
                "Non-Motor Vehicle_others",
                "Tiny_car",
                "unknown",
                "Flatbed_Trucks",
                "Car_transporter",
                "Tank_truck",
                "Garbage_truck",
                "Digger",
                "Loader",
                "Vehicle_light",
            ],
            "part": ["full"],
            "occlusion": [
                "full_visible",
                "occluded",
                "heavily_occluded",
                "invisible",
                "unknown",
            ],
            "ignore": ["no", "yes"],
            "confidence": ["High", "Middle", "Low", "VeryLow", "Unknown"],
            "Orientation": ["Unknown", "facade", "oblique", "Transverse"],
            "truncation": ["None", "High", "Middle", "Low", "VeryLow"],
        },
    )

    fake_res_data = [
        dict(pred_cls=(torch.randn([1, 44]))),
    ]

    pred_ret = attr_decoder(*fake_res_data)
    assert isinstance(pred_ret[0], Dict)
    assert len(pred_ret[0].keys()) == 2
