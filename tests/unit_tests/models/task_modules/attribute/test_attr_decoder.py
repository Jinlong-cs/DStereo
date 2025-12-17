from collections import OrderedDict

import pytest
import torch

from hat.core.data_struct.app_struct import DetObjects
from hat.models.task_modules.attribute import SoftmaxAttrDecoder
from hat.utils.package_helper import check_packages_available

# Optional requirement check
hatbc_available = check_packages_available("hatbc", raise_exception=False)

if hatbc_available:
    from hatbc.message import Instance


@pytest.mark.skipif(
    not hatbc_available,
    reason="hatbc is not available, skip",
)
def test_sofrmax_attr_decoder():
    task_name = "vehicle_attribute"
    task_type = "classification"

    attr_decoder = SoftmaxAttrDecoder(
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
    all_instances = pred_ret.to_hatbc_msg()
    assert isinstance(pred_ret, DetObjects)
    assert len(all_instances) == len(fake_res_data)
    assert isinstance(all_instances[0], Instance)
