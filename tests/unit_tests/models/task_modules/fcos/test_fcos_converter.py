# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.core.data_struct.app_struct import DetObjects
from hat.models.task_modules.fcos import FCOSConverter


@pytest.mark.parametrize(
    ["task_name", "cls_name_mapping"],
    [
        pytest.param("person_detection", {0: "person"}),
        pytest.param("cyclist", {0: "cyclist"}),
    ],
)
def test_fcos_converter(task_name, cls_name_mapping):

    if "_detection" not in task_name:
        with pytest.raises(AssertionError):
            FCOSConverter(
                task_name=task_name,
                cls_name_mapping=cls_name_mapping,
            )
    else:
        fcos_converter = FCOSConverter(
            task_name=task_name,
            cls_name_mapping=cls_name_mapping,
        )
        fake_det = torch.cat(
            (torch.randn((2, 5)), torch.randint(0, 10, (2, 1))), -1
        )
        fake_pred = {"pred_bboxes": [fake_det]}

        results = fcos_converter(fake_pred)
        assert isinstance(results[0], DetObjects)

        fake_det = torch.randn((2, 5))
        fake_pred = {"pred_bboxes": [fake_det]}

        with pytest.raises(AssertionError):
            fcos_converter(fake_pred)
