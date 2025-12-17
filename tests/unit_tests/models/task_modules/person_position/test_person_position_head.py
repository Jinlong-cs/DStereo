import pytest
import torch

from hat.models.task_modules.person_position.person_position_head import (
    PersonPostionHead,
)


@pytest.mark.parametrize(
    ["head_add_conv", "num_conv", "num_filter", "in_num_filter"],
    [
        pytest.param(True, 0, 128, 64),
        pytest.param(False, 0, 256, 128),
        pytest.param(True, 2, 128, 64),
        pytest.param(False, 2, 128, 64),
    ],
)
def test_person_position_head(
    head_add_conv, num_conv, num_filter, in_num_filter
):
    input = torch.randn((2, in_num_filter, 8, 8))
    head = PersonPostionHead(
        head_add_conv,
        4,
        4,
        num_conv=num_conv,
        num_filter=num_filter,
        in_num_filter=in_num_filter,
    )
    output = head(input)

    assert "pred_oms" in output
    assert "pred_dms" in output
