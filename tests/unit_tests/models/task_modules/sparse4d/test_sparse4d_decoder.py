import pytest
import torch

from hat.models.task_modules.sparse4d import SparseBox3DDecoder


@pytest.mark.parametrize(
    ["bs", "num_pred", "num_classes", "state_dims"],
    [
        pytest.param(1, 0, 1, 10),
        pytest.param(10, 300, 10, 10),
        pytest.param(10, 900, 1, 10),
        pytest.param(10, 100, 10, 11),
    ],
)
def test_sparse_box3d_target(bs, num_pred, num_classes, state_dims):
    num_output = 300 if num_pred > 300 else num_pred
    decoder = SparseBox3DDecoder(num_output=num_output)
    for _ in range(10):
        cls_pred = [torch.randn((bs, num_pred, num_classes))]
        box_pred = [torch.randn((bs, num_pred, state_dims))]
        outputs = decoder(cls_pred, box_pred)
        assert len(outputs) == bs
        for output in outputs:
            assert "boxes_3d" in output
            assert output["boxes_3d"].shape == (
                decoder.num_output,
                state_dims - 1,
            )
            assert "scores_3d" in output
            assert output["scores_3d"].shape == (decoder.num_output,)
            assert "labels_3d" in output
            assert output["labels_3d"].shape == (decoder.num_output,)
