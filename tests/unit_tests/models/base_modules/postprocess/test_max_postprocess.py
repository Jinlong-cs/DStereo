import pytest
import torch

from hat.registry import build_from_registry


@pytest.mark.parametrize("return_indices", [True, False])
def test_max_postprocess(return_indices):

    shape = (2, 9, 512, 960)
    max_shape = (2, 1, 512, 960)

    config = dict(
        type="MaxPostProcess",
        data_names=["bev3d_hm"],
        out_names=[["bev3d_hm", "bev3d_cls_id"]],
        dim=1,
        keepdim=True,
        return_indices=return_indices,
    )

    max_postprocess = build_from_registry(config)

    bev3d_hm = [torch.randn(shape)]
    pred_dict = dict()
    pred_dict["bev3d_hm"] = bev3d_hm
    pred_dict = max_postprocess(pred_dict)
    assert pred_dict["bev3d_hm"][0].shape == max_shape
    if return_indices:
        assert "bev3d_cls_id" in pred_dict
        assert pred_dict["bev3d_cls_id"][0].shape == max_shape

    bev3d_hm = torch.randn(shape)
    pred_dict = dict()
    pred_dict["bev3d_hm"] = bev3d_hm
    pred_dict = max_postprocess(pred_dict)
    assert pred_dict["bev3d_hm"].shape == max_shape
    if return_indices:
        assert "bev3d_cls_id" in pred_dict
        assert pred_dict["bev3d_cls_id"].shape == max_shape
