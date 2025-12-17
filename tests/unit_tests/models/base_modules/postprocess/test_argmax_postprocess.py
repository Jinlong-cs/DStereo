import torch

from hat.registry import build_from_registry


def test_argmax_postprocess():

    shape = (2, 9, 512, 960)
    argmax_shape = (2, 512, 960)

    config = dict(
        type="ArgmaxPostprocess", data_name="pred_segs", dim=1, keepdim=False
    )

    argmax_postprocess = build_from_registry(config)

    pred_segs = [torch.randn(shape)]
    pred_dict = dict()
    pred_dict["pred_segs"] = pred_segs
    pred_dict = argmax_postprocess(pred_dict)
    assert pred_dict["pred_segs"][0].shape == argmax_shape

    pred_segs = torch.randn(shape)
    pred_dict = dict()
    pred_dict["pred_segs"] = pred_segs
    pred_dict = argmax_postprocess(pred_dict)
    assert pred_dict["pred_segs"].shape == argmax_shape
