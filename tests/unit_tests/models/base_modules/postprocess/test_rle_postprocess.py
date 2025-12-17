import torch

from hat.registry import build_from_registry


def test_argmax_postprocess():

    shape = (2, 3, 256, 256)

    config = dict(
        type="RLEPostprocess", data_name="pred_segs", dtype=torch.int8
    )

    rle_postprocess = build_from_registry(config)

    pred_segs = [torch.randn(shape).argmax(dim=1, keepdim=True)]
    pred_dict = dict()
    pred_dict["pred_segs"] = pred_segs
    pred_dict = rle_postprocess(pred_dict)
    assert "pred_segs" in pred_dict
    rle_out = pred_dict["pred_segs"][0]
    assert (rle_out[1:][1::2]).sum() == shape[2] * shape[3]

    pred_segs = torch.randn(shape).argmax(dim=1, keepdim=True)
    pred_dict = dict()
    pred_dict["pred_segs"] = pred_segs
    pred_dict = rle_postprocess(pred_dict)
    assert "pred_segs" in pred_dict
    rle_out = pred_dict["pred_segs"]
    assert (rle_out[1:][1::2]).sum() == shape[2] * shape[3]
