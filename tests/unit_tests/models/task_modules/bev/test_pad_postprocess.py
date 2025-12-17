import torch

from hat.models.task_modules.bev.postprocess import ANCPadPostprocess


def test_pad_postprocess():
    data = torch.randn((1, 2, 576, 352))
    padding = [0, 160, 0, 0]
    pad_postprocess = ANCPadPostprocess(
        data_name="preds",
        padding=padding,
    )
    pred_dict = {"preds": data}
    output = pad_postprocess(pred_dict)
    assert output["preds"].shape[-2:] == (576, 512)
