import torch

from hat.models.task_modules.detr import DetrPostProcess


def test_detr_postprocess():
    targets = dict(
        ori_img=[torch.rand(32, 32, 3)],
        img_name="1.jpg",
        img_id=000,
    )
    outputs_class = torch.rand(6, 1, 100, 81)
    outputs_box = torch.rand(6, 1, 100, 4)

    module = DetrPostProcess()
    results = module(
        outs=(outputs_class, outputs_box),
        targets=targets,
    )
    assert len(results["pred_bboxes"]) == len(targets["ori_img"])
    assert results["pred_bboxes"][0].shape[-1] == 6
