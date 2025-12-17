import torch

from hat.models.task_modules.deform_detr import DeformDetrPostProcess


def test_detr_postprocess():
    batched_inputs = dict(
        ori_img=[torch.rand(32, 32, 3)],
        img_name="1.jpg",
        scale_factor=torch.tensor(
            [
                [
                    [1.2500, 0.0000, 266.5000],
                    [0.0000, 1.2500, 0.0000],
                    [0.0000, 0.0000, 1.0000],
                ]
            ]
        ),
        img_id=000,
        resized_shape=[[32, 32, 3]],
    )
    outputs_class = torch.rand(1, 90, 80)
    outputs_box = torch.rand(1, 90, 4)

    module = DeformDetrPostProcess(select_box_nums_for_evaluation=10)
    results = module(
        batched_inputs=batched_inputs,
        box_cls=outputs_class,
        box_pred=outputs_box,
    )
    assert len(results["pred_bboxes"]) == len(batched_inputs["ori_img"])
    assert results["pred_bboxes"][0].shape[-1] == 6
