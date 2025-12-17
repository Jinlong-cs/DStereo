import torch

from hat.models.task_modules.retinanet import (
    RetinanetMultiStrideFilter,
    RetinaNetPostProcess,
)


def test_retinanet_postprocess():
    anchors = [
        torch.rand(1, 36, 64, 64),
        torch.rand(1, 36, 32, 32),
        torch.rand(1, 36, 16, 16),
    ]
    scores = [
        torch.rand(1, 720, 64, 64),
        torch.rand(1, 720, 32, 32),
        torch.rand(1, 720, 16, 16),
    ]
    regs = [
        torch.rand(1, 36, 64, 64),
        torch.rand(1, 36, 32, 32),
        torch.rand(1, 36, 16, 16),
    ]
    filter = RetinanetMultiStrideFilter(
        strides=[8, 16, 32],
        threshold=-2.944,
    )
    post_process = RetinaNetPostProcess(
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=300,
        topk_candidates=1000,
    )
    preds = filter(
        cls_scores=scores,
        bbox_preds=regs,
    )
    predictions = post_process(
        boxes=anchors,
        preds=preds,
        image_shapes=[torch.tensor([1024, 1024, 3])],
    )
    assert len(predictions) == 1
    assert predictions[0].size(0) <= 300
    assert (predictions[0][:, 4] <= 1.0).all()
    assert (predictions[0][:, 5] <= 80).all()
