import numpy as np
import pytest
import torch
import torch.nn as nn

from hat.models.base_modules.postprocess import FilterModule

try:
    import hat_sim
except ImportError:
    hat_sim = None


def gen_fake_data(h, w, strides, num_classes):
    """Generate fake data for detection loss and target test."""
    cls_scores, bbox_preds, centernesses = [], [], []
    for stride in strides:
        assert (w % stride) == 0
        assert (h % stride) == 0
        stride_h = h // stride
        stride_w = w // stride
        cls_pred = torch.randn((1, num_classes, stride_h, stride_w))
        bbox_pred = torch.randn((1, 4, stride_h, stride_w))
        centerness_pred = torch.randn((1, 1, stride_h, stride_w))
        cls_scores.append(cls_pred)
        bbox_preds.append(bbox_pred)
        centernesses.append(centerness_pred)
    label = {}
    label["img_name"] = "dummy.jpg"
    label["img_id"] = 0
    label["gt_bboxes"] = [
        torch.Tensor(
            [
                [w // 8, h // 8, w // 3, h // 3],
                [w // 2, h // 2, w * 3 // 4, h * 3 // 4],
            ]
        ),
    ]
    label["gt_classes"] = [
        torch.Tensor([1, 1]).long(),
    ]
    label["layout"] = ["chw"]
    label["pad_shape"] = [[3, h, w]]
    label["scale_factor"] = [1.0]
    label["crop_offset"] = [[0, 0, 0, 0]]
    label["before_crop_shape"] = [[3, h, w]]

    # filter sim
    filters = nn.ModuleList(
        [
            FilterModule(
                threshold=0.0,
                idx_range=None,
            )
            for _ in range(len(strides))
        ]
    )

    mlvl_outputs = []
    for level in range(len(strides)):
        (
            per_level_cls_scores,
            per_level_bbox_preds,
            per_level_centernesses,
        ) = (cls_scores[level], bbox_preds[level], centernesses[level])
        filter_input = [
            per_level_cls_scores,
            per_level_bbox_preds,
            per_level_centernesses,
        ]
        for per_filter_input in filter_input:
            assert len(per_filter_input.shape) == 4, "should be in NCHW layout"
        filter_output = filters[level](*filter_input)
        per_sample_outs = []
        for i in range(len(filter_output)):
            (
                _,
                _,
                per_img_coord,
                per_img_score,
                per_img_bbox_pred,
                per_img_centerness,
            ) = filter_output[i]
            per_img_coord = torch.cat(
                [
                    per_img_coord[:, 1].view(-1, 1),
                    per_img_coord[:, 0].view(-1, 1),
                ],
                axis=1,
            )
            per_sample_outs.append(
                [
                    per_img_coord,
                    per_img_score,
                    per_img_bbox_pred,
                    per_img_centerness,
                ]
            )
        mlvl_outputs.append(per_sample_outs)

    return label, mlvl_outputs


@pytest.mark.skipif(True, reason="hat-sim is updating!")
def test_fcos_decoder():
    score_threshold = 0.05
    iou_threshold = 0.6
    input_shape = (512, 512)
    strides = [4, 8, 16, 32]
    num_classes = 10
    max_per_img = 5

    h, w = input_shape
    label, pred = gen_fake_data(h, w, strides, num_classes)

    # hat_sim
    decoder = hat_sim.FcosDecoder(
        score_threshold,
        max_per_img,
        iou_threshold,
        num_classes,
        input_shape[0],
        input_shape[1],
    )
    sim_output = np.zeros((max_per_img, 6), dtype=np.float32)
    coords = [x[0][0].numpy().astype(np.int32) for x in pred]
    scores = [x[0][1].numpy() for x in pred]
    bbox_pred = [x[0][2].numpy() for x in pred]
    centerness = [x[0][3].numpy() for x in pred]

    decoder.forward(coords, scores, bbox_pred, centerness, strides, sim_output)
    assert np.sum(sim_output) > 0


if __name__ == "__main__":
    pytest.main(["-s", __file__])
