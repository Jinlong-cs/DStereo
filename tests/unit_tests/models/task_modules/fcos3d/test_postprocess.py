import numpy as np
import torch

from hat.models.task_modules.fcos3d import FCOS3DBBoxCoder, FCOS3DPostProcess

cls_scores = [
    torch.rand(2, 10, 64, 112),
    torch.rand(2, 10, 32, 56),
    torch.rand(2, 10, 16, 28),
    torch.rand(2, 10, 8, 14),
    torch.rand(2, 10, 4, 7),
]

bbox_preds = [
    torch.rand(2, 9, 64, 112),
    torch.rand(2, 9, 32, 56),
    torch.rand(2, 9, 16, 28),
    torch.rand(2, 9, 8, 14),
    torch.rand(2, 9, 4, 7),
]

dir_cls_preds = [
    torch.rand(2, 2, 64, 112),
    torch.rand(2, 2, 32, 56),
    torch.rand(2, 2, 16, 28),
    torch.rand(2, 2, 8, 14),
    torch.rand(2, 2, 4, 7),
]

attr_preds = [
    torch.rand(2, 9, 64, 112),
    torch.rand(2, 9, 32, 56),
    torch.rand(2, 9, 16, 28),
    torch.rand(2, 9, 8, 14),
    torch.rand(2, 9, 4, 7),
]

centernesses = [
    torch.rand(2, 1, 64, 112),
    torch.rand(2, 1, 32, 56),
    torch.rand(2, 1, 16, 28),
    torch.rand(2, 1, 8, 14),
    torch.rand(2, 1, 4, 7),
]

img_metas = {
    "cam2img": [
        np.array(
            [
                [705.32769773, 0.0, 452.06162894],
                [0.0, 705.32769773, 280.66964855],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array(
            [
                [712.65485339, 0.0, 462.9046779],
                [0.0, 712.65485339, 268.66092731],
                [0.0, 0.0, 1.0],
            ]
        ),
    ]
}


def test_fcos3d_post_process():
    bbox_coder = FCOS3DBBoxCoder(
        code_size=9,
    )
    model = FCOS3DPostProcess(
        num_classes=10,
        use_direction_classifier=True,
        strides=[8, 16, 32, 64, 128],
        group_reg_dims=(2, 1, 3, 1, 2),
        pred_attrs=True,
        num_attrs=9,
        attr_background_label=9,
        bbox_coder=bbox_coder,
        bbox_code_size=9,
        dir_offset=0.7854,
        test_cfg=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=100,
            nms_thr=0.3,
            score_thr=0.05,
            min_bbox_size=0,
            max_per_img=100,
        ),
    )
    result_list = model(
        cls_scores=cls_scores,
        bbox_preds=bbox_preds,
        dir_cls_preds=dir_cls_preds,
        attr_preds=attr_preds,
        centernesses=centernesses,
        img_metas=img_metas,
        cfg=None,
        rescale=None,
    )
    assert len(result_list) == 2
    assert "ret" in result_list[0]
