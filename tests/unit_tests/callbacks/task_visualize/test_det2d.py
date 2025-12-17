import torch

from hat.callbacks.task_visualize import Det2dVisualize
from hat.core.data_struct.img_structures import ImgObjDet


def test_det2d():
    fake_data = torch.rand(2, 224, 224, 3)
    objects = [
        ImgObjDet(
            img=fake_data,
            img_id="img0",
            layout="hwc",
            color_space="rgb",
            img_height=224,
            img_width=224,
        ),
        ImgObjDet(
            img=fake_data,
            img_id="img1",
            layout="hwc",
            color_space="rgb",
            img_height=224,
            img_width=224,
        ),
    ]
    fake_batch = dict(structure=objects)

    model_outs = [
        [
            torch.rand(10, 7),
            torch.rand(10, 7),
        ],
        0,
    ]

    det2d = Det2dVisualize(viz_threshold=0.0)
    det2d.on_batch_end(0, fake_batch, model_outs)
