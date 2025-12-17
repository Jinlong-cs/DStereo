import torch

from hat.callbacks.save_eval_results import SaveDet2dResult
from hat.core.data_struct.img_structures import ImgObjDet


def test_save_det2d_result():
    fake_data = torch.rand(2, 3, 224, 224)
    objects = [
        ImgObjDet(
            img=fake_data,
            img_id="img0",
            layout="chw",
            color_space="rgb",
            img_height=224,
            img_width=224,
        ),
        ImgObjDet(
            img=fake_data,
            img_id="img1",
            layout="chw",
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

    det2d = SaveDet2dResult(output_dir="./")
    det2d.save_result(0, fake_batch, model_outs)
