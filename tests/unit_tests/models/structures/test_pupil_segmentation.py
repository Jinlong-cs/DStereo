import torch

from hat.registry import build_from_registry


def test_pupil_encoder():
    data = {}
    img_shape = (2, 3, 64, 64)
    config = dict(
        type="PupilSegNet",
        encoder=dict(
            type="PupilSegEncoder",
            in_c=3,
            chz=32,
            growth=1.5,
        ),
        decoder=dict(
            type="PupilSegDecoder",
            chz=32,
            out_c=1,
            growth=1.5,
        ),
        encoder_out_channels=192,
        mode="val",
    )
    pupil_seg_net = build_from_registry(config)
    data["img"] = torch.randn(img_shape)
    output = pupil_seg_net(data)
    assert output["ellipse_param_pred"].data.numpy().shape == (2, 5, 1, 1)
    assert output["mask_pred"].data.numpy().shape == (2, 1, 64, 64)
