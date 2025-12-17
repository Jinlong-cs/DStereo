import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


@pytest.mark.parametrize(
    [
        "input_h",
        "input_w",
        "alpha",
        "bn_kwargs",
        "j5_efficient",
    ],
    [
        pytest.param(
            192,
            320,
            0.75,
            {
                "eps": 2e-05,
                "momentum": 0.1,
            },
            True,
        ),
        pytest.param(
            192,
            320,
            0.75,
            {
                "eps": 2e-05,
                "momentum": 0.1,
            },
            False,
        ),
    ],
)
def test_gaze_task(input_h, input_w, alpha, bn_kwargs, j5_efficient):
    shared_feat_size = max(int(alpha * 1024), 1024)
    last_channel_out = int(256 * alpha)

    config = dict(
        type="GazeModel",
        backbone=dict(
            type="VargNetV2",
            num_classes=1000,
            include_top=False,
            bn_kwargs=bn_kwargs,
            alpha=alpha,
        ),
        head=dict(
            type="GazeEyeldmkHead",
            input_size=[input_w, input_h],
            last_channel_out=last_channel_out,
            shared_feat_size=shared_feat_size,
            gaze_head_params=dict(
                use_pool=True,
                dropout_ratio=0.5,
                channels=(shared_feat_size, 128, 128, 4),
                output_add_bias=False,
                bn_kwargs=bn_kwargs,
            ),
            eyeldmk_head_params=dict(
                alpha=0.25,
                bn_kwargs=bn_kwargs,
                channels=[128, 128, 16],
                J5_efficient=j5_efficient,
            ),
        ),
        losses=dict(
            type="GazeEyeldmkLoss",
            eye_ldmk_num=21,
            gaze_loss_name="wing",
            gaze_loss_weight=1,
            gaze_loss_params=dict(
                pitch_w=15,
                yaw_w=15,
                pitch_e=0.5,
                yaw_e=0.5,
                averaged_output=False,
                batch_axis=0,
            ),
            eye_ldmk_loss_name="wing",
            eye_ldmk_loss_weight=5,
            eye_ldmk_loss_params=dict(
                w=15,
                epsilon=0.1,
                averaged_output=False,
                batch_axis=0,
            ),
        ),
        deploy=True,
        compile_eyeldmk=True,
    )
    gaze_model = build_from_registry(config)
    x = {"img": torch.rand(4, 3, 192, 320)}
    batch_out = gaze_model(x)
    assert batch_out[0].shape == (4, 4, 1, 1)
    if j5_efficient:
        assert batch_out[1].shape == (4, 64, 1, 1)
    else:
        for i in range(1, 5):
            assert batch_out[i].shape == (4, 16, 1, 1)
    qat_test(gaze_model, x, with_quantized=False)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
