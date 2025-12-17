import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test
from tests.unit_tests.models.task_modules.hand3d.test_hand3d_decoder import (
    FLAT_HAND_MEAN,
    MANO_CENTER_INDEX,
    MANO_HAND_SIDE,
    MANO_ROOT,
    SMPL_HANDS_MEAN,
)

BATCH_SIZE = 16
INPUT_SIZE = 256
INPUT_CHANNELS = 4
BIFPN_CHANNELS = 64
ENCODING_CHANNELS = BIFPN_CHANNELS * 4
ENABLE_HEATMAP_HEAD = False
ENABLE_RENDER_HEAD = False


@pytest.mark.parametrize(
    [
        "mode",
    ],
    [pytest.param(["deploy", "infer"])],
)
def test_H3DStructure(mode):

    data = {
        "img": torch.randn(
            (BATCH_SIZE, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE)
        )
    }
    if mode == "deploy":
        decoder = None
    elif mode == "infer":
        decoder = (
            dict(
                type="H3DManoDecoder",
                intput_shape=(INPUT_SIZE, INPUT_SIZE),
                heatmap_stride=4,
                trans_coeff=2.5,
                min_distance=0.001,
                hand_scale_coeff=1,
                mano_layer=dict(
                    type="MANOLayer",
                    model_path=MANO_ROOT,
                    is_rhand=MANO_HAND_SIDE == "right",
                    center_idx=MANO_CENTER_INDEX,
                    flat_hand_mean=FLAT_HAND_MEAN,
                    smpl_hands_mean=SMPL_HANDS_MEAN,
                ),
                point_nerf=dict(
                    type="PointNeRF",
                    intput_shape=(INPUT_SIZE, INPUT_SIZE),
                    point_cloud_radius=0.025,
                    points_per_pixel=50,
                    text_feat_length=128,
                    position_encoding_length=4,
                    density_feat_length=12,
                ),
                enable_heatmap_head=ENABLE_HEATMAP_HEAD,
                enable_render_head=ENABLE_RENDER_HEAD,
                enable_handscale_head=False,
                enable_subdivide_points=False,
                infer_mode=False,
            ),
        )
    else:
        return

    model = dict(
        type="H3DStructure",
        backbone=dict(
            type="SNDRMobileNetV2",
            num_classes=0,
            bn_kwargs=dict(eps=1e-5, momentum=0.01),
            in_chls=[
                [32],
                [16, 24],
                [24, 32, 32],
                [32, 64, 64, 64, 64, 96, 96],
                [96, 160, 160, 160],
            ],
            out_chls=[
                [16],
                [24, 24],
                [32, 32, 32],
                [64, 64, 64, 64, 96, 96, 96],
                [160, 160, 160, 320],
            ],
            expand_ratio=6,
            input_channels=INPUT_CHANNELS,
            alpha=1.0,
            bias=True,
            include_top=False,
        ),
        neck=dict(
            type="FPN",
            in_strides=[2, 4, 8, 16, 32],
            in_channels=[16, 24, 32, 96, 320],
            out_strides=[8, 16, 32],
            out_channels=[64, 128, 192],
            bn_kwargs=dict(eps=1e-5, momentum=0.01),
            node_name="fpn_neck",
        ),
        encoder=dict(
            type="H3DFuseHighLowEncoder",
            in_channels=192,
            bifpn_channels=BIFPN_CHANNELS,
            encoding_channels=ENCODING_CHANNELS,
            bn_kwargs=dict(eps=1e-5, momentum=0.01),
            heatmap_encoder=dict(
                type="H3DHeatMapResEncoder",
                in_channels=BIFPN_CHANNELS,
                out_channels=ENCODING_CHANNELS,
                bn_kwargs=dict(eps=1e-5, momentum=0.01),
            ),
            global_avg_size=4,
        ),
        head=dict(
            type="H3DManoMultiFcHead",
            encoding_channel=ENCODING_CHANNELS,
            bn_kwargs=dict(eps=1e-5, momentum=0.01),
            scale_neurons=(256, 128, 1),
            pose_neurons=(256, 128, 90),
            shape_neurons=(256, 10),
            camera_base_neurons=(256,),
            rot_neurons=(256, 6),
            trans_neurons=(256, 3),
            text_fc_neurons=(256, 256, 128),
            text_reg_neurons=(1024, 778, 778),
            head_heatmap=None,
            head_heatmap_latent=True,
            enable_render_head=False,
            enable_handscale_head=False,
            num_joints=21,
            deploy=False,
        ),
        decoder=decoder,
        loss=None,
    )
    h3d_model = build_from_registry(model)
    outputs = h3d_model(data)

    if mode == "deploy":
        assert len(outputs) == 5
        assert "shape_mano" in outputs
        assert "pose_mano" in outputs
        assert "trans" in outputs
        assert "rot" in outputs
        assert "heatmap_latents" in outputs

        assert outputs["shape_mano"].shape == (BATCH_SIZE, 10)
        assert outputs["pose_mano"].shape == (BATCH_SIZE, 90)
        assert outputs["trans"].shape == (BATCH_SIZE, 3)
        assert outputs["rot"].shape == (BATCH_SIZE, 6)
        assert outputs["heatmap_latents"].shape == (BATCH_SIZE, 21, 32, 32)

    elif mode == "infer":
        assert "pred_trans" in outputs
        assert outputs["pred_trans"].shape == (BATCH_SIZE, 1, 3)

        assert "pred_ldmk3d_relat" in outputs
        assert outputs["pred_ldmk3d_relat"].shape == (BATCH_SIZE, 21, 3)

        assert "pred_ldmk3d" in outputs
        assert outputs["pred_ldmk3d"].shape == (BATCH_SIZE, 21, 3)

        assert "pred_verts" in outputs
        assert outputs["pred_verts"].shape == (BATCH_SIZE, 778, 3)

        assert "pred_pose" in outputs
        assert outputs["pred_pose"].shape == (BATCH_SIZE, 90)

        assert "pred_shape" in outputs
        assert outputs["pred_shape"].shape == (BATCH_SIZE, 10)

        assert "pred_ldmk" in outputs
        assert outputs["pred_ldmk"].shape == (BATCH_SIZE, 21, 2)

        assert "pred_ldmk_proj" in outputs
        assert outputs["pred_ldmk_proj"].shape == (BATCH_SIZE, 21, 2)

    else:
        pass

    qat_test(h3d_model, data, with_quantized=False)
