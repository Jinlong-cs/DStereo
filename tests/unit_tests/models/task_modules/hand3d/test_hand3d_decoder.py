import copy
import os

import torch

from hat.models.task_modules.hand3d.decoder.point_nerf import PointNeRF
from tests import HAT_BUCKET_PATH
from tests.unit_tests.models.task_modules.head_template import HeadTemplate

try:
    from kornia.geometry.subpix import spatial_soft_argmax2d
    from pytorch3d.structures import Meshes
    from smplx.body_models import MANO

    from hat.models.task_modules.hand3d.decoder.mano_decoder import (
        H3DManoDecoder,
    )
    from hat.models.task_modules.hand3d.manolayer import MANOLayer
except ImportError:
    MANO = None
    MANOLayer = None
    Meshes = None
    spatial_soft_argmax2d = None
    H3DManoDecoder = None


MANO_ROOT = os.path.join(
    HAT_BUCKET_PATH,
    "data/orig_data/hand3d/mano_params",
)

MANO_HAND_SIDE = "right"
MANO_CENTER_INDEX = 0
FLAT_HAND_MEAN = True
SMPL_HANDS_MEAN = None


class TestH3DManoDecoder(HeadTemplate):
    def get_inputs(self):
        self.batch_size = 16
        self.inputs = dict(
            shape_mano=torch.rand(self.batch_size, 10),
            pose_mano=torch.rand(self.batch_size, 90),
            scale_intr=torch.rand(self.batch_size, 1),
            trans=torch.rand(self.batch_size, 3),
            rot=torch.rand(self.batch_size, 6),
            heatmap_latents=torch.rand(self.batch_size, 21, 32, 32),
            text=torch.rand(self.batch_size, 778, 128),
        )
        self.quantized_inputs = {}
        self.data = dict(
            virtual_intrinsic=torch.rand(self.batch_size, 3, 3),
            trans_mat=torch.rand(self.batch_size, 3, 3),
        )

    def setup(self):
        super().setup()
        self.get_inputs()
        if MANOLayer is not None and H3DManoDecoder is not None:
            mano_layer = MANOLayer(
                model_path=MANO_ROOT,
                is_rhand=MANO_HAND_SIDE == "right",
                center_idx=MANO_CENTER_INDEX,
                flat_hand_mean=FLAT_HAND_MEAN,
                smpl_hands_mean=SMPL_HANDS_MEAN,
            )
            point_nerf = PointNeRF(
                intput_shape=(256, 256),
                point_cloud_radius=0.025,
                points_per_pixel=50,
                text_feat_length=128,
                position_encoding_length=4,
                density_feat_length=12,
            )
            self.model = H3DManoDecoder(
                intput_shape=(256, 256),
                heatmap_stride=4,
                trans_coeff=2.5,
                min_distance=0.001,
                hand_scale_coeff=1,
                enable_heatmap_head=False,
                enable_render_head=False,
                enable_handscale_head=False,
                enable_subdivide_points=False,
                num_verts_sampled=50,
                mano_layer=mano_layer,
                point_nerf=point_nerf,
                infer_mode=True,
            )
            self.float_model = copy.deepcopy(self.model)
            self.float_model.eval()
        else:
            self.float_model = None

    def test_float_model(self):
        if self.float_model is not None:
            outputs = self.float_model(self.inputs, self.data)

            assert "pred_trans" in outputs
            assert "pred_ldmk3d_relat" in outputs
            assert "pred_ldmk3d" in outputs
            assert "pred_verts" in outputs
            assert "pred_pose" in outputs
            assert "pred_shape" in outputs
            assert "pred_ldmk" in outputs
            assert "pred_ldmk_proj" in outputs

            assert outputs["pred_trans"].shape == (self.batch_size, 1, 3)
            assert outputs["pred_ldmk3d_relat"].shape == (
                self.batch_size,
                21,
                3,
            )
            assert outputs["pred_ldmk3d"].shape == (self.batch_size, 21, 3)
            assert outputs["pred_pose"].shape == (self.batch_size, 90)
            assert outputs["pred_shape"].shape == (self.batch_size, 10)
            assert outputs["pred_ldmk"].shape == (self.batch_size, 21, 2)
            assert outputs["pred_ldmk_proj"].shape == (self.batch_size, 21, 2)

    def test_qat_model(self):
        pass

    def test_quantize_model(self):
        pass

    def test_fuse_model(self):
        pass
