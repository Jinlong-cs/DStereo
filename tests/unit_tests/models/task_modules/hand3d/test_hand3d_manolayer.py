import copy

import torch

from tests.unit_tests.models.task_modules.hand3d.test_hand3d_decoder import (
    FLAT_HAND_MEAN,
    MANO_CENTER_INDEX,
    MANO_HAND_SIDE,
    MANO_ROOT,
    SMPL_HANDS_MEAN,
)
from tests.unit_tests.models.task_modules.head_template import HeadTemplate

try:
    from smplx.body_models import MANO

    from hat.models.task_modules.hand3d.manolayer import MANOLayer
except Exception:
    MANO = None
    MANOLayer = None


class TestMANOLayer(HeadTemplate):
    def get_inputs(self):
        self.batch_size = 16
        self.inputs = dict(
            global_orient=torch.rand(self.batch_size, 1, 3, 3),
            hand_pose=torch.rand(self.batch_size, 15, 3, 3),
            betas=torch.rand(self.batch_size, 10),
            pose2rot=False,
        )
        self.quantized_inputs = {}

    def setup(self):
        super().setup()
        self.get_inputs()
        if MANOLayer is not None:
            self.model = MANOLayer(
                model_path=MANO_ROOT,
                is_rhand=MANO_HAND_SIDE == "right",
                center_idx=MANO_CENTER_INDEX,
                flat_hand_mean=FLAT_HAND_MEAN,
                smpl_hands_mean=SMPL_HANDS_MEAN,
                batch_size=self.batch_size,
            )
            self.float_model = copy.deepcopy(self.model)
            self.float_model.eval()
        else:
            self.model = MANO
            self.float_model = copy.deepcopy(self.model)

    def test_float_model(self):
        if MANOLayer is not None:
            pred_verts, pred_joints = self.float_model(**self.inputs)

            assert pred_verts.shape == (self.batch_size, 778, 3)
            assert pred_joints.shape == (self.batch_size, 21, 3)

    def test_qat_model(self):
        pass

    def test_quantize_model(self):
        pass

    def test_fuse_model(self):
        pass
