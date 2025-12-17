import numpy as np
import torch

from hat.models.task_modules.gesture.multi_modality_head import (
    ActMultiModalityHead,
)
from tests.unit_tests.models.base import qtensor_test
from tests.unit_tests.models.task_modules.head_template import HeadTemplate

BATCHSIZE = 2
NUM_CLASS = 59


def setup_seed(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class TestMultiModeHeadForResultFusion(HeadTemplate):
    def setup(self):
        self.fusion_type = "result_fusion"
        self.expect_result = [0, 1]
        self._init_model_inputs()

    def _init_model_inputs(self):
        setup_seed()
        in_channels_dict = {
            "kps": 512,
            "frames": 512,
        }
        self.inputs_kps = torch.rand(BATCHSIZE, in_channels_dict["kps"], 1, 1)
        self.inputs_frames = torch.rand(
            BATCHSIZE, in_channels_dict["frames"], 1, 1
        )

        self.quantized_inputs_kps = torch.ones(
            BATCHSIZE, in_channels_dict["kps"], 1, 1, dtype=torch.int8
        )
        self.quantized_inputs_frames = torch.ones(
            BATCHSIZE, in_channels_dict["frames"], 1, 1, dtype=torch.int8
        )

        self.model = ActMultiModalityHead(
            num_classes=NUM_CLASS,
            in_channels_dict=in_channels_dict,
            fusion_type=self.fusion_type,
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs_kps, self.inputs_frames)
        for key_ret in self.expect_result:
            assert outputs[key_ret].shape[:2] == (BATCHSIZE, NUM_CLASS)

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs_kps, self.inputs_frames)
        fuse_outputs = self.fuse_model(self.inputs_kps, self.inputs_frames)
        for key_ret in self.expect_result:
            assert torch.allclose(
                outputs[key_ret], fuse_outputs[key_ret], atol=1e-5
            )

    def test_qat_model(self):
        qat_inputs_kps = qtensor_test(self.inputs_kps)
        qat_inputs_frames = qtensor_test(self.inputs_frames)
        qat_outputs = self.qat_model(qat_inputs_kps, qat_inputs_frames)
        for key_ret in self.expect_result:
            assert qat_outputs[key_ret].shape[:2] == (BATCHSIZE, NUM_CLASS)

    def test_quantize_model(self):
        quantized_inputs_kps = qtensor_test(self.quantized_inputs_kps)
        quantized_inputs_frames = qtensor_test(self.quantized_inputs_frames)

        quantized_outputs = self.quantized_model(
            quantized_inputs_kps, quantized_inputs_frames
        )
        for key_ret in self.expect_result:
            assert quantized_outputs[key_ret].shape[:2] == (
                BATCHSIZE,
                NUM_CLASS,
            )


class TestMultiModeHeadForFeatureFusion(TestMultiModeHeadForResultFusion):
    def setup(self):
        self.fusion_type = "feature_fusion"
        self.expect_result = [2]
        self._init_model_inputs()
