import pytest
import torch

from hat.models.task_modules.centerpoint.head import (
    CenterPointHead,
    DepthwiseSeparableCenterPointHead,
    VargCenterPointHead,
)
from tests.unit_tests.models.base import qtensor_test
from tests.unit_tests.models.task_modules.head_template import HeadTemplate

tasks = [
    dict(name="car", num_class=1, class_names=["car"]),
    dict(
        name="truck",
        num_class=2,
        class_names=["truck", "construction_vehicle"],
    ),
    dict(name="bus", num_class=2, class_names=["bus", "trailer"]),
    dict(name="barrier", num_class=1, class_names=["barrier"]),
    dict(name="bicycle", num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(
        name="pedestrian",
        num_class=2,
        class_names=["pedestrian", "traffic_cone"],
    ),
]


class TestDepthwiseSeparableCenterPointHead(HeadTemplate):
    def setup(self):
        super(TestDepthwiseSeparableCenterPointHead, self).setup()
        self.input_size = (512, 960)
        h_tmp = int(self.input_size[0] / 16)
        w_tmp = int(self.input_size[1] / 16)

        self.inputs = [torch.rand(1, 48, h_tmp, w_tmp)]
        self.quantized_inputs = [
            torch.ones(1, 48, h_tmp, w_tmp, dtype=torch.int8)
        ]
        self.model = DepthwiseSeparableCenterPointHead(
            in_channels=48,
            tasks=tasks,
            share_conv_channels=48,
            share_conv_num=1,
            common_heads=dict(
                reg=(2, 2),
                height=(1, 2),
                dim=(3, 2),
                rot=(2, 2),
                vel=(2, 2),
            ),
            head_conv_channels=48,
            num_heatmap_convs=2,
            final_kernel=3,
        )

        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)

        assert outputs[0]["reg"].shape == (1, 2, 32, 60)
        assert outputs[0]["height"].shape == (1, 1, 32, 60)
        assert outputs[0]["dim"].shape == (1, 3, 32, 60)
        assert outputs[0]["rot"].shape == (1, 2, 32, 60)
        assert outputs[0]["vel"].shape == (1, 2, 32, 60)
        assert outputs[0]["heatmap"].shape == (1, 1, 32, 60)

        assert outputs[1]["reg"].shape == (1, 2, 32, 60)
        assert outputs[1]["height"].shape == (1, 1, 32, 60)
        assert outputs[1]["dim"].shape == (1, 3, 32, 60)
        assert outputs[1]["rot"].shape == (1, 2, 32, 60)
        assert outputs[1]["vel"].shape == (1, 2, 32, 60)
        assert outputs[1]["heatmap"].shape == (1, 2, 32, 60)

        assert outputs[2]["reg"].shape == (1, 2, 32, 60)
        assert outputs[2]["height"].shape == (1, 1, 32, 60)
        assert outputs[2]["dim"].shape == (1, 3, 32, 60)
        assert outputs[2]["rot"].shape == (1, 2, 32, 60)
        assert outputs[2]["vel"].shape == (1, 2, 32, 60)
        assert outputs[2]["heatmap"].shape == (1, 2, 32, 60)

        assert outputs[3]["reg"].shape == (1, 2, 32, 60)
        assert outputs[3]["height"].shape == (1, 1, 32, 60)
        assert outputs[3]["dim"].shape == (1, 3, 32, 60)
        assert outputs[3]["rot"].shape == (1, 2, 32, 60)
        assert outputs[3]["vel"].shape == (1, 2, 32, 60)
        assert outputs[3]["heatmap"].shape == (1, 1, 32, 60)

        assert outputs[4]["reg"].shape == (1, 2, 32, 60)
        assert outputs[4]["height"].shape == (1, 1, 32, 60)
        assert outputs[4]["dim"].shape == (1, 3, 32, 60)
        assert outputs[4]["rot"].shape == (1, 2, 32, 60)
        assert outputs[4]["vel"].shape == (1, 2, 32, 60)
        assert outputs[4]["heatmap"].shape == (1, 2, 32, 60)

        assert outputs[5]["reg"].shape == (1, 2, 32, 60)
        assert outputs[5]["height"].shape == (1, 1, 32, 60)
        assert outputs[5]["dim"].shape == (1, 3, 32, 60)
        assert outputs[5]["rot"].shape == (1, 2, 32, 60)
        assert outputs[5]["vel"].shape == (1, 2, 32, 60)
        assert outputs[5]["heatmap"].shape == (1, 2, 32, 60)

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs)
        qat_outputs = self.qat_model(qat_inputs)

        assert qat_outputs[0]["reg"].shape == (1, 2, 32, 60)
        assert qat_outputs[0]["height"].shape == (1, 1, 32, 60)
        assert qat_outputs[0]["dim"].shape == (1, 3, 32, 60)
        assert qat_outputs[0]["rot"].shape == (1, 2, 32, 60)
        assert qat_outputs[0]["vel"].shape == (1, 2, 32, 60)
        assert qat_outputs[0]["heatmap"].shape == (1, 1, 32, 60)

        assert qat_outputs[1]["reg"].shape == (1, 2, 32, 60)
        assert qat_outputs[1]["height"].shape == (1, 1, 32, 60)
        assert qat_outputs[1]["dim"].shape == (1, 3, 32, 60)
        assert qat_outputs[1]["rot"].shape == (1, 2, 32, 60)
        assert qat_outputs[1]["vel"].shape == (1, 2, 32, 60)
        assert qat_outputs[1]["heatmap"].shape == (1, 2, 32, 60)

        assert qat_outputs[2]["reg"].shape == (1, 2, 32, 60)
        assert qat_outputs[2]["height"].shape == (1, 1, 32, 60)
        assert qat_outputs[2]["dim"].shape == (1, 3, 32, 60)
        assert qat_outputs[2]["rot"].shape == (1, 2, 32, 60)
        assert qat_outputs[2]["vel"].shape == (1, 2, 32, 60)
        assert qat_outputs[2]["heatmap"].shape == (1, 2, 32, 60)

        assert qat_outputs[3]["reg"].shape == (1, 2, 32, 60)
        assert qat_outputs[3]["height"].shape == (1, 1, 32, 60)
        assert qat_outputs[3]["dim"].shape == (1, 3, 32, 60)
        assert qat_outputs[3]["rot"].shape == (1, 2, 32, 60)
        assert qat_outputs[3]["vel"].shape == (1, 2, 32, 60)
        assert qat_outputs[3]["heatmap"].shape == (1, 1, 32, 60)

        assert qat_outputs[4]["reg"].shape == (1, 2, 32, 60)
        assert qat_outputs[4]["height"].shape == (1, 1, 32, 60)
        assert qat_outputs[4]["dim"].shape == (1, 3, 32, 60)
        assert qat_outputs[4]["rot"].shape == (1, 2, 32, 60)
        assert qat_outputs[4]["vel"].shape == (1, 2, 32, 60)
        assert qat_outputs[4]["heatmap"].shape == (1, 2, 32, 60)

        assert qat_outputs[5]["reg"].shape == (1, 2, 32, 60)
        assert qat_outputs[5]["height"].shape == (1, 1, 32, 60)
        assert qat_outputs[5]["dim"].shape == (1, 3, 32, 60)
        assert qat_outputs[5]["rot"].shape == (1, 2, 32, 60)
        assert qat_outputs[5]["vel"].shape == (1, 2, 32, 60)
        assert qat_outputs[5]["heatmap"].shape == (1, 2, 32, 60)

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.quantized_inputs)
        quantized_outputs = self.quantized_model(quantized_inputs)

        assert quantized_outputs[0]["reg"].shape == (1, 2, 32, 60)
        assert quantized_outputs[0]["height"].shape == (1, 1, 32, 60)
        assert quantized_outputs[0]["dim"].shape == (1, 3, 32, 60)
        assert quantized_outputs[0]["rot"].shape == (1, 2, 32, 60)
        assert quantized_outputs[0]["vel"].shape == (1, 2, 32, 60)
        assert quantized_outputs[0]["heatmap"].shape == (1, 1, 32, 60)

        assert quantized_outputs[1]["reg"].shape == (1, 2, 32, 60)
        assert quantized_outputs[1]["height"].shape == (1, 1, 32, 60)
        assert quantized_outputs[1]["dim"].shape == (1, 3, 32, 60)
        assert quantized_outputs[1]["rot"].shape == (1, 2, 32, 60)
        assert quantized_outputs[1]["vel"].shape == (1, 2, 32, 60)
        assert quantized_outputs[1]["heatmap"].shape == (1, 2, 32, 60)

        assert quantized_outputs[2]["reg"].shape == (1, 2, 32, 60)
        assert quantized_outputs[2]["height"].shape == (1, 1, 32, 60)
        assert quantized_outputs[2]["dim"].shape == (1, 3, 32, 60)
        assert quantized_outputs[2]["rot"].shape == (1, 2, 32, 60)
        assert quantized_outputs[2]["vel"].shape == (1, 2, 32, 60)
        assert quantized_outputs[2]["heatmap"].shape == (1, 2, 32, 60)

        assert quantized_outputs[3]["reg"].shape == (1, 2, 32, 60)
        assert quantized_outputs[3]["height"].shape == (1, 1, 32, 60)
        assert quantized_outputs[3]["dim"].shape == (1, 3, 32, 60)
        assert quantized_outputs[3]["rot"].shape == (1, 2, 32, 60)
        assert quantized_outputs[3]["vel"].shape == (1, 2, 32, 60)
        assert quantized_outputs[3]["heatmap"].shape == (1, 1, 32, 60)

        assert quantized_outputs[4]["reg"].shape == (1, 2, 32, 60)
        assert quantized_outputs[4]["height"].shape == (1, 1, 32, 60)
        assert quantized_outputs[4]["dim"].shape == (1, 3, 32, 60)
        assert quantized_outputs[4]["rot"].shape == (1, 2, 32, 60)
        assert quantized_outputs[4]["vel"].shape == (1, 2, 32, 60)
        assert quantized_outputs[4]["heatmap"].shape == (1, 2, 32, 60)

        assert quantized_outputs[5]["reg"].shape == (1, 2, 32, 60)
        assert quantized_outputs[5]["height"].shape == (1, 1, 32, 60)
        assert quantized_outputs[5]["dim"].shape == (1, 3, 32, 60)
        assert quantized_outputs[5]["rot"].shape == (1, 2, 32, 60)
        assert quantized_outputs[5]["vel"].shape == (1, 2, 32, 60)
        assert quantized_outputs[5]["heatmap"].shape == (1, 2, 32, 60)

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        for output, fuse_output in zip(outputs, fuse_outputs):
            for k in output.keys():
                assert torch.allclose(output[k], fuse_output[k], atol=1e-5)


class TestVargCenterPointHead(TestDepthwiseSeparableCenterPointHead):
    def setup(self):
        super(TestVargCenterPointHead, self).setup()
        self.input_size = (512, 960)
        h_tmp = int(self.input_size[0] / 16)
        w_tmp = int(self.input_size[1] / 16)

        self.inputs = [torch.rand(1, 48, h_tmp, w_tmp)]
        self.quantized_inputs = [
            torch.ones(1, 48, h_tmp, w_tmp, dtype=torch.int8)
        ]
        self.model = VargCenterPointHead(
            in_channels=48,
            tasks=tasks,
            share_conv_channels=48,
            share_conv_num=1,
            common_heads=dict(
                reg=(2, 2),
                height=(1, 2),
                dim=(3, 2),
                rot=(2, 2),
                vel=(2, 2),
            ),
            head_conv_channels=48,
            num_heatmap_convs=2,
            final_kernel=3,
        )

        self.build_model()


class TestCenterPointHead(TestDepthwiseSeparableCenterPointHead):
    def setup(self):
        super(TestCenterPointHead, self).setup()
        self.input_size = (512, 960)
        h_tmp = int(self.input_size[0] / 16)
        w_tmp = int(self.input_size[1] / 16)

        self.inputs = [torch.rand(1, 48, h_tmp, w_tmp)]
        self.quantized_inputs = [
            torch.ones(1, 48, h_tmp, w_tmp, dtype=torch.int8)
        ]
        self.model = CenterPointHead(
            in_channels=48,
            tasks=tasks,
            share_conv_channels=48,
            share_conv_num=1,
            common_heads=dict(
                reg=(2, 2),
                height=(1, 2),
                dim=(3, 2),
                rot=(2, 2),
                vel=(2, 2),
            ),
            head_conv_channels=48,
            num_heatmap_convs=2,
            final_kernel=3,
        )

        self.build_model()


if __name__ == "__main__":
    pytest.main(["-s", __file__])
