import torch

from hat.models.task_modules.reid import ReIDClsOutputBlock
from tests.unit_tests.models.base import ModelTemplate, qtensor_test


class TestReIDClsOutputBlockIncludeTop(ModelTemplate):
    def setup(self):
        super(TestReIDClsOutputBlockIncludeTop, self).setup()
        self.num_classes = 10
        self.inputs = [torch.rand(1, 16, 16, 16)]
        self.quantized_inputs = [torch.ones(1, 16, 64, 64, dtype=torch.int8)]

        include_top = True
        # test: int8_output=True and include_top=True
        self.model = ReIDClsOutputBlock(
            num_classes=self.num_classes,
            in_channels=16,
            feat_channels=16,
            int8_output=True,
            bn_kwargs={},
            pool_kernel_size=2,
            include_top=include_top,
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert len(outputs.shape) == 2
        assert outputs.shape[1] == self.num_classes

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs)
        qat_outputs = self.qat_model(qat_inputs)
        assert len(qat_outputs.shape) == 2
        assert qat_outputs.shape[1] == self.num_classes

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.quantized_inputs)
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert len(quantized_outputs.shape) == 2
        assert quantized_outputs.shape[1] == self.num_classes

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        for i in range(len(outputs)):
            assert all(
                torch.allclose(output, fuse_output, atol=1e-5)
                for output, fuse_output in zip(outputs[i], fuse_outputs[i])
            )


class TestReIDClsOutputBlockWithoutTop(ModelTemplate):
    def setup(self):
        super(TestReIDClsOutputBlockWithoutTop, self).setup()
        self.num_classes = 10
        self.inputs = [torch.rand(1, 16, 16, 16)]
        self.quantized_inputs = [torch.ones(1, 16, 16, 16, dtype=torch.int8)]

        include_top = False
        # test: int8_output=True and include_top=True
        self.model = ReIDClsOutputBlock(
            num_classes=self.num_classes,
            in_channels=16,
            feat_channels=16,
            int8_output=True,
            bn_kwargs={},
            pool_kernel_size=4,
            include_top=include_top,
        )
        self.build_model()
        self.expected_size = torch.Size([1, 16, 16 - 4 + 1, 16 - 4 + 1])

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert len(outputs.shape) == 4
        # print(f"--->output: {outputs.size()}")
        assert outputs.size() == self.expected_size

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs)
        qat_outputs = self.qat_model(qat_inputs)
        assert len(qat_outputs.shape) == 4
        assert qat_outputs.size() == self.expected_size

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.quantized_inputs)
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert len(quantized_outputs.shape) == 4
        assert quantized_outputs.size() == self.expected_size

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        for i in range(len(outputs)):
            assert all(
                torch.allclose(output, fuse_output, atol=1e-5)
                for output, fuse_output in zip(outputs[i], fuse_outputs[i])
            )


class TestReIDClsOutputBlockInt32(ModelTemplate):
    def setup(self):
        super(TestReIDClsOutputBlockInt32, self).setup()
        self.num_classes = 10
        self.inputs = [torch.rand(1, 16, 16, 16)]
        self.quantized_inputs = [torch.ones(1, 16, 16, 16, dtype=torch.int8)]

        # test: int8_output=False and include_top=True

        self.model = ReIDClsOutputBlock(
            num_classes=self.num_classes,
            in_channels=16,
            feat_channels=16,
            int8_output=False,
            bn_kwargs={},
            pool_kernel_size=4,
            include_top=True,
        )
        self.build_model()
        self.expected_size = torch.Size([1, 16, 16 - 4 + 1, 16 - 4 + 1])

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert len(outputs.shape) == 2
        assert outputs.shape[1] == self.num_classes

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs)
        qat_outputs = self.qat_model(qat_inputs)
        assert len(qat_outputs.shape) == 2
        assert qat_outputs.shape[1] == self.num_classes

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.quantized_inputs)
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert len(quantized_outputs.shape) == 2
        assert quantized_outputs.shape[1] == self.num_classes

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        for i in range(len(outputs)):
            assert all(
                torch.allclose(output, fuse_output, atol=1e-5)
                for output, fuse_output in zip(outputs[i], fuse_outputs[i])
            )
