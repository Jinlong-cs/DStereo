import torch

from hat.models.embeddings import PositionEmbeddingSine
from hat.models.task_modules.detr.head import DetrHead
from hat.models.task_modules.detr.transformer import Transformer
from tests.unit_tests.models.base import qtensor_test
from tests.unit_tests.models.task_modules.head_template import HeadTemplate


class TestDetrHead(HeadTemplate):
    def setup(self):
        super(TestDetrHead, self).setup()
        self.num_classes = 80
        self.embed_dims = 64
        self.num_heads = 8
        self.max_per_img = 10
        self.in_channels = 16
        self.num_decoder_layers = 6
        self.ff_channels = 64
        transformer = Transformer(
            embed_dims=self.embed_dims,
            num_heads=self.num_heads,
            num_decoder_layers=self.num_decoder_layers,
            feedforward_channels=self.ff_channels,
            return_intermediate_dec=True,
        )
        pos_embed = PositionEmbeddingSine(num_pos_feats=32)
        self.model = DetrHead(
            transformer=transformer,
            pos_embed=pos_embed,
            num_classes=self.num_classes,
            in_channels=self.in_channels,
            max_per_img=self.max_per_img,
        )
        img_meta = dict(
            batch_input_shape=[[512, 512]],
            img_shape=[[3, 512, 512]],
        )
        self.img_meta = img_meta
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs, self.img_meta)
        assert len(outputs[0]) == self.num_decoder_layers
        assert len(outputs[1]) == self.num_decoder_layers
        for cls_score in outputs[0]:
            assert isinstance(cls_score, torch.Tensor)
            assert len(cls_score) == len(self.inputs[-1])
            assert cls_score.shape[1] == self.max_per_img
            assert cls_score.shape[2] == self.num_classes + 1
        for bbox_pred in outputs[1]:
            assert isinstance(bbox_pred, torch.Tensor)
            assert len(bbox_pred) == len(self.inputs[-1])
            assert bbox_pred.shape[1] == self.max_per_img
            assert bbox_pred.shape[2] == 4

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs)
        qat_outputs = self.qat_model(qat_inputs, self.img_meta)
        assert len(qat_outputs[0]) == self.num_decoder_layers
        assert len(qat_outputs[1]) == self.num_decoder_layers
        for cls_score in qat_outputs[0]:
            assert isinstance(cls_score, torch.Tensor)
            assert len(cls_score) == len(self.inputs[-1])
            assert cls_score.shape[1] == self.max_per_img
            assert cls_score.shape[2] == self.num_classes + 1
        for bbox_pred in qat_outputs[1]:
            assert isinstance(bbox_pred, torch.Tensor)
            assert len(bbox_pred) == len(self.inputs[-1])
            assert bbox_pred.shape[1] == self.max_per_img
            assert bbox_pred.shape[2] == 4

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.quantized_inputs)
        quantized_outputs = self.quantized_model(
            quantized_inputs, self.img_meta
        )
        assert len(quantized_outputs[0]) == self.num_decoder_layers
        assert len(quantized_outputs[1]) == self.num_decoder_layers
        for cls_score in quantized_outputs[0]:
            assert isinstance(cls_score, torch.Tensor)
            assert len(cls_score) == len(self.inputs[-1])
            assert cls_score.shape[1] == self.max_per_img
            assert cls_score.shape[2] == self.num_classes + 1
        for bbox_pred in quantized_outputs[1]:
            assert isinstance(bbox_pred, torch.Tensor)
            assert len(bbox_pred) == len(self.inputs[-1])
            assert bbox_pred.shape[1] == self.max_per_img
            assert bbox_pred.shape[2] == 4

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs, self.img_meta)
        fuse_outputs = self.fuse_model(self.inputs, self.img_meta)
        for i in range(len(outputs)):
            assert all(
                torch.allclose(output, fuse_output, atol=1e-5)
                for output, fuse_output in zip(outputs[i], fuse_outputs[i])
            )
