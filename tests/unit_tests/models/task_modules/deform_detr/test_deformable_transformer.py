import torch

from hat.models.embeddings import PositionEmbeddingSine
from hat.models.task_modules.deform_detr.deformable_transformer import (
    DeformableDetrTransformer,
    DeformableDetrTransformerDecoder,
    DeformableDetrTransformerEncoder,
)
from tests.unit_tests.models.base import qtensor_test
from tests.unit_tests.models.task_modules.head_template import HeadTemplate


class TestDeformableTransformer(HeadTemplate):
    def setup(self):
        super(TestDeformableTransformer, self).setup()
        self.embed_dims = 64
        self.num_heads = 8
        self.num_decoder_layers = 6
        self.ff_channels = 64
        self.as_two_stage = False
        self.with_box_refine = False
        self.num_queries = 10
        self.model = DeformableDetrTransformer(
            encoder=DeformableDetrTransformerEncoder(
                embed_dim=self.embed_dims,
                num_heads=8,
                feedforward_dim=128,
                attn_dropout=0.1,
                ffn_dropout=0.1,
                num_layers=6,
                post_norm=False,
                num_feature_levels=4,
            ),
            decoder=DeformableDetrTransformerDecoder(
                embed_dim=self.embed_dims,
                num_heads=8,
                feedforward_dim=128,
                attn_dropout=0.1,
                ffn_dropout=0.1,
                num_layers=6,
                return_intermediate=True,
                num_feature_levels=4,
                as_two_stage=self.as_two_stage,
                with_box_refine=self.with_box_refine,
            ),
            num_feature_levels=4,
            two_stage_num_proposals=self.num_queries,
            as_two_stage=self.as_two_stage,
        )
        position_embedding = PositionEmbeddingSine(
            num_pos_feats=self.embed_dims // 2,
            temperature=10000,
            normalize=True,
            offset=-0.5,
        )
        spatial_shapes = [(32, 24), (16, 12), (8, 6), (4, 3)]
        self.multi_feats = []
        self.multi_mask = []
        self.multi_pos_embeddings = []
        for h, w in spatial_shapes:
            self.multi_feats.append(torch.rand(1, self.embed_dims, h, w))
            self.multi_mask.append(
                torch.randint(0, 2, size=(1, h, w)).to(torch.bool)
            )
            pos_embed = position_embedding(self.multi_mask[-1])
            # print("pos_embed.shape", pos_embed.shape)
            self.multi_pos_embeddings.append(pos_embed)
        self.query_embed = torch.rand(self.num_queries, self.embed_dims * 2)
        # self.pos_embed = torch.rand(1, self.embed_dims, 8, 8)
        self.quantized_multi_feats = []
        self.quantized_multi_mask = []
        self.quantized_pos_embed = []
        for h, w in spatial_shapes:
            self.quantized_multi_feats.append(
                torch.ones(1, self.embed_dims, h, w)
            )
            self.quantized_multi_mask.append(
                torch.randint(0, 2, size=(1, h, w)).to(torch.bool)
            )
            self.quantized_pos_embed.append(
                position_embedding(self.multi_mask[-1])
            )
        self.build_model(use_fx=True)

    def test_float_model(self):
        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embed
        outputs = self.float_model(
            self.multi_feats,
            self.multi_mask,
            self.multi_pos_embeddings,
            query_embeds,
        )
        assert len(outputs[0]) == self.num_decoder_layers
        assert outputs[0][0].shape == (1, self.num_queries, self.embed_dims)
        assert len(outputs[1]) == 0
        assert outputs[2].shape == (1, self.num_queries, 2)
        assert len(outputs[3]) == self.num_decoder_layers
        assert outputs[3][0].shape == (1, self.num_queries, 2)

    def test_qat_model(self):
        qat_multi_feats = [qtensor_test(f) for f in self.multi_feats]
        qat_query_embed = None
        if not self.as_two_stage:
            qat_query_embed = qtensor_test(self.query_embed)
        qat_outputs = self.qat_model(
            qat_multi_feats,
            self.multi_mask,
            self.multi_pos_embeddings,
            qat_query_embed,
        )
        assert len(qat_outputs[0]) == self.num_decoder_layers
        assert qat_outputs[0][0].shape == (
            1,
            self.num_queries,
            self.embed_dims,
        )
        assert len(qat_outputs[1]) == 0
        assert qat_outputs[2].shape == (1, self.num_queries, 2)
        assert len(qat_outputs[3]) == self.num_decoder_layers
        assert qat_outputs[3][0].shape == (1, self.num_queries, 2)

    def test_quantize_model(self):
        quantized_multi_feats = [
            qtensor_test(f) for f in self.quantized_multi_feats
        ]
        quantized_query_embed = None
        if not self.as_two_stage:
            quantized_query_embed = qtensor_test(self.query_embed)
        quantized_outputs = self.quantized_model(
            quantized_multi_feats,
            self.multi_mask,
            self.multi_pos_embeddings,
            quantized_query_embed,
        )

        assert len(quantized_outputs[0]) == self.num_decoder_layers
        assert quantized_outputs[0][0].shape == (
            1,
            self.num_queries,
            self.embed_dims,
        )
        assert len(quantized_outputs[1]) == 0
        assert quantized_outputs[2].shape == (1, self.num_queries, 2)
        assert len(quantized_outputs[3]) == self.num_decoder_layers
        assert quantized_outputs[3][0].shape == (1, self.num_queries, 2)

    def test_fuse_model(self):
        pass
