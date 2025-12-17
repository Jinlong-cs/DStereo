import copy

import numpy as np
import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


def test_motr_task():

    input_size = (200, 400)
    num_queries = 256
    num_classes = 1

    config = dict(
        type="Motr",
        backbone=dict(
            type="efficientnet",
            bn_kwargs={},
            model_type="b3",
            num_classes=1000,
            include_top=False,
            activation="relu",
            use_se_block=False,
        ),
        head=dict(
            type="MotrHead",
            transformer=dict(
                type="MotrDeformableTransformer",
                pos_embed=dict(
                    type="PositionEmbeddingSine",
                    num_pos_feats=128,
                    normalize=True,
                    temperature=20,
                ),
                d_model=256,
                num_queries=num_queries,
                dim_feedforward=1024,
                dropout=0.0,
                return_intermediate_dec=True,
                extra_track_attn=True,
                enc_n_points=1,
                dec_n_points=1,
            ),
            num_classes=num_classes,
            in_channels=[384],
            max_per_img=num_queries,
        ),
        criterion=dict(
            type="MotrCriterion",
            num_classes=num_classes,
        ),
        post_process=dict(
            type="MotrPostProcess",
        ),
        track_embed=dict(
            type="QueryInteractionModule",
            dim_in=256,
            hidden_dim=1024,
        ),
    )

    motr_model = build_from_registry(config)
    qat_motr_model = copy.deepcopy(motr_model)

    x = dict(
        frame_data_list=[
            dict(
                img=[torch.randn((1, 3, input_size[0], input_size[1]))],
                query_pos=torch.randn((1, 256, 2, 128), dtype=torch.float),
                mask_query=torch.ones(
                    (1, 1, 1, num_queries), dtype=torch.float
                ),
                ref_pts=torch.randn((1, 4, 2, 128), dtype=torch.float),
                gt_bboxes=[
                    torch.tensor(
                        [[10, 10, 20, 20], [20, 20, 40, 40]], dtype=torch.float
                    ),
                ],
                gt_classes=[torch.tensor([0, 0], dtype=torch.float)],
                gt_ids=[torch.tensor([0, 1], dtype=torch.float)],
                img_name=["test_img.jpg"],
                seq_name=["test_seq"],
                img_shape=[
                    np.array([3, input_size[0], input_size[1]]),
                ],
                scale_factor=torch.tensor(
                    [1.4253, 1.4264, 1.4253, 1.4264], dtype=torch.float
                ),
            ),
        ]
    )

    motr_model(x)

    qat_test(qat_motr_model, x, with_quantized=False)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
