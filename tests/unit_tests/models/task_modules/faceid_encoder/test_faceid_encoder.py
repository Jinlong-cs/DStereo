import torch

from hat.models.task_modules.faceid_encoder import FaceIDLargeVargNet


def test_faceid_large_vargnet():
    faceid_feature_net = FaceIDLargeVargNet(
        bn_kwargs=dict(eps=1e-3, momentum=0.01),
        bias=False,
        embedding_size=256,
        dropout=0.0,
        use_fp16=False,
        flat_output=False,
    )

    input_features = torch.randn(64, 3, 112, 112)

    output = faceid_feature_net(input_features)
    assert len(output.shape) == 4
    assert output.shape[0] == 64
    assert output.shape[1] == 256
