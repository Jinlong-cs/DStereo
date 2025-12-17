import torch

from hat.models.backbones.iresnet import IResNet100, IResNet180


def test_faceid_res100():
    b, c, h, w = (4, 3, 112, 112)
    embedding_size = 256
    faceid_feature_net = IResNet100(
        bn_kwargs=dict(eps=1e-3, momentum=0.01),
        bias=False,
        act_type="relu",
        embedding_size=embedding_size,
        dropout=0.0,
    )

    input_features = torch.randn(b, c, h, w)

    output = faceid_feature_net(input_features)
    assert len(output.shape) == 2
    assert output.shape[0] == b
    assert output.shape[1] == embedding_size


def test_faceid_res180():
    b, c, h, w = (4, 3, 112, 112)
    embedding_size = 256
    faceid_feature_net = IResNet180(
        bn_kwargs=dict(eps=1e-3, momentum=0.01),
        bias=False,
        act_type="relu",
        embedding_size=256,
        dropout=0.0,
    )

    input_features = torch.randn(b, c, h, w)

    output = faceid_feature_net(input_features)
    assert len(output.shape) == 2
    assert output.shape[0] == b
    assert output.shape[1] == embedding_size
