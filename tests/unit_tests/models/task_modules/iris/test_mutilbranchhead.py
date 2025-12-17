import pytest
import torch

from hat.registry import build_from_registry


@pytest.mark.parametrize(
    [
        "alpha",
        "classfier_num",
    ],
    [
        pytest.param(0.25, 2),
    ],
)
def test_iris_head(alpha, classfier_num):

    batch_size = 256
    dummy_data = torch.randn([batch_size, 64, 6, 10])

    config = dict(
        type="IrisMutilBranchHead",
        bn_kwargs={},
        alpha=alpha,
        classfier_num=classfier_num,
        use_pool=False,
        bias=True,
    )
    iris_model = build_from_registry(config)
    l_feat, r_feat = iris_model(dummy_data)
    assert l_feat.shape == (batch_size, classfier_num, 1, 1)
    assert r_feat.shape == (batch_size, classfier_num, 1, 1)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
