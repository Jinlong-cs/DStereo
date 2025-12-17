import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import profile_test, qat_test


@pytest.mark.parametrize(
    ["alpha", "classfier_num", "losses", "mod"],
    [
        pytest.param(0.25, 2, dict(type="CEWithLabelSmooth"), "training"),
    ],
)
def test_iris_task(alpha, classfier_num, losses, mod):

    batch_size = 2
    dummy_data = dict(
        img=torch.randn([batch_size, 3, 192, 320]),
        labels=[torch.randint(0, 2, (batch_size,)) for _ in range(7)],
    )
    config = dict(
        type="IrisClassifier",
        backbone=dict(
            type="VargNetV2",
            num_classes=1000,
            bn_kwargs={},
            alpha=alpha,
            include_top=False,
        ),
        head=dict(
            type="IrisMutilBranchHead",
            bn_kwargs={},
            alpha=alpha,
            classfier_num=classfier_num,
            use_pool=False,
            bias=True,
        ),
        losses=losses,
    )
    iris_model = build_from_registry(config)
    iris_model.training = True if mod == "train" else False
    if mod == "train":
        iris_model.training = True
        preds, losses, l_loss, r_loss = iris_model(dummy_data)
        assert len(preds) == 2
        assert preds[0].shape == (batch_size, classfier_num, 1, 1)
        assert isinstance(losses, torch.FloatTensor)
        assert l_loss + r_loss == losses
    else:
        iris_model.training = False
        preds, _ = iris_model(dummy_data)
        assert len(preds) == 2
        assert preds[0].shape == (batch_size, classfier_num, 1, 1)

    qat_test(iris_model, dummy_data)
    # Dont run profile_test before qat_test, strange bug will occur.
    profile_test(iris_model, dummy_data)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
