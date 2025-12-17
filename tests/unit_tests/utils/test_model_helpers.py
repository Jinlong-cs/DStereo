import torch

from hat.models.backbones.resnet import ResNet18
from hat.utils.model_helpers import (
    fuse_norm_recursively,
    get_binding_module,
    has_normalization,
)
from tests.unit_tests.base import BoringModel


def test_bn_fusion():
    x = torch.randn((1, 3, 224, 224))
    model = ResNet18(num_classes=1000, bn_kwargs={})
    assert has_normalization(model, check_list=["bn"])

    model_fused = fuse_norm_recursively(model, fuse_list=["bn"])
    assert not has_normalization(model_fused, check_list=["bn"])

    y = model(x)
    y_fused = model_fused(x)
    torch.testing.assert_allclose(y, y_fused)


def test_get_binding_module():
    model = BoringModel()
    assert get_binding_module(model) == model
    model_dp = torch.nn.parallel.DataParallel(model)
    assert model_dp != model
    assert get_binding_module(model_dp) == model


def test_has_batchnorm():
    model = ResNet18(num_classes=1000, bn_kwargs={})
    assert has_normalization(model, check_list=["bn"])

    model_fused = fuse_norm_recursively(model, fuse_list=["bn"])
    assert not has_normalization(model_fused, check_list=["bn"])
