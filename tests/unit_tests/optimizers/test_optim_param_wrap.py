import pytest
import torch
from torch.optim import SGD, Adam, AdamW, RMSprop

from hat.optimizers.optim_param_wrap import custom_param_optimizer


def build_test_model():
    test_model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
    )
    test_model[0].weight.data = torch.ones(test_model[0].weight.shape) * 1.1
    test_model[0].bias.data = torch.ones(test_model[0].bias.shape) * 2.1
    test_model[1].weight.data = torch.ones(test_model[1].weight.shape) * 3.1
    test_model[1].bias.data = torch.ones(test_model[1].bias.shape) * 4.1
    return test_model


@pytest.mark.parametrize(
    "optim_cls,optim_cfgs,custom_param_mapper,expected_param_settings",
    [
        (
            SGD,
            {"lr": 0.1, "momentum": 0.9},
            {torch.nn.Conv2d: {"lr": 0.01}},
            [
                {"lr": 0.01, "momentum": 0.9},
                {"lr": 0.01, "momentum": 0.9},
                {"lr": 0.1, "momentum": 0.9},
                {"lr": 0.1, "momentum": 0.9},
            ],
        ),
        (
            Adam,
            {"lr": 0.001, "betas": (0.9, 0.999)},
            {"norm_types": {"lr": 0.01}},
            [
                {"lr": 0.001, "betas": (0.9, 0.999)},
                {"lr": 0.001, "betas": (0.9, 0.999)},
                {"lr": 0.01, "betas": (0.9, 0.999)},
                {"lr": 0.01, "betas": (0.9, 0.999)},
            ],
        ),
        (
            SGD,
            {"lr": 0.1, "momentum": 0.9},
            {"weight": {"lr": 0.01}},
            [
                {"lr": 0.01, "momentum": 0.9},
                {"lr": 0.1, "momentum": 0.9},
                {"lr": 0.01, "momentum": 0.9},
                {"lr": 0.1, "momentum": 0.9},
            ],
        ),
        (
            RMSprop,
            {"lr": 0.1, "alpha": 0.9, "weight_decay": 0.05},
            {("norm_types", "bias"): {"weight_decay": 0.0}},
            [
                {"lr": 0.1, "weight_decay": 0.05},
                {"lr": 0.1, "weight_decay": 0.05},
                {"lr": 0.1, "weight_decay": 0.05},
                {"lr": 0.1, "weight_decay": 0.0},
            ],
        ),
        (
            AdamW,
            {"lr": 0.1, "weight_decay": 0.05},
            {("norm_types", "bias"): {"weight_decay": 0.0}},
            [
                {"lr": 0.1, "weight_decay": 0.05},
                {"lr": 0.1, "weight_decay": 0.05},
                {"lr": 0.1, "weight_decay": 0.05},
                {"lr": 0.1, "weight_decay": 0.0},
            ],
        ),
        (
            AdamW,
            {"lr": 0.1, "weight_decay": 0.05},
            {"no_such_param": {"weight_decay": 0.0}},
            [
                {"lr": 0.1, "weight_decay": 0.05},
                {"lr": 0.1, "weight_decay": 0.05},
                {"lr": 0.1, "weight_decay": 0.05},
                {"lr": 0.1, "weight_decay": 0.05},
            ],
        ),
        (
            AdamW,
            {"lr": 0.1, "weight_decay": 0.05},
            {
                "norm_types": {"weight_decay": 0.0},
                "bias": {"weight_decay": 0.0},
            },
            [
                {"lr": 0.1, "weight_decay": 0.05},
                {"lr": 0.1, "weight_decay": 0.0},
                {"lr": 0.1, "weight_decay": 0.0},
                {"lr": 0.1, "weight_decay": 0.0},
            ],
        ),
        (
            SGD,
            {
                "lr": 0.1,
                "weight_decay": 0.05,
            },
            {
                torch.nn.Conv2d: {"lr": 0.01},
                "norm_types": {"weight_decay": 0.0},
                (torch.nn.Conv2d, "bias"): {"weight_decay": 0.01},
            },
            [
                {"lr": 0.01, "weight_decay": 0.05},
                {"lr": 0.01, "weight_decay": 0.01},
                {"lr": 0.1, "weight_decay": 0.0},
                {"lr": 0.1, "weight_decay": 0.0},
            ],
        ),
    ],
)
def test_optim_param_wrapper(
    optim_cls, optim_cfgs, custom_param_mapper, expected_param_settings
):
    model = build_test_model()
    optimizer = custom_param_optimizer(
        optim_cls, model, optim_cfgs, custom_param_mapper
    )
    # Check param_groups
    params = []
    for param_group in optimizer.param_groups:
        for p in param_group["params"]:
            for k, value in expected_param_settings[
                p.int().flatten()[0] - 1
            ].items():
                assert param_group[k] == value
            params.append(p)

    assert len(set(params)) == len(params)
    assert set(model.parameters()) == set(params)

    model.train()
    # Check optimizer functionality
    # Generate random input and target
    input = torch.randn(1, 1, 32, 32)
    target = torch.randn(1, 64, 32, 32)
    output = model(input)
    loss = (output - target).pow(2).mean()
    loss.backward()

    # Check if gradients are not zero
    for param in model.parameters():
        assert param.grad is not None
        assert not torch.all(param.grad == 0)

    # Step the optimizer
    optimizer.step()
    optimizer.zero_grad()
    # Check if gradients are zero or None after step
    for param in model.parameters():
        # assert param.grad is not None
        if param.grad is not None:
            assert torch.all(param.grad == 0)
