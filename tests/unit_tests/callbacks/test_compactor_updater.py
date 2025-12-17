import torch

from hat.models.backbones.vargnasnet import VargNASBlockConfig
from hat.models.base_modules.conv_compactor import ConvCompactor2d
from hat.registry import build_from_registry


def test_compactor_update():

    nasnet_config_example = [
        VargNASBlockConfig(
            in_channels=32,
            out_channels=32,
            head_op="varg_k3f1",
            stack_ops=[],
            stack_ops_num=0,
            stride=1,
        ),  # noqa
        VargNASBlockConfig(
            in_channels=32,
            out_channels=24,
            head_op="vargr_k3",
            stack_ops=["varg_k3"],
            stack_ops_num=1,
            stride=2,
        ),  # noqa
        VargNASBlockConfig(
            in_channels=24,
            out_channels=40,
            head_op="vargr_k5",
            stack_ops=[],
            stack_ops_num=0,
            stride=2,
        ),  # noqa
        VargNASBlockConfig(
            in_channels=40,
            out_channels=56,
            head_op="vargr_k3",
            stack_ops=["varg_k5", "varg_k3"],
            stack_ops_num=2,
            stride=2,
        ),  # noqa
        VargNASBlockConfig(
            in_channels=56,
            out_channels=72,
            head_op="vargr_k3",
            stack_ops=["varg_k3", "varg_k3"],
            stack_ops_num=2,
            stride=1,
        ),  # noqa
        VargNASBlockConfig(
            in_channels=72,
            out_channels=96,
            head_op="vargr_k3",
            stack_ops=["varg_k3"],
            stack_ops_num=1,
            stride=2,
        ),  # noqa
        VargNASBlockConfig(
            in_channels=96,
            out_channels=160,
            head_op="vargr_k5",
            stack_ops=[],
            stack_ops_num=0,
            stride=1,
        ),  # noqa
    ]

    num_classes = 10
    bn_kwargs = dict(eps=1e-5, momentum=0.1)
    model_cfg = dict(
        type="VargNASNet",
        net_config=nasnet_config_example,
        num_classes=num_classes,
        bn_kwargs=bn_kwargs,
        include_top=False,
        flat_output=False,
    )

    compactor_update_callback_cfg = dict(
        type="CompactorUpdater",
        before_mask_iters=200,
        mask_interval=200,
        pruned_epsilon=1e-5,
        modules=["backbone"],
    )

    model = build_from_registry(model_cfg)
    compactorUpdater = build_from_registry(compactor_update_callback_cfg)
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=1e-4, weight_decay=1e-5
    )

    assert not has_compactor(model)
    compactorUpdater._add_compactor_recursively(model, optimizer)
    assert has_compactor(model)


def has_compactor(model):
    for _, m in model.named_modules():
        if isinstance(m, ConvCompactor2d):
            return True
    return False
