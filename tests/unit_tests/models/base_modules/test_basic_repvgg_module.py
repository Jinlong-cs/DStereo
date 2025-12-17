import torch
import torch.nn as nn

from hat.models.base_modules.basic_repvgg_module import (
    MultiBranchConvModule,
    MultiBranchModule,
    RepBlock,
)


def test_MultiBranchConvModule():
    fake_data = torch.rand([8, 16, 56, 56])
    bb_module = MultiBranchConvModule(
        16, 16, k_size_list=[0, 1, 3], bn_flag_list=[True, True, True]
    )
    bb_module.eval()
    output1 = bb_module(fake_data)
    assert output1 is not None


def test_RepBlock():
    fake_data = torch.rand([8, 8, 56, 56])
    module = MultiBranchConvModule(
        8, 8, k_size_list=[0, 1, 3], bn_flag_list=[True, True, True]
    )
    bb_module = RepBlock(
        module,
        act_layer=nn.ReLU(),
    )
    bb_module.eval()
    output1 = bb_module(fake_data)
    bb_module.switch_to_deploy()
    output2 = bb_module(fake_data)
    print(output1.shape, output2.shape)
    assert output1 is not None
    assert (
        output2 - output1
    ).norm() < 1e-4, f"{output1.shape} {output2.shape}"


def test_RepBlock2():
    fake_data = torch.rand([8, 8, 56, 56])
    module = MultiBranchConvModule(
        8, 8, k_size_list=[0, 1, 3], bn_flag_list=[False, False, True]
    )
    bb_module = RepBlock(
        module, norm_layer=nn.BatchNorm2d(8), act_layer=nn.ReLU()
    )
    bb_module.eval()
    output1 = bb_module(fake_data)
    bb_module.switch_to_deploy()
    output2 = bb_module(fake_data)
    assert output1 is not None
    assert (
        output2 - output1
    ).norm() < 1e-4, (
        f"{output1.shape} {output2.shape}, {output1.norm()}, {output2.norm()}"
    )


def test_RepBlock_pscale():
    fake_data = torch.rand([8, 8, 56, 56])
    conv_module = nn.Conv2d(8, 8, 3, 1, padding=1, bias=False)
    module2 = MultiBranchModule(
        [conv_module, nn.Identity(), nn.BatchNorm2d(8)],
        scale_list=[torch.ones([8, 1, 1]) * 0.001, 1, 1],
    )
    bb_module = RepBlock(module=module2)

    bb_module.eval()
    output1 = bb_module(fake_data)
    bb_module.switch_to_deploy()
    output2 = bb_module(fake_data)
    print(output1.shape, output2.shape, (output1 - output2).norm())
    assert output1 is not None
    assert (
        output2 - output1
    ).norm() < 1e-4, f" {output1.flatten()[:5]},{output2.flatten()[:5]}"


def test_RepBlock_multi_branch():
    fake_data = torch.rand([8, 8, 56, 56])
    conv_module = nn.Conv2d(8, 8, 3, 1, padding=1, bias=False)
    module2 = MultiBranchModule(
        [conv_module, nn.Identity(), nn.BatchNorm2d(8)],
        scale_list=[1, 1, 1],
    )
    bb_module = RepBlock(module=module2)

    bb_module.eval()
    output1 = bb_module(fake_data)
    bb_module.switch_to_deploy()
    output2 = bb_module(fake_data)
    print(output1.shape, output2.shape, (output1 - output2).norm())
    assert output1 is not None
    assert (
        output2 - output1
    ).norm() < 1e-4, f" {output1.flatten()[:5]},{output2.flatten()[:5]}"


def test_RepBlock_rep():
    fake_data = torch.rand([8, 8, 56, 56])
    module = MultiBranchConvModule(
        8, 8, k_size_list=[0, 1, 3], bn_flag_list=[False, False, True]
    )
    rep_module = RepBlock(module, deploy=False)
    module2 = MultiBranchModule(
        [rep_module, nn.Identity(), nn.BatchNorm2d(8)],
        scale_list=[1, 1, 1],
    )
    bb_module = RepBlock(module=module2)

    bb_module.eval()
    output1 = bb_module(fake_data)
    bb_module.module.branches[0].switch_to_deploy()
    output2 = bb_module(fake_data)
    bb_module.switch_to_deploy()
    output3 = bb_module(fake_data)
    assert output1 is not None
    assert (output2 - output1).norm() / (
        56 * 56 * 8 * 8
    ) < 2e-9, f"{output1.shape} {output2.shape}"
    assert (output3 - output2).norm() / (
        56 * 56 * 8 * 8
    ) < 2e-9, f"{(output2 - output1).norm()} {(output3 - output2).norm()}"


def test_RepBlock_repbn():
    fake_data = torch.rand([8, 8, 56, 56])
    module = MultiBranchConvModule(
        8, 8, k_size_list=[0, 1, 3], bn_flag_list=[False, False, True]
    )
    rep_module = RepBlock(module, norm_layer=nn.BatchNorm2d(8), deploy=False)
    module2 = MultiBranchModule(
        [rep_module, nn.Identity(), nn.BatchNorm2d(8)],
        scale_list=[1, 1, 1],
    )
    bb_module = RepBlock(module=module2)

    bb_module.eval()
    output1 = bb_module(fake_data)
    bb_module.module.branches[0].switch_to_deploy()
    output2 = bb_module(fake_data)
    bb_module.switch_to_deploy()
    output3 = bb_module(fake_data)
    # print(output1.shape, output2.shape, (output1 - output2).norm())
    assert output1 is not None
    assert (output2 - output1).norm() / (
        56 * 56 * 8 * 8
    ) < 2e-9, f"{output1.shape} {output2.shape}"
    assert (output3 - output2).norm() / (
        56 * 56 * 8 * 8
    ) < 2e-9, f"{output2.shape} {output3.shape}"
