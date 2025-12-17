import torch

from hat.optimizers.legacy_nadam_ex import LegacyNadamEx
from tests.data.toy_modules import ToyBackbone, ToyHead, ToyLoss, ToyModel


# TODO(mengyang.duan): refactor, maybe need to create and use OptimizerTemplate
def test_legacy_nadam_optimizer():
    model = ToyModel(
        backbone=ToyBackbone(strides=(1, 2), channels=(3, 8)),
        head=ToyHead(
            in_channels=8,
            fc_filter=16,
            num_classes=10,
            with_dequant=True,
        ),
        loss=ToyLoss(),
    )

    optim_params = [
        dict(
            lr=0.0001,
            weight_decay=0.0001,
            beta1=0.95,
            beta2=0.97,
            fused=False,
        ),
        dict(
            lr=0.0001,
            weight_decay=0.0001,
            beta1=0.95,
            beta2=0.97,
            fused=True,
        ),
    ]

    input = torch.randn((1, 3, 14, 14))
    label = torch.randint(0, 9, (1,))
    torch.manual_seed(1)

    res = []
    for optim_param in optim_params:

        optimizer = LegacyNadamEx(model.parameters(), **optim_param)
        print(optimizer.state_dict())
        for _ in range(5):
            optimizer.zero_grad()
            output = model(input, label)
            loss = output["loss"].sum()
            loss.backward()
            optimizer.step()
        res.append(model.named_parameters())

    for (k1, v1), (k2, v2) in zip(res[0], res[1]):
        assert k1 == k2
        assert torch.allclose(v1, v2)
