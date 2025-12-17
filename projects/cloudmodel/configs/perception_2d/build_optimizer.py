import torch


def build_optimizer(name="adamw", lr=0.0002, weight_decay=1e-2, **kwargs):
    if name == "adamw":
        optimizer = dict(
            type=torch.optim.AdamW,
            lr=lr,
            weight_decay=weight_decay,  # 1e-2
            **kwargs,
        )
    elif name == "adam":
        optimizer = dict(
            type=torch.optim.Adam,
            lr=lr,
            weight_decay=weight_decay,  # 5e-4,
            **kwargs,
        )
    elif name == "sgd":
        optimizer = dict(
            type=torch.optim.SGD,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,  # 5e-4,
            **kwargs,
        )
    else:
        raise NotImplementedError(name)

    return optimizer
