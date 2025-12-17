# Copyright (c) Horizon Robotics. All rights reserved.

import torch

from hat.models.task_modules.traj_pred.backbones.sgnet_backbone import (
    SGNetCvaeDecoder,
    SGNetEncoder,
)
from hat.models.task_modules.traj_pred.structures.sgnet import SGNet


def gen_example_sgnet_model():
    batch_size = 32
    enc_steps = 3
    dec_steps = 9
    pred_dim = 4
    hidden_size = 512
    latent_dim = 32
    k_value = 10

    data = torch.rand([batch_size, pred_dim, 1, enc_steps])
    rand_cvae_seed = torch.normal(
        0, 1, size=(batch_size, latent_dim, 1, k_value)
    )
    test_input = {
        "input_x": data,
        "rand_cvae_seed": rand_cvae_seed,
        "target_traj": None,
    }

    encoder = SGNetEncoder(
        in_channels=pred_dim,
        enc_steps=enc_steps,
        dec_steps=dec_steps,
        hidden_size=hidden_size,
        return_all_step=False,
    )
    decoder = SGNetCvaeDecoder(
        in_channels=pred_dim,
        out_channels=pred_dim,
        hidden_size=hidden_size,
        enc_steps=enc_steps,
        dec_steps=dec_steps,
        latent_dim=latent_dim,
        k_value=k_value,
        enc_with_all_steps=False,
    )
    net = SGNet(
        backbone=encoder,
        head=decoder,
        k_value=k_value,
        is_int_infer_model=True,
    )
    return net, test_input


def test_sgnet_structure():
    model, test_input = gen_example_sgnet_model()
    model.fuse_model()
    model.set_qconfig()
    output = model(test_input)

    batch_size = 32
    dec_steps = 9
    pred_dim = 4
    k_value = 10
    assert hasattr(output, "probabilities")
    assert hasattr(output, "pred_traj")
    assert output.probabilities.shape == (batch_size, 1, 1, k_value)
    assert output.pred_traj.shape == (batch_size, k_value, pred_dim, dec_steps)
