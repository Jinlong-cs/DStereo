import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import json
import os
import os.path as osp

import matplotlib.pyplot as plt


def extract_lut_from_ckpt(ckpt_path, save_dir):
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    coarse_x = state_dict["lut_net.coarse_x"].numpy()
    coarse_y = state_dict["lut_net.coarse_y"].numpy()
    coarse_y = np.concatenate([np.array([0]), coarse_y], axis=0)
    fine_x = state_dict["lut_net.fine_x"].numpy()
    fine_y = state_dict["lut_net.fine_y"].numpy()
    fine_y = np.concatenate([np.array([0]), fine_y], axis=0)

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    plt.subplot(2, 1, 1)
    plt.plot(coarse_x, coarse_y, "r-")
    plt.plot(coarse_x, coarse_y, "ro")
    plt.title("uniform lut")
    plt.subplot(2, 1, 2)
    plt.plot(fine_x, fine_y, "r-")
    plt.plot(fine_x, fine_y, "ro")
    plt.title("nonuniform lut")
    plt.tight_layout()
    plt.savefig(osp.join(save_dir, "lut.png"))
    plt.close()

    with open(osp.join(save_dir, "coarse_lut.json"), "w") as f:
        json.dump(
            {"coarse_x": coarse_x.tolist(), "coarse_y": coarse_y.tolist()}, f
        )
    with open(osp.join(save_dir, "fine_lut.json"), "w") as f:
        json.dump({"fine_x": fine_x.tolist(), "fine_y": fine_y.tolist()}, f)

    return fine_x, fine_y


def get_lutnet(
    in_channels,
    output_bit,
    lut_size,
    down_size=None,
    pregamma=1 / 3.0,
    track_coarse_steps=2000,
    norm_symmetric=True,
    alpha=0.9,
):
    model = dict(
        type="LutNet",
        in_channels=in_channels,
        output_bit=output_bit,
        lut_size=lut_size,
        down_size=down_size,
        pregamma=pregamma,
        track_coarse_steps=track_coarse_steps,
        norm_symmetric=norm_symmetric,
        alpha=alpha,
    )

    return model
