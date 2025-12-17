import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

KQ_NORM = 1


class LatentCode(nn.Module):
    def __init__(self, device, batch_size, eye_params_init):
        super(LatentCode, self).__init__()
        R_init = torch.tensor(
            eye_params_init.get("R", 7.8), dtype=torch.float64
        ).reshape(1, 1)
        K_init = torch.tensor(
            eye_params_init.get("K", 4.75), dtype=torch.float64
        ).reshape(1, 1)
        if eye_params_init.get("alpha_left") is not None:
            alpha_init = torch.tensor(
                eye_params_init.get("alpha_left", 5), dtype=torch.float64
            ).reshape(1, 1)
        else:
            alpha_init = torch.tensor(
                eye_params_init.get("alpha", 5), dtype=torch.float64
            ).reshape(1, 1)
        beta_init = torch.tensor(
            eye_params_init.get("beta", 1.5), dtype=torch.float64
        ).reshape(1, 1)
        self.register_parameter("R", Parameter(R_init.to(device)))
        self.register_parameter("K", Parameter(K_init.to(device)))
        self.register_parameter("alpha", Parameter(alpha_init.to(device)))
        self.register_parameter("beta", Parameter(beta_init.to(device)))

        if eye_params_init.get("kq_result") is None:
            kq_init = torch.tensor(
                [500 * KQ_NORM, 500 * KQ_NORM], dtype=torch.float64
            ).reshape(1, 2)
            self.register_parameter(
                "kq_result",
                Parameter(kq_init.repeat(batch_size, 1).to(device)),
            )
        else:
            kq_init = torch.tensor(
                eye_params_init.get("kq_result"), dtype=torch.float64
            ).reshape(-1, 2)
            self.register_parameter("kq_result", Parameter(kq_init.to(device)))

        if eye_params_init.get("pitch") is None:
            pitch_init = torch.tensor(0.1, dtype=torch.float64).reshape(1, 1)
            self.register_parameter(
                "pitch", Parameter(pitch_init.repeat(batch_size, 1).to(device))
            )
        else:
            pitch_init = torch.tensor(
                eye_params_init.get("pitch"), dtype=torch.float64
            ).reshape(-1, 1)
            self.register_parameter("pitch", Parameter(pitch_init.to(device)))

        if eye_params_init.get("yaw") is None:
            pitch_init = torch.tensor(0.1, dtype=torch.float64).reshape(1, 1)
            self.register_parameter(
                "yaw", Parameter(pitch_init.repeat(batch_size, 1).to(device))
            )
        else:
            pitch_init = torch.tensor(
                eye_params_init.get("yaw"), dtype=torch.float64
            ).reshape(-1, 1)
            self.register_parameter("yaw", Parameter(pitch_init.to(device)))

    def load_checkpoint(self, ckpt):
        for key, param in self.named_parameters():
            if key in ckpt.keys():
                param.data = ckpt[key]

    def freeze_params(self, param_list):
        for key, param in self.named_parameters():
            if key in param_list:
                param.requires_grad = False

    def active_params(self, param_list):
        for key, param in self.named_parameters():
            if key in param_list:
                param.requires_grad = True

    def forward(
        self,
    ):
        return [
            self.R,
            self.K,
            self.alpha,
            self.beta,
            self.kq_result,
            self.pitch,
            self.yaw,
        ]
