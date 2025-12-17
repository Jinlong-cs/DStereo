import kornia
import numpy as np
import torch
import torch.nn as nn

from hat.models.losses.pccr.pccr_losses import PCCRLoss
from hat.models.task_modules.PCCR_gaze.pccr import PCCRProcess
from projects.halo.cv.tools.pccr_calibration.latent import LatentCode


class Fitting(nn.Module):
    def __init__(
        self,
        lr,
        step_size,
        gamma,
        epochs,
        save_path,
        data_infos,
        loss_weights,
        eye_params_init,
        mode=None,
        batch_size=1,
        device="cuda:0",
        light_left=(-127.39130897, -2.84240908, -23.39666659),
        light_right=(97.37419513, -10.36709686, -22.68452026),
        cam_o=(0, 0, 0),
        camera_intrinsic=(
            1.8519191101732611e03,
            0.0,
            7.7160901419181175e02,
            0.0,
            1.8531620131166073e03,
            6.8977849735165626e02,
            0,
            0,
            1,
        ),
        distortion_parameter=(
            -3.2495881472030574e-01,
            6.1076092344351252e-01,
            -6.6795525414619838e-03,
            -1.7187268814147183e-04,
            -1.9507556131285542e00,
        ),
        pixel_width=1920,
        pixel_height=1080,
        scr_left_top_3d=(
            277.71341085045526,
            -317.036141472084,
            23.321810456505318,
        ),
        scr_left_bottom_3d=(
            283.57106459180403,
            4.5246415451806,
            -93.18071062266137,
        ),
        scr_right_bottom_3d=(
            -306.9330452158361,
            20.723089848946387,
            -87.24122621353507,
        ),
    ):
        super().__init__()
        self.mode = mode
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.camera_intrinsic = torch.tensor(
            camera_intrinsic, dtype=torch.float64
        ).reshape(1, 3, 3)
        self.distortion_parameter = torch.tensor(
            distortion_parameter, dtype=torch.float64
        ).reshape(1, 5)
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.epochs = epochs
        self.save_path = save_path

        self.cam_o = (
            torch.tensor(cam_o, dtype=torch.float64)
            .reshape(1, 3)
            .to(self.device)
        )
        self.light_left = (
            torch.tensor(light_left, dtype=torch.float64)
            .reshape(1, 3)
            .to(self.device)
        )
        self.light_right = (
            torch.tensor(light_right, dtype=torch.float64)
            .reshape(1, 3)
            .to(self.device)
        )
        self.pixel_width = (
            torch.tensor(pixel_width, dtype=torch.float64)
            .reshape(1, 1)
            .to(self.device)
        )
        self.pixel_height = (
            torch.tensor(pixel_height, dtype=torch.float64)
            .reshape(1, 1)
            .to(self.device)
        )
        self.scr_left_top_3d = (
            torch.tensor(scr_left_top_3d, dtype=torch.float64)
            .reshape(1, 3)
            .to(self.device)
        )
        self.scr_left_bottom_3d = (
            torch.tensor(scr_left_bottom_3d, dtype=torch.float64)
            .reshape(1, 3)
            .to(self.device)
        )
        self.scr_right_bottom_3d = (
            torch.tensor(scr_right_bottom_3d, dtype=torch.float64)
            .reshape(1, 3)
            .to(self.device)
        )

        self.params_range = {
            "R": [3, 20],
            "K": [2, 15],
            "alpha": [-10, 10],  # [-0.174, 0.175],
            "beta": [-5, 5],  # [-0.087, 0.088],
            "pitch": [-30, 30],  # [-0.7, 0.7],
            "yaw": [-20, 20],  # [-0.7, 0.7],
        }
        self.eye_params_typical = {
            "R": 7.8,
            "K": 4.75,
            "alpha": 5,  # 0.088,
            "beta": 1.5,  # 0.026,
        }

        self.loss_weights = loss_weights

        if self.mode is not None and "refitting" in self.mode:
            self.loss_weights_1 = {}
            self.loss_weights_1["in_range"] = loss_weights["in_range"]
            self.loss_weights_1["center_dis"] = loss_weights["center_dis"]
            self.loss_weights_1["eye3d_center_dis"] = loss_weights[
                "eye3d_center_dis"
            ]
            self.loss_weights_1["eye3d_allow_dis"] = loss_weights[
                "eye3d_allow_dis"
            ]

            self.loss_weights_2 = {}
            self.loss_weights_2["in_range"] = loss_weights["in_range"]
            self.loss_weights_2["pupil_dis"] = loss_weights["pupil_dis"]
            self.loss_weights_2["regularization"] = loss_weights[
                "regularization"
            ]

        else:
            self.loss_weights_1 = loss_weights
            self.loss_weights_2 = {}

        self.latent = LatentCode(self.device, batch_size, eye_params_init)
        self.pccr = PCCRProcess()
        self.pccr_loss = PCCRLoss(
            self.params_range,
            self.eye_params_typical,
            self.loss_weights_1,
            1,
        )
        self.data = self.prepare_data(data_infos)
        self.data_infos = data_infos

    def get_point_ccs(self, point):
        point_ics = kornia.geometry.calibration.undistort_points(
            point,
            self.camera_intrinsic,
            self.distortion_parameter,
            new_K=(torch.eye(3)[None, ...]),
        ).reshape(point.shape)
        homo_ones = torch.ones(
            list(point_ics.shape)[:-1] + [1],
            dtype=point_ics.dtype,
            device=point_ics.device,
        )
        point_ics = torch.cat([point_ics, homo_ones], dim=-1)
        # z=-1,norm, 像屏幕放在人的异面，跟文献中图对应起来
        point_ccs = -point_ics
        return point_ccs

    def prepare_data(self, data_infos):
        angle_list = []
        eye3d_list = []
        pog_list = []
        glint_wcs = []
        pb_wcs = []
        for tag_idx, tag_info in enumerate(data_infos):
            angle_list.extend(tag_info["left_angle"])
            eye3d_list.extend(tag_info["left_eye3d"])
            pog_list.extend(tag_info["pog_pixel"])
            glint_wcs.extend(tag_info["glint"])
            pb_wcs.extend(tag_info["pupil_boundary"])
            assert (
                len(angle_list)
                == len(eye3d_list)
                == len(pog_list)
                == len(glint_wcs)
                == len(pb_wcs)
            ), f"errors happend in idx {tag_idx} of {self.save_path}"
        assert self.batch_size == len(angle_list)

        glint_wcs_tensor = torch.tensor(glint_wcs, dtype=torch.float64)
        pb_wcs_tensor = torch.tensor(pb_wcs, dtype=torch.float64)

        pb_real_mask = torch.abs(
            pb_wcs_tensor - torch.tensor([-1, -1]).reshape(1, 1, 2)
        )
        pb_real_mask = torch.sum(pb_real_mask, dim=-1) > 1e-6
        assert pb_real_mask.sum(-1).min() > 0

        data = {
            "angle": torch.tensor(angle_list, dtype=torch.float64).to(
                self.device
            ),  # torch.randn((imgn, 2)),
            "eye3d": torch.tensor(eye3d_list, dtype=torch.float64).to(
                self.device
            ),  # torch.randn((imgn, 3)),
            "gaze_point": torch.tensor(pog_list, dtype=torch.float64).to(
                self.device
            ),  # torch.randn((imgn, 2)),
            "glint_ccs": self.get_point_ccs(glint_wcs_tensor).to(
                self.device
            ),  # torch.randn((imgn, 2, 3)),
            "pupil_boundary_ccs": self.get_point_ccs(pb_wcs_tensor).to(
                self.device
            ),  # torch.randn((imgn, pbn, 3)),
            "pb_real_mask": pb_real_mask.to(
                self.device
            ),  # torch.randn((imgn, pbn)),
        }

        return data

    def get_optimizer(self):
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": [
                        self.latent.R,
                        # self.latent.K,
                        # self.latent.alpha,
                        # self.latent.beta,
                    ],
                    "lr": self.lr[0],
                },
                {
                    "params": [
                        self.latent.kq_result,
                    ],
                    "lr": self.lr[1],
                },
                {
                    "params": [self.latent.pitch, self.latent.yaw],
                    "lr": self.lr[2],
                },
                {
                    "params": [
                        self.latent.K,
                        self.latent.alpha,
                        self.latent.beta,
                    ],
                    "lr": self.lr[3],
                },
            ]
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
        )

    def fit(self):
        if self.loss_weights["in_range"].get("R", 0) < 1e-6:
            self.latent.freeze_params("R")
            print("freeze R")
        if self.loss_weights["in_range"].get("K", 0) < 1e-6:
            self.latent.freeze_params("K")
            print("freeze K")
        if self.loss_weights["in_range"].get("alpha", 0) < 1e-6:
            self.latent.freeze_params("alpha")
            print("freeze alpha")
        if self.loss_weights["in_range"].get("beta", 0) < 1e-6:
            self.latent.freeze_params("beta")
            print("freeze beta")
        if self.loss_weights["in_range"].get("pitch", 0) < 1e-6:
            self.latent.freeze_params("pitch")
            print("freeze pitch")
        if self.loss_weights["in_range"].get("yaw", 0) < 1e-6:
            self.latent.freeze_params("yaw")
            print("freeze yaw")
        if self.loss_weights["in_range"].get("kq", 0) < 1e-6:
            self.latent.freeze_params("kq_result")
            print("freeze kq_result")

        if "refitting" in self.mode:
            print("----- refitting 2 steps ------")
            self.latent.freeze_params(["alpha", "beta", "R", "K"])
            self.latent.freeze_params(["pitch", "yaw"])

        for epoch in range(self.epochs):
            (
                R,
                K,
                alpha,
                beta,
                kq_result,
                pitch,
                yaw,
            ) = self.latent()

            in_pccr = {
                "R": R.repeat(self.batch_size, 1),
                "K": K.repeat(self.batch_size, 1),
                "alpha": alpha.repeat(self.batch_size, 1),
                "beta": beta.repeat(self.batch_size, 1),
                "pitch": pitch,
                "yaw": yaw,
                "kq_result": kq_result,
                "light_left": self.light_left.repeat(self.batch_size, 1),
                "light_right": self.light_right.repeat(self.batch_size, 1),
                "cam_o": self.cam_o.repeat(self.batch_size, 1),
                "glint_ccs_list": self.data["glint_ccs"],
                "pupil_boundary_ccs": self.data["pupil_boundary_ccs"],
                "pb_real_mask": self.data["pb_real_mask"],
                "pixel_width": self.pixel_width.repeat(self.batch_size, 1),
                "pixel_height": self.pixel_height.repeat(self.batch_size, 1),
                "scr_left_top_3d": self.scr_left_top_3d.repeat(
                    self.batch_size, 1
                ),
                "scr_left_bottom_3d": self.scr_left_bottom_3d.repeat(
                    self.batch_size, 1
                ),
                "scr_right_bottom_3d": self.scr_right_bottom_3d.repeat(
                    self.batch_size, 1
                ),
            }

            pccr_pred = self.pccr(in_pccr)
            pccr_pred["R"] = in_pccr["R"]
            pccr_pred["K"] = in_pccr["K"]
            pccr_pred["alpha"] = in_pccr["alpha"]
            pccr_pred["beta"] = in_pccr["beta"]
            pccr_pred["pitch"] = pitch
            pccr_pred["yaw"] = yaw
            pccr_pred["kq_result"] = kq_result
            in_loss = {
                "pred": pccr_pred,
                "label": self.data,
            }

            loss = self.pccr_loss(in_loss)

            for k, v in loss.items():
                if torch.isnan(v):
                    print("bad loss {}: {:.4f}".format(k, v.item()))
                    for kpa, pa in self.latent.named_parameters():
                        if pa.requires_grad:
                            print(kpa, pa.grad)

            self.optimizer.zero_grad()
            loss["total_loss"].backward()

            if "refitting" in self.mode and (
                epoch / self.epochs > 0.7
                or (
                    loss.get("center_dis", 100)
                    / self.loss_weights.get("center_dis", 1)
                )
                < 1
            ):
                self.latent.freeze_params(["kq_result"])
                self.latent.active_params(["pitch", "yaw"])
                self.pccr_loss.loss_weights = self.loss_weights_2

            self.optimizer.step()
            if epoch % 200 == 0 or (self.epochs - epoch < 2):
                print("************************************")
                print(
                    "Epoch: {:3d} / {:3d}, Lr: {:.5f},{:.5f}, Loss: {:.4f}".format(
                        epoch,
                        self.epochs,
                        self.optimizer.param_groups[0]["lr"],
                        self.optimizer.param_groups[-1]["lr"],
                        loss["total_loss"].item(),
                    )
                )
                for k, v in loss.items():
                    print("    loss {}: {:.4f}".format(k, v.item()))

                print("**************")
                print("R value: ", R[0].item())
                print("K value: ", K[0].item())
                print("alpha value: ", alpha[0].item())
                print("beta value: ", beta[0].item())
                print("kq_result value: ", kq_result.mean().item())
                print(
                    "eye3d_z value: ", self.data["eye3d"][:, -1].mean().item()
                )
                print("pitch value: ", pitch.mean().item())
                print("yaw value: ", yaw.mean().item())
                print("grad: ")
                for kpa, pa in self.latent.named_parameters():
                    if pa.requires_grad:
                        print(
                            kpa,
                            torch.abs(pa.grad).mean(),
                        )

            self.lr_scheduler.step()
        return loss

    def forward(
        self,
    ):
        self.get_optimizer()
        loss = self.fit()
        torch.save(self.latent.state_dict(), self.save_path)
        return loss, self.latent.state_dict()
