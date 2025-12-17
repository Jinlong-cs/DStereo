import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class HMRHead(nn.Module):
    """Head of hmr.

    "End-to-end Recovery of Human Shape and Pose", CVPR 2017.
    https://arxiv.org/abs/1712.06584

    Args:
        smpl_mean_params: Path of smpl mean params file.
        n_iter: Number of Iterative Error Feedback (IEF) loop. Defaults to 3.
        in_channels: Number of input featuremap channels. Defaults to 2048.
        avg_ks: Kernel size of average pooling. Defaults to 7.
    """

    def __init__(
        self,
        smpl_mean_params: str,
        n_iter: int = 3,
        in_channels: int = 2048,
        avg_ks: int = 7,
    ):
        self.inplanes = 64
        super(HMRHead, self).__init__()
        self.n_iter = n_iter
        npose = 24 * 6
        self.pose_quant = QuantStub()
        self.shape_quant = QuantStub()
        self.cam_quant = QuantStub()
        self.avgpool = nn.AvgPool2d(avg_ks, stride=1)
        self.fc1 = nn.Linear(in_channels + npose + 14, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 11)
        self.deccam = nn.Linear(1024, 3)
        self.cat_op = nn.quantized.FloatFunctional()
        self.add_op = nn.quantized.FloatFunctional()
        self.dequant = DeQuantStub()

        init_pose = torch.zeros((1, 144))
        init_shape = torch.zeros((1, 11))
        init_cam = torch.zeros((1, 3))

        if smpl_mean_params is not None:
            mean_params = np.load(smpl_mean_params)
            init_pose[0, :] = torch.from_numpy(
                mean_params["pose"][:]
            ).unsqueeze(0)
            init_shape[0, :10] = torch.from_numpy(
                mean_params["shape"][:].astype("float32")
            ).unsqueeze(0)
            init_cam[0, :] = torch.from_numpy(mean_params["cam"]).unsqueeze(0)

        self.register_buffer("init_pose", init_pose)
        self.register_buffer("init_shape", init_shape)
        self.register_buffer("init_cam", init_cam)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None):
        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        xf = self.avgpool(x)
        xf = xf.view(batch_size, -1)

        pred_pose = self.pose_quant(init_pose)
        pred_shape = self.shape_quant(init_shape)
        pred_cam = self.cam_quant(init_cam)

        for i in range(self.n_iter):  # noqa B007
            xc = self.cat_op.cat([xf, pred_pose, pred_shape, pred_cam], dim=1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)

            pred_pose = self.add_op.add(self.decpose(xc), pred_pose)
            pred_shape = self.add_op.add(self.decshape(xc), pred_shape)
            pred_cam = self.add_op.add(self.deccam(xc), pred_cam)

        pred_pose = self.dequant(pred_pose)
        pred_shape = self.dequant(pred_shape)
        pred_cam = self.dequant(pred_cam)

        return pred_pose, pred_shape, pred_cam

    def fuse_model(self):
        modules = [
            self.avgpool,
            self.fc1,
            self.drop1,
            self.fc2,
            self.drop2,
            self.decpose,
            self.decshape,
            self.deccam,
        ]

        for module in modules:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
