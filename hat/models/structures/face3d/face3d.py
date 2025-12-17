# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from collections import OrderedDict
from typing import Dict, Optional

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn
import torch.nn.functional as F
from horizon_plugin_pytorch.quantization import QuantStub

from hat.models.task_modules.face3d import batch_orth_proj, vertex_normals
from hat.registry import OBJECT_REGISTRY

__all__ = ["Face3dModel", "FaceMtlFace3dModel", "FaceMtlFace3DPostModule"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class Face3dModel(nn.Module):
    """The basic structure of Face3d model.

    Refer to paper `Accurate 3D Face Reconstruction with Weakly-Supervised
    Learning: From Single Image to Image Set` to get more details.

    Args:
        backbone: backbone module.
        head: face model parameter regression head.
        post_process: post processing to get face verts from coefficients.
        deploy: only return head pose during inference. Defaults to False.
        use_cliff_info: whether to use CLIFF info. Details refer to 'CLIFF:
            Carrying Location Information in Full Frames into Human Pose and
            Shape Estimation'. Defaults to False.
        use_camconv: whether to use CAMConv. Details refer to 'CAM-Convs:
            Camera-Aware Multi-Scale Convolutions for Single-View Depth'.
            Defaults to False.
    """

    DEFAULT_NUM_LDMK = 68

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        post_process: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        deploy: Optional[bool] = False,
        use_grid_sample: Optional[bool] = False,
        use_cliff_info: Optional[bool] = False,
        use_camconv: Optional[bool] = False,
    ):
        super(Face3dModel, self).__init__()
        self.use_cliff_info = use_cliff_info
        self.use_camconv = use_camconv
        self.backbone = backbone
        self.head = head
        self.post_process = post_process
        self.deploy = deploy
        self.loss = loss

        self.use_grid_sample = use_grid_sample
        if use_grid_sample:
            self.quan_img = QuantStub(scale=1 / 128)
            self.quan_grid = QuantStub()
            self.quan_posm = QuantStub()
            self.quantifunc = horizon.nn.quantized.FloatFunctional()

    def consistency_preprocess(self, data: Dict):
        """Process data for shape consistency constraints.

        Only used in training process.
        """
        if not self.training:
            return data
        _, nums, _ = data["gt_ldmk"].shape
        consistency_nums = nums // self.DEFAULT_NUM_LDMK
        if consistency_nums == 1:
            return data
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                if v.ndim == 4:
                    v_new = v.reshape(
                        v.shape[0] * consistency_nums,
                        v.shape[1] // consistency_nums,
                        v.shape[2],
                        v.shape[3],
                    )
                    if k not in ["gt_mask"]:
                        v_new = v_new.squeeze()
                elif v.ndim == 3:
                    v_new = v.reshape(
                        v.shape[0] * consistency_nums,
                        v.shape[1] // consistency_nums,
                        v.shape[2],
                    )
                    v_new = v_new.squeeze()
                else:
                    print(k, v)
                    assert 0
                    exit()
                data[k] = v_new
        return data

    def forward(self, data: Dict):
        self.consistency_preprocess(data)
        # preprocess image
        if self.use_grid_sample:
            img = data["img"]
            grid = data["grid_map"]
            position_map = data["position_map"]
            img = self.quan_img(img)
            grid = self.quan_grid(grid)
            grid = torch.permute(grid, (0, 2, 3, 1))
            position_map = self.quan_posm(position_map)
            normed_img = F.grid_sample(
                img,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            data["normed_img"] = normed_img
            image = self.quantifunc.cat((normed_img, position_map), dim=1)
        else:
            image = data["img"]

        # backbone and head forward
        if self.use_camconv:
            camconv_info = data["camconv_info"]
            image = torch.cat([image, camconv_info], 1)
        features = self.backbone(image)[-1]
        if self.use_cliff_info:
            (
                global_pose,
                jaw_pose,
                camera,
                shape,
                expression,
                texture,
                light,
            ) = self.head(features, data["cliff_info"])
        else:
            (
                global_pose,
                jaw_pose,
                camera,
                shape,
                expression,
                texture,
                light,
            ) = self.head(features)

        if self.deploy:
            return global_pose, jaw_pose, camera, shape, expression

        data["pred"] = {
            "global_pose": global_pose.reshape(-1, 3),
            "transl": camera.reshape(-1, 3),
            "jaw": jaw_pose.reshape(-1, 3),
            "shape": shape.reshape(-1, 100),
            "exp": expression.reshape(-1, 50),
            "tex": texture.reshape(-1, 50),
            "light": light.reshape(-1, 3, 9),
        }
        data["label"] = {
            "img_path": data["img_path"],
            "gt_ldmk": data.get("gt_ldmk", None),
            "gt_img": data.get("gt_img", None),
            "gt_mask": data.get("gt_mask", None),
            "gt_pose": data.get("gt_pose", None),
            "eye3d_left": data.get("eye3d_left", None),
            "eye3d_right": data.get("eye3d_right", None),
            "global_pose": data.get("global_pose", None),
            "transl": data.get("transl", None),
            "vir2real_rotmat": data.get("vir2real_rotmat", None)
            # "jaw": data.get("jaw", None),
            # "exp": data.get("exp", None),
            # "tex": data.get("tex", None),
            # "light": data.get("light", None),
            # "gt_cam_ldmk": data.get("gt_cam_ldmk", None),
            # "gt_depth": data["gt_depth"] * 1000,
            # "gt_verts": data["gt_verts"][..., :3] * 1000,
        }

        data = self.post_process(data)
        if self.training:
            loss = self.loss(data)
            return loss
        return data["pred"]

    def fuse_model(self):
        for module in [self.backbone, self.head]:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        if self.use_grid_sample:
            self.quan_grid.qconfig = (
                horizon.quantization.get_default_qat_qconfig(dtype="qint16")
            )
        for module in [self.backbone, self.head]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
        if self.loss is not None:
            self.loss.qconfig = None


@OBJECT_REGISTRY.register
class FaceMtlFace3dModel(nn.Module):
    """The structure of Face3d model for facemtl.

    Args:
        backbone: backbone module.
        head: FLAME parameter regression head.
        post_module: post process for facemtl face3d model
        deploy: only return the prediction during inference. Defaults to False.
        only_global_pose: if "true",
            only head pose will be return during inference.
            Which is valid only if deploy is true, and defaults to "true".
        loss_weights: loss weights for each branch. Defaults to None.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        post_module: Optional[nn.Module] = None,
        deploy: bool = False,
        only_global_pose: bool = True,
        loss_weights: Optional[Dict] = None,
    ):
        super(FaceMtlFace3dModel, self).__init__()
        self.backbone = backbone
        self.head = head
        self.post_module = post_module
        self.deploy = deploy
        self.only_global_pose = only_global_pose
        self.loss_weights = loss_weights

    def forward(self, data: dict):
        image = data["img"]
        outputs = OrderedDict()

        # backbone and head forward
        features = self.backbone(image)[-1]
        (
            global_pose,
            jaw_pose,
            camera,
            shape,
            expression,
            texture,
            light,
        ) = self.head(features)
        # NOTE: "gt_ldmk" is not in test_inputs. As a result,
        # in finetune stage, we save checkpoint for deploy only.
        # In pretrain stage, on the contrary, we save the whole model for
        # the finetune stage.
        if self.deploy or self.post_module is None:
            if self.only_global_pose:
                outputs.update({"pred_pose": global_pose})
            else:
                outputs.update(
                    {
                        "pred_pose": global_pose,
                        "pred_jaw_pose": jaw_pose,
                        "pred_camera": camera,
                        "pred_shape": shape,
                        "pred_expression": expression,
                    }
                )
            return outputs
        outputs.update(
            self.post_module(
                global_pose,
                jaw_pose,
                camera,
                shape,
                expression,
                texture,
                light,
                data,
                self.loss_weights,
            )
        )
        return outputs

    def fuse_model(self):
        for module in [self.backbone, self.head]:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.backbone, self.head]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()


@OBJECT_REGISTRY.register
class FaceMtlFace3DPostModule(nn.Module):
    """The post process for facemtl face3d model.

    Args:
        flame: FLAME module to get reconstructed face. Defaults to None.
        flame_tex: FLAME texture module to get albedo. Defaults to None.
        renderer: differential renderer for training. Defaults to None.
        lpips: feature extractor for perception loss. Defaults to None.
        deploy: only return  during inference. Defaults to False.
        face3d_loss_weights: loss weights for face3d. Defaults to 1.0.
    """

    def __init__(
        self,
        flame: Optional[nn.Module] = None,
        flame_tex: Optional[nn.Module] = None,
        renderer: Optional[nn.Module] = None,
        lpips: Optional[nn.Module] = None,
        face3d_loss_weight: float = 1.0,
    ):
        super(FaceMtlFace3DPostModule, self).__init__()
        self.flame = flame
        self.flame_tex = flame_tex
        self.renderer = renderer
        self.lpips = lpips
        self.face3d_loss_weight = face3d_loss_weight
        if self.flame is not None:
            for param in self.flame.parameters():
                param.requires_grad = False
        if self.flame_tex is not None:
            for param in self.flame_tex.parameters():
                param.requires_grad = False
        if self.lpips is not None:
            for param in self.lpips.parameters():
                param.requires_grad = False

    def forward(
        self,
        global_pose,
        jaw_pose,
        camera,
        shape,
        expression,
        texture,
        light,
        data,
        loss_weights,
    ):
        if loss_weights is None:
            loss_weights = {}
        # loss weight
        self.ldmk_weight = loss_weights.get("ldmk", 256.0)
        self.photo_weight = loss_weights.get("photo", 0.0)
        self.lpips_weight = loss_weights.get("lpips", 0.0)
        # regularization
        self.shape_reg_weight = loss_weights.get("shape_reg", 1e-3)
        self.exp_reg_weight = loss_weights.get("exp_reg", 1e-3)
        self.tex_reg_weight = loss_weights.get("tex_reg", 1e-3)

        target_keypoint = data.get("gt_ldmk", None)
        target_img = data.get("gt_img", None)
        target_mask = data.get("gt_mask", None)

        outputs = OrderedDict()
        global_pose = global_pose.reshape(-1, 3)
        jaw_pose = jaw_pose.reshape(-1, 3)
        camera = camera.reshape(-1, 3)
        shape = shape.reshape(-1, 100)
        expression = expression.reshape(-1, 50)
        texture = texture.reshape(-1, 50)
        light = light.reshape(-1, 3, 9)
        # flame forward
        # full_pose = head_pose + neck_pose + jaw_pose + left/right eyes
        zeros = torch.zeros_like(global_pose)
        full_pose = torch.cat(
            (global_pose, zeros, jaw_pose, zeros, zeros), dim=1
        )
        vertices, landmarks2d, landmarks3d = self.flame(
            full_pose=full_pose,
            shape_params=shape,
            expression_params=expression,
            pose2rot=True,
        )
        # vertices to normals
        normal = vertex_normals(
            vertices,
            self.flame.faces_tensor.unsqueeze(0).repeat(camera.size(0), 1, 1),
        )
        # weak perspective projection
        tf_landmarks3d = batch_orth_proj(landmarks3d, camera)
        tf_vertices = batch_orth_proj(vertices, camera)
        tf_landmarks3d[..., 1:] *= -1
        tf_vertices[..., 1:] *= -1

        # landmark loss
        loss_ldmk = self.ldmk_weight * F.l1_loss(
            tf_landmarks3d[..., :2] * target_keypoint[..., 2:],
            target_keypoint[..., :2] * target_keypoint[..., 2:],
        )
        outputs.update({"loss_ldmk": self.face3d_loss_weight * loss_ldmk})
        # image loss
        if self.renderer is not None:
            albedo = self.flame_tex(texture)
            # renderer forward
            predicted_img, predicted_mask = self.renderer(
                tf_vertices, normal, albedo, light
            )
            # bhwc to bchw
            predicted_img = (
                predicted_img.permute(0, 3, 1, 2).clamp(0, 1).contiguous()
            )
            predicted_mask = predicted_mask.permute(0, 3, 1, 2).contiguous()
            # blend result
            mask = predicted_mask * target_mask
            predicted_img = predicted_img * mask + target_img * (1 - mask)

            # photometric loss
            loss_photo = self.photo_weight * F.l1_loss(
                predicted_img, target_img
            )
            outputs.update(
                {"loss_photo": self.face3d_loss_weight * loss_photo}
            )
            # perception loss
            # https://github.com/richzhang/PerceptualSimilarity
            loss_lpips = (
                self.lpips_weight
                * self.lpips(predicted_img, target_img, normalize=True).mean()
            )
            outputs.update(
                {"loss_lpips": self.face3d_loss_weight * loss_lpips}
            )
        # regularization
        loss_shape = torch.norm(shape, dim=1).mean() * self.shape_reg_weight
        loss_exp = torch.norm(expression, dim=1).mean() * self.exp_reg_weight
        loss_tex = torch.norm(texture, dim=1).mean() * self.tex_reg_weight
        outputs.update({"loss_shape": self.face3d_loss_weight * loss_shape})
        outputs.update({"loss_exp": self.face3d_loss_weight * loss_exp})
        outputs.update({"loss_tex": self.face3d_loss_weight * loss_tex})
        total_loss = 0
        for _, val in outputs.items():
            total_loss += val
        outputs.update({"total": total_loss})
        return outputs

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        if self.lpips is not None:
            self.lpips.qconfig = None
        if self.flame is not None:
            self.flame.qconfig = None
        if self.flame_tex is not None:
            self.flame_tex.qconfig = None
        if self.renderer is not None:
            self.renderer.qconfig = None
