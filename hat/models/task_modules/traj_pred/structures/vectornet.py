# Copyright (c) Horizon Robotics. All rights reserved.

from collections import OrderedDict, namedtuple
from typing import Callable, Dict, Optional

import torch
from torch import nn

from hat.models.task_modules.traj_pred.structures import BasicTrajPredStructure
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = ["BasicVectorNet", "VectorNetV2"]


@OBJECT_REGISTRY.register
class BasicVectorNet(nn.Module):
    """Basic VectorNet structure implementation.

    This implementation follows the orginal paper of VectorNet. The
    used backbone should be "VectorNetBackbone", and the used head
    should be "BasicMlpDecoder".
    """

    def __init__(
        self,
        backbone,
        neck=None,
        heads=None,
        post_process=None,
        losses=None,
        itp_ratio: int = 1,
        enable_itp_gt: bool = False,
        is_int_infer_model: bool = False,
    ):
        """Initialize method.

        Args:
            backbone (dict): a dict for building backbone.
            (dict): a dict for building neck.
            heads (dict): the OrderedDict to build heads. Each element is a
                dict for building head.
            post_process (dict, optional): the post-processing model.
            losses (dict, optional): the losses. Defaults to None.
            itp_ratio (int, optional): the interpolating ratio of the ground
                truth. Defaults to 1.
            enable_itp_gt (bool, optional): whether to use interpolated ground
                truth instead of origin grounth truth. Defaults to False.
            is_int_infer_model (bool, optional): whether the model is for int
                inference. Default to False.
        """
        super(BasicVectorNet, self).__init__()
        self.itp_ratio = itp_ratio
        self.enable_itp_gt = enable_itp_gt
        self.is_int_infer_model = is_int_infer_model

        # Model structure.
        self.backbone = backbone
        self.neck = neck
        self.head_name_list = []
        if heads is not None:
            for head_name, head in heads.items():
                if head is None:
                    continue
                setattr(self, head_name, head)
                self.head_name_list.append(head_name)
        self.post_process = post_process

        # Loss.
        self.losses = None
        if losses is not None:
            self.losses = nn.ModuleList(_as_list(losses))

    def forward(self, data):
        """Forward.

        Args:
            data (Dict): the model input dictionary with the following keys:
                "struct_road_feats" (torch.Tensor, [num_obs, num_road_poly,
                    road_polyline_len, road_feat_channels]): the road polyline
                    features.
                "struct_traj_feats" (torch.Tensor, [num_obs, num_traj_poly,
                    traj_polyline_len, traj_feat_channels]): the trajectory
                    polyline features.
                "itp_fur_obs_trajs" (torch.Tensor, [num_obs, itp_traj_len, 2]):
                    the interpolated ground-truth.
                "itp_fut_obs_masks"  (torch.Tensor, [num_obs, itp_traj_len]):
                    the interpolated ground-truth masks.
                "future_trajectories" (torch.Tensor, [num_obs, ori_traj_len,
                    2]): the original ground-truth.
                "valid_masks" (torch.Tensor, [num_obs, ori_traj_len]): the
                    origin ground-truth masks.
                Note: the data dict should have at least one group trajs and
                    masks between original and interpolated results.
        """
        if not self.is_int_infer_model:
            road_input = data["struct_road_feats"].permute(0, 3, 1, 2)
            traj_input = data["struct_traj_feats"].permute(0, 3, 1, 2)
            data["struct_road_feats"] = road_input.cuda()
            data["struct_traj_feats"] = traj_input.cuda()
            road_mask = data["struct_road_masks"].cuda()
            traj_mask = data["struct_traj_masks"].cuda()
            concat_mask = torch.cat([traj_mask, road_mask], dim=1)
            attention_mask = torch.matmul(
                concat_mask[:, :, None], concat_mask[:, None, :]
            )
            attention_mask = attention_mask[:, None, :, :]
            data["attention_mask"] = attention_mask.cuda()
        feats = self.backbone(data)
        neck_feats = self.neck(feats) if self.neck else feats

        model_result = OrderedDict()
        if self.is_int_infer_model:
            for head_name in self.head_name_list:
                cur_head = getattr(self, head_name)
                cur_result = cur_head(neck_feats)
                cur_result = {
                    head_name + "_" + key: value
                    for key, value in cur_result.items()
                }
                model_result.update(cur_result)
            if self.post_process is not None:
                model_result.update(self.post_process(model_result))
            VectorNetOutput = namedtuple(
                "VectorNetOutput", model_result.keys()
            )
            model_result = VectorNetOutput(**model_result)
        else:
            for head_name in self.head_name_list:
                head_outs = OrderedDict()
                cur_head = getattr(self, head_name)
                head_ret = cur_head(neck_feats)
                head_outs.update(head_ret)
                # Choose interpolated ground-truth or down-sampled predicted
                # result in loss calculation.
                if self.enable_itp_gt:
                    head_outs["gts"] = data["itp_fut_obs_trajs"].cuda()
                    head_outs["masks"] = data["itp_fut_obs_masks"].cuda()
                else:
                    head_outs["gts"] = data["future_trajectories"].cuda()
                    head_outs["masks"] = data["valid_masks"]
                    head_outs["means"] = head_outs["means"][
                        :, :, (self.itp_ratio - 1) :: self.itp_ratio, :
                    ]
                    head_outs["scale_trils"] = head_outs["scale_trils"][
                        :, :, (self.itp_ratio - 1) :: self.itp_ratio, :, :
                    ]
                # For single-modal prediction, manually set the probabilities.
                if "log_anchors_probs" not in head_outs.keys():
                    num_obs, num_anchors = head_outs["means"].shape[:2]
                    head_outs["log_anchors_probs"] = torch.zeros(
                        [num_obs, num_anchors]
                    )

                if self.losses is not None:
                    for loss in self.losses:
                        head_outs.update(loss(head_outs))

                if self.post_process is not None:
                    head_outs.update(self.post_process(head_outs, data))

                head_results = {
                    head_name + "_" + key: value
                    for key, value in head_outs.items()
                }
                model_result.update(head_results)

                if (
                    "rendered_frames" in data
                    and data["rendered_frames"] is not None
                ):
                    model_result["rendered_frames"] = data["rendered_frames"]

        return model_result

    def fuse_model(self):
        for module in [self.backbone, self.neck]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()
        for head_name in self.head_name_list:
            head_cur = getattr(self, head_name)
            if hasattr(head_cur, "fuse_model"):
                head_cur.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        heads = [getattr(self, name) for name in self.head_name_list]
        for module in [self.backbone, self.neck] + heads:
            if module is None:
                continue
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()

        if self.losses is not None:
            for module in self.losses:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
                else:
                    module.qconfig = None


@OBJECT_REGISTRY.register
class VectorNetV2(BasicTrajPredStructure):
    """VectorNet version 2.

    This implementation follows the orginal paper of VectorNet. The
    used backbone should be "VectorNetBackbone", and the used head
    should be "BasicMlpDecoder".
    """

    def __init__(
        self,
        backbone: Callable,
        necks: Optional[Dict] = None,
        heads: Optional[Dict] = None,
        post_process: Optional[Callable] = None,
        losses: Optional[Dict] = None,
        is_int_infer_model: bool = False,
        # Belows are customized parameters.
        itp_ratio: int = 1,
        enable_itp_gt: bool = False,
    ):
        """Initialize method.

        Args:
            backbone: a dict for building backbone or callable instance.
            necks: the OrderedDict to build necks. Each element is a dict
                for building neck.
            heads: the OrderedDict to build heads. Each element is a dict
                for building head.
            post_process: the post-processing model.
            losses: the losses.
            is_int_infer_model: whether the model is for int inference.
            itp_ratio: the interpolating ratio of the ground truth.
            enable_itp_gt: whether to use interpolated ground truth instead
                of origin grounth truth.
        """
        self.itp_ratio = itp_ratio
        self.enable_itp_gt = enable_itp_gt
        kwargs = {
            "backbone": backbone,
            "necks": necks,
            "heads": heads,
            "post_process": post_process,
            "losses": losses,
            "is_int_infer_model": is_int_infer_model,
        }
        super(VectorNetV2, self).__init__(**kwargs)
        self.build_custom_structure()

    def custom_data_preprocess(self, data: Dict):  # noqa: D401
        """Customized data preprocess function.

        Args:
            data (Dict): the model input dict with the following keys: \
            "struct_road_feats" (torch.Tensor, [num_obs, num_road_poly, \
                road_polyline_len, road_feat_channels]): the road \
                polyline features. \
            "struct_traj_feats" (torch.Tensor, [num_obs, num_traj_poly, \
                traj_polyline_len, traj_feat_channels]): the trajectory \
                polyline features. \
            "itp_fur_obs_trajs" (torch.Tensor, [num_obs, itp_traj_len, 2]): \
                the interpolated ground-truth. \
            "itp_fut_obs_masks"  (torch.Tensor, [num_obs, itp_traj_len]): \
                the interpolated ground-truth masks. \
            "future_trajectories" (torch.Tensor, [num_obs, ori_traj_len, \
                2]): the original ground-truth. \
            "valid_masks" (torch.Tensor, [num_obs, ori_traj_len]): the \
                origin ground-truth masks. \
            Note: the data dict should have at least one group trajs and \
                masks between original and interpolated results. \

        Returns:
            data: the updated data dict.
            backbone_data: data for backbone input.
            gt_data: gt data for metric and postprocess.
        """
        gt_data = {}
        if not self.is_int_infer_model:
            road_input = data["struct_road_feats"].permute(0, 3, 1, 2)
            traj_input = data["struct_traj_feats"].permute(0, 3, 1, 2)
            data["struct_road_feats"] = road_input.cuda()
            data["struct_traj_feats"] = traj_input.cuda()
            road_mask = data["struct_road_masks"].cuda()
            traj_mask = data["struct_traj_masks"].cuda()
            concat_mask = torch.cat([traj_mask, road_mask], dim=1)
            attention_mask = torch.matmul(
                concat_mask[:, :, None], concat_mask[:, None, :]
            )
            attention_mask = attention_mask[:, None, :, :]
            data["attention_mask"] = attention_mask.cuda()

            # Choose interpolated ground-truth or down-sampled predicted
            # result in loss calculation.
            if self.enable_itp_gt:
                gt_data["gts"] = data["itp_fut_obs_trajs"].cuda()
                gt_data["masks"] = data["itp_fut_obs_masks"].cuda()
            else:
                gt_data["gts"] = data["future_trajectories"].cuda()
                gt_data["masks"] = data["valid_masks"]
            if "valid_track_ids" in data:
                gt_data["track_ids"] = data["valid_track_ids"]

            if "lat_behaviors" in data:
                gt_data["lat_behav_gts"] = data["lat_behaviors"]
            if "lon_behaviors" in data:
                gt_data["lon_behav_gts"] = data["lon_behaviors"]
            if "valid_behav_track_ids" in data:
                gt_data["behav_track_ids"] = data["valid_behav_track_ids"]

        return data, data, gt_data

    def custom_head_out_postprocess(self, head_outs: Dict, head_cls_name: str):
        """Customize post process function after head out.

        Args:
            head_outs: the output of the head.
            head_cls_name: the class name of the head.

        Returns:
            head_outs: the updated `head_outs`.
        """

        if head_cls_name in ["BasicMlpDecoder", "BasicAnchorBasedDecoder"]:
            if not self.is_int_infer_model:
                # Choose interpolated ground-truth or down-sampled predicted
                # result in loss calculation.
                if not self.enable_itp_gt:
                    if "means" in head_outs and "scale_trils" in head_outs:
                        head_outs["means"] = head_outs["means"][
                            :, :, (self.itp_ratio - 1) :: self.itp_ratio, :
                        ]
                        head_outs["scale_trils"] = head_outs["scale_trils"][
                            :, :, (self.itp_ratio - 1) :: self.itp_ratio, :, :
                        ]

                # For single-modal prediction, manually set the probabilities.
                if (
                    "means" in head_outs
                    and "log_anchors_probs" not in head_outs
                ):
                    num_obs, num_anchors = head_outs["means"].shape[:2]
                    head_outs["log_anchors_probs"] = torch.zeros(
                        [num_obs, num_anchors]
                    )

        return head_outs
