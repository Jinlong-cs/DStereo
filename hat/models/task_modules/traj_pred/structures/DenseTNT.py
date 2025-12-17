from collections import OrderedDict, namedtuple
from typing import Callable, Dict, Optional

import torch

from hat.models.task_modules.traj_pred.structures import BasicTrajPredStructure
from hat.registry import OBJECT_REGISTRY

__all__ = ["DenseTNT"]


@OBJECT_REGISTRY.register
class DenseTNT(BasicTrajPredStructure):
    """PyTorch-based DenseTNT implementation.

    The Backbone of DenseTNT is a vectornet. The output feature map of the
    backbone is sent to the head.
    """

    def __init__(
        self,
        backbone: Callable,
        necks: Optional[Dict] = None,
        heads: Optional[Dict] = None,
        post_process: Optional[Callable] = None,
        losses: Optional[Dict] = None,
        is_int_infer_model: bool = False,
        use_cuda: bool = True,
    ):
        """Initialize method.

        Args:
            backbone: a dict for building backbone.
            head: a dict for building head.
            neck: a dict for building neck.
            post_process: the post-processing model.
            losses: the losses class.
            is_int_infer_model: whether the model is for int inference.
            use_cuda: whether to use cuda.
        """
        kwargs = {
            "backbone": backbone,
            "necks": necks,
            "heads": heads,
            "post_process": post_process,
            "losses": losses,
            "is_int_infer_model": is_int_infer_model,
        }
        super(DenseTNT, self).__init__(**kwargs)
        self.build_custom_structure()
        self.epoch_id = 0
        self.use_cuda = use_cuda

    def get_epoch_id(self, epoch_id):
        self.epoch_id = epoch_id

    def custom_data_preprocess(self, data):
        """Customize data preprocess function.

        Args:
            data: (Dict): the model input dictionary with the following keys: \
                "struct_road_feats" (torch.Tensor, [num_obs, \
                    road_feat_channels,num_road_poly, road_polyline_len]): \
                    the road polyline features.
                "struct_traj_feats" (torch.Tensor, [num_obs, \
                    traj_feat_channels, num_traj_poly,traj_polyline_len]): \
                    the trajectory polyline features.
                "struct_road_masks" (torch.Tensor, [num_obj, num_road_poly]): \
                    Obstacle road element masks.
                "struct_traj_masks" (torch.Tensor, [num_obj, num_traj_poly]): \
                    Obstacle trajectory masks.
                "nearest_goal_idxs" (torch.Tensor, [num_obs,1,1,1]): \
                    the index of the goal nearest to end point.
                "nearest_road_idxs" (torch.Tensor, [num_obs,1,1,1]): \
                    the index of the road nearest to end point.
                "future_trajectories" (torch.Tensor, [num_obj, \
                    1,traj_len, 2]): \
                    the ground truth trajectories.
                "goal_coords" (torch.Tensor, [num_obs, 2, 1, num_goals]): \
                    the 2D coordinates of goals.
                "valid_masks" (torch.Tensor, [num_obj, traj_len]): \
                    the masks of future trajectories.
                "end_points" (torch.Tensor, [num_obj, 2,1,1]): \
                    the coordinates of future trajectories' end points.

        Returns:
            predict_topk_goals (torch.Tensor, [num_obs, 1, k, 2]): \
                the predicted topk goal coordinates.
            predict_topk_scores (torch.Tensor, [num_obs, k]): \
                the predicted topk goal scores.
            predict_traj_for_train (torch.Tensor, [num_obs, 1, \
                num_steps, 2]): \
                the predicted trajs in training stage.
            predict_trajs_for_val (torch.Tensor, [num_obs, k, \
                num_steps, 2]): \
                the predicted trajs in validation stage. \
            predict_road_scores (torch.Tensor, [num_obs, 1, 1, num_poly]): \
                the predicted road scores.
            predict_goal_scores (torch.Tensor, [num_obs, 1, 1, num_goals]): \
                the predicted goal scores.
            future_trajectories (torch.Tensor, [num_obj, traj_len, 2]): \
                the ground truth trajectories.
            ctx_trajectories (torch.Tensor, [num_obj, his_traj_len, 2]): \
                the history trajectories.
            valid_masks (torch.Tensor, [num_obj, traj_len]): \
                the masks of future trajectories.
            vis_lanes (List): the road polylines.
            goal_coords (torch.Tensor, [num_obs, 2, 1, num_goals]): \
                the 2D coordinates of goals.
            file_names (List): the obstacle's id.
            nearest_goal_idxs (torch.Tensor, [num_obs,1,1,1]): \
                the index of the goal nearest to end point.
            nearest_road_idxs (torch.Tensor, [num_obs,1,1,1]): \
                the index of the road nearest to end point.

            If self.is_int_infer_model is False, the output will also include
            some values for loss and metric calculation.

            data: the updated data dict.
            backbone_data: data for backbone input.
            gt_data: gt data for metric and postprocess.
        """
        gt_data = {}
        if not self.is_int_infer_model:
            road_mask = data["struct_road_masks"]
            traj_mask = data["struct_traj_masks"]
            concat_mask = torch.cat([traj_mask, road_mask], dim=1)
            attention_mask = torch.matmul(
                concat_mask[:, :, None], concat_mask[:, None, :]
            )
            attention_mask = attention_mask[:, None, :, :]
            data["attention_mask"] = attention_mask
            data["struct_road_feats"] = data["struct_road_feats"].permute(
                0, 3, 1, 2
            )
            data["struct_traj_feats"] = data["struct_traj_feats"].permute(
                0, 3, 1, 2
            )
            data["future_trajectories"] = data[
                "future_trajectories"
            ].unsqueeze(1)

            move_cuda_keys = [
                "struct_road_feats",
                "struct_traj_feats",
                "end_points",
                "goal_coords",
                "goal_masks",
                "valid_masks",
                "future_trajectories",
                "nearest_goal_idxs",
                "nearest_road_idxs",
                "diff_fut_trajs",
                "attention_mask",
            ]
            if self.use_cuda:
                for key in move_cuda_keys:
                    if key in data:
                        data[key] = data[key].cuda()

        return data, data, gt_data

    def custom_head_out_postprocess(self, head_outs: Dict, head_cls_name: str):
        """Customize post process function after head out.

        Args:
            head_outs: the output of the head.
            head_cls_name: the class name of the head.

        Returns:
            dec_result: the output of custom_head_out_postprocess.
        """
        dec_result = OrderedDict()
        common_result = {
            "predict_trajs": head_outs["predict_trajs"],
            "real_scores": head_outs["real_scores"],
        }
        dec_result.update(common_result)
        if not self.is_int_infer_model:
            # for vis and gt
            non_int_result = {
                "future_trajectories": head_outs[
                    "future_trajectories"
                ],  # (num_goals,1,num_fut_frams,2)
                "ctx_trajectories": head_outs[
                    "ctx_trajectories"
                ],  # (num_goals,1,num_his_frams,2)
                "goal_coords": head_outs[
                    "goal_coords"
                ],  # (num_obs, 2, 1, num_goals)
                "goal_masks": head_outs[
                    "goal_masks"
                ],  # (num_obs, 1, 1, num_goals)
                "nearest_goal_idxs": head_outs[
                    "nearest_goal_idxs"
                ],  # (num_obs, 1, 1, 1)
                "nearest_road_idxs": head_outs[
                    "nearest_road_idxs"
                ],  # (num_obs, 1, 1, 1)
                "valid_masks": head_outs[
                    "valid_masks"
                ],  # (num_obs,num_fut_frams)
                "classes": head_outs["batch_valid_class"],  # (num_obs,)
                "agent_classes": head_outs["agent_classes"],  # List[List[int]]
                "end_points": head_outs["end_points"],
            }
            # (num_obs,1,num_fut_frams,2)
            if "diff_fut_trajs" in head_outs:
                non_int_result["diff_fut_trajs"] = head_outs["diff_fut_trajs"]
            # (batch_size,3,512,512)
            if "rendered_frames" in head_outs:
                non_int_result["rendered_frames"] = head_outs[
                    "rendered_frames"
                ]
            if "vis_lanes" in head_outs:
                non_int_result["vis_lanes"] = head_outs["vis_lanes"]
            if "file_names" in head_outs:
                non_int_result["file_names"] = head_outs["file_names"]

            if "all_set_head_scores" in head_outs:
                dec_result["all_set_head_scores"] = head_outs[
                    "all_set_head_scores"
                ]
            if "all_set_head_pts" in head_outs:
                dec_result["all_set_head_pts"] = head_outs["all_set_head_pts"]
            if "set_pred_scores" in head_outs:
                dec_result["set_pred_scores"] = head_outs["set_pred_scores"]
            if "set_pred_goals" in head_outs:
                dec_result["set_pred_goals"] = head_outs["set_pred_goals"]
            if "select_set_head_pts" in head_outs:
                dec_result["select_set_head_pts"] = head_outs[
                    "select_set_head_pts"
                ]
            dec_result.update(non_int_result)
        return dec_result

    def forward(self, data):
        """Forward.

        Args:
            data: (Dict): the model input dictionary with the following keys: \
            future_trajectories (torch.Tensor, [num_obj, traj_len, 2]): \
                the ground truth trajectories.
            ctx_trajectories (torch.Tensor, [num_obj, his_traj_len, 2]): \
                the history trajectories.
            valid_masks (torch.Tensor, [num_obj, traj_len]): \
                the masks of future trajectories.
            vis_lanes (List): the road polylines. \
            goal_coords (torch.Tensor, [num_obs, 2, 1, num_goals]): \
                the 2D coordinates of goals.
            file_names (List): the obstacle's id.
            nearest_goal_idxs (torch.Tensor, [num_obs,1,1,1]): \
                the index of the goal nearest to end point.
            nearest_road_idxs (torch.Tensor, [num_obs,1,1,1]): \
                the index of the road nearest to end point.

        Returns:
            predict_topk_goals (torch.Tensor, [num_obs, 1, k, 2]): \
                the predicted topk goal coordinates.
            predict_topk_scores (torch.Tensor, [num_obs, k]): \
                the predicted topk goal scores.
            predict_traj_for_train (torch.Tensor, [num_obs, 1, \
                num_steps, 2]): \
                the predicted trajs in training stage.
            predict_trajs_for_val (torch.Tensor, [num_obs, k, \
                num_steps, 2]): \
                the predicted trajs in validation stage. \
            predict_road_scores (torch.Tensor, [num_obs, 1, 1, num_poly]): \
                the predicted road scores.
            predict_goal_scores (torch.Tensor, [num_obs, 1, 1, num_goals]): \
                the predicted goal scores.
            future_trajectories (torch.Tensor, [num_obj, traj_len, 2]): \
                the ground truth trajectories.
            ctx_trajectories (torch.Tensor, [num_obj, his_traj_len, 2]): \
                the history trajectories.
            valid_masks (torch.Tensor, [num_obj, traj_len]): \
                the masks of future trajectories.
            vis_lanes (List): the road polylines.
            goal_coords (torch.Tensor, [num_obs, 2, 1, num_goals]): \
                the 2D coordinates of goals.
            file_names (List): the obstacle's id.
            nearest_goal_idxs (torch.Tensor, [num_obs,1,1,1]): \
                the index of the goal nearest to end point.
            nearest_road_idxs (torch.Tensor, [num_obs,1,1,1]): \
                the index of the road nearest to end point.
        """
        data, backbone_data, _ = self.custom_data_preprocess(data)
        feats = self.backbone(backbone_data)
        data["feats"] = feats
        data["epoch_id"] = self.epoch_id
        neck_feats = {}
        for neck_name in self.neck_name_list:
            cur_neck = getattr(self, neck_name)
            neck_feats.update(cur_neck(data))
        if not len(neck_feats):
            graph_feat, road_feat, _, traj_feat = feats
            neck_feats = {
                "graph_feat": graph_feat,
                "road_feat": road_feat,
                "traj_feat": traj_feat,
            }
            neck_feats.update(data)

        model_result = OrderedDict()
        output_result = OrderedDict()
        model_result.update(data)
        if len(self.head_name_list) == 1:
            cur_head = getattr(self, self.head_name_list[0])
            head_output = cur_head(neck_feats)
            head_cls_name = self.head_name_list[0]
            model_result.update(head_output)
            output_result = self.custom_head_out_postprocess(
                model_result, head_cls_name
            )
        if self.post_process is not None:
            output_result.update(self.post_process(output_result))

        if self.is_int_infer_model:
            DenseTNTOutput = namedtuple("DenseTNTOutput", output_result.keys())
            output_result = DenseTNTOutput(**output_result)
        else:
            if self.losses is not None and self.training:
                for loss in self.losses:
                    cur_loss = loss(output_result)
                    if (
                        "total_loss" in output_result
                        and "total_loss" in cur_loss
                    ):
                        cur_loss["total_loss"] += output_result["total_loss"]
                    output_result.update(cur_loss)
        return output_result
