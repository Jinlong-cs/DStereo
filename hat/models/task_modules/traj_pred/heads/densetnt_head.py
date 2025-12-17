import random
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as HFF
from horizon_plugin_pytorch.quantization import (
    FixedScaleObserver,
    QuantStub,
    get_default_qat_qconfig,
)
from torch.quantization import DeQuantStub

from hat.models.task_modules.traj_pred.backbones.vectornet_backbone import (
    BasicGlobalGraph,
)
from hat.models.task_modules.traj_pred.backbones.vectornet_backbone import (
    SubGraphLayer as MLP,
)
from hat.models.task_modules.traj_pred.base_modules.cross_attention import (
    CrossAttention,
    CrossAttentionCut,
    qint16_freezescale_qconfig,
)
from hat.registry import OBJECT_REGISTRY

try:
    import sys

    sys.path.append("/home/users/shengzhe.dai/cnn1/DenseTNT-argoverse2/src/")
    sys.path.append("/running_package/HAT/DenseTNT-argoverse2/src/")
    sys.path.append("/running_package/code_package/DenseTNT-argoverse2/src/")
    import utils_cython
except:  # noqa
    warnings.warn(
        "Fail to load the cython optimization lib, if you do not use "
        "DenseTNT, do not care this warning."
    )
__all__ = ["DenseTNTHead"]


class DecoderResCat(nn.Module):
    """The DecoderRes structure with Cat.

    Original structure design
    """

    def __init__(self, hidden_size, in_features, out_channels=13):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(
            in_channels=in_features,
            out_channels=hidden_size,
            num_vec=1,
            use_relu=True,
            use_pool=False,
            use_layernorm=False,
            use_qint16=True,
            qconfig_func=qint16_freezescale_qconfig,
        )
        self.fc = nn.Conv2d(
            in_channels=hidden_size + in_features,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.cat_op = nn.quantized.FloatFunctional()

    def forward(self, hidden_states):
        hidden_states = self.cat_op.cat(
            [hidden_states, self.mlp(hidden_states)], dim=1
        )
        hidden_states = self.fc(hidden_states)
        return hidden_states

    def fuse_model(self):
        for module in [self.fc, self.mlp]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        self.qconfig = qint16_freezescale_qconfig()
        if hasattr(self.mlp, "set_qconfig"):
            self.mlp.set_qconfig()


class DecoderResAdd(nn.Module):
    """The DecoderRes structure with Add.

    Structure based on the DecoderResCat.
    Replace the cat operation in the decoder structure with the add operation.
    """

    def __init__(self, in_features, out_channels=13):
        super(DecoderResAdd, self).__init__()
        self.mlp = MLP(
            in_channels=in_features,
            out_channels=in_features,
            num_vec=1,
            use_relu=True,
            use_pool=False,
            use_layernorm=False,
            use_qint16=True,
            qconfig_func=qint16_freezescale_qconfig,
        )
        self.fc = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.add_op = nn.quantized.FloatFunctional()

    def forward(self, hidden_states):
        hidden_states = self.add_op.add(hidden_states, self.mlp(hidden_states))
        hidden_states = self.fc(hidden_states)
        return hidden_states

    def fuse_model(self):
        for module in [self.fc, self.mlp]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        self.qconfig = qint16_freezescale_qconfig()
        if hasattr(self.mlp, "set_qconfig"):
            self.mlp.set_qconfig()


@OBJECT_REGISTRY.register
class DenseTNTHead(nn.Module):
    """The DenseTNT Head.

    The structure is designed in the paper:
        DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets.
    """

    def __init__(
        self,
        k_values: List,
        in_channels: int = 128,
        hidden_size: int = 128,
        traj_feat_hidden_size: int = 128,
        road_feat_hidden_size: int = 128,
        traj_feat_num: int = 32,
        road_feat_num: int = 128,
        num_goals: int = 2048,
        num_dyn_obs: int = 32,
        num_road_ele: int = 128,
        num_fut_frame: int = 30,
        goal_coords_scale: float = 0.04,
        training_step: str = "float",
        use_lane_scoring: bool = False,
        use_optim_topk: bool = False,
        adjust_teacher_forcing: bool = False,
        use_set_predictor: bool = False,
        num_set_predictor_head: int = 4,
    ):
        """Initialize method.

        Args:
            k_values: list of top-k values to calculate the metric.
            in_channels: the input channels. This parameter can be
                int or Dict: 1) If the input of the head model is
                a single tensor, please set this parameter as int.
                2) If the input of the head (`data`) is a dict and
                the "real" input needs to concatenate different
                features in `data`, please set this parametr as a
                dict. The keys are the features in `data` and the
                values are the corresponding number of channels.
            hidden_size: the number of hidden units.
            traj_feat_hidden_size: the hidden_size of traj feat.
            road_feat_hidden_size: the hidden_size of road feat.
            traj_feat_num: the maximum number of traj elements.
            road_feat_num: the maximum number of road elements.
            num_goals: the number of sampled goals.
            num_dyn_obs: the max number of dynamic obstacles to
                predict.
            num_fut_frame: the number of future frames
            goal_coords_scale: the scale for input parameters
                like coordinates.
            training_step: the training step, in ["float", "qat",
                "int_infer"].
            use_lane_scoring: whether to use lane scoring during
                selecting goals.
            use_optim_topk: whether to get top k goals by the online
                optimization method instead of selecting by scores.
            adjust_teacher_forcing: whether to adjust the ratio
                of samples that use gt for supervise.
            use_set_predictor: only valid in stage-2 train, it means
                the model will generate goals by set predictor
                instead of directly use sampled goals.
            num_set_predictor_head: the number of set predictor heads.
        """
        super(DenseTNTHead, self).__init__()

        if type(in_channels) is int:
            self.in_channels = in_channels
            self.need_cat = False
        elif type(in_channels) is dict:
            self.in_channels = 0
            self.keys_to_cat = []
            for key, in_c in in_channels.items():
                self.in_channels += in_c
                self.keys_to_cat.append(key)

            self.need_cat = True
            self.feat_cat_op = nn.quantized.FloatFunctional()
        else:
            raise TypeError(
                "The param in_channels should be either int or dict."
            )

        self.k_max = max(k_values)
        self.traj_feat_hidden_size = traj_feat_hidden_size
        self.road_feat_hidden_size = road_feat_hidden_size
        self.graph_feat_hidden_size = self.in_channels
        self.num_goals = num_goals
        self.num_road_ele = num_road_ele
        self.num_dyn_obs = num_dyn_obs
        self.num_fut_frame = num_fut_frame
        self.goal_coords_scale = goal_coords_scale
        self.training_step = training_step
        self.use_lane_scoring = use_lane_scoring
        self.use_optim_topk = use_optim_topk
        self.num_k_roads = self.k_max
        self.use_set_predictor = use_set_predictor
        self.num_set_predictor_head = num_set_predictor_head
        self.adjust_teacher_forcing = adjust_teacher_forcing

        assert training_step in [
            "float",
            "calib",
            "qat",
            "int_infer",
        ], f"Undefined training step {training_step}."

        self.goal_quant = QuantStub(scale=None)
        self.end_points_quant = QuantStub(scale=None)
        self.scores_dequant = DeQuantStub()
        self.trajs_dequant = DeQuantStub()
        self.goal_dequant = DeQuantStub()
        self.need_score_quant = self.training_step == "qat" and use_optim_topk
        if self.need_score_quant:
            self.scores_quant = QuantStub(scale=None)

        # Structure of goal score head.
        # 注：下方goals_2D_mlps注释掉的部分，如果用户希望获得更小的float->
        #    qat的掉点，可以取消注释，但也会增加推理时延，不建议打开。
        self.goals_2D_mlps = nn.Sequential(
            MLP(
                in_channels=2,
                out_channels=hidden_size,
                num_vec=1,
                use_relu=True,
                use_pool=False,
                use_layernorm=False,
                # use_qint16=True,
                # qconfig_func=qint16_freezescale_qconfig,
            ),
            MLP(
                in_channels=hidden_size,
                out_channels=hidden_size,
                num_vec=1,
                use_relu=True,
                use_pool=False,
                use_layernorm=False,
                # use_qint16=True,
                # qconfig_func=qint16_freezescale_qconfig,
            ),
            MLP(
                in_channels=hidden_size,
                out_channels=hidden_size,
                num_vec=1,
                use_relu=True,
                use_pool=False,
                use_layernorm=False,
                # use_qint16=True,
                # qconfig_func=qint16_freezescale_qconfig,
            ),
        )
        assert traj_feat_hidden_size == road_feat_hidden_size, (
            "The hidden size of the trajectory feat and road feat "
            "should be the same for concatenating."
        )
        self.goals_2D_cross_attention = CrossAttentionCut(
            q_in_channels=hidden_size,
            k_in_channels=traj_feat_hidden_size,
            hidden_units=hidden_size,
            num_q=num_goals,
            num_k=traj_feat_num + road_feat_num,
            num_attention_heads=1,
            qconfig_func=qint16_freezescale_qconfig,
        )

        self.goals_2D_decoder = DecoderResAdd(
            in_features=hidden_size + self.graph_feat_hidden_size,
            out_channels=1,
        )

        self.goal_input_cat_op = nn.quantized.FloatFunctional()
        self.goal_decoder_cat_op = nn.quantized.FloatFunctional()

        self.score_head_ele = [
            "goals_2D_mlps",
            "goals_2D_cross_attention",
            "goals_2D_decoder",
        ]

        if self.use_lane_scoring:
            self.road_cross_attention = CrossAttention(
                q_in_channels=hidden_size,
                k_in_channels=traj_feat_hidden_size,
                hidden_units=hidden_size,
                num_q=num_road_ele,
                num_k=traj_feat_num,
                num_attention_heads=1,
                qconfig_func=qint16_freezescale_qconfig,
            )
            self.road_score_decoder = DecoderResCat(
                hidden_size=hidden_size,
                in_features=hidden_size * 2 + self.graph_feat_hidden_size,
                out_channels=1,
            )
            self.goals_2D_decoder_with_road = DecoderResCat(
                hidden_size=hidden_size,
                in_features=hidden_size * 3 + self.graph_feat_hidden_size,
                out_channels=1,
            )
            self.goals_road_cross_attention = CrossAttention(
                q_in_channels=hidden_size,
                k_in_channels=road_feat_hidden_size,
                hidden_units=hidden_size,
                num_q=num_goals,
                num_k=self.num_k_roads,
                num_attention_heads=1,
            )
            self.goal_road_traj_cat_op = nn.quantized.FloatFunctional()
            self.road_scores_dequant = DeQuantStub()
            self.score_head_ele += [
                "road_cross_attention",
                "road_score_decoder",
                "goals_2D_decoder_with_road",
                "goals_road_cross_attention",
            ]

        self.complete_traj_decoder = DecoderResAdd(
            in_features=hidden_size + self.graph_feat_hidden_size,
            out_channels=num_fut_frame * 2,
        )

        self.traj_input_cat_op = nn.quantized.FloatFunctional()
        self.traj_head_ele = [
            "complete_traj_decoder",
        ]

        # Structure of set predictor head.
        if self.use_set_predictor:
            # Set predictor sub network.
            self.set_feat_cat_op = nn.quantized.FloatFunctional()
            self.set_predict_point_feature = nn.Sequential(
                MLP(
                    in_channels=3,
                    out_channels=hidden_size,
                    num_vec=1,
                    use_relu=True,
                    use_pool=False,
                    use_layernorm=False,
                    use_qint16=True,
                    qconfig_func=qint16_freezescale_qconfig,
                ),
                MLP(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    num_vec=1,
                    use_relu=True,
                    use_pool=False,
                    use_layernorm=False,
                    use_qint16=True,
                    qconfig_func=qint16_freezescale_qconfig,
                ),
            )
            self.set_predict_encoders = BasicGlobalGraph(
                in_channels=hidden_size,
                hidden_units=hidden_size,
                num_poly=num_goals,
                num_attention_heads=1,
                need_scale=True,
                use_layernorm=True,
                qconfig_func=qint16_freezescale_qconfig,
            )
            # For set_predictor goal scoring.
            self.set_predict_cross_attn = CrossAttentionCut(
                q_in_channels=hidden_size,
                k_in_channels=traj_feat_hidden_size,
                hidden_units=hidden_size,
                num_q=self.k_max,
                num_k=traj_feat_num + road_feat_num,
                num_attention_heads=1,
                qconfig_func=qint16_freezescale_qconfig,
            )

            set_pred_out_c = 2 * self.k_max + 1
            self.set_predict_decoders = nn.ModuleList(
                [
                    DecoderResAdd(
                        in_features=hidden_size, out_channels=set_pred_out_c
                    )
                    for _ in range(self.num_set_predictor_head)
                ]
            )
            self.group_score_cat_op = nn.quantized.FloatFunctional()
            self.scale_mul_op = nn.quantized.FloatFunctional()
            self.inv_scale_mul_op = nn.quantized.FloatFunctional()
            self.scale_set_feat_cat_op = nn.quantized.FloatFunctional()
            self.inv_scale_quant = QuantStub(scale=None)
            self.scale_quant = QuantStub(scale=None)
            self.centric_sub_op = HFF()
            self.centric_add_op = nn.quantized.FloatFunctional()
            self.inv_centric_add_op = nn.quantized.FloatFunctional()
            self.set_head_cat_op = nn.quantized.FloatFunctional()
            self.set_head_ret_op = nn.quantized.FloatFunctional()
            self.set_head_scores_dequant = DeQuantStub()
            self.set_predictor_ele = [
                "set_predict_point_feature",
                "set_predict_encoders",
                "set_predict_cross_attn",
            ]

            self.freeze_stage_one_network()

    def forward(self, data: Dict):
        """Forward.

        Args:
            graph_feats (torch.Tensor, [num_obs, graph_hidden_size, \
                1, 1]): \
                the graph features.
            traj_feats (torch.Tensor, [num_obs, traj_feat_hidden_size, \
                1, num_dyn_obs]): \
                the traj features.
            road_feats (torch.Tensor, [num_obs, road_feat_hidden_size, \
                1, num_elements]): \
                the traj features.
            data (Dict): the model input dictionary with the following keys:
            is_int_infer_model (bool, optional):
                whether the model is for int inference.

        Returns:
            predictions: dictionary with the following keys:
                predict_goal_scores (torch.Tensor, [num_obs, 1, \
                    1, num_goals]): \
                    the predicted goal scores.
                predict_road_scores (torch.Tensor, [num_obs, 1, \
                    1, num_poly]): \
                    the predicted road scores.
                predict_trajs (torch.Tensor, [num_obs, k, \
                    num_fut_frame, 2]): \
                    the predicted trajs in validation stage.
                predict_traj_for_train (torch.Tensor, [num_obs, 1, \
                    num_fut_frame, 2]): \
                    the predicted trajs in training stage.
                predict_topk_scores (torch.Tensor, [num_obs, k]): \
                    the predicted topk goal scores.
                predict_topk_goals (torch.Tensor, [num_obs, 1, k, 2]): \
                    the predicted topk goal coordinates.
        """
        self.epoch_id = data["epoch_id"]
        if self.adjust_teacher_forcing:
            use_gt_prob = (1 - ((self.epoch_id - 0) / (29 - 0))) * (1 - 0) + 0
            random_number = random.random()
        else:
            use_gt_prob = 1
            random_number = 0
        force_to_use_gt = random_number < use_gt_prob and self.training
        if self.training_step == "int_infer":
            force_to_use_gt = False

        # Goal Head
        assert type(data) is dict
        if self.need_cat:
            graph_feats = [data[k] for k in self.keys_to_cat]
            graph_feats = self.feat_cat_op.cat(graph_feats, dim=1)
        else:
            graph_feats = data["graph_feat"]
        road_feats = data["road_feat"]
        traj_feats = data["traj_feat"]

        # (num_obs,2,1,num_goals)
        goal_coords = self.goal_quant(data["goal_coords"])

        predict_goal_scores, _ = self.real_get_scores(
            goal_coords,
            graph_feats,
            traj_feats,
            road_feats,
            for_set_predictor=False,
        )

        ret_dict = {}

        # Set predict
        traj_pred_coords, real_scores = None, None
        if self.use_set_predictor:
            (
                all_set_head_scores,
                all_set_head_pts,
                select_set_head_pts,
            ) = self.run_set_predictor(goal_coords, predict_goal_scores)
            traj_pred_coords = select_set_head_pts

            real_scores, _ = self.real_get_scores(
                select_set_head_pts,
                graph_feats,
                traj_feats,
                road_feats,
                for_set_predictor=True,
            )
        elif force_to_use_gt:
            traj_pred_coords = data["end_points"]
            real_scores = None
        else:
            traj_pred_coords = goal_coords
            real_scores = predict_goal_scores

        predict_trajs, predict_topk_goal_scores = self.real_cal_pred_trajs(
            traj_pred_coords,
            graph_feats,
            traj_feats,
            road_feats,
            real_scores,
            self.use_optim_topk,
        )

        # 注意：此时real_scores含义发生转换
        # 从输入给traj_pred模块的终点的概率值
        # 转换成traj_pred模块输出轨迹对应的概率值
        if predict_topk_goal_scores is not None:
            real_scores = predict_topk_goal_scores

        # save the parameters
        # predict_goal_scores: the scores of all end points
        # (num_obs, 1, 1, num_goals) or None
        if self.training_step != "int_infer":
            predict_goal_scores = self.scores_dequant(predict_goal_scores)
            ret_dict["predict_goal_scores"] = predict_goal_scores

        # real_scores: the scores of predicted topk trajactories
        # (num_obs, 1, 1, k_max)
        if real_scores is not None:
            real_scores = self.scores_dequant(real_scores)
            ret_dict["real_scores"] = real_scores
        else:  # force_to_use_gt = True when training
            ret_dict["real_scores"] = predict_goal_scores

        # predict_trajs: the site of predicted topk trajactories
        # (num_obs, k_max, num_fut_frams, 2)
        predict_trajs = self.trajs_dequant(predict_trajs)
        ret_dict["predict_trajs"] = predict_trajs

        # output of set_predictor: not need in int_infer
        if self.use_set_predictor and self.training_step != "int_infer":
            ret_dict["select_set_head_pts"] = self.trajs_dequant(
                select_set_head_pts
            )
            ret_dict["set_pred_goals"] = self.trajs_dequant(traj_pred_coords)
            ret_dict["set_pred_scores"] = real_scores
            ret_dict["all_set_head_scores"] = self.set_head_scores_dequant(
                all_set_head_scores
            )
            ret_dict["all_set_head_pts"] = [
                self.trajs_dequant(i) for i in all_set_head_pts
            ]

        return ret_dict

    def real_get_scores(
        self,
        goal_coords: torch.Tensor,
        graph_feats: torch.Tensor,
        traj_feats: torch.Tensor,
        road_feats: torch.Tensor,
        for_set_predictor: bool = False,
    ):
        """Calculate scores of all sampled goals.

        Args:
            goal_coords ([num_obs, 2, 1, num_sample_points]): sampled goals.
            graph_feats ([num_obs, hidden_size, 1, 1]): the feature extacted
                by the vectornet backbone.
            traj_feats ([num_obs, hidden_size, 1, num_traj]): the trajectory
                features extracted by the vectornet backbone.
            road_feats ([num_obs, hidden_size, 1, num_road]): the road
                element features extracted by the vectornet backbone.

        Returns:
            predict_goal_scores ([num_obs, 1, 1, num_sample_points]): the
                scores of all the goals.
        """
        if for_set_predictor:
            cross_attn = self.set_predict_cross_attn
            repeat_num = self.k_max
            decoder = (
                self.goals_2D_decoder
            )  # self.set_predict_goal_score_decoder
        else:
            cross_attn = self.goals_2D_cross_attention
            repeat_num = self.num_goals
            decoder = self.goals_2D_decoder

        # [num_obs, hidden_size, 1, num_sample_points]
        goals_2D_hidden = self.goals_2D_mlps(goal_coords)
        # [num_obs, hidden_size_2, 1, num_traj + num_road]
        cat_feats = self.goal_input_cat_op.cat(
            [traj_feats, road_feats], dim=-1
        )
        # [num_obs, hidden_size, 1, num_sample_points]
        goals_2D_hidden_attention = cross_attn(goals_2D_hidden, cat_feats)
        predict_road_scores = None
        if self.use_lane_scoring:
            road_hidden_attention = self.road_cross_attention(
                road_feats, traj_feats
            )
            predict_road_scores = self.road_score_decoder(
                torch.cat(
                    [
                        graph_feats.repeat(1, 1, 1, self.num_road_ele),
                        road_feats,
                        road_hidden_attention,
                    ],
                    dim=1,
                )
            )
            _, road_topk_ids = torch.topk(
                predict_road_scores, k=self.num_k_roads
            )
            road_topk_ids = road_topk_ids.repeat(
                1, self.road_feat_hidden_size, 1, 1
            )
            predict_topk_roads = torch.gather(road_feats, 3, road_topk_ids)
            goal_road_hidden_attention = self.goals_road_cross_attention(
                goals_2D_hidden, predict_topk_roads
            )
            goal_dec_input = self.goal_road_traj_cat_op.cat(
                [
                    graph_feats.repeat(1, 1, 1, self.num_goals),
                    goals_2D_hidden,
                    goals_2D_hidden_attention,
                    goal_road_hidden_attention,
                ],
                dim=1,
            )
            predict_goal_scores = self.goals_2D_decoder_with_road(
                goal_dec_input
            )
        else:
            graph_feat_expand = graph_feats.repeat(1, 1, 1, repeat_num)
            goal_dec_input = self.goal_decoder_cat_op.cat(
                [
                    graph_feat_expand,
                    goals_2D_hidden_attention,
                ],
                dim=1,
            )
            # [num_obs, 1, 1, num_sample_points]
            predict_goal_scores = decoder(goal_dec_input)
        return predict_goal_scores, predict_road_scores

    def real_cal_pred_trajs(
        self,
        goal_coords: torch.Tensor,
        graph_feats: torch.Tensor,
        traj_feats: torch.Tensor,
        road_feats: torch.Tensor,
        predict_goal_scores: Optional[torch.Tensor] = None,
        use_optim_topk: bool = False,
    ):
        """Calculate the trajectories based on the given goals.

        Args:
            goal_coords ([num_obs, 2, 1, num_sample_pts]): sampled goals.
            graph_feats ([num_obs, hidden_size, 1, 1]): the feature extacted
                by the vectornet backbone.
            traj_feats ([num_obs, hidden_size, 1, num_traj]): the trajectory
                features extracted by the vectornet backbone.
            road_feats ([num_obs, hidden_size, 1, num_road]): the road
                element features extracted by the vectornet backbone.
            predict_goal_scores ([num_obs, 1, 1, num_sample_pts]): the scores
                of all the goals. This value may be None, which means the param
                `goal_coords` is just the end point for trajectory prediction.
            use_optim_topk: whether to select top k goals by online
                optimization algorithm.

        Returns:
            predict_trajs ([num_obs, k_max, num_fut_frame, 2]): the prediction
                results.
        """
        num_topk = self.k_max

        if predict_goal_scores is None:
            # 这个分支意味着要使用gt作为input产生轨迹
            traj_head_input_goals = self.end_points_quant(goal_coords)
            traj_head_input_goals = traj_head_input_goals.repeat(
                1, 1, 1, num_topk
            )
            predict_topk_goal_scores = None
        elif self.use_set_predictor:
            # 这个分支在二阶段，不需要选top_k，省一次gather（因为只有cpu支持）
            traj_head_input_goals = goal_coords
            predict_topk_goal_scores = predict_goal_scores
        else:
            if use_optim_topk:
                if self.training_step == "qat":
                    goal_coords = goal_coords.dequantize()
                    predict_goal_scores = predict_goal_scores.dequantize()
                tmp_goal_coords = (
                    goal_coords.detach().cpu().numpy() / self.goal_coords_scale
                )
                tmp_goal_scores = predict_goal_scores.detach().cpu().numpy()
                num_obs = tmp_goal_coords.shape[0]
                batch_ans_points = np.zeros([num_obs, self.k_max, 2])
                batch_pred_probs = np.zeros([num_obs, self.k_max])
                tmp_device = goal_coords.device
                for i in range(num_obs):
                    tmp_sample_goals = tmp_goal_coords[i, :, 0, :].transpose(
                        1, 0
                    )
                    tmp_pred_goal_scores = np.exp(tmp_goal_scores[i, 0, 0, :])
                    tmp_pred_goal_scores /= np.sum(tmp_pred_goal_scores)
                    (
                        _,
                        ans_points,
                        pred_probs,
                    ) = utils_cython.get_optimal_targets(
                        tmp_sample_goals,
                        tmp_pred_goal_scores,
                        "xxx",
                        "MRminFDE",
                        0.1,
                        kwargs={
                            "cnt_sample": 9,
                            "MRratio": 0,
                            "num_step": 1000,
                        },
                    )
                    batch_ans_points[i] = ans_points
                    batch_pred_probs[i] = pred_probs
                batch_ans_points = batch_ans_points.transpose(0, 2, 1)[
                    :, :, None, :
                ]
                batch_ans_points = batch_ans_points * self.goal_coords_scale
                batch_pred_probs = batch_pred_probs[:, None, None, :]
                goal_coords = torch.FloatTensor(batch_ans_points).to(
                    tmp_device
                )
                predict_goal_scores = torch.FloatTensor(batch_pred_probs).to(
                    tmp_device
                )
                if self.training_step == "qat":
                    goal_coords = self.end_points_quant(goal_coords)
                    predict_goal_scores = self.scores_quant(
                        predict_goal_scores
                    )
            # (num_obs,1,1,k_max)
            predict_topk_goal_scores, topk_ids = torch.topk(
                predict_goal_scores, k=num_topk
            )
            # (num_obs,2,1,k_max)
            topk_ids = topk_ids.repeat(1, 2, 1, 1)
            # (num_obs,2,1,k_max)
            traj_head_input_goals = torch.gather(goal_coords, 3, topk_ids)

        # [num_obs, hidden_size, 1, k_max]
        # 这里复用了goal score head的特征提取器，因为是mlp，对于最后一个维度（前者是num_goals
        # 这里是k_max）不敏感，但不知道量化训练的时候，是否需要这里固定最后一个维度的尺寸，如果
        # 需要，那么这里不能复用goal_2D_mlps了。
        target_feature = self.goals_2D_mlps(traj_head_input_goals)
        graph_feat_expand = graph_feats.repeat(1, 1, 1, num_topk)
        # [num_obs, hidden_size_2, 1, k_max]
        hidden_extend = self.traj_input_cat_op.cat(
            # [graph_feat_expand, target_feature, hidden_attn], dim=1
            [graph_feat_expand, target_feature],
            dim=1,
        )
        # [num_obs, num_fut_frame * 2, 1, k_max]
        predict_trajs = self.complete_traj_decoder(hidden_extend)
        # [num_obs, k_max, num_fut_frame, 2]
        predict_trajs = predict_trajs.permute(0, 3, 1, 2)
        predict_trajs = predict_trajs.reshape(
            [-1, num_topk, self.num_fut_frame, 2]
        )
        return predict_trajs, predict_topk_goal_scores

    def run_set_predictor(
        self,
        goal_coords,
        predict_goal_scores,
    ):
        """Run set predictor to get candidate goals.

        Args:
            goal_coords ([num_obs, 2, 1, num_sample_points]): sampled goals.
            predict_goal_scores ([num_obs, 1, 1, num_sample_points]): the
                scores of all the goals.

        Returns:
            all_set_head_scores ([num_obs, num_head, 1, 1]): 所有set predictor
                head输出的对应head的打分
            all_set_head_pts (List of [num_obs, 2, 1, k_points]): 所有set
                predictor head的输出结果
            set_ret ([num_obs, 2, 1, num_sample_points]): padding到num_goals
                的set predictor输出, 目的是复用goal score部分的网络
        """
        # Set predict
        # (num_obs, 3, 1, num_goals)
        inv_scale = torch.tensor(
            [1 / self.goal_coords_scale],
            device=goal_coords.device,
            dtype=torch.float32,
        )
        inv_scale = self.inv_scale_quant(inv_scale)
        scale = torch.tensor(
            [self.goal_coords_scale],
            device=goal_coords.device,
            dtype=torch.float32,
        )
        scale = self.scale_quant(scale)

        max_point_idx = torch.argmax(
            predict_goal_scores, -1, keepdim=True
        ).repeat(1, 2, 1, 1)
        scaled_goal_coords = self.inv_scale_mul_op.mul(goal_coords, inv_scale)
        centric_trans_coords = torch.gather(
            scaled_goal_coords, 3, max_point_idx
        )

        scaled_goal_coords = self.centric_sub_op.sub(
            scaled_goal_coords, centric_trans_coords
        )
        set_vec_3d = self.scale_set_feat_cat_op.cat(
            [scaled_goal_coords, predict_goal_scores], dim=1
        )
        points_feature = self.set_predict_point_feature(set_vec_3d)
        set_enc = self.set_predict_encoders(points_feature)
        all_set_head_scores = []
        all_set_head_pts = []
        for _, set_decoder in enumerate(self.set_predict_decoders):
            # [num_obs, 13, 1, 1]
            tmp_set_dec = set_decoder(set_enc)
            tmp_group_score = tmp_set_dec[:, 0:1, :, :]
            tmp_traj_points = tmp_set_dec[:, 1:, :, :].reshape(
                [-1, 2, 1, self.k_max]
            )

            tmp_traj_points = self.centric_add_op.add(
                tmp_traj_points, centric_trans_coords
            )
            all_set_head_scores.append(tmp_group_score)
            all_set_head_pts.append(
                self.scale_mul_op.mul(tmp_traj_points, scale)
            )

        all_set_head_scores = self.group_score_cat_op.cat(
            all_set_head_scores, dim=1
        )
        if self.num_set_predictor_head == 1:
            select_set_head_pts = all_set_head_pts[0]
        else:
            index = (
                torch.argmax(all_set_head_scores, dim=1)
                .unsqueeze(1)
                .repeat(1, 2, self.num_set_predictor_head, self.k_max)
            )
            all_set_head_pts_cat = self.set_head_cat_op.cat(
                all_set_head_pts, dim=2
            )
            select_set_head_pts = torch.gather(all_set_head_pts_cat, 2, index)
            select_set_head_pts = select_set_head_pts[:, :, 0:1, :]

        # TODO (shengzhe.dai): 如果此处验证了要只保留一个head的输出，取消这里
        # 变量名的all_前缀。
        return (
            all_set_head_scores,
            all_set_head_pts,
            select_set_head_pts,
        )

    def freeze_stage_one_network(self):
        for key in (
            self.score_head_ele
            + self.traj_head_ele
            + ["set_predict_cross_attn"]
        ):
            if hasattr(self, key):
                element = getattr(self, key)
                for _, param in element.named_parameters():
                    param.requires_grad = False

    def freeze_stage_two_network(self):
        if not self.use_set_predictor:
            warnings.warn(
                "The user wants to freeze stage two nework, "
                "But the parameter `use_set_predictor` is False."
            )
            return
        for key in self.set_predictor_ele:
            if hasattr(self, key):
                element = getattr(self, key)
                for _, param in element.named_parameters():
                    param.requires_grad = False
        for element in self.set_predict_decoders:
            for _, param in element.named_parameters():
                param.requires_grad = False

    def fuse_model(self):
        modules = [
            self.goals_2D_cross_attention,
            self.goals_2D_decoder,
            self.complete_traj_decoder,
        ]
        seq_modules = [
            self.goals_2D_mlps,
        ]
        if self.use_set_predictor:
            modules.append(self.set_predict_encoders)
            modules.append(self.set_predict_cross_attn)
            seq_modules.append(self.set_predict_point_feature)
            seq_modules.append(self.set_predict_decoders)

        for module in modules:
            if hasattr(module, "fuse_model"):
                module.fuse_model()
        for mod in seq_modules:
            for m in mod:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        self.qconfig = qint16_freezescale_qconfig()
        QINT16_MAX = 32768.0
        self.goal_quant.qconfig = get_default_qat_qconfig(
            dtype="qint16",
        )

        self.end_points_quant.qconfig = get_default_qat_qconfig(
            dtype="qint16",
        )
        if self.need_score_quant:
            self.scores_quant.qconfig = get_default_qat_qconfig(
                dtype="qint16",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": 40.0 / QINT16_MAX,
                },
            )

        modules = [
            self.goals_2D_cross_attention,
            self.goals_2D_decoder,
            self.complete_traj_decoder,
        ]
        seq_modules = [
            self.goals_2D_mlps,
        ]
        if self.use_set_predictor:
            modules.append(self.set_predict_encoders)
            modules.append(self.set_predict_cross_attn)
            seq_modules.append(self.set_predict_point_feature)
            seq_modules.append(self.set_predict_decoders)

        for module in modules:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()
        for mod in seq_modules:
            for m in mod:
                if hasattr(m, "set_qconfig"):
                    m.set_qconfig()
