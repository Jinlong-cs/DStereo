from typing import Dict, List

# from hat.models.task_modules.traj_pred.backbones.vectornet_backbone import (
#     SubGraphLayer as MLP,
# )
import torch.nn as nn
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from torch.quantization import DeQuantStub

from hat.models.task_modules.traj_pred.base_modules.linear import MLP
from hat.models.task_modules.traj_pred.base_modules.traj_qconfig import (
    qint16_qconfig,
)
from hat.registry import OBJECT_REGISTRY

__all__ = ["DenseTNTPlusHead"]


@OBJECT_REGISTRY.register
class DenseTNTPlusHead(nn.Module):
    """The DenseTNT Head.

    The structure is designed in the paper:
        DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets.
    """

    def __init__(
        self,
        k_values: List,
        in_channels: int = 128,
        hidden_size: int = 128,
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
        """
        super(DenseTNTPlusHead, self).__init__()

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
        self.graph_feat_hidden_size = self.in_channels

        self.scores_dequant = DeQuantStub()
        self.trajs_dequant = DeQuantStub()

        self.endpts_hidden_size = hidden_size
        self.endpoint_predictor = MLP(
            input_dim=self.graph_feat_hidden_size,
            output_dim=5 * 2,
            H_dim=self.k_max,
            W_dim=1,
            p_drop=0.0,
            hidden_dim=self.endpts_hidden_size,
            residual=True,
        )
        self.endpoint_cat1 = FF()
        self.endpoint_refiner = MLP(
            input_dim=self.graph_feat_hidden_size + 2,
            output_dim=2,
            H_dim=self.k_max,
            W_dim=1,
            p_drop=0.0,
            hidden_dim=self.endpts_hidden_size,
            residual=True,
        )
        self.endpoint_add = FF()
        self.traj_input_cat = FF()
        self.trajs_decoder = MLP(
            input_dim=self.graph_feat_hidden_size + 2,
            output_dim=11 * 2,
            H_dim=self.k_max,
            W_dim=1,
            p_drop=0.0,
            hidden_dim=self.endpts_hidden_size,
            residual=True,
        )
        self.scores_decoder = MLP(
            input_dim=self.graph_feat_hidden_size + 2,
            output_dim=1,
            H_dim=self.k_max,
            W_dim=1,
            p_drop=0.0,
            hidden_dim=self.endpts_hidden_size,
            residual=True,
        )
        self.traj_endpts_cat = FF()

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
                real_scores (torch.Tensor, [num_obs, k, \
                    1, 1]): \
                    the predicted scores.
                predict_trajs (torch.Tensor, [num_obs, k, \
                    num_fut_frame, 2]): \
                    the predicted trajs.
        """
        assert type(data) is dict
        ret_dict = {}

        if self.need_cat:
            graph_feats = [data[k] for k in self.keys_to_cat]
            graph_feats = self.feat_cat_op.cat(
                graph_feats, dim=1
            )  # [bs, 128+3, 1, 1] NCHW
        else:
            graph_feats = data["graph_feat"]

        endpoints = self.endpoint_predictor(
            graph_feats  # [bs, 128+3, 1, 1]
        ).view(
            -1, 2, self.k_max, 1
        )  # [bs, 5*2, 1, 1]--> [bs, 2, 5, 1]

        graph_feats_expand = graph_feats.repeat(
            1, 1, self.k_max, 1
        )  # [bs, 128+3, 1, 1]--> [bs, 128+3, 5, 1]

        refine_embed = self.endpoint_cat1.cat(
            [graph_feats_expand, endpoints.detach()], dim=1
        )  # [bs, 128+3+2, 5, 1]

        offsets = self.endpoint_refiner(refine_embed)  # [bs, 2, 5, 1]

        endpoints = self.endpoint_add.add(endpoints, offsets)  # [bs, 2, 5, 1]

        traj_embed = self.traj_input_cat.cat(
            [graph_feats_expand, endpoints.detach()],
            dim=1,
        )  # [bs, 128+3+2, 5, 1]

        predict_trajs = self.trajs_decoder(traj_embed)  # [bs, 11*2, 5, 1]
        predict_trajs = predict_trajs.permute(
            0, 2, 3, 1  # [bs, 11*2, 5, 1] --> [bs, 5, 1, 11*2]
        ).reshape(
            -1, self.k_max, 11, 2  # [bs, 5, 1, 11*2]-->[bs, 5, 11, 2]
        )

        predict_trajs = self.traj_endpts_cat.cat(
            [predict_trajs, endpoints.permute(0, 2, 3, 1)],
            dim=2,
        )  # [bs, 5, 12, 2]

        # [bs, 1, 5, 1] -->  [bs, 5, 1, 1]
        predict_scores = self.scores_decoder(traj_embed).permute(0, 2, 3, 1)
        predict_scores = self.scores_dequant(predict_scores)
        predict_trajs = self.trajs_dequant(predict_trajs)
        # save results
        ret_dict["real_scores"] = predict_scores
        ret_dict["predict_trajs"] = predict_trajs

        return ret_dict

    def set_qconfig(self):
        self.qconfig = qint16_qconfig()
        modules = [
            self.endpoint_predictor,
            self.endpoint_refiner,
            self.trajs_decoder,
            self.scores_decoder,
        ]
        for module in modules:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()
