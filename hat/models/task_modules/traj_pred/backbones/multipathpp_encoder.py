# Copyright (c) Horizon Robotics. All rights reserved.
#
# Implementation of encoder of Multipath++ model.
# https://arxiv.org/abs/2111.14973
#
# For any implementation details please refer to developer guide:
# https://horizonrobotics.feishu.cn/docs/doccnBr8XXeUQG3MeNHAr0tcZxb
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from hat.models.task_modules.traj_pred.structures.multi_context_gating import (
    MultiContextGating,
)
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class MTPPlusEncoder(nn.Module):
    def __init__(
        self,
        use_LSTM: bool,
        target_pos_vel_split: int,
        target_agent_feat_size: int,
        target_agent_emb_size: int,
        target_agent_mcg_layers: int,
        target_agent_mcg_hidden_size: int,
        position_time_size: int,
        nbr_feat_size: int,
        nbr_emb_size: int,
        nbr_agent_mcg_layers: int,
        nbr_agent_mcg_hidden_size: int,
        nbr_enc_size: int,
        custom_lane_encode: bool,
        node_feat_size: int,
        node_emb_size: int,
        node_enc_size: int,
        lane_only: bool,
        road_agent_mcg_layers: int,
        road_agent_mcg_hidden_size: int,
        max_node_count: Optional[int] = None,
    ):
        """GRU based encoder from PGP.

        Lane node features and agent histories encoded using GRUs.
        Additionally, agent-node attention layers infuse each node encoding
        with nearby agent context.
        Finally, GAT layers aggregate local context at each node.

        Args:
            use_LSTM: whether to use LSTM to sequence encoding. If not,
                will use GRU instead.

            target_pos_vel_split: whether to use separate encoding layers
                for target agent's position and velocity.
            target_agent_emb_size: target agent embedding size. Will be
                the output size of LSTM/GRU.
            target_agent_mcg_layers: MCG layers used for target agent
                encoding.
            target_agent_mcg_hidden_size: hidden layer size of MCG layers
                used for target agent encoding.

            position_time_size: sptial-temporal mixed feature size. Used
                in defining MultiContextGating module.

            nbr_feat_size: neighboring agent feature size.
            nbr_emb_size: neighboring agent embedding size. Will be the
                output size of LSTM/GRU.
            nbr_agent_mcg_layers: MCG layers used for neighboring agent
                encoding.
            nbr_agent_mcg_hidden_size: hidden layer size of MCG layers
                used for neighboring agent encoding.
            nbr_enc_size: neighboring agent encode size.

            custom_lane_encode: whether to use custom lane encode.

            node_feat_size: lane node feature size.
            node_emb_size: lane node embedding size.
            node_enc_size: lane node encoding size.

            lane_only: whether the input maps contains lane only.
            road_agent_mcg_layers: number of MCG layers for road agents.
            road_agent_mcg_hidden_size: hidden size of MCG layeres for
                road agents.
            max_node_count: max number of lane nodes. Defaults to None, which
                uses all lane nodes.
        """
        super().__init__()

        self.use_LSTM = use_LSTM
        self.custom_lane_encode = custom_lane_encode
        self.pos_vel_split = target_pos_vel_split

        # Target agent encoder
        if self.use_LSTM:
            self.target_agent_history_enc = nn.LSTM(
                target_agent_feat_size,
                target_agent_emb_size,
                batch_first=True,
            )
        else:
            # Use splited two GRU to encode history position and speed
            if self.pos_vel_split:
                self.target_agent_pos_enc = nn.GRU(
                    target_agent_feat_size // 2,
                    target_agent_emb_size // 2,
                    batch_first=True,
                )
                self.target_agent_vel_enc = nn.GRU(
                    target_agent_feat_size // 2,
                    target_agent_emb_size // 2,
                    batch_first=True,
                )
            else:
                self.target_agent_history_enc = nn.GRU(
                    target_agent_feat_size,
                    target_agent_emb_size,
                    batch_first=True,
                )
        self.agent_feat_size = target_agent_feat_size

        self.target_agent_MCG = MultiContextGating(
            target_agent_mcg_layers,
            position_time_size,
            position_time_size,
            target_agent_mcg_hidden_size,
        )

        # Surrounding agent encoder
        self.nbr_emb = nn.Linear(nbr_feat_size + 1, nbr_emb_size)
        if self.use_LSTM:
            self.nbr_enc = nn.LSTM(
                nbr_emb_size, nbr_enc_size, batch_first=True
            )
        else:
            self.nbr_enc = nn.GRU(nbr_emb_size, nbr_enc_size, batch_first=True)

        self.neighbor_agent_MCG = MultiContextGating(
            nbr_agent_mcg_layers,
            nbr_enc_size,
            target_agent_mcg_hidden_size + target_agent_emb_size,
            nbr_agent_mcg_hidden_size,
        )

        # Node encoders
        self.lane_only = lane_only
        self.max_node_count = max_node_count
        if self.custom_lane_encode:
            self.node_emb = nn.Linear(node_feat_size, node_enc_size)
        else:
            self.node_emb = nn.Linear(node_feat_size, node_emb_size)
            self.node_encoder = nn.GRU(
                node_emb_size, node_enc_size, batch_first=True
            )

        self.road_agent_MCG = MultiContextGating(
            road_agent_mcg_layers,
            node_enc_size,
            target_agent_mcg_hidden_size
            + target_agent_emb_size
            + nbr_agent_mcg_hidden_size,
            road_agent_mcg_hidden_size,
        )

        # Non-linearities
        self.leaky_relu = nn.LeakyReLU()

    def target_agent_history_encode(
        self, agent_feature: torch.Tensor, agent_feat_size: int
    ) -> torch.Tensor:
        """Encode target agent's history trajectory.

        Args:
            agent_feature: [B, t_h, target_agent_feat_size] tensor.
            agent_feat_size: first this number of features will be used during
                the encoding.

        Returns:
            torch.Tensor: _description_
        """
        device = agent_feature.device
        if self.use_LSTM:
            _, (target_hist_emb, _) = self.target_agent_history_enc(
                agent_feature[:, :, :agent_feat_size]
            )
        else:
            if self.pos_vel_split:
                _, target_pos_emb = self.target_agent_pos_enc(
                    agent_feature[:, :, :2]
                )
                _, target_vel_emb = self.target_agent_vel_enc(
                    agent_feature[:, :, 2:4]
                )
            else:
                _, target_hist_emb = self.target_agent_history_enc(
                    agent_feature[:, :, :agent_feat_size]
                )
                target_hist_emb = target_hist_emb.squeeze(0)
        # time_offset is hard-coded to cope with WAYMO data, which contains
        # 10 history steps and 1 current steps, thus 11 timesteps.
        time_offset = (
            torch.linspace(-1, 0, 11)
            .unsqueeze(0)
            .repeat(agent_feature.shape[0], 1)
            .to(device)
        )
        position_time = torch.cat(
            (agent_feature[:, :, :2], time_offset.unsqueeze(2)), dim=-1
        )
        _, mcg_context = self.target_agent_MCG(position_time, None)
        if self.pos_vel_split:
            return torch.cat(
                (
                    target_pos_emb.squeeze(0),
                    target_vel_emb.squeeze(0),
                    mcg_context,
                ),
                dim=-1,
            )
        else:
            return torch.cat((target_hist_emb, mcg_context), dim=-1)

    def forward(self, inputs: Dict) -> Dict:
        """Forward module.

        Abbreviation definition:
            B: batch size
            MN: max number of nodes.
            MP: max number of poses.
            MV: max number of vehicles.
            MPED: max number of pedestrians.
            t_h: time horizon.

        Args:
            inputs: dictionary with the following structure:
                - target_agent_representation: [B, t_h, target_agent_feat_size]
                - map_representation:
                    - lane_node_feats: [B, MN, MP, node_feat_size]
                    - lane_node_masks: [B, MN, MP, node_feat_size]
                    - (optional) s_next: Edge look-up table pointing to
                      destination node from source node.
                    - (optional) edge_type: Look-up table with edge type.
                - surrounding_agent_representation:
                    - vehicles: [B, MV, t_h, nbr_feat_size]
                    - vehicle_masks: [B, MV, t_h, nbr_feat_size]
                    - pedestrians: [B, MPED, t_h, nbr_feat_size]
                    - pedestrian_masks: [B, MPED, t_h, nbr_feat_size]
                - agent_node_masks:
                    - vehicles: [B, MN, MV]
                    - pedestrians: [B, MN, MPED]
                - (optional) init_node: initial node in the lane graph based
                    on track history.
                - (optional) node_seq_gt: ground truth node sequence for
                    pre-training.
        """
        # Encode target agent
        target_agent_feats = inputs["target_agent_representation"]
        device = target_agent_feats.device
        target_agent_mcg_context = self.target_agent_history_encode(
            target_agent_feats, self.agent_feat_size
        )

        # Encode surrounding agents
        nbr_vehicle_feats = inputs["surrounding_agent_representation"][
            "vehicles"
        ]
        nbr_vehicle_feats = torch.cat(
            (
                nbr_vehicle_feats[:, :, :, : self.agent_feat_size],
                torch.zeros_like(nbr_vehicle_feats[:, :, :, 0:1]).to(device),
            ),
            dim=-1,
        )
        nbr_vehicle_masks = inputs["surrounding_agent_representation"][
            "vehicle_masks"
        ]
        nbr_vehicle_embedding = self.leaky_relu(
            self.nbr_emb(nbr_vehicle_feats)
        )
        nbr_vehicle_enc = self.variable_size_gru_encode(
            nbr_vehicle_embedding,
            nbr_vehicle_masks,
            self.nbr_enc,
            self.use_LSTM,
        )

        nbr_ped_feats = inputs["surrounding_agent_representation"][
            "pedestrians"
        ]
        nbr_ped_feats = torch.cat(
            (
                nbr_ped_feats[:, :, :, : self.agent_feat_size],
                torch.ones_like(nbr_ped_feats[:, :, :, 0:1]).to(device),
            ),
            dim=-1,
        )
        nbr_ped_masks = inputs["surrounding_agent_representation"][
            "pedestrian_masks"
        ]
        nbr_ped_embedding = self.leaky_relu(self.nbr_emb(nbr_ped_feats))
        nbr_ped_enc = self.variable_size_gru_encode(
            nbr_ped_embedding, nbr_ped_masks, self.nbr_enc, self.use_LSTM
        )

        nbr_cyc_feats = inputs["surrounding_agent_representation"]["cyclists"]
        nbr_cyc_feats = torch.cat(
            (
                nbr_cyc_feats[:, :, :, : self.agent_feat_size],
                torch.ones_like(nbr_cyc_feats[:, :, :, 0:1]).to(device),
            ),
            dim=-1,
        )
        nbr_cyc_masks = inputs["surrounding_agent_representation"][
            "cyclist_masks"
        ]
        nbr_cyc_embedding = self.leaky_relu(self.nbr_emb(nbr_cyc_feats))
        nbr_cyc_enc = self.variable_size_gru_encode(
            nbr_cyc_embedding, nbr_cyc_masks, self.nbr_enc, self.use_LSTM
        )

        # Agent-node attention
        nbr_encodings = torch.cat(
            (nbr_vehicle_enc, nbr_ped_enc, nbr_cyc_enc), dim=1
        )
        _, nbr_mcg_context = self.neighbor_agent_MCG(
            nbr_encodings, target_agent_mcg_context
        )
        agent_enc = torch.cat(
            (target_agent_mcg_context, nbr_mcg_context), dim=1
        )

        # Encode lane nodes
        if self.max_node_count is not None and not self.lane_only:
            lane_node_feats = inputs["map_representation"]["lane_node_feats"][
                :, : self.max_node_count, :, :
            ]
            lane_node_masks = inputs["map_representation"]["lane_node_masks"][
                :, : self.max_node_count, :, :
            ]
        else:
            lane_node_feats = inputs["map_representation"]["lane_node_feats"]
            lane_node_masks = inputs["map_representation"]["lane_node_masks"]

        if self.custom_lane_encode:
            lane_node_embedding = self.road_network_encode(
                lane_node_feats, lane_node_masks, self.lane_only
            )
            lane_node_enc = self.leaky_relu(self.node_emb(lane_node_embedding))
        else:
            lane_node_embedding = self.leaky_relu(
                self.node_emb(lane_node_feats)
            )
            lane_node_enc = self.variable_size_gru_encode(
                lane_node_embedding, lane_node_masks, self.node_encoder
            )

        _, road_network_context = self.road_agent_MCG(lane_node_enc, agent_enc)
        context_encoding = torch.cat(
            (target_agent_mcg_context, nbr_mcg_context, road_network_context),
            dim=1,
        )

        return context_encoding

    @staticmethod
    def road_network_encode(
        lane_node_feats: torch.Tensor,
        lane_node_masks: torch.Tensor,
        lane_only: bool = False,
    ) -> torch.Tensor:
        """Return road network encoding in reference of Multipath++.

        Args:
            lane_node_feats: [B, MN, MP, node_feat_size] tensor.
            lane_node_masks: [B, MN, MP, node_feat_size] tensor.
            lane_only: whether the input maps contain lane only.
        """
        # Initialize device
        device = lane_node_feats.device
        with torch.no_grad():
            # filter invalid pose by adding 1000 to distance
            lane_pose_distance = (
                torch.sqrt(
                    lane_node_feats[:, :, :, 0] ** 2
                    + lane_node_feats[:, :, :, 1] ** 2
                )
                + lane_node_masks[:, :, :, 0] * 1e3
            )
            closest_lane_pose_index = (
                torch.argmin(lane_pose_distance, dim=-1)
                .unsqueeze(dim=2)
                .type(torch.int64)
            )
            batch_size = lane_node_feats.shape[0]
            max_num = lane_node_feats.shape[1]
            if lane_only:
                road_feats = torch.zeros((batch_size, max_num, 12)).to(device)
            else:
                road_feats = torch.zeros((batch_size, max_num, 28)).to(device)

            # r
            lane_closest_pose_x = torch.gather(
                lane_node_feats[:, :, :, 0], 2, closest_lane_pose_index
            ).squeeze(dim=2)
            lane_closest_pose_y = torch.gather(
                lane_node_feats[:, :, :, 1], 2, closest_lane_pose_index
            ).squeeze(dim=2)
            road_feats[:, :, 0] = torch.clip(
                torch.gather(
                    lane_pose_distance, 2, closest_lane_pose_index
                ).squeeze(dim=2),
                min=1e-3,
            )
            road_feats[:, :, 1] = lane_closest_pose_x / road_feats[:, :, 0]
            road_feats[:, :, 2] = lane_closest_pose_y / road_feats[:, :, 0]

            # b - a
            last_valid_index = (
                torch.sum(1 - lane_node_masks[:, :, :, 0], dim=-1) - 1
            )
            last_valid_index = torch.clip(
                last_valid_index.unsqueeze(dim=2).type(torch.int64), min=0
            )
            lane_start_pose_x = lane_node_feats[:, :, 0, 0].squeeze()
            lane_start_pose_y = lane_node_feats[:, :, 0, 1].squeeze()
            lane_end_pose_x = torch.gather(
                lane_node_feats[:, :, :, 0], 2, last_valid_index
            ).squeeze(dim=2)
            lane_end_pose_y = torch.gather(
                lane_node_feats[:, :, :, 1], 2, last_valid_index
            ).squeeze(dim=2)
            road_feats[:, :, 5] = torch.clip(
                torch.sqrt(
                    (lane_end_pose_x - lane_start_pose_x) ** 2
                    + (lane_end_pose_y - lane_start_pose_y) ** 2
                ),
                min=1e-3,
            )
            road_feats[:, :, 3] = (
                lane_end_pose_x - lane_start_pose_x
            ) / road_feats[:, :, 5]
            road_feats[:, :, 4] = (
                lane_end_pose_y - lane_start_pose_y
            ) / road_feats[:, :, 5]

            # b - r
            road_feats[:, :, 6] = torch.sqrt(
                (lane_end_pose_x - lane_closest_pose_x) ** 2
                + (lane_end_pose_y - lane_closest_pose_y) ** 2
            )

            # a tangent vector
            road_feats[:, :, 7] = torch.cos(
                lane_node_feats[:, :, 0, 2].squeeze()
            )
            road_feats[:, :, 8] = torch.sin(
                lane_node_feats[:, :, 0, 2].squeeze()
            )

            # one hot encoding
            if lane_only:
                road_feats[:, :, 9] = torch.gather(
                    lane_node_feats[:, :, :, 3], 2, closest_lane_pose_index
                ).squeeze(dim=2)
                road_feats[:, :, 10] = torch.gather(
                    lane_node_feats[:, :, :, 4], 2, closest_lane_pose_index
                ).squeeze(dim=2)
                road_feats[:, :, 11] = torch.gather(
                    lane_node_feats[:, :, :, 5], 2, closest_lane_pose_index
                ).squeeze(dim=2)
            else:
                road_feats[:, :, 9:28] = lane_node_feats[:, :, 0, 3:22]

            m = torch.isnan(road_feats)
            if torch.any(m):
                road_feats[m] = 0
                print("Found nan in road_feats: ", m.sum())

            return road_feats

    @staticmethod
    def variable_size_gru_encode(
        feat_embedding: torch.Tensor,
        masks: torch.Tensor,
        gru: nn.GRU,
        use_LSTM=False,
    ) -> torch.Tensor:
        """Return GRU encoding for a batch of inputs.

        Each sample in the batch is a set of a variable number
        of sequences, of variable lengths.

        Args:
            feat_embedding: a feature embedding tensor.
            masks: a mask tensor indicating which values are valid. See
                descriptions in forward function for more details.
            gru: an LSTM or a GRU module.
            use_LSTM: whether the used module is an LSTM module.
                Defaults to False.
        """
        # Initialize device
        device = feat_embedding.device

        # Form a large batch of all sequences in the batch
        masks_for_batching = ~masks[:, :, :, 0].bool()
        masks_for_batching = (
            masks_for_batching.any(dim=-1).unsqueeze(2).unsqueeze(3)
        )
        feat_embedding_batched = torch.masked_select(
            feat_embedding, masks_for_batching
        )
        feat_embedding_batched = feat_embedding_batched.view(
            -1, feat_embedding.shape[2], feat_embedding.shape[3]
        )

        # Pack padded sequences
        seq_lens = torch.sum(1 - masks[:, :, :, 0], dim=-1)
        seq_lens_batched = seq_lens[seq_lens != 0].cpu()
        if len(seq_lens_batched) != 0:
            feat_embedding_packed = pack_padded_sequence(
                feat_embedding_batched,
                seq_lens_batched,
                batch_first=True,
                enforce_sorted=False,
            )

            # Encode
            if use_LSTM:
                _, (encoding_batched, _) = gru(feat_embedding_packed)
            else:
                _, encoding_batched = gru(feat_embedding_packed)
            encoding_batched = encoding_batched.squeeze(0)

            # Scatter back to appropriate batch index
            masks_for_scattering = masks_for_batching.squeeze(3).repeat(
                1, 1, encoding_batched.shape[-1]
            )
            encoding = (
                torch.zeros(masks_for_scattering.shape)
                .to(device)
                .type(encoding_batched.type())
            )
            encoding = encoding.masked_scatter(
                masks_for_scattering, encoding_batched
            )

        else:
            batch_size = feat_embedding.shape[0]
            max_num = feat_embedding.shape[1]
            hidden_state_size = gru.hidden_size
            encoding = torch.zeros(
                (batch_size, max_num, hidden_state_size)
            ).to(device)

        return encoding
