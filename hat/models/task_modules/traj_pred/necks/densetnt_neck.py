# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict

import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub

from hat.registry import OBJECT_REGISTRY
from hat.utils import qconfig_manager

__all__ = ["DenseTNTNeck"]


@OBJECT_REGISTRY.register
class DenseTNTNeck(nn.Module):
    """DenseTNTNeck neck implementation.

    The vectornet neck for quanting extra features, including
    state vectors.
    """

    def __init__(
        self,
        is_int_infer_model: bool = False,
        use_state_vectors: bool = True,
        backbone_output_all_feats: bool = True,
        use_cuda: bool = True,
    ):
        """Initialize method.

        Args:
            is_int_infer_model: whether the model is for int inference.
            use_state_vectors: whether to use state_vectors.
            backbone_output_all_feats: whether to output all features.
            use_cuda: whether to use cuda.
        """
        super(DenseTNTNeck, self).__init__()
        self.is_int_infer_model = is_int_infer_model
        self.use_state_vectors = use_state_vectors
        self.backbone_output_all_feats = backbone_output_all_feats
        self.use_cuda = use_cuda

        # Build graph.
        if self.use_state_vectors:
            self.state_vector_quant = QuantStub(scale=None)

    def forward(self, data: Dict):
        """Forward.

        Args:
            data: the model input dictionary with the following keys:
            feats (List): the features from backbone.

        Returns:
            results (Dict): features including:
            state_vector_feat (torch.Tensor, [num_obj, num_c_2, 1, 1]): \
                state vector feature.
        """

        if self.backbone_output_all_feats:
            graph_feat, road_feat, _, traj_feat = data["feats"]
            neck_feats = {
                "graph_feat": graph_feat,
                "road_feat": road_feat,
                "traj_feat": traj_feat,
            }
        else:
            neck_feats = {
                "graph_feat": data["feats"],
            }

        if self.use_state_vectors:
            if "state_vectors" in data:
                state_vectors = data["state_vectors"]
            if "behav_state_vectors" in data:
                state_vectors = data["behav_state_vectors"]
            if not self.is_int_infer_model and self.use_cuda:
                state_vectors = state_vectors.cuda()
            state_vectors = self.state_vector_quant(state_vectors)
            neck_feats["state_vector_feat"] = state_vectors

        neck_feats.update(data)
        return neck_feats

    def set_qconfig(self):
        self.qconfig = qconfig_manager.get_default_qat_qconfig()
