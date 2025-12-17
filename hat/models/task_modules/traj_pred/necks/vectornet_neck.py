# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, List, Optional

import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub

from hat.registry import OBJECT_REGISTRY
from hat.utils import qconfig_manager

__all__ = ["VectorNetNeck"]


@OBJECT_REGISTRY.register
class VectorNetNeck(nn.Module):
    """VectorNetNeck neck implementation.

    The vectornet neck for quanting extra features, including
    state vectors.
    """

    def __init__(
        self,
        is_int_infer_model: bool = False,
        use_state_vectors: Optional[List] = True,
    ):
        """Initialize method.

        Args:
            is_int_infer_model: whether the model is for int inference.
            use_state_vectors: whether to use state_vectors.
        """
        super(VectorNetNeck, self).__init__()
        self.is_int_infer_model = is_int_infer_model
        self.use_state_vectors = use_state_vectors

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

        neck_feats = {
            "graph_feat": data["feats"],
        }

        if self.use_state_vectors:
            if "state_vectors" in data:
                state_vectors = data["state_vectors"]
            if "behav_state_vectors" in data:
                state_vectors = data["behav_state_vectors"]
            if not self.is_int_infer_model:
                state_vectors = state_vectors.cuda()
            state_vectors = self.state_vector_quant(state_vectors)
            neck_feats["state_vector_feat"] = state_vectors
            neck_feats["behav_state_vector_feat"] = state_vectors
        return neck_feats

    def set_qconfig(self):
        self.qconfig = qconfig_manager.get_default_qat_qconfig()
