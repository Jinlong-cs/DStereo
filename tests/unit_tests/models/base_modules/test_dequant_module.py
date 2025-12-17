from collections import OrderedDict

import torch

from hat.registry import build_from_registry


def test_dequant_module():

    config = dict(type="DequantModule", data_names=["bev3d_hm"])

    dequant_module = build_from_registry(config)
    bev3d_hm = torch.randn((2, 1, 512, 960))
    bev3d_cls_id = torch.ones((2, 1, 512, 960))
    cls_id_dtype = bev3d_cls_id.dtype

    pred_dict = OrderedDict()
    pred_dict["bev3d_hm"] = bev3d_hm
    pred_dict["bev3d_cls_id"] = bev3d_cls_id

    pred_dict = dequant_module(pred_dict)
    assert pred_dict["bev3d_hm"].dtype == torch.float32
    assert pred_dict["bev3d_cls_id"].dtype == cls_id_dtype
