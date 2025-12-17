import json

import torch

from hat.registry import build_from_registry


def test_TraceRcnnModule():

    module = dict(
        type="TraceRcnnModule",
        obj_type="vehicle",
        quant_module=dict(
            type="FPNQuant",
            stride_num=4,
        ),
        fpn_desc=dict(
            type="AddDesc",
            per_tensor_desc=[
                json.dumps(
                    dict(
                        task="bifpn",
                        size=[
                            1,
                            32,
                            512 // s,
                            512 // s,
                        ],
                        stride=s,
                    )
                )
                for s in [8, 16, 32, 64]
            ],
        ),
        roi_desc=dict(
            type="AddDesc",
            per_tensor_desc=[
                json.dumps(
                    dict(
                        task="fcos_vehicle_rois",
                        size=[100, 4],
                        class_name=["vehicle"],
                    )
                )
            ],
        ),
        roi_module=dict(
            type="RoIModule",
            output_head_out=True,
            roi_key="pred_bboxes",
            roi_feat_extractor=dict(
                type="MultiScaleRoIAlign",
                output_size=(8, 8),
                feature_strides=[8, 16, 32, 64],
                canonical_level=5,
                aligned=None,
            ),
            head=dict(
                type="ExtSequential",
                modules=[
                    dict(
                        type="RCNNVarGNetShareHead",
                        bn_kwargs={"eps": 1e-5, "momentum": 0.1},
                        roi_out_channel=32,
                        gc_num_filter=128,
                        pw_num_filter=128,
                        pw_num_filter2=128,
                        group_base=8,
                        factor=1,
                        stride=2,
                    ),
                    dict(
                        type="RCNNVarGNetSplitHead",
                        num_fg_classes=1,
                        bn_kwargs={"eps": 1e-5, "momentum": 0.1},
                        with_background=False,
                        in_channel=128,
                        pw_num_filter2=96,
                        with_box_reg=False,
                        with_tracking_feat=False,
                    ),
                ],
            ),
            target=None,
            loss=None,
            postprocess=None,
        ),
    )
    trace_rcnn_module = build_from_registry(module)
    trace_rcnn_module.eval()
    input = {
        "feats_input": [
            torch.randn(1, 32, 512 // stride, 512 // stride)
            for stride in [8, 16, 32, 64]
        ],
        "person_rois_input": [torch.randn(1, 4)],
        "vehicle_rois_input": [torch.randn(1, 4)],
        "rear_rois_input": [torch.randn(1, 4)],
    }

    output = trace_rcnn_module(input)
    assert "rcnn_cls_pred" in output
