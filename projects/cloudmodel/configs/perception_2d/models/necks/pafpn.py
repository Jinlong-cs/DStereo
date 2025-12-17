from hatbc.workflow.trace import GraphTracer

from hat.models.necks.pafpn import PAFPN


def build_neck_pafpn(
    in_stride2channels, out_stride2channels, node_name="neck"
):
    in_strides = list(in_stride2channels.keys())
    start_level = in_strides.index(list(out_stride2channels.keys())[0])
    pafpn_neck = dict(
        type="PAFPN",
        in_channels=list(in_stride2channels.values()),
        out_channels=out_stride2channels,
        out_strides=list(out_stride2channels.keys()),
        start_level=start_level,
        add_extra_convs="on_output",  # use P5
        num_outs=len(out_stride2channels),
        relu_before_extra_convs=True,
        norm_cfg={"norm_type": "gn", "num_groups": 32},
        node_name=node_name,
    )
    return dict(
        neck=pafpn_neck,
        stride2channels=out_stride2channels,
    )


# register network in GraphTracer
GraphTracer.register_basic_types(PAFPN)
