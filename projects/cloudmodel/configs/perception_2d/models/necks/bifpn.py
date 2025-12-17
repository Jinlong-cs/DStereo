from hatbc.workflow.trace import GraphTracer

from hat.models.necks.bifpn import BiFPN


def build_neck_bifpn(
    in_stride2channels,
    out_stride2channels,
    num_outs=5,
    neck_stack=3,
    node_name="neck",
):
    in_strides = list(in_stride2channels.keys())
    out_strides = list(out_stride2channels.keys())
    start_level = in_strides.index(list(out_stride2channels.keys())[0])
    bifpn_neck = dict(
        type="BiFPN",
        fpn_name="bifpn_sum",
        in_strides=in_strides,
        out_strides=out_strides,
        stride2channels=in_stride2channels,
        out_channels=out_stride2channels,
        start_level=start_level,
        end_level=-1,
        num_outs=num_outs,
        stack=neck_stack,
        node_name=node_name,
    )
    return dict(
        neck=bifpn_neck,
        stride2channels=out_stride2channels,
    )


# register network in GraphTracer
GraphTracer.register_basic_types(BiFPN)
