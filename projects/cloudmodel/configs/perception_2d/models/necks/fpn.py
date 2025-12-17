from hatbc.workflow.trace import GraphTracer

from hat.models.necks.fpn import FPN


def build_neck_fpn(in_stride2channels, out_stride2channels, node_name="neck"):
    fpn_neck = dict(
        type="FPN",
        in_strides=list(in_stride2channels.keys()),
        in_channels=list(in_stride2channels.values()),
        out_strides=list(out_stride2channels.keys()),
        out_channels=list(out_stride2channels.values()),
        node_name=node_name,
    )
    return dict(
        neck=fpn_neck,
        stride2channels=out_stride2channels,
    )


# register network in GraphTracer
GraphTracer.register_basic_types(FPN)
